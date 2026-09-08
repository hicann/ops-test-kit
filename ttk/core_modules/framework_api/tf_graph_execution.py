#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TF graph mode execution via tf.function.

Corresponds to torch's graph_execution.py (torch.compile + torchair).
tf.function is TF's native graph compilation — npu_device handles NPU
dispatch internally, no separate compiler backend needed.
"""

import logging

from ttk.core_modules.deterministic import resolve_deterministic_level
from ttk.core_modules.npu_preprocess import invoke_npu_preprocess

from .profiling_utils import compute_output_md5, finalize_det_status, prepare_device_args
from .tf_graph_network import TfGraphWrapper

WARMUP_COUNT = 1


def _build_input_signature(testcase, dynamic):
    """Build tf.TensorSpec list from testcase tensor_view_shapes/dtypes.

    static (dynamic=False): fixed shapes → corresponds to -c/--const
    dynamic (dynamic=True):  None dimensions → corresponds to -d/--dynamic

    Returns (signature, sig_idx): signature is the TensorSpec list (or None),
    sig_idx lists the flat tensor index each signature entry binds to.
    """
    import tensorflow as tf

    from ttk.utilities.dtypes import str_to_tf_dtype

    from .tf_stateful import get_mutable_param_indexes

    sig = []
    sig_idx = []
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = testcase.flat_tensor_dtypes
    # const 来源的 0-D 位不进 signature：GE infershape 要求其以 Const 节点
    # 进入图，由 TfGraphWrapper 闭包（Python 标量）而非 Placeholder 提供
    const_indexes = getattr(testcase, "const_input_indexes", None) or set()
    # mutable-ref 位不进 signature：tf.function 按 TensorSpec 追踪会把
    # tf.Variable 降级为 SymbolicTensor，state_ops.*scatter 系列随即在
    # ref.handle/_lazy_read 上崩溃；ref 只能经闭包捕获保持 Variable 身份
    # (见 tf_stateful.py 与 TfGraphWrapper._build_kw_function)
    dist = testcase.tensor_list_dist or ()
    mutable_idx = set(get_mutable_param_indexes(testcase.api_name)) if not any(d > 0 for d in dist) else set()
    for idx, (shape, dtype_str) in enumerate(zip(flat_shapes, flat_dtypes)):
        if shape is None or idx in const_indexes or idx in mutable_idx:
            continue
        dims = list(shape) if not dynamic else [None] * len(shape)
        tf_dtype = str_to_tf_dtype(dtype_str)
        if tf_dtype is None:
            logging.warning(f"Cannot map dtype {dtype_str} to tf.dtype, skipping input_signature")
            return None, ()
        sig.append(tf.TensorSpec(dims, tf_dtype))
        sig_idx.append(idx)
    return (sig if sig else None), sig_idx


def _execute_tf_graph(
    testcase,
    backend,
    dev_id,
    switches,
    plan,
    resolved,
    is_tensor_method,
    is_inplace,
    raw_inputs,
    dynamic,
    is_aclgraph=False,
    deterministic_level=None,
):
    """Execute API in TF graph mode via tf.function with profiling.

    Args:
        testcase: TestcaseE2e
        backend: Backend instance (NpuTfBackend or CpuTfBackend)
        dev_id: device ID
        switches: SWITCHES
        plan: ParamPlan
        resolved: resolved API callable
        is_tensor_method: unused placeholder (always False for TF; kept for
            signature parity with torch's _execute_graph so the caller can
            use a single graph_fn variable for both frameworks)
        is_inplace: unused placeholder (always False for TF; same reason)
        raw_inputs: numpy input arrays
        dynamic: True for dynamic shape graph, False for static shape graph

    Returns:
        (list of numpy arrays, ProfileResult, det_status) on success, or ([], None, None) on failure
    """
    if is_aclgraph:
        logging.warning("aclgraph mode not supported for TF, skipping")
        return [], None, None

    mode_str = "dynamic" if dynamic else "static"
    logging.info(f"Executing TF graph mode: {mode_str}")

    try:
        args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)
        if backend.is_npu():
            invoke_npu_preprocess(
                testcase,
                switches,
                plan,
                args,
                kwargs,
                device_scope=lambda: backend.device_scope(dev_id),
            )

        input_signature, sig_idx = _build_input_signature(testcase, dynamic)
        wrapper = TfGraphWrapper(
            resolved,
            input_signature=input_signature,
            dynamic=dynamic,
            api_name=testcase.api_name,
            call_args=args,
            call_kwargs=kwargs,
            sig_idx=sig_idx,
        )

        from .tf_stateful import is_ref_variable

        # mutable-ref 变量经闭包捕获，resource 句柄在 trace 时固化，无法像 eager
        # 路径那样在窗口外换新克隆、窗口内纯引用替换。改为两阶段(基准测试常规
        # 模式)：Phase 1 带复位跑正确性(窗口外)，Phase 2 纯图执行跑计时(窗口内
        # 无复位 op，采集天然干净，无需事后按 op 名剔除)。计时有效性依据：kernel
        # 耗时只与 shape/dtype 相关，与数据值无关，Phase 2 的状态累积不影响计时
        mutable_backups = [(v, v.read_value()) for v in (*args, *kwargs.values()) if is_ref_variable(v)]

        def reset_mutable_refs():
            for var, backup in mutable_backups:
                var.assign(backup)

        profiling_enabled = bool(getattr(switches, "TASK_PROFILING", True))
        if deterministic_level is None:
            deterministic_level = resolve_deterministic_level(switches, testcase)
        deterministic = deterministic_level > 0
        run_count = switches.run_time
        if switches.warmup and profiling_enabled:
            for _ in range(WARMUP_COUNT):
                reset_mutable_refs()
                wrapper(*args, **kwargs)
            backend.synchronize(dev_id)

        from .profiler import ProfilerConfig, get_profiler

        profiler = get_profiler(
            testcase.api_name,
            backend,
            ProfilerConfig(
                testcase_name=testcase.testcase_name,
                root_path=switches.root_path,
                dev_id=dev_id,
                enabled=profiling_enabled,
            ),
        )
        md5_list = []
        result = None
        if mutable_backups:
            # Phase 1 正确性: 每轮复位回初值, 产出比对结果与确定性 MD5
            for _ in range(run_count):
                reset_mutable_refs()
                result = wrapper(*args, **kwargs)
                if deterministic:
                    backend.synchronize(dev_id)
                    md5_list.append(compute_output_md5(backend.result_to_numpy(result)))
            backend.synchronize(dev_id)
            # Phase 2 性能: 窗口内仅图执行
            if profiling_enabled and run_count:
                with profiler:
                    for _ in range(run_count):
                        wrapper(*args, **kwargs)
                    backend.synchronize(dev_id)
        else:
            with profiler:
                for _ in range(run_count):
                    result = wrapper(*args, **kwargs)
                    if deterministic:
                        backend.synchronize(dev_id)
                        run_nps = backend.result_to_numpy(result)
                        md5_list.append(compute_output_md5(run_nps))
                backend.synchronize(dev_id)

        perf = profiler.result(backend, run_count)
        result_nps = backend.result_to_numpy(result)
        det_status = finalize_det_status(md5_list, testcase.testcase_name)
    except Exception as e:
        logging.error(f"TF graph {mode_str} execution failed: {e}", exc_info=True)
        return [], None, None

    del args, kwargs
    return result_nps, perf, det_status
