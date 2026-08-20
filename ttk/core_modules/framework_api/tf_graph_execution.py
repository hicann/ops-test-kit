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

from .profiling_utils import prepare_device_args
from .tf_graph_network import TfGraphWrapper

WARMUP_COUNT = 1


def _build_input_signature(testcase, dynamic):
    """Build tf.TensorSpec list from testcase tensor_view_shapes/dtypes.

    static (dynamic=False): fixed shapes → corresponds to -c/--const
    dynamic (dynamic=True):  None dimensions → corresponds to -d/--dynamic
    """
    import tensorflow as tf
    from ttk.utilities.dtypes import str_to_tf_dtype

    sig = []
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = testcase.flat_tensor_dtypes
    for shape, dtype_str in zip(flat_shapes, flat_dtypes):
        if shape is None:
            continue
        dims = list(shape) if not dynamic else [None] * len(shape)
        tf_dtype = str_to_tf_dtype(dtype_str)
        if tf_dtype is None:
            logging.warning(f"Cannot map dtype {dtype_str} to tf.dtype, skipping input_signature")
            return None
        sig.append(tf.TensorSpec(dims, tf_dtype))
    return sig if sig else None


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
        (list of numpy arrays, ProfileResult) on success, or ([], None) on failure
    """
    if is_aclgraph:
        logging.warning("aclgraph mode not supported for TF, skipping")
        return [], None

    mode_str = "dynamic" if dynamic else "static"
    logging.info(f"Executing TF graph mode: {mode_str}")

    try:
        args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)

        input_signature = _build_input_signature(testcase, dynamic)
        wrapper = TfGraphWrapper(resolved, input_signature=input_signature, dynamic=dynamic, api_name=testcase.api_name)

        if switches.warmup:
            for _ in range(WARMUP_COUNT):
                wrapper(*args, **kwargs)
            backend.synchronize(dev_id)

        from .profiler import get_profiler

        profiler = get_profiler(
            testcase.api_name, backend, testcase_name=testcase.testcase_name, root_path=switches.root_path,
            dev_id=dev_id,
        )
        run_count = switches.run_time
        result = None
        with profiler:
            for _ in range(run_count):
                result = wrapper(*args, **kwargs)
            backend.synchronize(dev_id)

        perf = profiler.result(backend, run_count)
        result_nps = backend.result_to_numpy(result)
    except Exception as e:
        logging.error(f"TF graph {mode_str} execution failed: {e}", exc_info=True)
        return [], None

    del args, kwargs
    return result_nps, perf
