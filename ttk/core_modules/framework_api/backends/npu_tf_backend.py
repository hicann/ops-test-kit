#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""NPU TF backend — uses npu_device plugin for Ascend NPU.

Corresponds to NpuTorchBackend (torch_npu). npu_device.open() must be called
BEFORE any TF eager operations (tf.convert_to_tensor etc.), because it swaps
the global TF context to _ContextWithDefaultDevice with NPU as default device.
Calling it after the context is already initialized will not take effect.

open() is therefore deferred to ``set_device(dev_id)`` — called by
``_do_profile`` before input generation (the same hook and timing as
torch_npu.npu.set_device) — rather than __init__: E2E workers are forked
from the parent process, and an open() in the parent leaves the children
with dead GE threads inherited across fork (aclprofStart segfaults). The
parent only queries metadata (has_device / device_type) and must not
touch the device.

as_default() monkey-patches ops.device to _device_consistent_with_context,
which ignores the argument device path and always uses ctx.default_device
(NPU:0).  Therefore to_device / device_scope need not (and cannot) place
tensors on a specific NPU via tf.device — all ops auto-dispatch to NPU.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

from .tf_backend import TfBackend


class NpuTfBackend(TfBackend):
    """NPU TF backend via npu_device plugin."""

    tf_device_type = "NPU"
    _segment_name = "npu"

    _opened_device = None

    def __init__(self):
        # open 延迟到 set_device(dev_id)（_do_profile 在输入生成前统一调用，
        # 与 torch 的 torch_npu.npu.set_device 同一时机、同一语义）。此处不能
        # open：E2E worker 经 fork 拉起，父进程 open 后 C++ GE 线程不随 fork
        # 存活，子进程继承的句柄已坏（aclprofStart 段错误）。父进程仅做元数据
        # 查询（has_device/device_type），不需要设备。
        pass

    def is_npu(self) -> bool:
        return True

    def is_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("npu_device") is not None
        except Exception:
            return False

    def device_count(self) -> int:
        return 1

    def to_device(self, tensor, dev_id=0, preserve_stride=False):
        return self.from_numpy(tensor)

    def synchronize(self, dev_id=0):
        pass

    def device_scope(self, dev_id=0):
        return nullcontext()

    def set_device(self, dev_id: int = 0):
        self._ensure_npu_opened(dev_id)

    def _ensure_npu_opened(self, dev_id):
        if NpuTfBackend._opened_device is None:
            import npu_device

            handle = npu_device.open(dev_id)
            handle.as_default()
            NpuTfBackend._opened_device = dev_id
        elif NpuTfBackend._opened_device != dev_id:
            raise RuntimeError(
                f"npu_device only supports one device; already opened "
                f"{NpuTfBackend._opened_device}, cannot open {dev_id}"
            )

    def device_name(self, dev_id=0):
        try:
            from ttk.utilities.platform import get_npu_hw_info

            from ...dsmi import DSMIInterface

            platform = DSMIInterface().get_chip_info(dev_id).get_complete_platform()
            return get_npu_hw_info(platform).get("short_soc_version", platform)
        except Exception:
            return self._segment_name or "NPU"

    def soc_series(self):
        return self.device_name()

    def supports_graph_mode(self) -> bool:
        return True

    def wrap_eager_callable(self, resolved):
        """Wrap API in tf.function so eager ops dispatch to NPU kernels.

        npu_device registers NPU as a custom device whose execute callback
        only triggers GE graph compilation (and thus real NPU kernel launches)
        for ops inside tf.function.  Bare eager calls fall back to CPU.
        Wrapping with tf.function(autograph=False) preserves single-op
        semantics (no input_signature, TF auto-traces by actual shape) while
        ensuring the op runs on NPU.

        tf.raw_ops.* require keyword args; we generate a wrapper that binds
        positional inputs to the API's tensor parameter names at call time,
        so tf.function tracing passes them as kwargs. Non-tensor params use
        the API's own defaults.
        """
        import inspect

        import tensorflow as tf

        try:
            sig = inspect.signature(resolved)
            param_names = [
                name
                for name, p in sig.parameters.items()
                if name != "name"
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            ]
        except (ValueError, TypeError):
            param_names = []

        if param_names:
            names = param_names

            def wrapper(*args, **kwargs):
                call_kwargs = {}
                for i, name in enumerate(names):
                    if i < len(args) and args[i] is not None:
                        call_kwargs[name] = args[i]
                call_kwargs.update(kwargs)
                return resolved(**call_kwargs)
        else:
            wrapper = resolved

        return tf.function(autograph=False)(wrapper)

    def set_deterministic_level(self, level):
        try:
            import npu_device

            cfg = npu_device.global_options()
            cfg.deterministic = level
            npu_device.global_options()
        except Exception as e:
            logging.warning(f"Failed to set TF deterministic: {e}")
