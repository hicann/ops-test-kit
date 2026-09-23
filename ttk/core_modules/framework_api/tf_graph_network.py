#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TfGraphWrapper — tf.function wrapper for TF graph mode testing.

Corresponds to torch's GraphNetwork (torch.nn.Module) + torch.compile.
tf.function is TF's native graph compilation mechanism — no separate
compiler backend needed (npu_device handles NPU dispatch internally).
"""


class TfGraphWrapper:
    """Wrap a TF API callable in tf.function for graph-mode execution.

    For static shape (-c): input_signature with fixed TensorSpec shapes.
    For dynamic shape (-d): input_signature with None dimensions.

    tf.raw_ops.* ops require keyword args; the wrapper binds positional
    inputs to the API's tensor parameter names via inspect.signature,
    so tf.function tracing passes them as kwargs.
    """

    def __init__(
        self,
        api_func,
        input_signature=None,
        dynamic=False,
        api_name=None,
        call_args=None,
        call_kwargs=None,
        sig_idx=None,
    ):
        import tensorflow as tf

        from .tf_stateful import get_resource_param_names

        self._api_func = api_func
        self._dynamic = dynamic
        self._api_name = api_name
        self._input_signature = input_signature
        self._param_names = self._extract_tensor_param_names(api_func, api_name)
        self._sig_param_names = None
        # tf.raw_ops.Resource* 的 DT_RESOURCE 位: trace 内 tf.Variable 须显式
        # 取 .handle(Variable 在 op 边界被解引用为值, resource 校验必挂)
        self._resource_names = set(get_resource_param_names(api_name)) if api_name else set()

        if self._param_names and input_signature is not None:
            self._tf_func, sig_param_names = self._build_kw_function(
                api_func,
                self._param_names,
                input_signature,
                call_args,
                call_kwargs,
                sig_idx,
                resource_names=self._resource_names,
            )
            self._sig_param_names = set(sig_param_names)
        elif input_signature is not None:
            self._tf_func = tf.function(api_func, input_signature=input_signature, autograph=False)
            self._tf_func.get_concrete_function()
        else:
            self._tf_func = tf.function(api_func, autograph=False)

    @staticmethod
    def _build_kw_function(
        api_func, param_names, input_signature, call_args=None, call_kwargs=None, sig_idx=None, resource_names=None
    ):
        """Build tf.function with explicit named params matching input_signature.

        tf.raw_ops.* require keyword args; we generate a wrapper with explicit
        parameter names (matching param_names) so input_signature binds correctly,
        and the wrapper forwards them as kwargs to the API. Params outside the
        signature are baked into the closure: Python scalars trace as Const
        nodes (what GE infershape requires), tf.Variable keeps its identity via
        resource capture (what state_ops mutable-ref APIs require — a Variable
        bound to a TensorSpec slot degrades to a SymbolicTensor and crashes on
        ref._lazy_read).

        sig_idx maps signature entry j to param_names[sig_idx[j]] (flat tensor
        index == tensor param position; mutable/const positions are excluded
        from the signature by the caller). Falls back to binding the first
        len(input_signature) params when absent.

        resource_names (tf.raw_ops.Resource* 的 DT_RESOURCE 位): call 时把
        tf.Variable 换成 .handle; 无输出 raw op 在 trace 下返回 Operation,
        tf.function 拒绝 — 降级为 None(eager 下本就返回 None)。
        """
        import tensorflow as tf

        call_args = list(call_args or [])
        if sig_idx and all(i < len(param_names) for i in sig_idx):
            tensor_names = [param_names[i] for i in sig_idx]
            sig_pos = set(sig_idx)
        else:
            tensor_names = param_names[: len(input_signature)]
            sig_pos = set(range(len(input_signature)))
        closure_values = {}
        for i, name in enumerate(param_names):
            if i in sig_pos:
                continue
            if i < len(call_args):
                closure_values[name] = call_args[i]
            elif call_kwargs and name in call_kwargs:
                closure_values[name] = call_kwargs[name]

        def wrapper(*args):
            kwargs = dict(zip(tensor_names, args))
            kwargs.update(closure_values)
            if resource_names:
                for name in resource_names:
                    v = kwargs.get(name)
                    if isinstance(v, tf.Variable):
                        kwargs[name] = v.handle
            r = api_func(**kwargs)
            return None if isinstance(r, tf.Operation) else r

        tf_func = tf.function(wrapper, input_signature=input_signature, autograph=False)
        tf_func.get_concrete_function()
        return tf_func, tensor_names

    @staticmethod
    def _extract_tensor_param_names(api_func, api_name):
        """Extract tensor parameter names from the API signature."""
        import inspect

        try:
            sig = inspect.signature(api_func)
            names = []
            for name, p in sig.parameters.items():
                if name == "name":
                    continue
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    names.append(name)
                elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                    break
            return names if names else None
        except (ValueError, TypeError):
            return None

    def __call__(self, *args, **kwargs):
        if self._param_names:
            call_kwargs = {}
            for i, name in enumerate(self._param_names):
                if i < len(args) and args[i] is not None:
                    call_kwargs[name] = args[i]
                elif name in kwargs and kwargs[name] is not None:
                    call_kwargs[name] = kwargs[name]
            if self._sig_param_names is not None:
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in self._sig_param_names}
            else:
                n_sig = len(self._input_signature) if self._input_signature else len(call_kwargs)
                call_kwargs = {k: v for i, (k, v) in enumerate(call_kwargs.items()) if i < n_sig}
            return self._tf_func(*call_kwargs.values())
        return self._tf_func(*args, **kwargs)
