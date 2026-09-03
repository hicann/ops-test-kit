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

    def __init__(self, api_func, input_signature=None, dynamic=False, api_name=None, call_args=None, call_kwargs=None):
        import tensorflow as tf

        self._api_func = api_func
        self._dynamic = dynamic
        self._api_name = api_name
        self._input_signature = input_signature
        self._param_names = self._extract_tensor_param_names(api_func, api_name)

        if self._param_names and input_signature is not None:
            self._tf_func = self._build_kw_function(
                api_func, self._param_names, input_signature, call_args, call_kwargs
            )
        elif input_signature is not None:
            self._tf_func = tf.function(api_func, input_signature=input_signature, autograph=False)
            self._tf_func.get_concrete_function()
        else:
            self._tf_func = tf.function(api_func, autograph=False)

    @staticmethod
    def _build_kw_function(api_func, param_names, input_signature, call_args=None, call_kwargs=None):
        """Build tf.function with explicit named params matching input_signature.

        tf.raw_ops.* require keyword args; we generate a wrapper with explicit
        parameter names (matching param_names) so input_signature binds correctly,
        and the wrapper forwards them as kwargs to the API. Non-tensor params
        (scalars from attributes) are baked into the closure as Python values —
        tf.function traces them as Const nodes, which is also what GE infershape
        passes (e.g. CombinedNonMaxSuppression) require.
        """
        import tensorflow as tf

        n_sig = len(input_signature)
        tensor_names = param_names[:n_sig]
        scalar_values = {}
        call_args = list(call_args or [])
        for i, name in enumerate(param_names):
            if i < n_sig:
                continue
            if i < len(call_args):
                scalar_values[name] = call_args[i]
            elif call_kwargs and name in call_kwargs:
                scalar_values[name] = call_kwargs[name]

        def wrapper(*args):
            kwargs = dict(zip(tensor_names, args))
            kwargs.update(scalar_values)
            return api_func(**kwargs)

        tf_func = tf.function(wrapper, input_signature=input_signature, autograph=False)
        tf_func.get_concrete_function()
        return tf_func

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
            n_sig = len(self._input_signature) if self._input_signature else len(call_kwargs)
            call_kwargs = {k: v for i, (k, v) in enumerate(call_kwargs.items()) if i < n_sig}
            return self._tf_func(*call_kwargs.values())
        return self._tf_func(*args, **kwargs)
