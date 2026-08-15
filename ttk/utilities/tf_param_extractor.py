#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TF parameter extractor — parse TF op signatures via inspect.signature.

TF ops are standard Python functions, so inspect.signature works directly
(unlike torch which needs aten schema / pyi stub parsing).
"""

import inspect
import logging
from typing import Optional

from ttk.utilities.simple_param_extractor import APIParamInfo, OverloadInfo, ParamInfo
from ttk.utilities.func_dispatch import resolve_callable_str


_TF_SCALAR_TYPES = {"int", "float", "bool", "str", "Number", "Scalar"}

_TF_NON_TENSOR_PARAM_NAMES = frozenset(
    {
        "name",
        "name_",
    }
)


def _is_tensor_param(p: inspect.Parameter, api_name: str) -> bool:
    """Determine if a TF op parameter is a tensor parameter.

    Strategy:
    1. If annotation is tf.Tensor/tf.Variable → True.
    2. If param name is a known non-tensor name (name, axis, dims, etc.) → False.
    3. If annotation is a scalar type (int, float, bool, str) → False.
    4. tf.raw_ops.* convention: all params except 'name' are tensors.
    5. If annotation is Annotated/Any/empty → treat positional params as tensors.
    """
    import tensorflow as tf
    import typing

    ann = p.annotation
    if ann is not inspect.Parameter.empty:
        try:
            if isinstance(ann, type) and issubclass(ann, (tf.Tensor, tf.Variable)):
                return True
        except TypeError:
            pass
        ann_name = getattr(ann, "__name__", str(ann))
        if ann_name in ("int", "float", "bool", "str", "dtype"):
            return False
        if p.name in _TF_NON_TENSOR_PARAM_NAMES:
            return False
        if api_name.startswith("tf.raw_ops.") or api_name.startswith("tensorflow.raw_ops."):
            return True
        return p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    if p.name in _TF_NON_TENSOR_PARAM_NAMES:
        return False
    if api_name.startswith("tf.raw_ops.") or api_name.startswith("tensorflow.raw_ops."):
        return True
    return p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)


def _infer_param_type(p: inspect.Parameter, api_name: str) -> str:
    """Infer type string for a TF op parameter."""
    if _is_tensor_param(p, api_name):
        return "Tensor"
    ann = p.annotation
    if ann is inspect.Parameter.empty:
        return "Number"
    ann_name = getattr(ann, "__name__", str(ann))
    if ann_name in ("int", "float", "bool", "str"):
        return ann_name
    if ann_name in ("list", "tuple"):
        return "Number"
    return ann_name if ann_name in _TF_SCALAR_TYPES else "Number"


def extract_tf_params(api_name: str) -> Optional[APIParamInfo]:
    """Parse a TF op's signature and return APIParamInfo.

    Uses inspect.signature to extract parameters. TF ops (tf.raw_ops.*,
    tf.nn.*, tf.math.*) are standard Python callables.

    Args:
        api_name: e.g. 'tf.raw_ops.Add', 'tf.nn.relu'

    Returns:
        APIParamInfo with a single overload, or None on failure.
    """
    try:
        func = resolve_callable_str(api_name)
    except Exception as e:
        logging.warning(f"Cannot resolve TF api {api_name}: {e}")
        return None

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        logging.warning(f"Cannot get signature for {api_name}: {e}")
        return None

    params = []
    is_raw_ops = api_name.startswith(("tf.raw_ops.", "tensorflow.raw_ops."))
    for name, p in sig.parameters.items():
        is_kw_only = p.kind == inspect.Parameter.KEYWORD_ONLY
        is_var_pos = p.kind == inspect.Parameter.VAR_POSITIONAL
        has_default = p.default is not inspect.Parameter.empty
        is_tensor = _is_tensor_param(p, api_name)

        # tf.raw_ops ops require keyword args even when inspect shows POSITIONAL_OR_KEYWORD
        if is_raw_ops and name != "name":
            is_kw_only = True

        # The 'name' parameter in TF ops is always a string, not a tensor
        if name == "name" and not is_tensor:
            is_kw_only = True

        pi = ParamInfo(
            name=name,
            type=_infer_param_type(p, api_name),
            default=p.default if has_default else None,
            is_optional=has_default or is_var_pos,
            is_keyword_only=is_kw_only,
            is_var_positional=is_var_pos,
        )
        params.append(pi)

    overload = OverloadInfo(params=params)
    info = APIParamInfo(
        api_name=api_name,
        params=params,
        source="tf_inspect",
        overloads=[overload],
    )
    logging.debug(f"Parsed {api_name}: {len(params)} params from tf_inspect")
    return info
