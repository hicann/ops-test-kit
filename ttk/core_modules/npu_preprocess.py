#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Framework-neutral invocation contract for the optional NPU preprocess hook."""

import inspect
from contextlib import nullcontext

from ttk.test_spec import get_spec_attr


def resolve_npu_preprocess(testcase, switches):
    """Resolve a hook without setting up a device when the operator has none."""
    return get_spec_attr(
        testcase.api_name,
        "npu_preprocess",
        switches.plugin_path,
    )


def _parameter_names(plan):
    overload_params = getattr(plan, "overload_params", None)
    if overload_params is not None:
        return {parameter.name for parameter in overload_params}
    return {entry[1] for entry in (getattr(plan, "param_layout", None) or ()) if len(entry) > 1}


def _hook_extras(testcase, switches, plan):
    parameter_names = _parameter_names(plan)
    attributes = dict(getattr(testcase, "attributes", None) or {})
    extra = {name: value for name, value in attributes.items() if name not in parameter_names and name != "context"}
    extra.update(
        {
            "testcase_name": getattr(testcase, "testcase_name", None),
            "short_soc_version": getattr(switches, "short_soc_version", None),
            "tensor_formats": getattr(testcase, "tensor_formats", None),
            "tensor_dtypes": getattr(testcase, "tensor_dtypes", None),
            "scalar_dtypes": getattr(testcase, "scalar_dtypes", None),
            "input_ranges": getattr(testcase, "input_data_ranges", None),
        }
    )
    for name in ("batch_axis", "batch_slice_info", "batch_seed"):
        value = getattr(testcase, name, None)
        if value is not None:
            extra[name] = value
    return extra


def invoke_npu_preprocess(
    testcase,
    switches,
    plan,
    args,
    kwargs,
    *,
    func=None,
    device_scope=None,
):
    """Invoke one context-free hook and enforce its public return contract."""
    if func is None:
        func = resolve_npu_preprocess(testcase, switches)
    if func is None:
        return False

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU_PREPROCESS_FAILURE: cannot inspect TestSpec.npu_preprocess: {exc}") from exc
    if "context" in signature.parameters:
        raise RuntimeError(
            "NPU_PREPROCESS_FAILURE: TestSpec.npu_preprocess must not declare a framework context parameter"
        )

    hook_kwargs = {name: value for name, value in dict(kwargs or {}).items() if name != "context"}
    extra = _hook_extras(testcase, switches, plan)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    for name, value in extra.items():
        if accepts_kwargs or name in signature.parameters:
            hook_kwargs.setdefault(name, value)

    scope = device_scope() if device_scope is not None else nullcontext()
    try:
        with scope:
            result = func(*args, **hook_kwargs)
    except Exception as exc:
        if str(exc).startswith("NPU_PREPROCESS_FAILURE:"):
            raise
        raise RuntimeError(f"NPU_PREPROCESS_FAILURE: {exc}") from exc
    if result is not None:
        raise RuntimeError("NPU_PREPROCESS_FAILURE: TestSpec.npu_preprocess must return None")
    return True
