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
import logging
from contextlib import nullcontext

from ttk.core_modules.deterministic import BATCH_RELATION_FIELDS, batch_relation_kwargs
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
    extra = {
        name: value
        for name, value in attributes.items()
        if name not in parameter_names and name not in (*BATCH_RELATION_FIELDS, "context")
    }
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
    # Batch relation fields form one contract.  Do not expose a partial triple
    # to an operator hook when this is only a deterministic execution case.
    extra.update(batch_relation_kwargs(testcase))
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
    """Invoke one context-free hook and enforce its None-or-device-tensor-map contract."""
    if func is None:
        func = resolve_npu_preprocess(testcase, switches)
    if func is None:
        return None

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
    if result is None:
        return None
    if not isinstance(result, dict):
        raise RuntimeError(
            "NPU_PREPROCESS_FAILURE: TestSpec.npu_preprocess must return None or dict{param_name: tensor}"
        )
    for name, tensor in result.items():
        if not isinstance(name, str) or tensor is None or not hasattr(tensor, "shape") or not hasattr(tensor, "device"):
            raise RuntimeError(f"NPU_PREPROCESS_FAILURE: returned slot '{name}' is not a tensor")
    return result


def _tensor_param_bindings(testcase, plan, args, kwargs):
    """Map tensor parameter names to flat slots and their built call locations."""
    top_slots = []
    flat_idx = 0
    for shape in testcase.tensor_view_shapes or ():
        count = len(shape) if shape and isinstance(shape[0], (tuple, list)) else 1
        top_slots.append(tuple(range(flat_idx, flat_idx + count)))
        flat_idx += count

    bindings = {}
    for name, entries in plan.tensor_param_bindings(testcase.tensor_view_shapes or ()).items():
        if len(entries) != 1:
            # A named *args region cannot be addressed by one dict key.
            top_index, container, key, _is_vararg = entries[0]
            bindings[name] = (top_slots[top_index], container, key, True)
            continue
        top_index, container, key, is_vararg = entries[0]
        bindings[name] = (top_slots[top_index], container, key, is_vararg)
    return bindings


def _tensor_device(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _tensor_device(item)
            if device is not None:
                return device
        return None
    return getattr(value, "device", None)


def apply_npu_preprocess_result(testcase, plan, args, kwargs, returned):
    """Patch device-side call arguments materialized by npu_preprocess."""
    if returned is None:
        return []
    bindings = _tensor_param_bindings(testcase, plan, args, kwargs)
    dyn_indexes = set(testcase.dyn_input_slot_indexes)

    dynamic_names = {
        name
        for name, (slots, _container, _key, is_vararg) in bindings.items()
        if not is_vararg and len(slots) == 1 and slots[0] in dyn_indexes
    }
    for name in returned:
        binding = bindings.get(name)
        if binding is None or binding[3] or len(binding[0]) != 1 or binding[0][0] not in dyn_indexes:
            raise RuntimeError(f"NPU_PREPROCESS_FAILURE: returned slot '{name}' is not declared as a supported -1 slot")
    missing = sorted(dynamic_names - set(returned))
    if missing:
        raise RuntimeError(f"NPU_PREPROCESS_FAILURE: dynamic slots {missing} not materialized by npu_preprocess")
    mapped_dyn = {
        slots[0] for slots, _container, _key, is_vararg in bindings.values() if not is_vararg and len(slots) == 1
    }
    unmapped = sorted(dyn_indexes - mapped_dyn)
    if unmapped:
        raise RuntimeError(
            "NPU_PREPROCESS_FAILURE: dynamic slot(s) "
            f"{unmapped} have no supported API tensor parameter (TensorList and *args slots are unsupported)"
        )

    target_device = _tensor_device(args) or _tensor_device(tuple(kwargs.values()))
    patched = []
    for name, tensor in returned.items():
        returned_device = _tensor_device(tensor)
        if target_device is not None and returned_device is not None and str(returned_device) != str(target_device):
            raise RuntimeError(
                f"NPU_PREPROCESS_FAILURE: returned slot '{name}' is on {returned_device}, expected {target_device}"
            )
        slots, container, key, _is_vararg = bindings[name]
        if container == "args":
            args[key] = tensor
        else:
            kwargs[key] = tensor
        patched.append((slots[0], name, tensor))

    logging.info(
        "[%s] npu_preprocess resolved dynamic shapes: %s",
        getattr(testcase, "testcase_name", ""),
        {name: tuple(tensor.shape) for _idx, name, tensor in patched},
    )
    return patched


def _aclnn_tensor_param_bindings(testcase, plan):
    """Map ACLNN tensor parameter names to top-level and flat testcase slots."""
    bindings = {}
    top = 0
    flat = 0
    for kind, name, _acl_type, _default in plan.param_layout:
        if kind != "tensor":
            continue
        shape = testcase.tensor_view_shapes[top]
        count = len(shape) if shape and isinstance(shape[0], (tuple, list)) else 1
        bindings[name] = (top, tuple(range(flat, flat + count)))
        top += 1
        flat += count
    return bindings


def apply_aclnn_npu_preprocess_result(testcase, plan, returned):
    """Materialize dynamic ACLNN inputs on host after their NPU preprocess."""
    if returned is None:
        return []

    import torch

    from ttk.utilities import str_to_torch_dtype

    bindings = _aclnn_tensor_param_bindings(testcase, plan)
    dyn_indexes = set(testcase.dyn_input_slot_indexes)
    dynamic_names = {name for name, (_top, slots) in bindings.items() if len(slots) == 1 and slots[0] in dyn_indexes}
    for name in returned:
        binding = bindings.get(name)
        if binding is None or len(binding[1]) != 1 or binding[1][0] not in dyn_indexes:
            raise RuntimeError(
                f"NPU_PREPROCESS_FAILURE: returned ACLNN slot '{name}' is not declared as a supported -1 slot"
            )
    missing = sorted(dynamic_names - set(returned))
    if missing:
        raise RuntimeError(f"NPU_PREPROCESS_FAILURE: dynamic ACLNN slots {missing} not materialized")
    mapped_dyn = {slots[0] for _top, slots in bindings.values() if len(slots) == 1}
    if dyn_indexes - mapped_dyn:
        raise RuntimeError("NPU_PREPROCESS_FAILURE: dynamic ACLNN slot inside TensorList is not supported")

    dynamic_tensors = {}
    patched = []
    for name, tensor in returned.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"NPU_PREPROCESS_FAILURE: returned ACLNN slot '{name}' is not a torch.Tensor")
        top, slots = bindings[name]
        flat = slots[0]
        host_tensor = tensor.detach().to("cpu").contiguous()
        expected_dtype = str_to_torch_dtype(testcase.flat_tensor_dtypes[flat])
        if expected_dtype is None or host_tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"NPU_PREPROCESS_FAILURE: returned ACLNN slot '{name}' has dtype {host_tensor.dtype}, "
                f"expected {expected_dtype}"
            )
        dynamic_tensors[flat] = host_tensor
        patched.append((flat, name, host_tensor))

    # Keep the CPU-side None marker intact.  ACLNN needs a host copy only to
    # create its device tensor; golden, dump, and third-party paths must not
    # mistake a NPU-derived metadata buffer for a user input.
    testcase._aclnn_dynamic_tensors = dynamic_tensors
    logging.info(
        "[%s] npu_preprocess resolved ACLNN dynamic shapes: %s",
        getattr(testcase, "testcase_name", ""),
        {name: tuple(tensor.shape) for _idx, name, tensor in patched},
    )
    return patched
