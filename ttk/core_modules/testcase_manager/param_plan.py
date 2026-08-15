#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""
Parameter plan: overload matching, value coercion, and argument assembly.

Independent of framework_api — depends only on utilities.simple_param_extractor
for ParamInfo/APIParamInfo data types.

Usage:
    plan = ParamPlan(api_name, overload_params, oidx, output_tensor_indexes, attributes)
    args, kwargs, extra = plan.build_args(nested_tensors)
"""

import ast
import logging
import re

from ttk.utilities.dtypes import str_to_torch_dtype


def safe_eval_division(s):
    if "/" not in s:
        return None
    parts = s.split("/")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]) / float(parts[1])
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def score_attr_type_compatibility(overload_params, attr_values):
    """Score how well attribute *values* match their expected param types in an overload.

    Used as a tiebreaker when multiple overloads have the same key-overlap score.
    For each (param, attr_value) pair, a lightweight isinstance check determines
    whether the runtime value is compatible with the declared param type, without
    invoking the full coerce_value pipeline.

    Scoring rules (per matched attribute):
        - numeric types (int/float/Number/Scalar): +1 if value is int/float/bool
        - str type: +1 if value is a str (and not parseable as a number)
        - bool type: +1 if value is bool
        - tuple/list types: +1 if value is tuple/list
        - otherwise: +0 (conservative — don't penalise unknowns)

    Args:
        overload_params: list of ParamInfo for one overload signature.
        attr_values: dict of {param_name: raw_value} from CSV attributes.

    Returns:
        int — number of attributes whose values are type-compatible.
    """
    score = 0
    for p in overload_params:
        if p.is_tensor_like or p.name == "out":
            continue
        if p.name not in attr_values:
            continue
        raw = attr_values[p.name]
        ptype = p.type

        if "|" in ptype:
            score += 1
            continue

        if ptype in ("int",):
            if isinstance(raw, (int, bool)):
                score += 1
        elif ptype in ("float", "Number", "Scalar"):
            if isinstance(raw, (int, float, bool)):
                score += 1
        elif ptype == "bool":
            if isinstance(raw, bool):
                score += 1
        elif ptype == "str":
            if isinstance(raw, str):
                score += 1
        elif "tuple" in ptype or "list" in ptype or ptype == "torch.Size":
            if isinstance(raw, (tuple, list)):
                score += 1
        else:
            score += 1
    return score


def match_overload(api_name, input_tensor_count, attributes=None, tensor_distribution=None, api_info=None):
    """Match testcase to a specific overload signature.

    Matching criteria (in order):
        1. Input tensor count (excluding out) must fit required..total range
        2. Tensor distribution must match (TensorList vs Tensor)
        3. Attribute key overlap — prefer overload whose non-tensor param names
           have the most overlap with provided attributes
        4. (Tiebreaker) Attribute value type compatibility — prefer overload
           whose declared param types best match the runtime attribute values

    Args:
        api_info: **Required**. APIParamInfo from FrameworkApiInfoKeeper.get() or equivalent.

    Returns:
        (overload_params, overload_index) or (None, -1)
    """
    info = api_info
    if info is None:
        return None, -1

    attrs = set(attributes.keys()) if attributes else set()
    attr_values = attributes or {}
    candidates = []

    for oidx, ov_info in enumerate(info.overloads):
        overload_params = ov_info.params
        input_tensors = [p for p in overload_params if p.is_tensor_like and p.name != "out"]
        has_var = any(getattr(p, "is_var_positional", False) for p in input_tensors)
        required = sum(1 for p in input_tensors if not p.is_optional and not getattr(p, "is_var_positional", False))
        scalar_cover = sum(1 for p in input_tensors if p.name in attrs and p.name != "self")
        effective_count = input_tensor_count + scalar_cover
        if has_var:
            if effective_count < required:
                continue
        else:
            total = len(input_tensors)
            if effective_count < required or effective_count > total:
                continue

        if tensor_distribution is not None:
            type_ok = True
            non_var_tensors = [p for p in input_tensors if not getattr(p, "is_var_positional", False)]
            for idx in range(min(input_tensor_count, len(non_var_tensors))):
                if tensor_distribution[idx]:
                    param = non_var_tensors[idx]
                    if not param.is_tensor_list:
                        type_ok = False
                        break
                else:
                    param = non_var_tensors[idx]
                    if param.is_tensor_list:
                        type_ok = False
                        break
            if not type_ok:
                continue

        non_tensor_names = {p.name for p in overload_params if not p.is_tensor_like and p.name != "out"}
        key_score = len(attrs & non_tensor_names)
        candidates.append((key_score, oidx, overload_params))

    if not candidates:
        return None, -1

    candidates.sort(key=lambda x: -x[0])
    top_key_score = candidates[0][0]
    tied = [(ks, oidx, op) for ks, oidx, op in candidates if ks == top_key_score]

    if len(tied) > 1:
        value_scored = [(ks, score_attr_type_compatibility(op, attr_values), oidx, op) for ks, oidx, op in tied]
        value_scored.sort(key=lambda x: -x[1])
        return value_scored[0][3], value_scored[0][2]

    return tied[0][2], tied[0][1]


def coerce_value(raw, target_type):
    """Coerce a raw value from CSV attributes to the expected Python type.

    Supports union types expressed as 'type1|type2' (e.g., 'int|tuple of ints').
    For union types, tries each member type in order and returns the first
    successful coercion. If all fail, raises ValueError with details.
    Single (non-union) types are handled by the original logic unchanged.
    """
    if "|" in target_type:
        errors = []
        for member_type in target_type.split("|"):
            try:
                return coerce_value(raw, member_type)
            except (ValueError, TypeError) as e:
                errors.append(str(e))
        raise ValueError(
            f"Cannot coerce {raw!r} to union type {target_type}: none of the members succeeded: {'; '.join(errors)}"
        )
    if raw is None:
        return None
    if isinstance(raw, str) and raw == "None":
        return None
    if type(raw).__module__ != "builtins":
        return raw
    if target_type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() in ("true", "1")
        return bool(raw)
    if target_type in ("int",):
        if isinstance(raw, int):
            return raw
        if isinstance(raw, (tuple, list)):
            return tuple(int(v) for v in raw)
        if isinstance(raw, str):
            _REDUCTION_STR_TO_INT = {"none": 0, "mean": 1, "sum": 2, "elementwise_mean": 1}
            if raw.lower() in _REDUCTION_STR_TO_INT:
                return _REDUCTION_STR_TO_INT[raw.lower()]
            else:
                torch_obj = str_to_torch_dtype(raw)
                if torch_obj is not None:
                    return torch_obj
        try:
            return int(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot coerce {raw!r} to int: {e}") from e
    if target_type in ("float", "Number", "Scalar"):
        if isinstance(raw, (int, float, bool)):
            return raw
        if isinstance(raw, (tuple, list)):
            return tuple(float(v) for v in raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except (ValueError, TypeError):
                result = safe_eval_division(raw)
                if result is not None:
                    return result
                raise ValueError(f"Cannot coerce {raw!r} to {target_type}: not a numeric value")
        return float(raw)
    if target_type == "str":
        if (
            isinstance(raw, str)
            and len(raw) >= 2
            and ((raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'"))
        ):
            return raw[1:-1]
        return str(raw)
    if target_type in ("ScalarType", "Dtype", "torch.dtype"):
        obj = str_to_torch_dtype(raw)
        if obj is not None:
            return obj
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}: not a torch/torch_npu dtype.")
    _ARRAY_TYPE_RE = re.compile(r"^(int|float|bool)\[(\d*)\]?\??$")
    m = _ARRAY_TYPE_RE.match(target_type)
    if m:
        elem_type = m.group(1)
        if isinstance(raw, (tuple, list)):
            return tuple(coerce_value(v, elem_type) for v in raw)
        if isinstance(raw, bool if elem_type == "bool" else (int if elem_type == "int" else float)):
            return (coerce_value(raw, elem_type),)
        if isinstance(raw, str):
            if not raw or raw in ("[]", "()"):
                return ()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (tuple, list)):
                    return tuple(coerce_value(v, elem_type) for v in parsed)
                return (coerce_value(parsed, elem_type),)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Cannot coerce {raw!r} to {target_type}: {e}") from e
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}")
    if "tuple" in target_type or "list" in target_type or target_type == "torch.Size":
        if isinstance(raw, (tuple, list)):
            return raw
        if isinstance(raw, int):
            return (raw,)
        if isinstance(raw, float):
            return (raw,)
        if isinstance(raw, str):
            if not raw or raw in ("[]", "()"):
                return ()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (tuple, list)):
                    return tuple(parsed)
                return (parsed,)
            except (ValueError, SyntaxError):
                try:
                    return (int(raw),)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Cannot coerce {raw!r} to {target_type}: {e}") from e
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}")
    if isinstance(raw, str) and raw.startswith("torch."):
        import torch

        attr_name = raw.split(".", 1)[1] if "." in raw else raw
        obj = getattr(torch, attr_name, None)
        if obj is not None:
            return obj
    if target_type in ("torch.memory_format", "memory_format", "Layout", "torch.layout") and isinstance(raw, str):
        import torch

        obj = getattr(torch, raw, None)
        if obj is not None:
            return obj
        raise ValueError(f"Cannot coerce {raw!r} to {target_type}: not a torch attribute")
    if target_type in ("Device", "torch.device"):
        return raw
    raise ValueError(f"Cannot coerce {raw!r} to {target_type}: unsupported type")


class ParamPlan:
    """Cached parameter assembly plan — resolved once, reused by profiling/golden/input.

    Decides overload matching and param layout based on testcase configuration
    (tensor count, attributes, output_tensor_indexes, distribution).
    The actual tensor objects (numpy vs device tensor) are plugged in at build time.

    Usage:
        plan = testcase.get_param_plan()
        args, kwargs, _ = plan.build_args(nested_tensors)
        result = resolved_api(*args, **kwargs)
    """

    __slots__ = (
        "api_name",
        "overload_params",
        "overload_index",
        "output_tensor_indexes",
        "attributes",
    )

    def __init__(self, api_name, overload_params, overload_index, output_tensor_indexes, attributes):
        self.api_name = api_name
        self.overload_params = overload_params
        self.overload_index = overload_index
        self.output_tensor_indexes = output_tensor_indexes
        self.attributes = attributes

    def build_args(self, nested_tensors):
        """Assemble (*args, **kwargs) from tensors using this plan.

        Tensor positions are filled from nested_tensors (any type: numpy, torch, etc).
        Scalar positions use values from attributes/defaults, coerced to correct types.

        Returns:
            (args_list, kwargs_dict, extra_attrs) where extra_attrs contains
            attributes not matched by any API parameter name.
        """
        out_indices = set(self.output_tensor_indexes or ())
        from ttk.core_modules.framework_api.framework_detector import is_inplace_tensor_method

        is_inplace = is_inplace_tensor_method(self.api_name) if self.api_name else False
        if is_inplace:
            input_tensors = list(nested_tensors)
        else:
            input_tensors = [t for i, t in enumerate(nested_tensors) if i not in out_indices]
        out_tensors = [nested_tensors[i] for i in sorted(out_indices)]
        out_iter = iter(out_tensors)

        tensor_queue = list(input_tensors)
        attrs = dict(self.attributes) if self.attributes else {}
        args = []
        kwargs = {}

        for param in self.overload_params:
            if param.is_tensor_like and param.name == "out":
                if param.is_tensor_list:
                    collected = [t for t in out_iter if t is not None]
                    if collected:
                        kwargs["out"] = collected
                    elif not param.is_keyword_only:
                        args.append(None)
                else:
                    val = next(out_iter, None)
                    if val is not None:
                        kwargs["out"] = val
                    elif not param.is_keyword_only:
                        args.append(None)
            elif param.is_keyword_only:
                if param.is_tensor_like:
                    if tensor_queue:
                        kwargs[param.name] = tensor_queue.pop(0)
                elif param.name in attrs:
                    kwargs[param.name] = coerce_value(attrs[param.name], param.type)
                elif param.default is not None:
                    val = coerce_value(param.default, param.type)
                    kwargs[param.name] = val
            elif param.is_tensor_like:
                if getattr(param, "is_var_positional", False):
                    args.extend(tensor_queue)
                    tensor_queue.clear()
                elif param.name in attrs and param.name != "self" and not tensor_queue:
                    raw = attrs[param.name]
                    try:
                        args.append(coerce_value(raw, param.type))
                    except (ValueError, TypeError):
                        logging.warning(
                            f"{self.api_name}: scalar fallback for param '{param.name}' "
                            f"(declared type={param.type}, value={raw!r})"
                        )
                        args.append(coerce_value(raw, "Number"))
                elif tensor_queue:
                    val = tensor_queue.pop(0)
                    if param.is_tensor and isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    args.append(val)
                elif param.is_optional:
                    args.append(None)
                else:
                    raise ValueError(
                        f"{self.api_name}: not enough tensors for param '{param.name}' "
                        f"(queue empty, {len(args)} args built so far)"
                    )
            elif param.name in attrs:
                val = coerce_value(attrs[param.name], param.type)
                args.append(val)
            elif param.default is not None:
                args.append(coerce_value(param.default, param.type))
            else:
                args.append(None)

        param_names = {p.name for p in self.overload_params}
        extra_attrs = {k: v for k, v in attrs.items() if k not in param_names}
        return args, kwargs, extra_attrs


def build_positional_args(
    api_name, nested_tensors, attributes, output_tensor_indexes, tensor_distribution=None, api_info=None
):
    """Build (positional_args, kwargs) based on matched API signature.

    Convenience wrapper — creates a one-shot ParamPlan.
    For repeated calls (profiling/golden), prefer testcase.get_param_plan() + plan.build_args().

    Args:
        api_info: **Required**. APIParamInfo for the target API.

    Returns:
        (args_list, kwargs_dict, matched_overload_index) or raises ValueError
    """
    overload_params, oidx = match_overload(
        api_name,
        input_tensor_count=sum(1 for i, _ in enumerate(nested_tensors) if i not in set(output_tensor_indexes or ())),
        attributes=attributes,
        tensor_distribution=tensor_distribution,
        api_info=api_info,
    )
    if overload_params is None:
        raise ValueError(
            f"Cannot match overload for {api_name} "
            f"with {sum(1 for i, _ in enumerate(nested_tensors) if i not in set(output_tensor_indexes or ()))} input tensors"
        )

    plan = ParamPlan(api_name, overload_params, oidx, output_tensor_indexes, attributes)
    args, kwargs, _ = plan.build_args(nested_tensors)
    return args, kwargs, oidx
