#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared TestSpec pre-compare and custom-compare execution."""

import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

from ...utilities.container_utils import apply_as_list, deep_flatten

__all__ = ["CompareContext", "apply_pre_compare", "try_custom_compare"]


@dataclass(frozen=True)
class CompareContext:
    """Replay-safe input context available to an explicitly opted-in compare hook."""

    api_name: Optional[str]
    testcase_name: Optional[str]
    input_tensors: Any
    input_scalars: Any
    attributes: Mapping[str, Any]
    csv_fields: Mapping[str, Any]


def _can_customize(outputs, goldens):
    return bool(outputs) and bool(goldens) and not any(
        isinstance(golden, str) for golden in goldens
    )


def _fold_outputs(testcase, outputs, goldens):
    output_dist = getattr(testcase, "output_dist", None)
    if output_dist is None:
        output_dist = getattr(testcase, "output_distribution", None)
    if output_dist:
        return apply_as_list(outputs, output_dist), apply_as_list(goldens, output_dist)
    return list(outputs), list(goldens)


def apply_pre_compare(testcase, outputs, goldens, func):
    """Apply a TestSpec pre_compare function to flat output and golden lists."""
    if func is None or not _can_customize(outputs, goldens):
        return

    nested_outputs, nested_goldens = _fold_outputs(testcase, outputs, goldens)
    transformed = func(*nested_outputs, *nested_goldens)
    if transformed is None:
        return

    output_count = len(nested_outputs)
    expected_count = output_count + len(nested_goldens)
    if not isinstance(transformed, (list, tuple)) or len(transformed) != expected_count:
        actual_count = len(transformed) if hasattr(transformed, "__len__") else "?"
        raise ValueError(
            f"[{testcase.testcase_name}] pre_compare returned len={actual_count}, "
            f"expected {expected_count} (npu={output_count} + golden={len(nested_goldens)})"
        )

    flat_outputs = deep_flatten(transformed[:output_count])
    flat_goldens = deep_flatten(transformed[output_count:])
    if len(flat_outputs) != len(outputs) or len(flat_goldens) != len(goldens):
        raise ValueError(
            f"[{testcase.testcase_name}] pre_compare unfolded len mismatch: "
            f"npu={len(flat_outputs)}/{len(outputs)}, "
            f"golden={len(flat_goldens)}/{len(goldens)} (check tensor-list nesting)"
        )
    outputs[:] = flat_outputs
    goldens[:] = flat_goldens


def _compare_kwargs(testcase):
    kwargs = {}
    for name in (
        "batch_consistency_id",
        "batch_axis",
        "batch_seed",
        "batch_slice_info",
    ):
        value = getattr(testcase, name, None)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _read_only_mapping(value):
    return MappingProxyType(dict(value if value is not None else {}))


def _compare_context(testcase):
    api_name = getattr(testcase, "api_name", None)
    if api_name is None:
        api_name = getattr(testcase, "op_name", None)
    input_tensors = getattr(testcase, "tensors", None)
    if input_tensors is None:
        input_tensors = getattr(testcase, "input_arrays", None)
    return CompareContext(
        api_name=api_name,
        testcase_name=getattr(testcase, "testcase_name", None),
        input_tensors=input_tensors,
        input_scalars=getattr(testcase, "scalars", ()),
        attributes=_read_only_mapping(getattr(testcase, "attributes", None)),
        csv_fields=_read_only_mapping(getattr(testcase, "original_dict", None)),
    )


def try_custom_compare(testcase, outputs, goldens, func):
    """Run a TestSpec compare function and normalize its public result contract."""
    if func is None or not _can_customize(outputs, goldens):
        return None

    nested_outputs, nested_goldens = _fold_outputs(testcase, outputs, goldens)
    kwargs = _compare_kwargs(testcase)
    signature = inspect.signature(func)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs:
        kwargs = {
            name: value for name, value in kwargs.items()
            if name in signature.parameters
        }

    context_param = signature.parameters.get("compare_context")
    if context_param is not None:
        if context_param.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("compare_context must be a keyword or keyword-only parameter")
        kwargs["compare_context"] = _compare_context(testcase)

    result = func(*nested_outputs, *nested_goldens, **kwargs)

    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"[{testcase.testcase_name}] compare must return dict or list[dict]")
    if not result:
        raise ValueError(f"[{testcase.testcase_name}] compare returned empty list")

    items = deep_flatten(result) if any(isinstance(item, (list, tuple)) for item in result) else result
    precisions = []
    passes = []
    log_lines = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"[{testcase.testcase_name}] compare output[{index}] is not a dict"
            )
        missing_keys = [
            key for key in ("pass", "precision")
            if key not in item
        ]
        if missing_keys:
            formatted_keys = ", ".join(repr(key) for key in missing_keys)
            raise ValueError(
                f"[{testcase.testcase_name}] compare output[{index}] "
                f"missing required key(s): {formatted_keys}"
            )
        precision = item["precision"]
        precisions.append(f"{precision}%" if isinstance(precision, (int, float)) else str(precision))
        passes.append(bool(item["pass"]))
        if item.get("error_info"):
            log_lines.append(f"Output {index}: {item['error_info']}")

    log_data = "\n".join(log_lines) + ("\n" if log_lines else "")
    return ",".join(precisions), log_data, all(passes)
