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

import numpy as np

from .comparison import compare
from ...utilities.container_utils import apply_as_list, deep_flatten

__all__ = ["CompareContext", "apply_pre_compare", "try_custom_compare",
           "compare_with_hooks"]


def _copy_goldens(goldens):
    """比对钩子可能就地改写 golden，先复制以免污染后续使用（如 dump）。"""
    copied = []
    for golden in goldens:
        if isinstance(golden, np.ndarray):
            copied.append(golden.copy())
        elif hasattr(golden, "clone"):
            copied.append(golden.clone())
        else:
            copied.append(golden)
    return copied


def _reshape_outputs_for_hooks(outputs, goldens):
    """按 golden 的形状还原输出：比对内部按 flatten 处理，钩子里算子作者按真实形状写。

    实现与 kernel 通路原版逐字一致（上提复用不改语义）。
    """
    reshaped = list(outputs)
    for index, (output, golden) in enumerate(zip(reshaped, goldens)):
        if not isinstance(output, np.ndarray) or not hasattr(golden, "shape"):
            continue
        golden_shape = tuple(golden.shape)
        if output.size == int(np.prod(golden_shape, dtype=np.int64)):
            reshaped[index] = output.reshape(golden_shape)
    return reshaped


def compare_with_hooks(testcase, outputs, goldens, output_dtypes,
                       standards, third_parties, pre_compare, custom_compare):
    """带 TestSpec 钩子（pre_compare / compare）的比对入口，供各测试通路共用。

    【为何要做】
    算子自实现 compare 是某些算子的刚需，而非锦上添花。例如 NonZeroWithValue 的三个
    输出都是静态 max-size buffer，有效长度由 count 给出，尾部预留区在 NPU 上未定义；
    默认整块比对会把这段未定义内存算进判定，通过率恰好退化成输入的非零密度，与内核
    对错无关。它必须靠 Spec.compare 只比有效前缀 [0:N]。
    这套钩子逻辑原先只写在 kernel 通路（npu/op/comparison.py::_compare_mode）里，
    aclnn/e2e 也各自接了，唯独 GEIR 通路没接——同一个算子换条通路验证，自实现 compare
    就静默失效，判定结果不可信。此处上提为公共函数，供各通路共用，避免继续复制。

    【实现逻辑】
    1) 没有钩子、或输出尚未落地成真实数组（占位字符串/None）→ 直接走默认 compare，
       保持原行为；
    2) 有钩子 → 先把输出还原成 golden 的形状（钩子里按真实形状书写），复制一份 golden
       防止钩子就地改写污染后续使用，跑 pre_compare；
    3) 再试 custom compare：返回非 None 即采信其判定，跳过标准判据；返回 None 表示
       算子不接管这次比对，回落到默认 compare。

    【实现效果】
    GEIR 通路获得与 kernel/aclnn/e2e 一致的钩子能力；未声明钩子的算子走的仍是原来那条
    默认路径，行为逐位不变。
    """
    hooks_enabled = testcase is not None and (pre_compare is not None or custom_compare is not None)
    has_runtime_output = outputs and not any(
        isinstance(output, (str, type(None))) for output in outputs
    )
    if not hooks_enabled or not has_runtime_output:
        return compare(outputs, goldens, output_dtypes,
                       standards=standards, third_parties=third_parties)

    mode_outputs = _reshape_outputs_for_hooks(outputs, goldens)
    mode_goldens = _copy_goldens(goldens)
    apply_pre_compare(testcase, mode_outputs, mode_goldens, pre_compare)
    custom_result = try_custom_compare(testcase, mode_outputs, mode_goldens, custom_compare)
    if custom_result is not None:
        precision, logging_data, passed = custom_result
        return precision, logging_data, passed, {}
    return compare(mode_outputs, mode_goldens, output_dtypes,
                   standards=standards, third_parties=third_parties)


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
