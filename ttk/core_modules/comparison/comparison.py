#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Standard Packages
import gc
from typing import List, Tuple, Union

import numpy as np

from ...utilities import get

# Third-party Packages
from .registry import FAIL_REASONS, ComparisonRegister

__all__ = ["compare"]


def compare(
    outputs: Union[Tuple[np.ndarray], List[np.ndarray]],
    goldens: Union[Tuple[np.ndarray], List[np.ndarray]],
    output_dtypes: tuple,
    *,
    standards,
    third_parties=None,
) -> (str, str, bool, dict):
    """逐输出比对。standards: resolve_tolerance 产出（与 outputs 同长）。
    返回 (precision_str, log, is_pass, metrics_dict)。"""
    # 触发 @register_comparison 装饰器（注册到 ComparisonRegister.registry）
    from .binary_equal import BinaryComparison  # noqa: F401
    from .cosine_similarity import CosineSimilarityComparison  # noqa: F401
    from .cross_check import CrossCheckComparison  # noqa: F401
    from .is_close import CloseComparison  # noqa: F401
    from .quant import QuantComparison  # noqa: F401
    from .re_quantize import ReQuantizeComparison  # noqa: F401
    from .stat_rel_err import StatRelErrComparison  # noqa: F401

    if not outputs:
        return ("UNKNOWN", "Output or Golden data is empty, compare result UNKNOWN\n", False, {})

    if third_parties and not isinstance(outputs[0], str) and len(third_parties) < len(outputs):
        return (
            "COMPARE_FAILURE",
            f"third_parties({len(third_parties)}) < outputs({len(outputs)})",
            False,
            {"reason": FAIL_REASONS["third_party_count_mismatch"]},
        )

    total_precision, total_pass, total_log, total_metrics = [], [], "", {}
    for idx, data_pair in enumerate(zip(outputs, goldens)):
        output, golden = data_pair
        if isinstance(output, str):
            total_precision.append(f"{output}")
            total_pass.append(_filter_fake_fail(output))
            continue
        if isinstance(golden, str):
            total_precision.append(f"{golden}")
            total_pass.append(_filter_fake_fail(golden))
            continue
        if golden is None:
            total_precision.append("SUPPRESSED")
            total_pass.append(True)
            continue
        if output is None:
            total_precision.append("NO_OUTPUT")
            total_pass.append(False)
            continue

        token = standards[idx].token
        comparison = ComparisonRegister.registry.get(token.lower())
        if comparison is None:
            raise ValueError(f"Comparison standard [{token}] is not recognized.")
        c = comparison(
            output,
            golden,
            idx,
            get(output_dtypes, idx),
            standards[idx].params,
            third_party=get(third_parties, idx) if third_parties else None,
        )

        precision, log, is_pass, metrics = c.compare()
        total_precision.append(precision)
        total_pass.append(is_pass)
        total_log += log
        total_metrics[idx] = metrics
        gc.collect()

    return ",".join(total_precision), total_log, all(total_pass), total_metrics


def _filter_fake_fail(output: str):
    if output in (
        "DYN_OFF",
        "STC_OFF",
        "CST_OFF",
        "BIN_OFF",
        "DYN_UNSUPPORTED",
        "STC_UNSUPPORTED",
        "CST_UNSUPPORTED",
        "BIN_UNSUPPORTED",
        "DYN_OPERATOR_NOT_FOUND",
        "STC_OPERATOR_NOT_FOUND",
        "CST_OPERATOR_NOT_FOUND",
        "BIN_OPERATOR_NOT_FOUND",
        "SUPPRESSED",
        "DYN_INPUT_MISSING",
    ):
        return True
    else:
        return False
