#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Batch consistency comparison — level=2 cross-testcase slice MD5 check.

Groups testcases by batch_consistency_id, extracts output slices per
batch_axis/batch_slice_info, and compares MD5 of corresponding slices
across testcases in the same group.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _extract_slice(arr, batch_axis, batch_slice_info):
    """Extract a slice from an array using batch_axis and batch_slice_info.

    batch_axis:    nested list of axis positions, e.g. [[0], [1]]
    batch_slice_info: nested list of (start, stop, step) triples, e.g. [[[0,5,1]], [[2,8,1]]]

    Returns the sliced array (views, not copies).
    """
    if arr is None or not isinstance(arr, np.ndarray):
        return arr
    if batch_axis is None or batch_slice_info is None:
        return arr

    result = arr
    try:
        for axis_group, slice_group in zip(batch_axis, batch_slice_info):
            for axis, slc in zip(axis_group, slice_group):
                if axis is None or slc is None:
                    continue
                start, stop, step = slc[0], slc[1], slc[2]
                result = np.take(result, range(start, stop, step), axis=axis)
    except (IndexError, ValueError, TypeError) as e:
        logging.warning(f"Batch slice extraction failed: {e}")
        return arr
    return result


def _compute_md5(arr) -> str:
    """Compute MD5 hex digest of a numpy array's bytes."""
    if arr is None:
        return "None"
    if isinstance(arr, np.ndarray):
        return hashlib.md5(arr.tobytes()).hexdigest()
    return hashlib.md5(str(arr).encode()).hexdigest()


def compare_batch_consistency(
    collected_results: List[Tuple[Any, Any]],
) -> List[Dict]:
    """Compare output slices across testcases sharing the same batch_consistency_id.

    Args:
        collected_results: list of (testcase, result) pairs.

    Returns:
        List of per-group comparison result dicts:
          {
            "batch_consistency_id": id,
            "members": [{"testcase": name, "md5": hash}, ...],
            "pass": bool,
          }
    """
    groups: Dict[Any, List[Tuple[Any, Any]]] = {}
    for testcase, result in collected_results:
        bcid = getattr(testcase, "batch_consistency_id", None)
        if bcid is None:
            continue
        groups.setdefault(bcid, []).append((testcase, result))

    results = []
    for bcid, members in groups.items():
        if len(members) < 2:
            continue

        member_infos = []
        md5_set = set()

        for testcase, result in members:
            batch_axis = getattr(testcase, "batch_axis", None)
            batch_slice_info = getattr(testcase, "batch_slice_info", None)

            output_bytes = getattr(result, "output_bytes", None)
            if output_bytes is None:
                output_bytes = getattr(testcase, "prof_result", None)
                if output_bytes is not None:
                    output_bytes = getattr(output_bytes, "output_bytes", None)

            if output_bytes and isinstance(output_bytes, (list, tuple)):
                arr = output_bytes[0] if len(output_bytes) > 0 else None
                if isinstance(arr, (bytes, bytearray)):
                    sliced_md5 = hashlib.md5(bytes(arr)).hexdigest()
                else:
                    sliced = _extract_slice(arr, batch_axis, batch_slice_info)
                    sliced_md5 = _compute_md5(sliced)
            else:
                sliced_md5 = "NO_OUTPUT"

            member_infos.append({
                "testcase": testcase.testcase_name,
                "md5": sliced_md5,
            })
            md5_set.add(sliced_md5)

        passed = len(md5_set) == 1
        if not passed:
            logging.error(
                f"Batch consistency FAIL for group {bcid}: "
                f"{[m['testcase'] + '=' + m['md5'][:8] for m in member_infos]}"
            )
        else:
            logging.info(
                f"Batch consistency PASS for group {bcid}: "
                f"{len(member_infos)} testcases, MD5={list(md5_set)[0][:8]}"
            )

        results.append({
            "batch_consistency_id": str(bcid),
            "members": member_infos,
            "pass": passed,
        })

    return results
