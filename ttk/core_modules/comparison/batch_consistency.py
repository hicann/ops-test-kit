#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Batch consistency comparison for level-3 cross-testcase output slices.

Groups testcases by batch_consistency_id, extracts output slices per
batch_axis/batch_slice_info, and compares MD5 of corresponding slices
across testcases in the same group.
"""

import hashlib
import logging
from typing import Any, Dict, List, Tuple

import numpy as np


def _is_slice_triple(value):
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(isinstance(item, (int, np.integer)) for item in value)
    )


def _normalize_axis_slices(value):
    if _is_slice_triple(value):
        return (tuple(value),)
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("batch slice group must contain at least one slice")
    if not all(_is_slice_triple(item) for item in value):
        raise ValueError("batch slice must be a (start, stop, step) integer triple")
    return tuple(tuple(item) for item in value)


def _extract_slice(arr, batch_axis, batch_slice_info):
    """Extract all relation samples from an array in deterministic order.

    batch_axis:    nested list of axis positions, e.g. [[0], [1]]
    batch_slice_info: nested list of (start, stop, step) triples, e.g. [[[0,5,1]], [[2,8,1]]]

    Returns the relation samples and an optional unsupported reason.
    """
    if arr is None or not isinstance(arr, np.ndarray):
        return None, "UNSLICEABLE_OUTPUT"
    if batch_axis is None or batch_slice_info is None:
        return None, "MISSING_SLICE"

    try:
        if len(batch_axis) != len(batch_slice_info):
            raise ValueError("batch metadata top-level counts differ")
        active_groups = []
        for axis_group, slice_group in zip(batch_axis, batch_slice_info):
            if slice_group is None:
                continue
            if axis_group is None or len(axis_group) != len(slice_group):
                raise ValueError("batch axis and slice counts differ")
            normalized = tuple(
                (axis, _normalize_axis_slices(axis_slices)) for axis, axis_slices in zip(axis_group, slice_group)
            )
            sample_counts = {len(axis_slices) for _axis, axis_slices in normalized}
            if len(sample_counts) != 1:
                raise ValueError("batch axes contain different relation counts")
            active_groups.append(normalized)

        # Output 0 can only have one relation declaration. Applying independent
        # input-slot declarations to it would validate unrelated output bytes.
        if len(active_groups) != 1:
            raise ValueError("output has zero or multiple active relation groups")

        normalized = active_groups[0]
        if len({axis for axis, _axis_slices in normalized}) != len(normalized):
            raise ValueError("batch relation repeats an output axis")
        sample_count = len(normalized[0][1])
        samples = []
        for sample_index in range(sample_count):
            result = arr
            for axis, axis_slices in normalized:
                if not isinstance(axis, (int, np.integer)):
                    raise TypeError("batch axis must be an integer")
                axis_index = int(axis)
                if axis_index < 0 or axis_index >= result.ndim:
                    raise IndexError(f"batch axis {axis_index} is outside output rank {result.ndim}")
                start, stop, step = axis_slices[sample_index]
                start, stop, step = int(start), int(stop), int(step)
                if step <= 0 or start < 0 or stop <= start or stop > result.shape[axis_index]:
                    raise ValueError(
                        f"invalid output slice {(start, stop, step)} for axis {axis_index} "
                        f"with extent {result.shape[axis_index]}"
                    )
                result = np.take(result, range(start, stop, step), axis=axis_index)
            samples.append(np.ascontiguousarray(result))
        if len({sample.shape for sample in samples}) != 1:
            raise ValueError("batch relation samples produce different output shapes")
        return tuple(samples), None
    except (IndexError, ValueError, TypeError) as e:
        logging.warning(f"Batch slice extraction failed: {e}")
        return None, "INVALID_SLICE"


def _compute_md5(arr) -> str:
    """Compute MD5 hex digest of a numpy array's bytes."""
    if arr is None:
        return "None"
    if isinstance(arr, np.ndarray):
        return hashlib.md5(arr.tobytes()).hexdigest()  # noqa: S324 - non-security data comparison
    return hashlib.md5(str(arr).encode()).hexdigest()  # noqa: S324 - non-security data comparison


def _sliceable_output(testcase, result):
    """Return output 0 only when the worker retained shape-aware data."""
    output_bytes = getattr(result, "output_bytes", None)
    if output_bytes is None:
        prof_result = getattr(testcase, "prof_result", None)
        output_bytes = getattr(prof_result, "output_bytes", None)
    if not isinstance(output_bytes, (list, tuple)) or not output_bytes:
        return None, "NO_OUTPUT"

    output = output_bytes[0]
    if output is None:
        return None, "NO_OUTPUT"
    if not isinstance(output, np.ndarray):
        # Raw bytes have no dtype or shape, so batch_axis cannot be applied.
        return None, "UNSLICEABLE_OUTPUT"
    return output, None


def _member_label(member):
    """Keep sentinel states readable instead of truncating NO_OUTPUT."""
    sample_md5s = member.get("sample_md5s", ())
    if member["status"] == "ok" and len(sample_md5s) > 1:
        digests = ",".join(digest[:8] for digest in sample_md5s)
        return f"{member['testcase']}=[{digests}]"
    digest = member["md5"]
    digest = digest[:8] if member["status"] == "ok" else digest
    return f"{member['testcase']}={digest}"


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
            "members": [{"testcase": name, "md5": hash,
                         "sample_md5s": (hash, ...), "status": status}, ...],
            "pass": bool,
            "supported": bool,
            "reason": optional_reason,
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
        unsupported = []

        for testcase, result in members:
            batch_axis = getattr(testcase, "batch_axis", None)
            batch_slice_info = getattr(testcase, "batch_slice_info", None)

            arr, unsupported_reason = _sliceable_output(testcase, result)
            if unsupported_reason is None:
                sliced, unsupported_reason = _extract_slice(arr, batch_axis, batch_slice_info)
            if unsupported_reason is None:
                sample_md5s = tuple(_compute_md5(sample) for sample in sliced)
                sliced_md5 = sample_md5s[0]
                status = "ok"
            else:
                sample_md5s = ()
                sliced_md5 = unsupported_reason
                status = "unsupported"
                unsupported.append(unsupported_reason)

            member_infos.append(
                {
                    "testcase": testcase.testcase_name,
                    "md5": sliced_md5,
                    "sample_md5s": sample_md5s,
                    "status": status,
                }
            )

        supported = not unsupported
        sample_md5_groups = []
        if supported:
            sample_counts = {len(member["sample_md5s"]) for member in member_infos}
            if len(sample_counts) != 1:
                supported = False
                unsupported.append("RELATION_COUNT_MISMATCH")
            else:
                sample_count = sample_counts.pop()
                sample_md5_groups = [
                    {member["sample_md5s"][index] for member in member_infos} for index in range(sample_count)
                ]
        # Different samples within one case represent different relations. Only
        # hashes at the same relation position are comparable across cases.
        passed = supported and all(len(sample_md5s) == 1 for sample_md5s in sample_md5_groups)
        labels = [_member_label(member) for member in member_infos]
        if not supported:
            logging.warning(
                "Batch consistency unsupported for group %s: %s; reason=%s",
                bcid,
                labels,
                ",".join(sorted(set(unsupported))),
            )
        elif not passed:
            logging.error(f"Batch consistency FAIL for group {bcid}: {labels}")
        else:
            digests = ",".join(next(iter(sample_md5s))[:8] for sample_md5s in sample_md5_groups)
            logging.info(
                f"Batch consistency PASS for group {bcid}: {len(member_infos)} testcases, relation MD5=[{digests}]"
            )

        results.append(
            {
                "batch_consistency_id": str(bcid),
                "members": member_infos,
                "pass": passed,
                "supported": supported,
                "reason": None if supported else ",".join(sorted(set(unsupported))),
            }
        )

    return results
