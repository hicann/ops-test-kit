#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare FA batch relations from a TTK phase-one result CSV and NPU dumps.

The script deliberately has no plugin import.  It consumes the result CSV emitted
by phase one and the raw files in ``NPU_DUMP_PATH`` so operator assets remain
responsible only for constructing equivalent inputs.
"""

import argparse
import ast
import csv
import hashlib
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPORT_HEADERS = (
    "testcase_name",
    "api_name",
    "result",
    "SHA-256",
    "errors",
    "batch_consistency_id",
    "batch_axis",
    "batch_slice_info",
    "batch_seed",
    "case_number",
    "tensor_view_shapes",
    "tensor_dtypes",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "inplace_input_indexes",
)

# Phase two needs the original case description, not a display-only --title CSV.
REQUIRED_RESULT_HEADERS = (
    "testcase_name",
    "api_name",
    "precision_status",
    "tensor_view_shapes",
    "tensor_dtypes",
    "attributes",
    "input_data_ranges",
    "batch_consistency_id",
    "batch_axis",
    "batch_slice_info",
    "batch_seed",
)

BYTE_WIDTHS = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}


@dataclass(frozen=True)
class Relation:
    """One logical Q-output slice selected by a batch relation."""

    axes: Tuple[int, ...]
    slices: Tuple[Tuple[int, int, int], ...]
    seed: int


@dataclass
class Sample:
    """A raw output slice plus the context needed to prove comparability."""

    row: Dict[str, str]
    profile: "BatchProfile"
    relation: Relation
    value: bytes
    shape: Tuple[int, ...]
    dtype: str
    context: Dict[str, Any]

    @property
    def group_key(self):
        # Slice offsets and lengths are intentionally not group identity.  The
        # established contract is seed + logical axis; context validates shape.
        return self.profile.name, self.relation.axes, self.relation.seed


def parse_cell(row: Dict[str, str], name: str, default=None):
    """Parse a TTK FREE_EVAL/DICT result field with an actionable error."""
    value = row.get(name)
    if value is None or not str(value).strip():
        return default
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        testcase_name = row.get("testcase_name", "<unknown>")
        raise ValueError(f"{testcase_name}: invalid {name}: {error}") from error


def is_enabled(row: Dict[str, str]) -> bool:
    value = row.get("is_enabled")
    if value is None or not value.strip():
        return True
    normalized = value.strip().title()
    try:
        return bool(ast.literal_eval(normalized))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"{row.get('testcase_name', '<unknown>')}: invalid is_enabled") from error


def normalize_value(value):
    """Create a deterministic, JSON-like comparison value from nested fields."""
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [normalize_value(item) for item in value]
    return value


def product(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"invalid output shape {tuple(shape)!r}")
        result *= dimension
    return result


def byte_width(dtype: str) -> int:
    normalized = str(dtype).lower()
    if normalized.startswith("torch."):
        normalized = normalized[len("torch.") :]
    width = BYTE_WIDTHS.get(normalized)
    if width is None:
        raise ValueError(f"unsupported raw output dtype {dtype!r}")
    return width


def complete_batch_metadata(row: Dict[str, str]):
    """Return parsed metadata only when all three batch fields are provided.

    A deterministic-only case may set neither field.  Treat a partial triple in
    the same way instead of turning phase two into an accidental execution gate.
    """
    names = ("batch_axis", "batch_slice_info", "batch_seed")
    values = tuple(parse_cell(row, name) for name in names)
    if any(value is None for value in values):
        return None
    return values


def slice_indices(value: Tuple[int, int, int]) -> range:
    """Return the logical positions selected by one positive-step slice."""
    return range(*value)


def slices_overlap(left: Tuple[int, int, int], right: Tuple[int, int, int]) -> bool:
    """Check overlap using selected positions, not contiguous span bounds."""
    left_indices, right_indices = slice_indices(left), slice_indices(right)
    if len(left_indices) > len(right_indices):
        left_indices, right_indices = right_indices, left_indices
    return any(index in right_indices for index in left_indices)


def parse_slice(testcase_name: str, axis: int, value) -> Tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{testcase_name}: invalid logical axis {axis} slice {value!r}")
    if not all(type(item) is int for item in value):
        raise ValueError(f"{testcase_name}: logical slices must contain integers")
    start, stop, step = (int(item) for item in value)
    result = (start, stop, step)
    if step <= 0 or start < 0 or not slice_indices(result):
        raise ValueError(f"{testcase_name}: logical slice must have a non-negative start and non-empty positive step")
    return result


def parse_relations(row: Dict[str, str]) -> Optional[List[Relation]]:
    """Parse the five FA operators' Q-only batch contract from one result row."""
    metadata = complete_batch_metadata(row)
    if metadata is None:
        return None
    batch_axis, batch_slices, batch_seed = metadata
    testcase_name = row["testcase_name"]
    if not (
        isinstance(batch_axis, (tuple, list))
        and isinstance(batch_slices, (tuple, list))
        and isinstance(batch_seed, (tuple, list))
    ):
        raise ValueError(f"{testcase_name}: batch metadata must use nested sequences")
    if not (len(batch_axis) == len(batch_slices) == len(batch_seed)):
        raise ValueError(f"{testcase_name}: batch metadata top-level counts differ")
    if (
        not batch_axis
        or not isinstance(batch_axis[0], (tuple, list))
        or any(type(axis) is not int for axis in batch_axis[0])
        or tuple(batch_axis[0]) not in ((0,), (0, 1))
    ):
        raise ValueError(f"{testcase_name}: Q batch relation must use logical axes (0,) or (0, 1)")
    if batch_slices[0] is None or batch_seed[0] is None:
        raise ValueError(f"{testcase_name}: Q batch relation requires slices and seeds")
    if any(value is not None for value in batch_slices[1:]) or any(value is not None for value in batch_seed[1:]):
        raise ValueError(f"{testcase_name}: only Q batch relations are supported")

    axes = tuple(int(axis) for axis in batch_axis[0])
    slice_groups = batch_slices[0]
    seed_groups = batch_seed[0]
    if not (isinstance(slice_groups, (tuple, list)) and isinstance(seed_groups, (tuple, list))):
        raise ValueError(f"{testcase_name}: Q batch groups must be nested sequences")
    if len(slice_groups) != len(axes) or len(seed_groups) != len(axes):
        raise ValueError(f"{testcase_name}: Q batch groups do not match the declared axes")
    if not slice_groups or not isinstance(slice_groups[0], (tuple, list)):
        raise ValueError(f"{testcase_name}: Q batch relation is empty")
    sample_count = len(slice_groups[0])
    if not sample_count or any(
        not isinstance(group, (tuple, list)) or len(group) != sample_count for group in (*slice_groups, *seed_groups)
    ):
        raise ValueError(f"{testcase_name}: Q batch samples have inconsistent counts")

    relations = []
    for sample_index in range(sample_count):
        slices = []
        seed = None
        for group_index, axis in enumerate(axes):
            slices.append(parse_slice(testcase_name, axis, slice_groups[group_index][sample_index]))
            seed_value = seed_groups[group_index][sample_index]
            if type(seed_value) is not int:
                raise ValueError(f"{testcase_name}: Q relation seed must be an integer")
            if seed is not None and seed != seed_value:
                raise ValueError(f"{testcase_name}: B and S relation seeds must be identical")
            seed = int(seed_value)
        if axes == (0, 1) and len(slice_indices(slices[0])) != 1:
            raise ValueError(f"{testcase_name}: logical (B, S) needs exactly one B per sample")
        relations.append(Relation(axes, tuple(slices), seed))

    for index, relation in enumerate(relations):
        for candidate in relations[index + 1 :]:
            if relation.axes != candidate.axes:
                continue
            first_b, second_b = relation.slices[0], candidate.slices[0]
            if not slices_overlap(first_b, second_b):
                continue
            if relation.axes == (0,):
                raise ValueError(f"{testcase_name}: Q relation samples overlap logical B positions")
            first_s, second_s = relation.slices[1], candidate.slices[1]
            if slices_overlap(first_s, second_s):
                raise ValueError(f"{testcase_name}: Q relation samples overlap logical S positions")
    return relations


def validate_batch_consistency_id(row: Dict[str, str], relations: Sequence[Relation]):
    """Validate full IDs; accept checked length IDs from historical result CSVs."""
    value = parse_cell(row, "batch_consistency_id")
    if value is None:
        raise ValueError(f"{row['testcase_name']}: complete batch metadata needs batch_consistency_id")
    if not isinstance(value, (tuple, list)) or len(value) != 1:
        raise ValueError(f"{row['testcase_name']}: batch_consistency_id must contain one Q relation group")
    id_groups = value[0]
    axes = relations[0].axes
    if not isinstance(id_groups, (tuple, list)) or len(id_groups) != len(axes):
        raise ValueError(f"{row['testcase_name']}: batch_consistency_id axis groups do not match Q axes")
    legacy_id = False
    for group_index, axis in enumerate(axes):
        ids = id_groups[group_index]
        if not isinstance(ids, (tuple, list)) or len(ids) != len(relations):
            raise ValueError(f"{row['testcase_name']}: batch_consistency_id sample count does not match Q relations")
        for relation_id, relation in zip(ids, relations):
            parts = str(relation_id).split("_")
            if len(parts) not in (4, 5):
                raise ValueError(f"{row['testcase_name']}: invalid batch_consistency_id relation {relation_id!r}")
            try:
                actual = tuple(int(part) for part in parts)
            except ValueError as error:
                raise ValueError(
                    f"{row['testcase_name']}: invalid batch_consistency_id relation {relation_id!r}"
                ) from error
            expected = (relation.seed, axis, *relation.slices[group_index])
            if len(parts) == 4:
                selection = slice_indices(relation.slices[group_index])
                expected = (relation.seed, axis, len(selection), selection.step)
                legacy_id = True
            if actual != expected:
                raise ValueError(
                    f"{row['testcase_name']}: batch_consistency_id relation {relation_id!r} does not match {expected!r}"
                )
    if legacy_id:
        logging.warning(
            "%s: reading legacy length-based batch_consistency_id; new phase-one results use "
            "seed_axis_start_stop_step. Comparison groups still use seed + axis.",
            row["testcase_name"],
        )


class BatchProfile:
    """Operator-family geometry and relation-context adapter."""

    def __init__(self, name: str, api_suffixes: Iterable[str], input_count: int, output_dtype=None):
        self.name = name
        self.api_suffixes = frozenset(api_suffixes)
        self.input_count = input_count
        self.output_dtype = output_dtype

    @staticmethod
    def _attributes(row):
        attributes = parse_cell(row, "attributes", {})
        if not isinstance(attributes, dict):
            raise ValueError(f"{row['testcase_name']}: attributes must be a dictionary")
        attributes = dict(attributes)
        aliases = {
            "layoutQOptional": "layout_q",
            "layoutKOptional": "layout_k",
            "layoutKvOptional": "layout_kv",
            "layout_query": "layout_q",
            "layout_key": "layout_k",
            "maxSeqlenQ": "max_seqlen_q",
            "quantMode": "quant_mode",
            "maskMode": "mask_mode",
            "cmpRatio": "cmp_ratio",
            "returnValue": "return_value",
            "softmaxScale": "softmax_scale",
            "oriMaskMode": "ori_mask_mode",
            "cmpMaskMode": "cmp_mask_mode",
            "oriWinLeft": "ori_win_left",
            "oriWinRight": "ori_win_right",
            "topkValueMode": "topk_value_mode",
            "returnSoftmaxLse": "return_softmax_lse",
            "ropeHeadDim": "rope_head_dim",
        }
        for source, target in aliases.items():
            if source in attributes:
                if target in attributes and attributes[target] != attributes[source]:
                    raise ValueError(f"{row['testcase_name']}: conflicting {source}/{target} attributes")
                attributes[target] = attributes.pop(source)
        return attributes

    def _input_row(self, row):
        """Keep ACLNN output slots out of the input-relation context."""
        if not row.get("api_name", "").startswith("aclnn"):
            return row
        shapes = parse_cell(row, "tensor_view_shapes", ())
        dtypes = parse_cell(row, "tensor_dtypes", ())
        output_indexes = parse_cell(row, "output_tensor_indexes", ())
        expected_indexes = (self.input_count, self.input_count + 1)
        if (
            not all(isinstance(value, (tuple, list)) for value in (shapes, dtypes, output_indexes))
            or tuple(output_indexes) != expected_indexes
            or len(shapes) != self.input_count + 2
            or len(dtypes) != len(shapes)
        ):
            raise ValueError(f"{row['testcase_name']}: ACLNN requires output_tensor_indexes={expected_indexes}")
        normalized = dict(row)
        normalized["tensor_view_shapes"] = repr(shapes[: self.input_count])
        normalized["tensor_dtypes"] = repr(dtypes[: self.input_count])
        return normalized

    def _validate_output_metadata(self, row, shape, dtype):
        if not row.get("api_name", "").startswith("aclnn"):
            return
        shapes = parse_cell(row, "tensor_view_shapes")
        dtypes = parse_cell(row, "tensor_dtypes")
        output_shape = shapes[self.input_count]
        if (
            not isinstance(output_shape, (tuple, list))
            or tuple(output_shape) != tuple(shape)
            or str(dtypes[self.input_count]) != dtype
        ):
            raise ValueError(f"{row['testcase_name']}: ACLNN output 0 shape/dtype does not match Q geometry")

    def supports(self, api_name: str) -> bool:
        return api_name.rsplit(".", 1)[-1] in self.api_suffixes

    def make_samples(self, row: Dict[str, str], dump_dir: Path) -> List[Sample]:
        relations = parse_relations(row)
        if relations is None:
            return []
        validate_batch_consistency_id(row, relations)
        input_row = self._input_row(row)
        geometry = self.geometry(input_row)
        output_shape, output_dtype = geometry["output_shape"], geometry["output_dtype"]
        self._validate_output_metadata(row, output_shape, output_dtype)
        output_bytes = self._read_output(row, dump_dir, output_shape, output_dtype)
        samples = []
        for relation in relations:
            value, shape = self._extract(input_row, relation, geometry, output_bytes)
            samples.append(
                Sample(row, self, relation, value, shape, output_dtype, self._context(input_row, relation, geometry))
            )
        return samples

    def geometry(self, row):
        raise NotImplementedError

    def _context(self, row, relation, geometry):
        raise NotImplementedError

    @staticmethod
    def _extract(row, relation, geometry, output_bytes):
        batch_slice = relation.slices[0]
        batch_indices = slice_indices(batch_slice)
        if batch_slice[1] > geometry["batch_size"]:
            raise ValueError(f"{row['testcase_name']}: logical B slice exceeds output batch")
        sequence_slice = relation.slices[1] if relation.axes == (0, 1) else None
        output_shape = geometry["output_shape"]
        bsnd = geometry["layout_q"] == "BSND"
        feature_shape = output_shape[2:] if bsnd else output_shape[1:]
        token_bytes = product(feature_shape) * byte_width(geometry["output_dtype"])
        chunks, lengths = [], []
        for batch_index in batch_indices:
            storage_length = geometry["q_storage_lengths"][batch_index]
            effective_length = geometry["q_effective_lengths"][batch_index]
            if sequence_slice is not None:
                if sequence_slice[1] > effective_length:
                    raise ValueError(f"{row['testcase_name']}: logical S slice exceeds Q effective length")
                positions = slice_indices(sequence_slice)
            else:
                # BSND indexers exclude padding; TND B relations retain the physical prefix span.
                positions = range(effective_length if bsnd else storage_length)
            lengths.append(len(positions))
            offset = batch_index * output_shape[1] if bsnd else geometry["q_prefix"][batch_index]
            if positions.step == 1:
                chunks.append(
                    output_bytes[(offset + positions.start) * token_bytes : (offset + positions.stop) * token_bytes]
                )
            else:
                chunks.extend(
                    output_bytes[(offset + index) * token_bytes : (offset + index + 1) * token_bytes]
                    for index in positions
                )
        if not all(lengths) or (bsnd and len(set(lengths)) != 1):
            raise ValueError(f"{row['testcase_name']}: B relation needs positive, equal BSND Q effective lengths")
        shape = (len(lengths), lengths[0], *feature_shape) if bsnd else (sum(lengths), *feature_shape)
        return b"".join(chunks), shape

    @staticmethod
    def _read_output(row: Dict[str, str], dump_dir: Path, shape: Sequence[int], dtype: str) -> bytes:
        testcase_name = row["testcase_name"]
        if row.get("precision_status", "").upper() != "PASS":
            raise ValueError(f"{testcase_name}: precision_status is not PASS")
        eager_precision = row.get("eager_precision") or ""
        if "NO_OUTPU" in eager_precision:
            raise ValueError(f"{testcase_name}: phase one did not produce a device output")
        # ACLNN honors dump_file_prefix while the E2E dumper retains the testcase
        # name. Prefer the explicit prefix, then retain E2E/cache compatibility.
        prefix = str(row.get("dump_file_prefix") or "").strip()
        dump_names = (prefix, testcase_name) if prefix and prefix != testcase_name else (testcase_name,)
        output_paths = tuple(dump_dir / f"{name}_output_0.bin" for name in dump_names)
        output_path = next((path for path in output_paths if path.is_file()), None)
        if output_path is None:
            expected_paths = ", ".join(str(path) for path in output_paths)
            raise ValueError(f"{testcase_name}: missing output dump; expected one of {expected_paths}")
        output_bytes = output_path.read_bytes()
        expected_bytes = product(shape) * byte_width(dtype)
        if len(output_bytes) != expected_bytes:
            raise ValueError(
                f"{testcase_name}: output bytes={len(output_bytes)}, expected={expected_bytes} for {tuple(shape)!r}/{dtype}"
            )
        return output_bytes


class MlaProfile(BatchProfile):
    """SMLA/QSMLA/MQSMLA output-0 relation handling."""

    PREFIX_ATTRIBUTE_NAMES = (
        "cu_seqlens_q_values",
        "cu_seqlens_ori_kv_values",
        "cu_seqlens_cmp_kv_values",
    )
    STATIC_SEQUENCE_ATTRIBUTE_NAMES = frozenset(("q_datarange", "ori_kv_datarange", "cmp_kv_datarange"))
    BLOCK_COUNT_ATTRIBUTE_NAMES = frozenset(("block_num1", "block_num2"))
    PRIVATE_ATTRIBUTE_NAMES = frozenset(("batch_deterministic_level",))

    def geometry(self, row):
        shapes = parse_cell(row, "tensor_view_shapes")
        dtypes = parse_cell(row, "tensor_dtypes")
        if not shapes or not dtypes or shapes[0] is None:
            raise ValueError(f"{row['testcase_name']}: Q tensor metadata is required")
        output_shape = tuple(shapes[0])
        attributes = self._attributes(row)
        layout_q = attributes.get("layout_q", "BSND")
        if layout_q not in ("BSND", "TND"):
            raise ValueError(f"{row['testcase_name']}: unsupported layout_q={layout_q!r}")
        if len(output_shape) != (4 if layout_q == "BSND" else 3):
            raise ValueError(f"{row['testcase_name']}: Q shape does not match layout_q={layout_q}")
        prefixes = {}
        for name in self.PREFIX_ATTRIBUTE_NAMES:
            value = attributes.get(name)
            if value is not None:
                prefixes[name] = [int(item) for item in value]
        if layout_q == "TND":
            q_prefix = prefixes.get("cu_seqlens_q_values")
            if not q_prefix or q_prefix[0] != 0 or q_prefix[-1] != output_shape[0]:
                raise ValueError(f"{row['testcase_name']}: TND Q prefix must span output T")
            batch_size = len(q_prefix) - 1
        else:
            q_prefix = None
            batch_size = output_shape[0]
        for name, value in prefixes.items():
            if len(value) != batch_size + 1 or value[0] != 0:
                raise ValueError(f"{row['testcase_name']}: {name} must contain B + 1 prefix values")
            is_q_prefix = name == "cu_seqlens_q_values"
            invalid = (
                any(right <= left for left, right in zip(value, value[1:]))
                if is_q_prefix
                else any(right < left for left, right in zip(value, value[1:]))
            )
            if invalid:
                order = "strictly increasing" if is_q_prefix else "non-decreasing"
                raise ValueError(f"{row['testcase_name']}: {name} must be {order}")
        q_lengths = (
            [right - left for left, right in zip(q_prefix, q_prefix[1:])]
            if q_prefix is not None
            else [output_shape[1]] * batch_size
        )
        return {
            "attributes": attributes,
            "input_dtypes": dtypes,
            "batch_size": batch_size,
            "layout_q": layout_q,
            "q_prefix": q_prefix,
            "q_storage_lengths": q_lengths,
            "q_effective_lengths": q_lengths,
            "output_shape": output_shape,
            "output_dtype": self.output_dtype or str(dtypes[0]),
        }

    @staticmethod
    def _normalize_prefix(value: Sequence[int], batch_indices: Sequence[int]):
        normalized = [0]
        for batch_index in batch_indices:
            normalized.append(normalized[-1] + int(value[batch_index + 1]) - int(value[batch_index]))
        return normalized

    def _context(self, row, relation, geometry):
        attributes, batch_size = geometry["attributes"], geometry["batch_size"]
        batch_indices = slice_indices(relation.slices[0])
        sequence_slice = relation.slices[1] if relation.axes == (0, 1) else None
        normalized = {}
        for key, value in attributes.items():
            if key in self.PRIVATE_ATTRIBUTE_NAMES or key in ("B", "T1", "T2", "T3"):
                continue
            if key in self.PREFIX_ATTRIBUTE_NAMES and isinstance(value, (list, tuple)):
                normalized[key] = self._normalize_prefix(value, batch_indices)
            elif key in self.STATIC_SEQUENCE_ATTRIBUTE_NAMES:
                normalized[key] = normalize_value(value)
            elif key in self.BLOCK_COUNT_ATTRIBUTE_NAMES and attributes.get("layout_kv") != "PA_BBND":
                continue
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                normalized[key] = normalize_value([value[index] for index in batch_indices])
            else:
                normalized[key] = normalize_value(value)
        masks = [attributes.get("ori_mask_mode")]
        if attributes.get("has_cmp_kv", True):
            masks.append(attributes.get("cmp_mask_mode"))
        sequence_context = None
        if sequence_slice is not None:
            sequence_indices = slice_indices(sequence_slice)
            sequence_context = (
                [0, len(sequence_indices)] if all(value in (None, 0) for value in masks) else list(sequence_indices)
            )
        return {
            "relation_batch_count": len(batch_indices),
            "relation_sequence_slice": sequence_context,
            "input_dtypes": normalize_value(geometry["input_dtypes"]),
            "attributes": normalized,
        }


class IndexerProfile(BatchProfile):
    """LI_V2/QLI_V2 TopK-index output-0 relation handling."""

    @staticmethod
    def _vector(attributes, name, batch_size, default):
        value = attributes.get(f"{name}_values")
        values = [int(item) for item in value] if value is not None else list(default)
        if len(values) != batch_size:
            raise ValueError(f"{name}_values length does not equal B={batch_size}")
        return values

    @staticmethod
    def _prefix_lengths(attributes, name: str, batch_size: int):
        value = attributes.get(f"{name}_values")
        if value is None:
            return None
        value = [int(item) for item in value]
        if len(value) != batch_size + 1 or value[0] != 0:
            raise ValueError(f"{name}_values must contain B + 1 prefix values")
        if any(right <= left for left, right in zip(value, value[1:])):
            raise ValueError(f"{name}_values must be strictly increasing")
        return [right - left for left, right in zip(value, value[1:])]

    @classmethod
    def _effective_lengths(cls, attributes, name: str, batch_size: int, storage_lengths=None):
        """Resolve optional seqused values without changing physical storage geometry."""
        default = storage_lengths if storage_lengths is not None else [0] * batch_size
        lengths = cls._vector(attributes, name, batch_size, default)
        if any(length < 0 for length in lengths):
            raise ValueError(f"{name}_values must not contain negative lengths")
        if storage_lengths is not None and any(length > capacity for length, capacity in zip(lengths, storage_lengths)):
            raise ValueError(f"{name}_values exceeds its physical storage length")
        return lengths

    def geometry(self, row):
        shapes = parse_cell(row, "tensor_view_shapes")
        dtypes = parse_cell(row, "tensor_dtypes")
        attributes = self._attributes(row)
        if len(shapes) != self.input_count:
            raise ValueError(f"{row['testcase_name']}: expected LI_V2/QLI_V2 direct input slots")
        q_shape = tuple(shapes[0])
        k_shape = tuple(shapes[1])
        layout_q = attributes.get("layout_q", "BSND")
        layout_k = attributes.get("layout_k", "BSND")
        if layout_q == "BSND":
            batch_size, q_extent, q_heads, head_dim = q_shape
            q_prefix = None
            q_storage_lengths = [q_extent] * batch_size
        elif layout_q == "TND":
            q_extent, q_heads, head_dim = q_shape
            q_prefix = [int(item) for item in attributes.get("cu_seqlens_q_values", ())]
            if len(q_prefix) < 2 or q_prefix[0] != 0 or q_prefix[-1] != q_extent:
                raise ValueError(f"{row['testcase_name']}: TND Q prefix must span the Q tensor")
            batch_size = len(q_prefix) - 1
            q_storage_lengths = self._prefix_lengths(attributes, "cu_seqlens_q", batch_size)
            if q_storage_lengths is None or sum(q_storage_lengths) != q_extent:
                raise ValueError(f"{row['testcase_name']}: invalid TND Q prefix")
        else:
            raise ValueError(f"{row['testcase_name']}: unsupported layout_q={layout_q}")
        if layout_k == "BSND":
            if int(k_shape[0]) != batch_size:
                raise ValueError(f"{row['testcase_name']}: key B does not match Q B")
            key_heads = int(k_shape[2])
            k_storage_lengths = [int(k_shape[1])] * batch_size
            block_size = None
        elif layout_k == "TND":
            key_heads = int(k_shape[1])
            k_storage_lengths = self._prefix_lengths(attributes, "cu_seqlens_k", batch_size)
            if k_storage_lengths is None or sum(k_storage_lengths) != int(k_shape[0]):
                raise ValueError(f"{row['testcase_name']}: invalid TND K prefix")
            block_size = None
        elif layout_k == "PA_BBND":
            block_size = int(k_shape[1])
            key_heads = int(k_shape[2])
            k_storage_lengths = None
        else:
            raise ValueError(f"{row['testcase_name']}: unsupported layout_k={layout_k}")
        topk = int(attributes.get("topk", attributes.get("sparse_count")))
        output_shape = (batch_size, q_extent, key_heads, topk) if layout_q == "BSND" else (q_extent, key_heads, topk)
        return {
            "attributes": attributes,
            "input_dtypes": tuple(dtypes),
            "batch_size": int(batch_size),
            "q_heads": int(q_heads),
            "key_heads": key_heads,
            "head_dim": int(head_dim),
            "q_storage_lengths": q_storage_lengths,
            "q_effective_lengths": self._effective_lengths(attributes, "seqused_q", batch_size, q_storage_lengths),
            "k_storage_lengths": k_storage_lengths,
            "k_effective_lengths": self._effective_lengths(attributes, "seqused_k", batch_size, k_storage_lengths),
            "residual": self._vector(attributes, "cmp_residual_k", batch_size, [0] * batch_size),
            "layout_q": layout_q,
            "layout_k": layout_k,
            "block_size": block_size,
            "q_prefix": q_prefix,
            "output_shape": tuple(output_shape),
            "output_dtype": "int32",
            "topk": topk,
        }

    def _context(self, row, relation, geometry):
        batch_indices = slice_indices(relation.slices[0])
        context = {
            key: geometry[key]
            for key in ("layout_q", "layout_k", "q_heads", "key_heads", "head_dim", "topk", "block_size")
        }
        for key in ("q_storage_lengths", "q_effective_lengths", "k_storage_lengths", "k_effective_lengths", "residual"):
            values = geometry[key]
            context[key] = tuple(values[index] for index in batch_indices) if values is not None else None
        if relation.axes == (0, 1):
            context["q_storage_lengths"] = context["q_effective_lengths"] = (len(slice_indices(relation.slices[1])),)
        return {
            **context,
            "input_dtypes": normalize_value(geometry["input_dtypes"]),
            "attributes": {
                key: value
                for key, value in geometry["attributes"].items()
                if key
                not in (
                    "batch_deterministic_level",
                    "seqused_q_values",
                    "seqused_k_values",
                    "cu_seqlens_q_values",
                    "cu_seqlens_k_values",
                    "cmp_residual_k_values",
                )
                and not isinstance(value, (list, tuple, dict))
            },
        }


PROFILES = (
    MlaProfile("SMLA", ("sparse_flash_mla", "sparse_flash_mla_ttk", "aclnnSparseFlashMla"), 18),
    MlaProfile(
        "QSMLA",
        ("quant_sparse_flash_mla", "quant_sparse_flash_mla_ttk", "aclnnQuantSparseFlashMla"),
        21,
        "bfloat16",
    ),
    MlaProfile(
        "MQSMLA",
        ("mixed_quant_sparse_flash_mla", "mixed_quant_sparse_flash_mla_ttk", "aclnnMixedQuantSparseFlashMla"),
        18,
    ),
    IndexerProfile("LI_V2", ("lightning_indexer", "lightning_indexer_v2", "aclnnLightningIndexerV2"), 11),
    IndexerProfile(
        "QLI_V2", ("quant_lightning_indexer", "quant_lightning_indexer_v2", "aclnnQuantLightningIndexerV2"), 13
    ),
)


def find_profile(api_name: str) -> Optional[BatchProfile]:
    return next((profile for profile in PROFILES if profile.supports(api_name)), None)


def report_record(row: Dict[str, str], status: str, digest: str, errors: Sequence[str], case_number: int):
    return {
        **{name: row.get(name, "") for name in REPORT_HEADERS},
        "result": status,
        "SHA-256": digest,
        "errors": "; ".join(errors),
        "case_number": case_number,
    }


def write_report(path: Path, records: Sequence[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.stem}.", suffix=".csv", delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=REPORT_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_result_headers(fieldnames: Optional[Sequence[str]]):
    """Reject display-only phase-one CSVs before replacing a final report."""
    if fieldnames is None:
        raise ValueError("phase-one result CSV has no header")
    missing = [name for name in REQUIRED_RESULT_HEADERS if name not in fieldnames]
    if missing:
        raise ValueError(
            "phase-one result CSV is missing required columns: "
            f"{', '.join(missing)}; do not use a --title-trimmed CSV for phase two"
        )


def compare_result_csv(result_path: Path, final_path: Path, dump_dir: Path):
    with result_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        validate_result_headers(reader.fieldnames)
        rows = list(reader)

    testcase_names = set()
    for row in rows:
        testcase_name = row.get("testcase_name")
        if not testcase_name:
            continue
        if testcase_name in testcase_names:
            raise ValueError(
                "phase-one result CSV has duplicate testcase_name "
                f"{testcase_name!r}; rerun phase one without duplicate or appended rows"
            )
        testcase_names.add(testcase_name)

    samples = []
    records = []
    for row in rows:
        if not row.get("testcase_name") or not is_enabled(row) or complete_batch_metadata(row) is None:
            continue
        profile = find_profile(row.get("api_name", ""))
        if profile is None:
            records.append(report_record(row, "FAIL", "", ("unsupported FA API for unified batch comparison",), 1))
            continue
        try:
            samples.extend(profile.make_samples(row, dump_dir))
        except (OSError, ValueError) as error:
            records.append(report_record(row, "FAIL", "", (str(error),), 1))

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.group_key].append(sample)
    for group_samples in grouped.values():
        group_samples.sort(key=lambda sample: (sample.row["testcase_name"], sample.relation.slices))
        reference = group_samples[0]
        errors = []
        for sample in group_samples[1:]:
            if sample.shape != reference.shape or sample.dtype != reference.dtype:
                errors.append(
                    f"{sample.row['testcase_name']}: relation shape/dtype differs from {reference.row['testcase_name']}"
                )
            if sample.context != reference.context:
                errors.append(
                    f"{sample.row['testcase_name']}: relation context differs from {reference.row['testcase_name']}"
                )
            if sample.value != reference.value:
                errors.append(
                    f"{sample.row['testcase_name']}: raw output bytes differ from {reference.row['testcase_name']}"
                )
        case_number = len({sample.row["testcase_name"] for sample in group_samples})
        status = "FAIL" if errors else ("PASS" if len(group_samples) > 1 else "NOT_APPLICABLE")
        if len(group_samples) == 1:
            errors.append("one relation sample: no same-case or cross-case peer")
        for sample in group_samples:
            records.append(
                report_record(sample.row, status, hashlib.sha256(sample.value).hexdigest(), errors, case_number)
            )

    records.sort(key=lambda record: (record["testcase_name"], record["batch_seed"], record["batch_axis"]))
    write_report(final_path, records)
    failed = sum(record["result"] == "FAIL" for record in records)
    passed = sum(record["result"] == "PASS" for record in records)
    not_applicable = sum(record["result"] == "NOT_APPLICABLE" for record in records)
    return failed == 0, {"pass": passed, "fail": failed, "not_applicable": not_applicable, "records": len(records)}


def build_parser():
    parser = argparse.ArgumentParser(description="Compare FA TTK batch-consistency output dumps.")
    parser.add_argument("--result", required=True, help="Phase-one TTK result CSV")
    parser.add_argument("--final_result", required=True, help="Final relation report CSV")
    return parser


def main():
    args = build_parser().parse_args()
    dump_path = os.environ.get("NPU_DUMP_PATH")
    if not dump_path:
        print("batch consistency comparison failed: NPU_DUMP_PATH is not set")
        return 1
    try:
        passed, summary = compare_result_csv(Path(args.result), Path(args.final_result), Path(dump_path))
    except (OSError, ValueError, csv.Error) as error:
        print(f"batch consistency comparison failed: {error}")
        return 1
    print(
        "batch consistency report: "
        f"PASS={summary['pass']} FAIL={summary['fail']} NOT_APPLICABLE={summary['not_applicable']} "
        f"records={summary['records']} final_result={args.final_result}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
