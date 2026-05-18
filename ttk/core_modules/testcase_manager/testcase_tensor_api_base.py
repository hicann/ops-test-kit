#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""
Shared base class for aclnn (op_api) and framework_api (e2e) testcase structures.
"""


__all__ = ["TensorApiTestcaseBase"]


from .testcase_base import TestcaseBase
from ...utilities import get, shape_stride
from ...utilities.container_utils import (
    infer_list_distribution_from_nesting, flatten_nested_sequence
)


class TensorApiTestcaseBase(TestcaseBase):
    """Base class with shared tensor property accessors, distribution inference,
    field normalization, and per-index accessors.

    Inherited by:
      - TestcaseAclnn (aclnn)
      - TestcaseE2e (framework_api / e2e)

    Not used by:
      - TestcaseOp (kernel-level, no tensor fields)
    """

    __slots__ = (
        "api_name",
        "tensor_view_shapes",
        "tensor_dtypes",
        "tensor_formats",
        "tensor_storage_shapes",
        "tensor_view_offsets",
        "tensor_view_strides",
        "output_tensor_indexes",
        "np_storages",
        "prof_result",
        "_inferred_tensor_list_dist",
        "_param_plan_cache",
        "_flat_tensor_view_shapes",
        "_flat_tensor_dtypes",
        "_flat_tensor_formats",
        "_flat_tensor_storage_shapes",
        "_flat_tensor_view_offsets",
        "_flat_tensor_view_strides",
        "_flat_input_data_ranges",
        "_flat_precision_tolerances",
        "_flat_absolute_precision",
        "_pure_output_indexes",
        "_is_torch_dtype_support",
    )

    _normalized_tensor_fields = (
        'tensor_dtypes', 'tensor_formats', 'tensor_view_offsets',
        'tensor_view_strides', 'tensor_storage_shapes',
    )

    def __init__(self):
        super().__init__()
        self.api_name = None
        self.tensor_view_shapes = None
        self.tensor_dtypes = None
        self.tensor_formats = ()
        self.tensor_storage_shapes = ()
        self.tensor_view_offsets = ()
        self.tensor_view_strides = ()
        self.output_tensor_indexes = ()
        self.np_storages = None
        self.prof_result = None
        self._inferred_tensor_list_dist = None
        self._param_plan_cache = None
        self._flat_tensor_view_shapes = None
        self._flat_tensor_dtypes = None
        self._flat_tensor_formats = None
        self._flat_tensor_storage_shapes = None
        self._flat_tensor_view_offsets = None
        self._flat_tensor_view_strides = None
        self._flat_input_data_ranges = None
        self._flat_precision_tolerances = None
        self._flat_absolute_precision = None
        self._pure_output_indexes = None
        self._is_torch_dtype_support = None

    @property
    def op_name(self):
        return self.api_name

    # ========== Distribution ==========

    def _get_tensor_list_distribution(self):
        if self._inferred_tensor_list_dist is None and self.tensor_view_shapes:
            self._inferred_tensor_list_dist = infer_list_distribution_from_nesting(
                self.tensor_view_shapes)
        return self._inferred_tensor_list_dist or ()

    # ========== Flat properties ==========

    @property
    def flat_tensor_view_shapes(self):
        if self._flat_tensor_view_shapes is not None:
            return self._flat_tensor_view_shapes
        if not self.tensor_view_shapes:
            return self.tensor_view_shapes
        self._flat_tensor_view_shapes = flatten_nested_sequence(self.tensor_view_shapes)
        return self._flat_tensor_view_shapes

    @property
    def flat_tensor_dtypes(self):
        if self._flat_tensor_dtypes is not None:
            return self._flat_tensor_dtypes
        if not self.tensor_dtypes:
            return self.tensor_dtypes
        # Leaf values are str ('float32').  After _normalize_compressed_fields
        # the nesting matches distribution exactly, but a tuple of strs like
        # ('float32', 'float32') is indistinguishable from a leaf — so we must
        # use _flatten_by_distribution to respect TensorList boundaries.
        dist = self._get_tensor_list_distribution()
        if dist:
            self._flat_tensor_dtypes = self._flatten_by_distribution(
                self.tensor_dtypes, dist)
        else:
            self._flat_tensor_dtypes = self.tensor_dtypes
        return self._flat_tensor_dtypes

    @property
    def flat_tensor_formats(self):
        if self._flat_tensor_formats is not None:
            return self._flat_tensor_formats
        if not self.tensor_formats:
            return self.tensor_formats
        # Leaf values are str ('ND', 'NCHW').  Same rationale as
        # flat_tensor_dtypes — use _flatten_by_distribution.
        dist = self._get_tensor_list_distribution()
        if dist:
            self._flat_tensor_formats = self._flatten_by_distribution(
                self.tensor_formats, dist)
        else:
            self._flat_tensor_formats = self.tensor_formats
        return self._flat_tensor_formats

    @property
    def flat_tensor_storage_shapes(self):
        if self._flat_tensor_storage_shapes is not None:
            return self._flat_tensor_storage_shapes
        if not self.tensor_storage_shapes:
            return self.tensor_storage_shapes
        # Leaf values are shape tuples (e.g. (2, 3)) — must use
        # _flatten_by_distribution to respect TensorList boundaries,
        # otherwise flatten_nested_sequence would split the shape tuple.
        dist = self._get_tensor_list_distribution()
        if dist:
            self._flat_tensor_storage_shapes = self._flatten_by_distribution(
                self.tensor_storage_shapes, dist)
        else:
            self._flat_tensor_storage_shapes = self.tensor_storage_shapes
        return self._flat_tensor_storage_shapes

    @property
    def flat_tensor_view_offsets(self):
        if self._flat_tensor_view_offsets is not None:
            return self._flat_tensor_view_offsets
        if not self.tensor_view_offsets:
            return self.tensor_view_offsets
        # Leaf values are int (0, 1, …).  Same rationale as flat_tensor_dtypes
        # — use _flatten_by_distribution to respect TensorList boundaries.
        dist = self._get_tensor_list_distribution()
        if dist:
            self._flat_tensor_view_offsets = self._flatten_by_distribution(
                self.tensor_view_offsets, dist)
        else:
            self._flat_tensor_view_offsets = self.tensor_view_offsets
        return self._flat_tensor_view_offsets

    @property
    def flat_tensor_view_strides(self):
        if self._flat_tensor_view_strides is not None:
            return self._flat_tensor_view_strides
        if not self.tensor_view_strides:
            return self.tensor_view_strides
        # Leaf values are shape tuples (e.g. (1, 2, 3)) — must use
        # _flatten_by_distribution to respect TensorList boundaries,
        # otherwise flatten_nested_sequence would split the stride tuple.
        dist = self._get_tensor_list_distribution()
        if dist:
            self._flat_tensor_view_strides = self._flatten_by_distribution(
                self.tensor_view_strides, dist)
        else:
            self._flat_tensor_view_strides = self.tensor_view_strides
        return self._flat_tensor_view_strides

    @property
    def flat_input_data_ranges(self):
        """Flat per-tensor ranges.

        If ``_normalize_input_data_ranges`` has already been called (via
        ``_normalize_compressed_fields``), the nested structure is guaranteed
        and a simple ``_flatten_by_distribution`` suffices.  When that hasn't
        happened yet we fall back to the same broadcast / pad logic that the
        normalize step uses so that callers that bypass validation still work.
        """
        if self._flat_input_data_ranges is not None:
            return self._flat_input_data_ranges
        if not self.input_data_ranges:
            return self.input_data_ranges
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.input_data_ranges
        # Already flat (no TensorList, or already one-to-one).
        if len(self.input_data_ranges) == flat_count:
            self._flat_input_data_ranges = self.input_data_ranges
        else:
            dist = self._get_tensor_list_distribution()
            if dist and self._is_range_field_already_nested(self.input_data_ranges, dist):
                # Normalized nested structure → simple flatten (fast path).
                self._flat_input_data_ranges = self._flatten_by_distribution(
                    self.input_data_ranges, dist)
            elif dist:
                # Compressed or partially-specified — normalize then flatten.
                self._normalize_input_data_ranges(dist)
                self._flat_input_data_ranges = self._flatten_by_distribution(
                    self.input_data_ranges, dist)
            else:
                self._flat_input_data_ranges = self.input_data_ranges
        return self._flat_input_data_ranges

    @property
    def flat_precision_tolerances(self):
        """Flat per-output precision tolerances, expanded by distribution."""
        if self._flat_precision_tolerances is not None:
            return self._flat_precision_tolerances
        if not self.precision_tolerances:
            return self.precision_tolerances
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.precision_tolerances
        if len(self.precision_tolerances) == flat_count:
            self._flat_precision_tolerances = self.precision_tolerances
        else:
            dist = self._get_tensor_list_distribution()
            if dist:
                self._flat_precision_tolerances = self._flatten_by_distribution(
                    self.precision_tolerances, dist)
            else:
                self._flat_precision_tolerances = self.precision_tolerances
        return self._flat_precision_tolerances

    @property
    def flat_absolute_precision(self):
        """Flat per-output absolute precision, or single float if not nested."""
        if self._flat_absolute_precision is not None:
            return self._flat_absolute_precision
        if not isinstance(self.absolute_precision, tuple):
            return self.absolute_precision
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.absolute_precision
        if len(self.absolute_precision) == flat_count:
            self._flat_absolute_precision = self.absolute_precision
        else:
            dist = self._get_tensor_list_distribution()
            if dist:
                self._flat_absolute_precision = self._flatten_by_distribution(
                    self.absolute_precision, dist)
            else:
                self._flat_absolute_precision = self.absolute_precision
        return self._flat_absolute_precision

    # ========== Per-flat-index accessors ==========

    def is_torch_dtype_support(self) -> bool:
        """Check if all dtypes in testcase are supported by the current torch natively."""
        if self._is_torch_dtype_support is not None:
            return self._is_torch_dtype_support
        from ttk.utilities.dtypes import is_torch_native_dtype
        result = True
        for dtype in self.flat_tensor_dtypes:
            if dtype is not None and not is_torch_native_dtype(dtype):
                result = False
                break
        self._is_torch_dtype_support = result
        return result

    # ========== Legacy per-flat-index accessors (kept for backward compat) ==========

    def flat_storage_shape(self, idx: int):
        flat_shapes = self.flat_tensor_view_shapes
        view_shape = flat_shapes[idx] if idx < len(flat_shapes) else None
        s_shapes = self.flat_tensor_storage_shapes
        if not s_shapes or idx >= len(s_shapes):
            return view_shape
        val = s_shapes[idx]
        if val is None:
            return view_shape
        return val

    def flat_view_stride(self, idx: int):
        flat_shapes = self.flat_tensor_view_shapes
        view_shape = flat_shapes[idx] if idx < len(flat_shapes) else None
        if view_shape is None:
            return None
        strides = self.flat_tensor_view_strides or ()
        if idx < len(strides):
            s = strides[idx]
            if s is not None and s != ():
                return s
        return shape_stride(view_shape)

    def flat_view_offset(self, idx: int):
        flat_shapes = self.flat_tensor_view_shapes
        view_shape = flat_shapes[idx] if idx < len(flat_shapes) else None
        if view_shape is None:
            return None
        offsets = self.flat_tensor_view_offsets or ()
        if idx < len(offsets):
            val = offsets[idx]
            if val is not None:
                return val
        return 0

    # ========== Pure output indexes ==========

    @property
    def pure_output_indexes(self):
        if self._pure_output_indexes is not None:
            return self._pure_output_indexes
        dist = self._get_tensor_list_distribution()
        flat_output = set()
        for idx in (self.output_tensor_indexes or ()):
            flat_idx = sum(max(d, 1) for d in dist[:idx])
            count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
            flat_output.update(range(flat_idx, flat_idx + count))
        self._pure_output_indexes = sorted(flat_output)
        return self._pure_output_indexes

    # ========== Normalize ==========

    def _normalize_compressed_fields(self):
        if not self.is_valid or not self.tensor_view_shapes:
            return
        dist = self._get_tensor_list_distribution()
        if dist:
            for field_name in self._normalized_tensor_fields:
                self._normalize_field_by_dist(field_name, dist)
            self._normalize_input_data_ranges(dist)

    def _normalize_field_by_dist(self, field_name, dist):
        field = getattr(self, field_name)
        if not field:
            return
        if self._is_field_already_nested(field, dist):
            return
        flat_count = sum(max(d, 1) for d in dist)
        if len(field) == flat_count:
            flat = field
        elif len(field) == 1:
            val = field[0]
            if isinstance(val, (tuple, list)) and len(val) == 1:
                val = val[0]
            flat = (val,) * flat_count
        elif len(field) <= len(dist):
            # Fewer values than params — broadcast the last value to fill.
            padded = field + (field[-1],) * (len(dist) - len(field))
            flat = self._flatten_by_distribution(padded, dist)
        else:
            flat = self._flatten_by_distribution(field, dist)
        result = []
        idx = 0
        for d in dist:
            if d > 0:
                result.append(tuple(flat[idx:idx + d]))
                idx += d
            else:
                result.append(flat[idx])
                idx += 1
        setattr(self, field_name, tuple(result))
        cache_attr = f'_flat_{field_name}'
        if hasattr(self, cache_attr):
            setattr(self, cache_attr, None)

    @staticmethod
    def _is_field_already_nested(field, dist):
        if len(field) != len(dist):
            return False
        for val, num in zip(field, dist):
            if num > 0:
                if not isinstance(val, (tuple, list)) or len(val) != num:
                    return False
        return True

    @staticmethod
    def _is_range_field_already_nested(field, dist):
        """Like _is_field_already_nested but for range tuples.

        A range expression like (None, 1.0) has scalar (non-tuple) elements,
        while a range list like ((None, 1.0), (-1.0, 1.0)) has tuple elements.
        Only the latter counts as "already nested" for range fields.
        """
        if len(field) != len(dist):
            return False
        for val, num in zip(field, dist):
            if num > 0:
                if not isinstance(val, (tuple, list)) or len(val) != num:
                    return False
                if num > 1 and val:
                    first = val[0]
                    if not isinstance(first, (tuple, list)):
                        return False
        return True

    def _normalize_input_data_ranges(self, dist):
        """Normalize input_data_ranges to a fully nested per-param structure.

        After normalization, ``self.input_data_ranges`` has length ``len(dist)``
        and every TensorList position (num > 0) holds a tuple of individual
        range expressions whose count equals *num*.

        Processing steps:
          1. Per-param alignment  – match user-supplied ranges to params,
             broadcasting or padding with ``(None, None)`` as needed.
          2. TensorList expand    – for each param whose dist entry is > 0,
             expand a single range or short list into *num* individual ranges,
             padding with ``(None, None)`` when the user supplied fewer entries
             than the TensorList expects.
          3. Re-nest              – rebuild the per-param nested structure so
             that downstream ``flat_input_data_ranges`` can simply flatten.
        """
        field = self.input_data_ranges
        # Keep empty — downstream defaults every tensor to (None, None).
        if not field:
            return
        # Already fully nested (len matches dist, TensorList entries have
        # correct sub-counts) — nothing to do.
        if self._is_range_field_already_nested(field, dist):
            return

        # ---------- Step 1: align to per-param ----------
        # len(field) == 1: single range expression broadcast to all params.
        # len(field) < len(dist): missing params padded with (None, None).
        # len(field) >= len(dist): truncate to param count.
        if len(field) == 1:
            per_param = [field[0]] * len(dist)
        elif len(field) < len(dist):
            per_param = list(field) + [(None, None)] * (len(dist) - len(field))
        else:
            per_param = list(field[:len(dist)])

        # ---------- Step 2: expand TensorList positions ----------
        result = []
        for i, num in enumerate(dist):
            val = per_param[i]
            if num == 0:
                # Single-tensor param — use as-is.
                result.append(val)
                continue

            # Distinguish "single range" from "list of ranges".
            # A range expression is a tuple of scalars, e.g. (None, 1.0) or
            # (0.0, 5.0, 2.0).  A "list of ranges" has tuple elements, e.g.
            # ((0.0, 0.5), (0.5, 1.0)).
            is_range_list = (isinstance(val, (tuple, list))
                             and val
                             and isinstance(val[0], (tuple, list)))

            if not is_range_list:
                # Single range (r) → broadcast to num copies.
                result.append(tuple([val] * num))
            elif len(val) == 1:
                # (r,) → broadcast the sole element.
                result.append(tuple([val[0]] * num))
            elif len(val) >= num:
                # Enough or extra — truncate.
                result.append(tuple(val[:num]))
            else:
                # Not enough — pad remaining with (None, None).
                result.append(tuple(val) + ((None, None),) * (num - len(val)))

        # ---------- Step 3: write back ----------
        self.input_data_ranges = tuple(result)
        self._flat_input_data_ranges = None

    @staticmethod
    def _flatten_by_distribution(values, distribution):
        result = []
        vi = 0
        for num in distribution:
            val = values[vi]
            vi += 1
            if num > 0:
                if isinstance(val, (tuple, list)):
                    if len(val) == num:
                        result.extend(val)
                    elif len(val) == 1:
                        result.extend([val[0]] * num)
                    else:
                        result.extend(val)
                else:
                    result.extend([val] * num)
            else:
                result.append(val)
        return tuple(result)
