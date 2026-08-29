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


from ...utilities import shape_stride
from ...utilities.container_utils import deep_flatten, flatten_nested_sequence, infer_list_distribution_from_nesting
from .testcase_base import TestcaseBase


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
        "inplace_input_indexes",
        "np_storages",
        "prof_result",
        "_tensor_list_dist",
        "_output_dist",
        "_param_plan_cache",
        "_flat_tensor_view_shapes",
        "_flat_tensors",
        "_flat_scalars",
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
        "_is_tf_dtype_support",
        "const_input_indexes",
    )

    _scalar_tensor_fields = (
        "tensor_dtypes",
        "tensor_formats",
        "tensor_view_offsets",
    )
    _shape_tensor_fields = (
        "tensor_view_strides",
        "tensor_storage_shapes",
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
        self.inplace_input_indexes = ()
        self.np_storages = None
        self.prof_result = None
        self._tensor_list_dist = None
        self._output_dist = None
        self._param_plan_cache = None
        self._flat_tensor_view_shapes = None
        self._flat_tensors = None
        self._flat_scalars = None
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
        self._is_tf_dtype_support = None
        self.const_input_indexes = set()

    @property
    def op_name(self):
        return self.api_name

    # ========== Distribution ==========

    @property
    def tensor_list_dist(self):
        """TensorList distribution inferred from tensor_view_shapes.

        Cached on first access. Returns () when no tensor_view_shapes.
        """
        if self._tensor_list_dist is None and self.tensor_view_shapes:
            self._tensor_list_dist = infer_list_distribution_from_nesting(self.tensor_view_shapes)
        return self._tensor_list_dist or ()

    @property
    def output_dist(self):
        """Output tensor distribution: projection of tensor_list_dist onto output_tensor_indexes.

        Cached on first access. Returns () when no output tensors.
        """
        if self._output_dist is None:
            dist = self.tensor_list_dist
            if dist and self.output_tensor_indexes:
                self._output_dist = tuple(dist[i] for i in self.output_tensor_indexes if i < len(dist))
            else:
                self._output_dist = ()
        return self._output_dist

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
        dist = self.tensor_list_dist
        if dist:
            self._flat_tensor_dtypes = self._flatten_by_distribution(self.tensor_dtypes, dist)
        else:
            self._flat_tensor_dtypes = self.tensor_dtypes
        return self._flat_tensor_dtypes

    @property
    def flatten_tensors(self):
        if self._flat_tensors is not None:
            return self._flat_tensors
        self._flat_tensors = deep_flatten(self.tensors) if self.tensors is not None else None
        return self._flat_tensors

    @property
    def flatten_scalars(self):
        if self._flat_scalars is not None:
            return self._flat_scalars
        self._flat_scalars = deep_flatten(self.scalars) if self.scalars is not None else None
        return self._flat_scalars

    @property
    def flat_tensor_formats(self):
        if self._flat_tensor_formats is not None:
            return self._flat_tensor_formats
        if not self.tensor_formats:
            return self.tensor_formats
        # Leaf values are str ('ND', 'NCHW').  Same rationale as
        # flat_tensor_dtypes — use _flatten_by_distribution.
        dist = self.tensor_list_dist
        if dist:
            self._flat_tensor_formats = self._flatten_by_distribution(self.tensor_formats, dist)
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
        dist = self.tensor_list_dist
        if dist:
            self._flat_tensor_storage_shapes = self._flatten_by_distribution(self.tensor_storage_shapes, dist)
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
        dist = self.tensor_list_dist
        if dist:
            self._flat_tensor_view_offsets = self._flatten_by_distribution(self.tensor_view_offsets, dist)
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
        dist = self.tensor_list_dist
        if dist:
            self._flat_tensor_view_strides = self._flatten_by_distribution(self.tensor_view_strides, dist)
        else:
            self._flat_tensor_view_strides = self.tensor_view_strides
        return self._flat_tensor_view_strides

    @property
    def flat_input_data_ranges(self):
        """Flat per-tensor ranges.

        Assumes _normalize_compressed_fields has already expanded compressed
        forms.  Simply flattens the normalized nested structure.
        """
        if self._flat_input_data_ranges is not None:
            return self._flat_input_data_ranges
        if not self.input_data_ranges:
            return self.input_data_ranges
        dist = self.tensor_list_dist
        if dist:
            self._flat_input_data_ranges = self._flatten_by_distribution(self.input_data_ranges, dist)
        else:
            self._flat_input_data_ranges = self.input_data_ranges
        return self._flat_input_data_ranges

    @property
    def flat_precision_tolerances(self):
        """Flat per-output precision tolerances. Assumes normalize already done."""
        if self._flat_precision_tolerances is not None:
            return self._flat_precision_tolerances
        if not self.precision_tolerances:
            return self.precision_tolerances
        odist = self.output_dist
        if odist:
            self._flat_precision_tolerances = self._flatten_by_distribution(self.precision_tolerances, odist)
        else:
            self._flat_precision_tolerances = self.precision_tolerances
        return self._flat_precision_tolerances

    @property
    def flat_absolute_precision(self):
        """Flat per-output absolute precision. Assumes normalize already done."""
        if self._flat_absolute_precision is not None:
            return self._flat_absolute_precision
        if not isinstance(self.absolute_precision, tuple):
            return self.absolute_precision
        odist = self.output_dist
        if odist:
            self._flat_absolute_precision = self._flatten_by_distribution(self.absolute_precision, odist)
        else:
            self._flat_absolute_precision = self.absolute_precision
        return self._flat_absolute_precision

    def invalidate_flat_cache(self, *fields):
        """Invalidate flat cache(s) by field name.

        Example: invalidate_flat_cache("tensor_dtypes") sets self._flat_tensor_dtypes = None.
        Uses hasattr guard: nonexistent _flat_{field} is safely skipped.
        """
        for field in fields:
            attr = f"_flat_{field}"
            if hasattr(self, attr):
                setattr(self, attr, None)

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

    def is_tf_dtype_support(self) -> bool:
        """Check if all dtypes in testcase are supported by TF natively."""
        if self._is_tf_dtype_support is not None:
            return self._is_tf_dtype_support
        from ttk.utilities.dtypes import is_tf_native_dtype

        result = True
        for dtype in self.flat_tensor_dtypes:
            if dtype is not None and not is_tf_native_dtype(dtype):
                result = False
                break
        self._is_tf_dtype_support = result
        return result

    def is_dtype_support(self) -> bool:
        """Framework-aware dtype support check."""
        from ttk.core_modules.framework_api.framework_detector import detect_framework

        framework = detect_framework(self.api_name)
        if framework == "tf":
            return self.is_tf_dtype_support()
        return self.is_torch_dtype_support()

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
        dist = self.tensor_list_dist
        flat_output = set()
        for idx in self.output_tensor_indexes or ():
            flat_idx = sum(max(d, 1) for d in dist[:idx])
            count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
            flat_output.update(range(flat_idx, flat_idx + count))
        self._pure_output_indexes = sorted(flat_output)
        return self._pure_output_indexes

    # ========== Normalize ==========

    def _normalize_compressed_fields(self):
        if not self.is_valid or not self.tensor_view_shapes:
            return
        dist = self.tensor_list_dist
        if dist:
            for field_name in self._scalar_tensor_fields:
                self._normalize_scalar_field_by_dist(field_name, dist)
            for field_name in (*self._shape_tensor_fields, "input_data_ranges"):
                self._normalize_range_field_by_dist(field_name, dist)
            odist = self.output_dist
            if odist:
                self._normalize_scalar_field_by_dist("absolute_precision", odist)
                self._normalize_range_field_by_dist("precision_tolerances", odist)

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
