#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.


"""
Testcase structure for framework_api tests.
"""

import logging

from ttk.core_modules.testcase_manager.testcase_tensor_api_base import TensorApiTestcaseBase
from ttk.core_modules.testcase_manager.field_types import FIELD_TYPES
from ttk.core_modules.testcase_manager.param_plan import ParamPlan, match_overload
from ttk.utilities import get, shape_stride
from ttk.utilities.container_utils import flatten_nested_sequence


class TestcaseE2e(TensorApiTestcaseBase):
    """Testcase structure for framework-level API testing (torch/tf).

    Supports nested tensor_view_shapes for TensorList grouping:
      - Flat:  ((2,3),(3,4)) → 2 independent tensors
      - Nested: (((2,3),(3,4)),(5,6)) → TensorList of 2 + single tensor

    Compressed field syntax (after parsing, before validate):
      - ('float32',) broadcasts to all tensors
      - ('float32',('int8','int16')) per-param compression via dist

    Distribution is inferred from tensor_view_shapes nesting:
      dist = (2, 0) means param-0 is TensorList(2), param-1 is single Tensor
    """

    __slots__ = (
        "attributes",
        "golden_api",
        "golden_data",
        "tensors",
        "_api_info_cache",
        "batch_axis",
        "batch_slice_info",
        "batch_seed",
        "batch_consistency_id",
        "golden_mode_override",
    )

    identity_headers = {
        **TensorApiTestcaseBase.identity_headers,
        "api_name": (FIELD_TYPES.STRING, None),
    }

    tensor_property_headers = {
        "tensor_view_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None),
        "tensor_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None),
    }

    optional_tensor_headers = {
        "tensor_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
        "tensor_storage_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
        "tensor_view_offsets": (FIELD_TYPES.INT_CONTAINER_NESTED, None, ()),
        "tensor_view_strides": (FIELD_TYPES.SHAPE_STRIDE, None, ()),
        "output_tensor_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
        "inplace_input_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
    }

    attr_headers = {
        "attributes": (FIELD_TYPES.DICT, None, {}),
    }

    batch_consistency_headers = {
        "batch_axis": (FIELD_TYPES.FREE_EVAL, None, None),
        "batch_slice_info": (FIELD_TYPES.FREE_EVAL, None, None),
        "batch_seed": (FIELD_TYPES.FREE_EVAL, None, None),
    }

    golden_headers = {
        "golden_api": (FIELD_TYPES.STRING, None, ""),
    }

    special_property_headers = {
        **TensorApiTestcaseBase.special_property_headers,
    }

    option_headers = {
        **TensorApiTestcaseBase.option_headers,
    }

    complete_headers = {
        **identity_headers,
        **tensor_property_headers,
        **optional_tensor_headers,
        **attr_headers,
        **golden_headers,
        **special_property_headers,
        **batch_consistency_headers,
        **option_headers,
    }

    def __init__(self):
        super().__init__()
        self.attributes = {}
        self.golden_api = None
        self.golden_data = None
        self.tensors = None
        self._api_info_cache = None
        self.batch_axis = None
        self.batch_slice_info = None
        self.batch_seed = None
        self.batch_consistency_id = None
        # cross_check 判据下由 profiling 侧临时置 "Promote",令 golden 抬成高精度真值。
        # 必须在 __slots__ 中声明:该类及其基类均无 __dict__,否则赋值抛 AttributeError,
        # 被调用侧 try/except 咽掉后 Promote 静默空转。范式同 TestcaseOp / TestcaseAclnn。
        self.golden_mode_override = None

    @classmethod
    def get_all_visible_headers(cls):
        return tuple(cls.complete_headers.keys())

    def get_api_info(self):
        """Get cached APIParamInfo for this testcase's api_name."""
        if self._api_info_cache is not None:
            return self._api_info_cache
        if not self.api_name:
            return None
        try:
            from ttk.core_modules.framework_api.framework_api_info_keeper import (
                FrameworkApiInfoKeeper,
            )

            self._api_info_cache = FrameworkApiInfoKeeper().get(self.api_name)
        except Exception:
            self._api_info_cache = None
        return self._api_info_cache

    def get_param_plan(self):
        """Resolve and cache parameter assembly plan for this testcase."""
        if self._param_plan_cache is not None:
            return self._param_plan_cache
        if not self.api_name:
            return None

        try:
            info = self.get_api_info()
            if info is None:
                logging.warning(f"[{self.testcase_name}] get_api_info returned None for {self.api_name}")
                return None

            out_indices = set(self.output_tensor_indexes or ())
            top_count = len(self.tensor_view_shapes or ())
            if self._is_inplace_tensor_method(self.api_name):
                input_count = top_count
            else:
                input_count = sum(
                    1 for i in range(top_count) if self.tensor_view_shapes[i] is not None and i not in out_indices
                )

            dist = self.tensor_list_dist
            tensor_distribution = [d > 0 for d in dist] if dist else None

            params, oidx = match_overload(
                self.api_name,
                input_count,
                attributes=self.attributes,
                tensor_distribution=tensor_distribution,
                api_info=info,
            )
            if params is None:
                return None

            plan = ParamPlan(
                api_name=self.api_name,
                overload_params=params,
                overload_index=oidx,
                output_tensor_indexes=self.output_tensor_indexes,
                attributes=self.attributes,
            )
            self._param_plan_cache = plan
            return plan
        except Exception:
            return None

    # ========== Flat properties ==========

    @property
    def flat_tensor_view_shapes(self):
        """Flatten nested tensor_view_shapes to a flat tuple of shapes.

        (((3,3),(3,2)),(3,5)) -> ((3,3),(3,2),(3,5))
        """
        if not self.tensor_view_shapes:
            return self.tensor_view_shapes
        return flatten_nested_sequence(self.tensor_view_shapes)

    @property
    def flat_tensor_dtypes(self):
        """Flatten nested tensor_dtypes to per-tensor dtypes matching flat_tensor_view_shapes."""
        if not self.tensor_dtypes:
            return self.tensor_dtypes
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.tensor_dtypes
        if len(self.tensor_dtypes) == flat_count:
            return self.tensor_dtypes
        dist = self.tensor_list_dist
        if dist and len(self.tensor_dtypes) == len(dist):
            return self._flatten_by_distribution(self.tensor_dtypes, dist)
        if len(self.tensor_dtypes) == 1:
            val = self.tensor_dtypes[0]
            if isinstance(val, (tuple, list)) and len(val) == 1:
                val = val[0]
            return (val,) * flat_count
        return flatten_nested_sequence(self.tensor_dtypes)

    @property
    def flat_tensor_formats(self):
        """Flatten nested tensor_formats to per-tensor format strings."""
        if not self.tensor_formats:
            return self.tensor_formats
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.tensor_formats
        if len(self.tensor_formats) == flat_count:
            return self.tensor_formats
        dist = self.tensor_list_dist
        if dist and len(self.tensor_formats) == len(dist):
            return self._flatten_by_distribution(self.tensor_formats, dist)
        if len(self.tensor_formats) == 1:
            return (self.tensor_formats[0],) * flat_count
        return flatten_nested_sequence(self.tensor_formats)

    @property
    def flat_tensor_storage_shapes(self):
        """Flatten nested tensor_storage_shapes to per-tensor storage shapes."""
        if not self.tensor_storage_shapes:
            return self.tensor_storage_shapes
        flat_count = len(self.flat_tensor_view_shapes)
        if len(self.tensor_storage_shapes) == flat_count:
            return self.tensor_storage_shapes
        return flatten_nested_sequence(self.tensor_storage_shapes)

    @property
    def flat_tensor_view_offsets(self):
        """Flatten nested tensor_view_offsets to per-tensor offsets."""
        if not self.tensor_view_offsets:
            return self.tensor_view_offsets
        flat_count = len(self.flat_tensor_view_shapes)
        if flat_count == 0:
            return self.tensor_view_offsets
        if len(self.tensor_view_offsets) == flat_count:
            return self.tensor_view_offsets
        dist = self.tensor_list_dist
        if dist and len(self.tensor_view_offsets) == len(dist):
            return self._flatten_by_distribution(self.tensor_view_offsets, dist)
        if len(self.tensor_view_offsets) == 1:
            return (self.tensor_view_offsets[0],) * flat_count
        return flatten_nested_sequence(self.tensor_view_offsets)

    @property
    def flat_tensor_view_strides(self):
        """Flatten nested tensor_view_strides to per-tensor strides."""
        if not self.tensor_view_strides:
            return self.tensor_view_strides
        flat_count = len(self.flat_tensor_view_shapes)
        if len(self.tensor_view_strides) == flat_count:
            return self.tensor_view_strides
        return flatten_nested_sequence(self.tensor_view_strides)

    # ========== Per-flat-index accessors ==========

    def flat_storage_shape(self, idx: int):
        """Get storage shape for flat tensor at idx. Falls back to view shape."""
        flat_shapes = self.flat_tensor_view_shapes
        s_shapes = self.flat_tensor_storage_shapes
        if not s_shapes:
            return flat_shapes[idx]
        return get(s_shapes, idx, out_of_range=flat_shapes[idx])

    def flat_view_stride(self, idx: int):
        """Get view stride for flat tensor at idx. Auto-compute from shape if not specified."""
        strides = self.flat_tensor_view_strides or ()
        s = get(strides, idx, out_of_range=())
        if not s:
            s = shape_stride(self.flat_tensor_view_shapes[idx])
        return s

    def flat_view_offset(self, idx: int):
        """Get view offset for flat tensor at idx. Default 0."""
        offsets = self.flat_tensor_view_offsets or ()
        return get(offsets, idx, out_of_range=0)

    # ========== Legacy per-top-level accessors (kept for backward compat) ==========

    def storage_shape(self, idx: int):
        v_shapes = self.tensor_view_shapes
        s_shapes = self.tensor_storage_shapes
        return get(s_shapes, idx, out_of_range=v_shapes[idx])

    def view_stride(self, idx: int):
        strides = self.tensor_view_strides or ()
        s = get(strides, idx, out_of_range=())
        if not s:
            s = shape_stride(self.tensor_view_shapes[idx])
        return s

    def view_offset(self, idx: int):
        offsets = self.tensor_view_offsets or ()
        return get(offsets, idx, out_of_range=0)

    # ========== actual_input_data_ranges ==========

    @property
    def actual_input_data_ranges(self):
        return self.input_data_ranges

    @property
    def pure_output_indexes(self):
        base = super().pure_output_indexes
        if self._is_inplace_tensor_method(self.api_name) and 0 in base:
            return sorted(set(base) - {0})
        return base

    # ========== Validate ==========

    def validate(self):
        super().validate()
        if not self.api_name:
            self.is_valid = False
            self.fail_reason = "api_name is empty"
            return
        if not self.tensor_view_shapes:
            info = self._try_get_api_info_for_factory()
            if info is None:
                return
        if self.tensor_view_shapes and not self.tensor_dtypes:
            self.is_valid = False
            self.fail_reason = "tensor_dtypes is empty"
            return
        self._normalize_compressed_fields()
        self._auto_fill_inplace_tensor_method()
        self._check_top_level_counts()
        self._check_tensor_configuration()
        self._check_output_configuration()
        self._generate_batch_consistency_id()

    def _try_get_api_info_for_factory(self):
        """Check if API is a factory function (no required input tensors)."""
        info = self.get_api_info()
        if info is None:
            self.is_valid = False
            self.fail_reason = "tensor_view_shapes is empty"
            return None
        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like] if not p.is_optional) for ov in info.overloads
        )
        if required_min == 0:
            return info
        self.is_valid = False
        self.fail_reason = "tensor_view_shapes is empty"
        return None

    @staticmethod
    def _is_inplace_tensor_method(api_name):
        from ttk.core_modules.framework_api.framework_detector import is_inplace_tensor_method

        return is_inplace_tensor_method(api_name)

    def _auto_fill_inplace_tensor_method(self):
        if not self._is_inplace_tensor_method(self.api_name):
            return
        if not self.output_tensor_indexes:
            self.output_tensor_indexes = (0,)

    def _generate_batch_consistency_id(self):
        """根据 batch_seed batch和 batch_slice_info 的切片长度生成 batch_consistency_id。

        相同 batch_seed 且切片长度相同的用例，生成相同 id,
        标识这些用例的输出切片可以做 batch 一致性比较。
        """
        if self.batch_seed is None:
            self.batch_consistency_id = None
            return

        if self.batch_axis is None or self.batch_slice_info is None:
            self.batch_consistency_id = None
            return

        slice_key = []
        for axis_pos, slices, seed in zip(self.batch_axis, self.batch_slice_info, self.batch_seed):
            if axis_pos is None or slices is None or seed is None:
                continue
            slice_axes = []
            for axis_idx, slices_idx, seed_idx in zip(axis_pos, slices, seed):
                if axis_idx is None or slices_idx is None or seed_idx is None:
                    slice_id = "None"
                    slice_axes.append(slice_id)
                    continue
                slice_lens = []
                for sl, seed_value in zip(slices_idx, seed_idx):
                    if sl is None:
                        continue
                    start = sl[0]
                    stop = sl[1]
                    step = sl[2]
                    if step <= 0 or start < 0 or stop < 0:
                        length = 0
                    else:
                        length = stop - start if stop > start else 0
                    slice_id = f"{seed_value}_{axis_idx}_{start}_{stop}_{step}"
                    if length == 0:
                        logging.warning(
                            f"testcase: {self.testcase_name}, slice_id is: {slice_id}, slice is:{sl} this slice is Invalid"
                        )
                    slice_lens.append(slice_id)
                slice_axes.append(tuple(slice_lens))
            slice_key.append(tuple(slice_axes))
        self.batch_consistency_id = tuple(slice_key)

    def _check_tensor_configuration(self):
        """Validate tensor parameters match API definition in count and type."""
        if not self.is_valid:
            return
        info = self.get_api_info()
        if info is None:
            self.is_valid = False
            self.fail_reason = "API_PARSE_FAIL"
            return
        if self._check_input_count_exceeded(info):
            return
        if self._is_factory_api(info):
            return
        if self._check_all_tensors_output(info):
            return
        self._check_overload_match(info)

    def _check_input_count_exceeded(self, info):
        """Fail if non-output tensor count exceeds API's max input tensor parameters.

        Overloads containing a VAR_POSITIONAL (*args) tensor parameter are exempt
        because they accept an unbounded number of tensors.
        """
        out_indices = set(self.output_tensor_indexes or ())
        top_count = len(self.tensor_view_shapes or ())
        input_count = sum(1 for i in range(top_count) if i not in out_indices)
        # If any overload has a VAR_POSITIONAL tensor param, it can accept
        # any number of tensors — skip the count check entirely.
        has_var_pos = any(
            any(getattr(p, "is_var_positional", False) for p in ov.params if p.is_tensor_like and p.name != "out")
            for ov in info.overloads
        )
        if has_var_pos:
            return False
        max_input = max(sum(1 for p in ov.params if p.is_tensor_like and p.name != "out") for ov in info.overloads)
        if input_count > max_input:
            self.is_valid = False
            self.fail_reason = "INPUT_COUNT_EXCEEDED"
            logging.error(
                f"[{self.testcase_name}] API [{self.api_name}] has at most {max_input} input tensor "
                f"parameters (excluding out), but testcase configured {input_count} "
                f"input tensor(s) (excluding {len(out_indices)} output tensor(s)). "
                f"(source: {info.source})"
            )
            return True
        return False

    def _is_factory_api(self, info):
        """Return True if API requires no input tensors (factory function like torch.zeros)."""
        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )
        return required_min == 0

    def _check_all_tensors_output(self, info):
        """Fail if all tensors are marked as output with none left as input."""
        out_indices = set(self.output_tensor_indexes or ())
        top_count = len(self.tensor_view_shapes)
        if self._is_inplace_tensor_method(self.api_name):
            input_count = top_count
        else:
            input_count = sum(1 for i in range(top_count) if i not in out_indices)
        if input_count > 0:
            return False
        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )
        self.is_valid = False
        self.fail_reason = "ALL_TENSORS_MARKED_OUTPUT"
        logging.error(
            f"[{self.testcase_name}] API [{self.api_name}] requires at least {required_min} input tensor "
            f"parameters (excluding out), but all {top_count} tensor(s) are "
            f"marked as output (output_tensor_indexes={sorted(out_indices)}). "
            f"(source: {info.source})"
        )
        return True

    @staticmethod
    def _classify_input_types(input_shapes):
        """Classify each input shape as nested (TensorList) or None placeholder."""
        nested_flags = []
        has_none = []
        for element in input_shapes:
            if element is None:
                nested_flags.append(False)
                has_none.append(True)
            else:
                is_nested = (
                    isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list))
                )
                nested_flags.append(is_nested)
                has_none.append(False)
        return nested_flags, has_none

    def _count_scalar_attrs_for_tensor_params(self, info):
        """Count tensor-like params satisfied by scalar values in attributes."""
        if not self.attributes:
            return 0
        attr_keys = set(self.attributes.keys())
        count = 0
        for ov in info.overloads:
            ov_count = 0
            for p in ov.layout.input_params:
                if p.name in attr_keys and p.name != "self":
                    ov_count += 1
            count = max(count, ov_count)
        return count

    def _check_overload_match(self, info):
        """Fail if input tensors don't match any overload in count/type/nesting."""
        out_indices = set(self.output_tensor_indexes or ())
        if self._is_inplace_tensor_method(self.api_name):
            input_shapes = list(self.tensor_view_shapes)
        else:
            input_shapes = [s for i, s in enumerate(self.tensor_view_shapes) if i not in out_indices]
        input_count = len(input_shapes)
        nested_flags, has_none = self._classify_input_types(input_shapes)

        matched, _, oidx = info.match_overload(input_count, nested_flags, has_none)
        if matched:
            self._check_required_attrs(info, oidx)
            return

        scalar_attr_count = self._count_scalar_attrs_for_tensor_params(info)
        if scalar_attr_count > 0:
            effective_count = input_count + scalar_attr_count
            matched, _, oidx = info.match_overload(effective_count, None, None)
            if matched:
                self._check_required_attrs(info, oidx)
                return

        required_min = min(
            sum(1 for p in [pp for pp in ov.params if pp.is_tensor_like and pp.name != "out"] if not p.is_optional)
            for ov in info.overloads
        )
        required_min -= scalar_attr_count
        count_matched = info.match_overload(input_count, None, None)
        if count_matched[0]:
            self.is_valid = False
            self.fail_reason = "PARAM_TYPE_MISMATCH"
            logging.error(
                f"API [{self.api_name}] input tensor count matches an overload, "
                f"but type (Tensor/TensorList) does not. "
                f"nested={nested_flags}. "
                f"(source: {info.source})"
            )
        elif input_count > info.tensor_count:
            self.is_valid = False
            self.fail_reason = "TENSOR_COUNT_MISMATCH"
            logging.error(
                f"API [{self.api_name}] has at most {info.tensor_count} tensor parameters "
                f"in any overload, but testcase configured {input_count} input tensors "
                f"(excluding {len(out_indices)} output tensors). "
                f"(source: {info.source})"
            )
        else:
            self.is_valid = False
            self.fail_reason = "TENSOR_COUNT_MISMATCH"
            logging.error(
                f"API [{self.api_name}] requires at least {required_min} input tensor "
                f"parameters (excluding out), but testcase configured {input_count}. "
                f"(source: {info.source})"
            )

    def _check_required_attrs(self, info, oidx):
        """Fail if matched overload has required non-tensor params missing from attributes.

        If the matched overload has missing attrs but another overload also matches
        the tensor count without missing attrs, skip the check (build_args will try all).
        """
        ov = info.overloads[oidx]
        attrs = set(self.attributes.keys()) if self.attributes else set()
        missing = [
            p.name
            for p in ov.params
            if not p.is_tensor_like
            and not p.is_keyword_only
            and p.name != "out"
            and not p.is_optional
            and p.name not in attrs
        ]
        if not missing:
            return

        out_indices = set(self.output_tensor_indexes or ())
        top_count = len(self.tensor_view_shapes or ())
        input_count = (
            top_count
            if self._is_inplace_tensor_method(self.api_name)
            else sum(1 for i in range(top_count) if i not in out_indices)
        )

        for alt_oidx, alt_ov in enumerate(info.overloads):
            if alt_oidx == oidx:
                continue
            alt_tensors = [p for p in alt_ov.params if p.is_tensor_like and p.name != "out"]
            alt_req = sum(1 for p in alt_tensors if not p.is_optional)
            alt_total = len(alt_tensors)
            if not (alt_req <= input_count <= alt_total):
                continue
            alt_missing = [
                p.name
                for p in alt_ov.params
                if not p.is_tensor_like
                and not p.is_keyword_only
                and p.name != "out"
                and not p.is_optional
                and p.name not in attrs
            ]
            if not alt_missing:
                return

        self.is_valid = False
        self.fail_reason = "MISSING_REQUIRED_ATTR"
        logging.error(
            f"[{self.testcase_name}] API [{self.api_name}] overload[{oidx}] requires "
            f"non-tensor attribute(s) {missing} but attributes only have "
            f"{sorted(attrs)}. (source: {info.source})"
        )

    def _check_output_configuration(self):
        """Validate output_tensor_indexes against API's out parameter definition.

        Checks:
        1. Indexes within top-level range
        2. If API has a required out parameter, testcase must provide output tensors
        3. If API has a Tensor[] out, output count must match out_expected_count
        """
        if not self.is_valid:
            return
        info = self.get_api_info()
        if not self.output_tensor_indexes:
            if info is not None:
                any_out_required = any(ov.layout.is_out_required for ov in info.overloads)
                if any_out_required:
                    self.is_valid = False
                    self.fail_reason = "MISSING_REQUIRED_OUTPUT"
                    logging.error(
                        f"[{self.testcase_name}] API [{self.api_name}] has overloads "
                        f"with required 'out' parameter, but testcase provides no "
                        f"output_tensor_indexes. (source: {info.source})"
                    )
            return
        top_count = len(self.tensor_view_shapes)
        out_of_range = [i for i in self.output_tensor_indexes if i < 0 or i >= top_count]
        if out_of_range:
            self.is_valid = False
            self.fail_reason = "OUTPUT_INDEX_INVALID"
            logging.error(f"output_tensor_indexes {out_of_range} out of range [0, {top_count})")
            return
        if info is None:
            return
        out_count = len(self.output_tensor_indexes)
        matching_overloads = []
        for oidx, ov in enumerate(info.overloads):
            layout = ov.layout
            if layout.out_param is None:
                if out_count == 0:
                    matching_overloads.append(oidx)
            elif layout.is_out_tensor_list:
                if layout.out_expected_count > 0:
                    # Exact validation: we know the expected out tensor count
                    if layout.is_out_required and out_count != layout.out_expected_count:
                        continue
                    elif not layout.is_out_required and out_count not in (0, layout.out_expected_count):
                        continue
                else:
                    # Loose validation: TensorList but unknown count (e.g. TypeError path)
                    if layout.is_out_required and out_count == 0:
                        continue
                matching_overloads.append(oidx)
            else:
                if layout.is_out_required and out_count != 1:
                    continue
                elif not layout.is_out_required and out_count > 1:
                    continue
                matching_overloads.append(oidx)
        if not matching_overloads:
            any_out_required = any(ov.layout.is_out_required for ov in info.overloads)
            any_tensor_list_out = any(ov.layout.is_out_tensor_list for ov in info.overloads)
            if any_out_required and any_tensor_list_out:
                expected = set(ov.layout.out_expected_count for ov in info.overloads if ov.layout.is_out_required)
                self.is_valid = False
                self.fail_reason = "OUTPUT_COUNT_MISMATCH"
                logging.error(
                    f"[{self.testcase_name}] API [{self.api_name}] requires exactly "
                    f"{expected} output tensor(s) for Tensor[] 'out', but testcase "
                    f"provides {out_count}. (source: {info.source})"
                )
            elif any_out_required:
                self.is_valid = False
                self.fail_reason = "OUTPUT_COUNT_MISMATCH"
                logging.error(
                    f"[{self.testcase_name}] API [{self.api_name}] requires exactly "
                    f"1 output tensor, but testcase provides {out_count}. "
                    f"(source: {info.source})"
                )

    def _check_top_level_counts(self):
        """Validate that all fields have matching top-level count after normalization."""
        if not self.is_valid:
            return
        top_count = len(self.tensor_view_shapes)
        if len(self.tensor_dtypes) != top_count:
            self.is_valid = False
            self.fail_reason = "DTYPES_COUNT_MISMATCH"
            logging.error(
                f"[{self.testcase_name}] tensor_dtypes top-level count ({len(self.tensor_dtypes)}) "
                f"!= tensor_view_shapes count ({top_count})"
            )
            return
        if self.tensor_formats and len(self.tensor_formats) != top_count:
            self.is_valid = False
            self.fail_reason = "FORMATS_COUNT_MISMATCH"
            logging.error(
                f"[{self.testcase_name}] tensor_formats top-level count ({len(self.tensor_formats)}) "
                f"!= tensor_view_shapes count ({top_count})"
            )
            return
        if self.tensor_storage_shapes and len(self.tensor_storage_shapes) != top_count:
            self.is_valid = False
            self.fail_reason = "STORAGE_SHAPES_COUNT_MISMATCH"
            logging.error(
                f"[{self.testcase_name}] tensor_storage_shapes top-level count ({len(self.tensor_storage_shapes)}) "
                f"!= tensor_view_shapes count ({top_count})"
            )
