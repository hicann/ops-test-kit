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
API Testcase Structure
"""


__all__ = ["TestcaseAclnn", "AclnnParamPlan"]


# Standard Packages
import copy
import logging
import numpy.random
from typing import Dict, List, Optional, Tuple, Union, Any
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

# Third-party Packages
from .testcase_tensor_api_base import TensorApiTestcaseBase
from .field_types import FIELD_TYPES
from ..aclnn import OpApiInfoKeeper, OpApiInfo
from ...utilities import get, shape_stride, shape_product_with_strides
from ...utilities import shape_product, parse_dtype, get_dtype_width
from ...utilities.container_utils import infer_list_distribution_from_nesting, flatten_nested_sequence


class AclnnParamPlan:
    """Cached parameter assembly plan for aclnn (op_api) testcases.

    Single source of truth for param layout, resolved once per testcase,
    reused by custom input, custom golden, and profiling.

    Unlike framework_api's ParamPlan (which handles overloads, keyword-only,
    out= separation), aclnn has a single C function signature with all params
    positional. This plan captures:
      - param_layout: ordered list of (kind, name, acl_type, default)
      - tensor_count / scalar_count: for quick validation
    """

    TENSOR = 'tensor'
    SCALAR = 'scalar'
    OTHER = 'other'

    __slots__ = (
        'api_name', 'param_layout', 'tensor_count', 'scalar_count',
    )

    def __init__(self, api_name: str, op_api_info: OpApiInfo):
        self.api_name = api_name
        self.param_layout: List[Tuple[str, str, str, Any]] = []
        self.tensor_count = 0
        self.scalar_count = 0

        for param_name, param_info in op_api_info.params.items():
            acl_type = param_info["type"]
            default = param_info["default"]
            if "aclTensor" in acl_type:
                self.param_layout.append((self.TENSOR, param_name, acl_type, default))
                self.tensor_count += 1
            elif "aclScalar" in acl_type:
                self.param_layout.append((self.SCALAR, param_name, acl_type, default))
                self.scalar_count += 1
            else:
                self.param_layout.append((self.OTHER, param_name, acl_type, default))

    def build_args(self, tensors, scalars, attributes):
        """Build positional args list for custom plugin calls.

        Iterates param_layout in C header order, consuming tensors and scalars
        from their respective queues, and looking up attribute values.

        Args:
            tensors: nested tensor list (from context.tensors)
            scalars: nested scalar list (from context.scalars)
            attributes: dict of API parameters

        Returns:
            list of args in C header function signature order
        """
        args = []
        tensor_queue = list(tensors)
        scalar_queue = list(scalars)

        for kind, name, acl_type, default in self.param_layout:
            if kind == self.TENSOR:
                args.append(tensor_queue.pop(0))
            elif kind == self.SCALAR:
                args.append(scalar_queue.pop(0))
            else:
                args.append(attributes.get(name, default))

        return args


class TestcaseAclnn(TensorApiTestcaseBase):
    """
    Structure for Op Api Profiling
    """

    __slots__ = (
        # === aclnn-specific testcase configurations === #
        "output_inplace_indexes",
        # scalar
        "scalar_dtypes",
        # input attributes
        "attributes",
        # other help configuration
        "scalar_data_ranges",
        # specify dump file prefix
        "dump_file_prefix",
        # manual configurations
        "manual_tensor_binaries",
        # used for compare.
        "manual_golden_binaries",
        # === Runtime parameters below === #
        # torch.Tensor or sequence of torch.Tensor with strides & offsets.
        "tensors",
        "scalars",
        "flatten_tensors",
        "flatten_scalars",
        # list of torch.Tensor with strides & offset or numpy.ndarray (consider as view shape)
        "golden_tensors",
        # private
        "_output_dtypes",
        "_output_view_shapes",
        "_output_view_offsets",
        "_output_view_strides",
        "_output_storage_shapes",
        "_actual_scalar_data_ranges",
        "_pure_attrs",
        "_pure_output_indexes",
        "_inferred_scalar_list_dist",
    )

    identity_headers: Dict[str, tuple] = {
        **TensorApiTestcaseBase.identity_headers,
        "api_name": (FIELD_TYPES.STRING, None),  # Required
    }
    tensor_property_headers: Dict[str, tuple] = {
        "tensor_view_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
        "tensor_formats": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ("ND",)),
        "tensor_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
        "tensor_storage_shapes": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
        "tensor_view_offsets": (FIELD_TYPES.INT_CONTAINER_NESTED, None, ()),
        "tensor_view_strides": (FIELD_TYPES.SHAPELIKE_STC_NESTED, None, ()),
    }
    output_property_headers: Dict[str, tuple] = {
        "output_tensor_indexes": (FIELD_TYPES.INT_CONTAINER, None, None),  # required
        "output_inplace_indexes": (FIELD_TYPES.INT_CONTAINER, None, ()),
    }
    attr_property_headers: Dict[str, tuple] = {
        "attributes": (FIELD_TYPES.DICT, None, {}),
    }
    scalar_property_headers: Dict[str, tuple] = {
        "scalar_dtypes": (FIELD_TYPES.STRING_SCALAR_NESTED, None, ()),
    }
    special_property_headers: Dict[str, tuple] = {
        **TensorApiTestcaseBase.special_property_headers,
        "scalar_data_ranges": (FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED, None, ((None, None),)),
    }
    property_headers: Dict[str, tuple] = {
        **tensor_property_headers, **output_property_headers,
        **attr_property_headers, **scalar_property_headers,
        **special_property_headers
    }
    option_headers: Dict[str, tuple] = {
        **TensorApiTestcaseBase.option_headers,
        # Manually controlled property
        "dump_file_prefix": (FIELD_TYPES.STRING, None, None),
        "manual_tensor_binaries": (FIELD_TYPES.FREE_EVAL, None, ()),
        "manual_golden_binaries": (FIELD_TYPES.STRING_CONTAINER, None, ()),
    }
    complete_headers: Dict[str, tuple] = {**identity_headers, **property_headers, **option_headers}

    def __init__(self):
        super().__init__()
        # output properties
        self.output_inplace_indexes: Optional[tuple] = None
        # scalar or scalarList dtype
        self.scalar_dtypes: Optional[tuple] = None
        # attributes
        self.attributes: Optional[dict] = None
        # others
        self.scalar_data_ranges = None
        # Manual controlled parameters
        self.dump_file_prefix = None
        self.manual_tensor_binaries: Optional[Tuple[str, ...]] = None
        self.manual_golden_binaries = None
        # End of testcase valid configurations
        self.tensors = None
        self.scalars = None
        # flatten tensors and scalars
        self.flatten_tensors = None
        self.flatten_scalars = None
        self.golden_tensors = None
        # Test Runtime Attributes
        # private parameters
        self._output_dtypes: Optional[tuple] = None
        self._output_view_shapes: Optional[tuple] = None
        self._output_view_offsets: Optional[tuple] = None
        self._output_view_strides: Optional[tuple] = None
        self._output_storage_shapes: Optional[tuple] = None
        self._actual_scalar_data_ranges = None
        self._pure_attrs: Optional[dict] = None
        self._pure_output_indexes: Optional[tuple] = None
        self._inferred_scalar_list_dist: Optional[tuple] = None

    @property
    def tensor_bytes(self):
        """Calculate total bytes of all flat tensors based on storage shape and dtype width."""
        bytes_lst: list = []
        for idx, vs in enumerate(self.flat_tensor_view_shapes):
            if vs is None:
                continue
            try:
                bytes_lst.append(shape_product(self.flat_storage_shape(idx)) *
                                 get_dtype_width(get(self.flat_tensor_dtypes, idx))
                                 )
            except:
                bytes_lst.append(0)
        return sum(bytes_lst)

    @property
    def output_dtypes(self) -> tuple:
        """Resolve output dtypes from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes:
          - Single tensor → single dtype string
          - TensorList → tuple of dtype strings
        """
        if self._output_dtypes is None:
            dist = self._get_tensor_list_distribution()
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                dtypes = self.flat_tensor_dtypes[flat_idx:flat_idx + count]
                result.append(dtypes[0] if count == 1 else dtypes)
            self._output_dtypes = tuple(result)
        return self._output_dtypes

    @property
    def flat_output_dtypes(self):
        """Flatten output dtypes to one-per-tensor, matching output_shapes order."""
        if not self.output_dtypes:
            return self.output_dtypes
        result = []
        for element in self.output_dtypes:
            if isinstance(element, (tuple, list)) and element and isinstance(element[0], (tuple, list, str)) and not isinstance(element, str):
                result.extend(element)
            else:
                result.append(element)
        return tuple(result)

    @property
    def output_view_shapes(self) -> tuple:
        """Resolve output view shapes from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_view_shapes is None:
            dist = self._get_tensor_list_distribution()
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                shapes = self.flat_tensor_view_shapes[flat_idx:flat_idx + count]
                result.append(shapes[0] if count == 1 else shapes)
            self._output_view_shapes = tuple(result)
        return self._output_view_shapes

    @property
    def flat_output_view_shapes(self):
        """Flatten output view shapes to one-per-tensor, matching output_shapes order."""
        return flatten_nested_sequence(self.output_view_shapes) if self.output_view_shapes else self.output_view_shapes

    @property
    def output_view_offsets(self) -> tuple:
        """Resolve output view offsets from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_view_offsets is None:
            dist = self._get_tensor_list_distribution()
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                if count == 1:
                    result.append(self.flat_view_offset(flat_idx))
                else:
                    result.append(tuple(self.flat_view_offset(flat_idx + i) for i in range(count)))
            self._output_view_offsets = tuple(result)
        return self._output_view_offsets

    @property
    def flat_output_view_offsets(self):
        """Flatten output view offsets to one-per-tensor, matching output_shapes order."""
        if not self.output_view_offsets:
            return self.output_view_offsets
        result = []
        for element in self.output_view_offsets:
            if isinstance(element, (tuple, list)):
                result.extend(element)
            else:
                result.append(element)
        return tuple(result)

    @property
    def output_view_strides(self) -> tuple:
        """Resolve output view strides from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_view_strides is None:
            dist = self._get_tensor_list_distribution()
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                if count == 1:
                    result.append(self.flat_view_stride(flat_idx))
                else:
                    result.append(tuple(self.flat_view_stride(flat_idx + i) for i in range(count)))
            self._output_view_strides = tuple(result)
        return self._output_view_strides

    @property
    def flat_output_view_strides(self):
        """Flatten output view strides to one-per-tensor, matching output_shapes order."""
        if not self.output_view_strides:
            return self.output_view_strides
        result = []
        for element in self.output_view_strides:
            if isinstance(element, (tuple, list)) and element and isinstance(element[0], (tuple, list)):
                result.extend(element)
            else:
                result.append(element)
        return tuple(result)

    @property
    def output_storage_shapes(self) -> tuple:
        """Resolve output storage shapes from nested output_tensor_indexes through distribution.

        Returns nested structure matching output_tensor_indexes.
        """
        if self._output_storage_shapes is None:
            dist = self._get_tensor_list_distribution()
            result = []
            for idx in self.output_tensor_indexes:
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                if count == 1:
                    result.append(self.flat_storage_shape(flat_idx))
                else:
                    result.append(tuple(self.flat_storage_shape(flat_idx + i) for i in range(count)))
            self._output_storage_shapes = tuple(result)
        return self._output_storage_shapes

    @property
    def flat_output_storage_shapes(self):
        """Flatten output storage shapes to one-per-tensor, matching output_shapes order."""
        if not self.output_storage_shapes:
            return self.output_storage_shapes
        result = []
        for element in self.output_storage_shapes:
            if isinstance(element, (tuple, list)) and element and isinstance(element[0], (tuple, list)):
                result.extend(element)
            else:
                result.append(element)
        return tuple(result)

    @property
    def actual_scalar_data_ranges(self):
        """Return scalar data ranges as tuple of (min, max) pairs, stripping extra fields."""
        data_range = self._actual_scalar_data_ranges or self.scalar_data_ranges
        return data_range if not data_range else tuple([tuple(dr[:2]) for dr in data_range])

    @property
    def full_scalar_data_ranges(self):
        """Return full scalar data ranges including extra fields (e.g., distribution type)."""
        return self._actual_scalar_data_ranges or self.scalar_data_ranges

    @property
    def pure_attrs(self) -> dict:
        """Return attributes that are pure API parameters (not tensor/scalar overrides)."""
        if self._pure_attrs is None:
            self._pure_attrs = {}
            op_api_info = OpApiInfoKeeper().info_of(self.api_name)
            for k, v in self.attributes.items():
                if k not in op_api_info.params:
                    continue
                if k in op_api_info.tensors or k in op_api_info.scalars:
                    continue
                self._pure_attrs.update({k: v})
        return self._pure_attrs

    @property
    def pure_output_indexes(self):
        """Return flat indexes of output tensors that are NOT inplace.

        Computed from nested output_tensor_indexes and output_inplace_indexes
        expanded through distribution to flat indexes.
        """
        if self._pure_output_indexes is None:
            dist = self._get_tensor_list_distribution()
            flat_output = set()
            for idx in self.output_tensor_indexes:
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                flat_output.update(range(flat_idx, flat_idx + count))
            flat_inplace = set()
            for idx in (self.output_inplace_indexes or ()):
                flat_idx = sum(max(d, 1) for d in dist[:idx])
                count = dist[idx] if idx < len(dist) and dist[idx] > 0 else 1
                flat_inplace.update(range(flat_idx, flat_idx + count))
            self._pure_output_indexes = sorted(flat_output - flat_inplace)
        return self._pure_output_indexes

    @actual_scalar_data_ranges.setter
    def actual_scalar_data_ranges(self, value):
        self._actual_scalar_data_ranges = value

    def validate(self):
        """Run all validation checks on testcase configuration."""
        super().validate()
        self._check_api_name()
        self._normalize_compressed_fields()
        self._check_tensor_parm()
        self._parse_tensor_dtypes()
        self._check_tensor_list_configuration()
        self._check_scalar_list_configuration()
        self._auto_fill_output_inplace_indices()
        self._check_output_configuration()
        self._parse_scalar_dtypes()
        self._check_params_count()

    def _normalize_compressed_fields(self):
        """Expand compressed tensor/scalar fields to fully specified nested format.

        After normalization each field matches tensor_view_shapes nesting exactly:
          - Per-TensorList: fully expanded, no len-1 inner tuples
          - Top-level: fully expanded, no len-1 broadcast
        e.g. tensor_dtypes ('float32',) with dist (2,0) -> (('float32','float32'),'float32')
        """
        if not self.is_valid or not self.tensor_view_shapes:
            return
        dist = self._get_tensor_list_distribution()
        if dist:
            for field_name in ('tensor_dtypes', 'tensor_formats', 'tensor_view_offsets',
                               'tensor_view_strides', 'tensor_storage_shapes'):
                self._normalize_field_by_dist(field_name, dist)
            self._normalize_input_data_ranges(dist)
        sdist = self._get_scalar_list_distribution()
        if sdist:
            self._normalize_field_by_dist('scalar_dtypes', sdist)

    def _get_scalar_list_distribution(self):
        """Derive ScalarList distribution from API info + scalar_dtypes.

        Uses API info to determine which positions are aclScalarList*.
        For ScalarList positions, takes element count from scalar_dtypes.
        For Scalar positions, marks as 0 (supports compressed format).

        Returns a tuple where each element corresponds to a top-level parameter:
          >0 means ScalarList with that many scalars, 0 means single scalar.
        Cached in _inferred_scalar_list_dist.
        """
        if self._inferred_scalar_list_dist is not None:
            return self._inferred_scalar_list_dist
        if not self.api_name or not self.scalar_dtypes:
            return ()
        try:
            op_api_info = OpApiInfoKeeper().info_of(self.api_name)
        except Exception:
            return ()
        if not op_api_info.scalars:
            return ()
        result = []
        for idx, name in enumerate(op_api_info.scalars):
            param_type = op_api_info.params[name]["type"]
            if param_type == "aclScalarList*":
                element = self.scalar_dtypes[idx] if idx < len(self.scalar_dtypes) else ()
                if isinstance(element, (tuple, list)):
                    result.append(len(element))
                elif isinstance(element, str):
                    # 单个 dtype string 视为 ScalarList 中包含 1 个 scalar
                    result.append(1)
                else:
                    result.append(0)
            else:
                result.append(0)
        self._inferred_scalar_list_dist = tuple(result)
        return self._inferred_scalar_list_dist

    @property
    def flat_scalar_dtypes(self):
        """Flatten nested scalar_dtypes to per-scalar dtypes.

        Uses API-derived distribution to distinguish ScalarList from Scalar.
        ScalarList elements are flattened to individual dtype strings.
        Scalar elements are kept as-is.
        """
        if not self.scalar_dtypes:
            return self.scalar_dtypes
        dist = self._get_scalar_list_distribution()
        if not dist:
            return self.scalar_dtypes
        flat_count = sum(max(d, 1) for d in dist)
        if len(self.scalar_dtypes) == flat_count:
            return self.scalar_dtypes
        if len(self.scalar_dtypes) == len(dist):
            result = []
            for i, d in enumerate(dist):
                element = self.scalar_dtypes[i]
                if d > 0:
                    if isinstance(element, (tuple, list)):
                        result.extend(element)
                    elif isinstance(element, str):
                        result.append(element)
                    else:
                        result.append(element)
                else:
                    result.append(element)
            return tuple(result)
        if len(self.scalar_dtypes) == 1:
            return (self.scalar_dtypes[0],) * flat_count
        return self.scalar_dtypes

    @property
    def flat_scalar_data_ranges(self):
        """Flatten nested scalar_data_ranges to per-scalar (min, max) pairs.

        Range tuples (None, 1.0) are atomic values, same as flat_input_data_ranges.
        """
        if not self.scalar_data_ranges:
            return self.scalar_data_ranges
        dist = self._get_scalar_list_distribution()
        if not dist:
            return self.scalar_data_ranges
        flat_count = sum(max(d, 1) for d in dist)
        if len(self.scalar_data_ranges) == flat_count:
            return self.scalar_data_ranges
        if len(self.scalar_data_ranges) == 1:
            return (self.scalar_data_ranges[0],) * flat_count
        is_nested = any(
            isinstance(e, (tuple, list)) and len(e) > 0 and isinstance(e[0], (tuple, list))
            for e in self.scalar_data_ranges if e is not None
        )
        if is_nested:
            return flatten_nested_sequence(self.scalar_data_ranges)
        if len(self.scalar_data_ranges) == len(dist):
            result = []
            for i, num in enumerate(dist):
                val = self.scalar_data_ranges[i]
                if num > 0:
                    result.extend([val] * num)
                else:
                    result.append(val)
            return tuple(result)
        return flatten_nested_sequence(self.scalar_data_ranges)

    @staticmethod
    def _expand_by_distribution(values, distribution):
        """Expand a short tuple to match flattened tensor count.

        e.g. values=('float32',), distribution=(2,0) -> ('float32','float32')
        """
        if not values or not distribution:
            return values
        result = []
        vi = 0
        for num in distribution:
            if num == 0:
                result.append(get(values, vi))
                vi += 1
            else:
                for _ in range(num):
                    result.append(get(values, vi))
                vi += 1
        return tuple(result)

    def is_torch_dtype_support(self) -> bool:
        """Check if all dtypes in testcase are supported by the current torch natively."""
        from ttk.utilities.dtypes import is_torch_native_dtype
        for dtype in self.flat_tensor_dtypes:
            if dtype is not None and not is_torch_native_dtype(dtype):
                return False
        for dtype in self.scalar_dtypes:
            if dtype is not None and not is_torch_native_dtype(dtype):
                return False
        return True

    def get_param_plan(self):
        """Resolve and cache parameter assembly plan for this testcase.

        The plan captures the C function signature layout once,
        then custom input, custom golden, and profiling all reuse it
        — same param order everywhere.

        Returns:
            AclnnParamPlan or None (if API info unavailable)
        """
        if self._param_plan_cache is not None:
            return self._param_plan_cache
        if not self.api_name:
            return None
        try:
            op_api_info = OpApiInfoKeeper().info_of(self.api_name)
            if op_api_info is None:
                return None
            plan = AclnnParamPlan(self.api_name, op_api_info)
            self._param_plan_cache = plan
            return plan
        except Exception:
            return None

    @staticmethod
    def _tensor_param_valid(view_shape, view_stride, view_offset, storage_shape):
        """Validate that view_shape/stride/offset fit within storage_shape."""
        if not view_stride and not view_offset and not storage_shape:
            return True
        if view_offset < 0:
            logging.error(f"Negative view_offset is not supported: {view_offset}")
            return False
        if not view_stride:
            view_stride = shape_stride(view_shape)
        if not storage_shape:
            storage_shape = view_shape
        if len(view_shape) != len(view_stride):
            logging.error(f"Rank of view_shape/view_strides should be same: "
                          f"{view_shape} vs {view_stride}")
            return False
        for idx, v in enumerate(view_stride):
            if v < 0:
                logging.error(f"Negative view_strides are not supported: {view_stride}.")
                return False
        view_numel = shape_product_with_strides(view_shape, view_stride)
        if view_numel == 0:
            # (a tensor with arbitrary 0 dims)'s storage can have any numel
            return True
        storage_numel = shape_product(storage_shape)
        if view_numel + view_offset > storage_numel:
            logging.error(f"view_shape ({view_shape}), view_strides ({view_stride}), view_offset ({view_offset}) "
                          f"requiring a storage with elements {view_numel + view_offset} "
                          f"is out of bounds for storage with elements {storage_numel} ({storage_shape})")
            return False
        return True

    @staticmethod
    def _infer_storage_shape(shape, stride, offset):
        """Infer minimum storage shape from view shape, stride and offset."""
        min_storage_size = shape_product_with_strides(shape, stride)

        def find_original_dims(_stride):
            dims = []
            for i in range(len(_stride) - 1):
                if _stride[i] == 0:
                    dims.append(1)
                elif _stride[i + 1] == 0:
                    dims.append(_stride[i])
                elif _stride[i] % _stride[i + 1] == 0:
                    dims.append(_stride[i] // _stride[i + 1])
                else:
                    dims.append(1)
            dims.append(_stride[-1] or 1)
            return dims

        original_dims = find_original_dims(stride)
        while True:
            total_elements = 1
            for dim in original_dims:
                total_elements *= dim
            if total_elements >= min_storage_size:
                break
            original_dims[0] += 1

        original_dims[-1] += offset
        return tuple(original_dims)

    def _random_non_contiguous(self, view_shape, offset):
        """Generate random non-contiguous stride and storage shape for a given view shape."""
        if len(view_shape) == 1:
            if view_shape[0] >= 1024 * 1024 * 512:  # 2GB for fp32, 1GB for bfp16/fp16
                return (1,), view_shape + offset
            else:
                return (2,), self._infer_storage_shape(view_shape, (2,), offset)
        elif all([view_shape[idx] == view_shape[0]
                  for idx in range(len(view_shape))]):
            stride = list(shape_stride(view_shape))
            ex_idx = sorted(numpy.random.choice(len(stride), size=2, replace=False))
            stride[ex_idx[0]], stride[ex_idx[1]] = stride[ex_idx[1]], stride[ex_idx[0]]
            ss = list(copy.deepcopy(view_shape))
            ss[-1] += offset
            return tuple(stride), tuple(ss)
        else:
            vs = list(view_shape)
            ex_idx = sorted(numpy.random.choice(len(vs), size=2, replace=False))
            vs[ex_idx[0]], vs[ex_idx[1]] = vs[ex_idx[1]], vs[ex_idx[0]]
            stride = shape_stride(vs)
            vs[-1] += offset
            return stride, tuple(vs)

    def _check_api_name(self):
        """Validate api_name is specified and exists in OpApiInfo."""
        if not self.api_name:
            self.is_valid = False
            self.fail_reason = "API_NAME_MISSING"
            logging.error("api_name must be specified.")
        if not OpApiInfoKeeper().has_api(self.api_name):
            self.is_valid = False
            self.fail_reason = "OP_API_NOT_FOUND"
            logging.error(f"OpApi [{self.api_name}] is not found.")

    def _check_tensor_parm(self):
        """Validate view/stride/offset/storage consistency for each flat tensor."""
        if not self.is_valid:
            return
        self._check_shape_nesting(self.tensor_view_strides, "tensor_view_strides")
        self._check_shape_nesting(self.tensor_storage_shapes, "tensor_storage_shapes")
        if not self.is_valid:
            return
        flat_strides = self.flat_tensor_view_strides or ()
        flat_offsets = self.flat_tensor_view_offsets or ()
        flat_storages = self.flat_tensor_storage_shapes or ()
        for idx, view_shape in enumerate(self.flat_tensor_view_shapes):
            if view_shape is None:
                continue
            view_stride = get(flat_strides, idx, out_of_range=())
            view_offset = get(flat_offsets, idx, out_of_range=0)
            storage_shape = get(flat_storages, idx, out_of_range=())
            if not self._tensor_param_valid(view_shape, view_stride,
                                            view_offset, storage_shape):
                self.is_valid = False
                self.fail_reason = "TENSOR_PARAM_INVALID"
                return

    def _check_shape_nesting(self, field, field_name):
        """Validate that field's top-level count matches tensor_view_shapes when nested.

        view_strides and storage_shapes must either be empty, flat (match flat_count),
        or have the same top-level count as tensor_view_shapes.
        """
        if not self.is_valid or not field or not self.tensor_view_shapes:
            return
        top_count = len(self.tensor_view_shapes)
        if len(field) not in (0, 1, len(self.flat_tensor_view_shapes), top_count):
            self.is_valid = False
            self.fail_reason = "TENSOR_PARAM_INVALID"
            logging.error(f"{field_name} length [{len(field)}] is inconsistent with "
                          f"tensor_view_shapes top-level count [{top_count}].")

    def _build_non_contiguous_tensor(self):
        """Auto-generate random non-contiguous strides and storage shapes for tensors that lack them."""
        def __fill(_fill_size: int, _fill_ele: Any,
                   _to_fill: Union[list, tuple]):
            if not _to_fill:
                _to_fill = [_fill_ele] * _fill_size
            _num = len(_to_fill)
            if _num < _fill_size:
                _to_fill = list(_to_fill).append([_fill_ele] * (_fill_size - _num))
            return list(_to_fill)

        def _random_offset():
            return numpy.random.randint(0, 32)

        if not self.is_valid:
            return
        flat_shapes = self.flat_tensor_view_shapes
        tensor_num = len(flat_shapes)
        view_strides = __fill(tensor_num, (), self.flat_tensor_view_strides)
        view_offsets = __fill(tensor_num, None, self.flat_tensor_view_offsets)
        storage_shapes = __fill(tensor_num, (), self.flat_tensor_storage_shapes)
        for idx, view_shape in enumerate(flat_shapes):
            # if storage shape is specified or scalar tensor. skip
            if view_shape is None or len(storage_shapes[idx]) > 0 or len(view_shape) == 0:
                # if storage shape is specified or scalar tensor. skip
                if view_offsets[idx] is None:
                    view_offsets[idx] = 0
                continue
            if view_offsets[idx] is None:
                view_offsets[idx] = _random_offset()
            if view_offsets[idx] < 0:
                # invalid !! leave it to validate later.
                continue
            if not view_strides[idx]:
                # neither storage shape or stride is specified. random it.
                view_strides[idx], storage_shapes[idx] = self._random_non_contiguous(view_shape,
                                                                                     view_offsets[idx])
            else:
                # if storage shape is not specified. try to infer it.
                if len(view_shape) != len(view_strides[idx]):
                    # invalid !! leave it to validate later.
                    continue
                if any([d < 0 for d in view_strides[idx]]):
                    # invalid !! leave it to validate later.
                    continue
                storage_shapes[idx] = self._infer_storage_shape(view_shape, view_strides[idx],
                                                                view_offsets[idx])
        self.tensor_view_strides = tuple(view_strides)
        self.tensor_view_offsets = tuple(view_offsets)
        self.tensor_storage_shapes = tuple(storage_shapes)

    @staticmethod
    def _recursively_parse(field, parse_fn):
        """Recursively apply parse_fn to each element, preserving nesting structure."""
        if not field:
            return field
        result = []
        for item in field:
            if isinstance(item, (tuple, list)):
                result.append(tuple(parse_fn(d) for d in item))
            else:
                result.append(parse_fn(item))
        return tuple(result)

    def _parse_tensor_dtypes(self):
        """Parse and normalize tensor dtype strings to standard form, preserving nesting."""
        if not self.is_valid:
            return
        try:
            if self.tensor_dtypes:
                self.tensor_dtypes = self._recursively_parse(self.tensor_dtypes, parse_dtype)
        except:
            self.is_valid = False
            self.fail_reason = "TENSOR_DTYPES_INVALID"
            logging.exception(f"Tensor dtypes parse failed: {self.tensor_dtypes}")

    def _parse_scalar_dtypes(self):
        """Parse and normalize scalar dtype strings to standard form, preserving nesting."""
        if not self.is_valid:
            return
        try:
            if self.scalar_dtypes:
                self.scalar_dtypes = self._recursively_parse(self.scalar_dtypes, parse_dtype)
        except:
            self.is_valid = False
            self.fail_reason = "SCALAR_DTYPES_INVALID"
            logging.exception(f"Scalar dtypes parse failed: {self.scalar_dtypes}")

    def _check_param_configuration(self, case_values, api_param_names, kind, is_nested_fn):
        """Generic check for tensor/scalar parameter count and type matching against API definition.

        Args:
            case_values: Nested testcase values (tensor_view_shapes or scalar_dtypes).
            api_param_names: Ordered parameter names from API info.
            kind: 'Tensor' or 'Scalar' for error messages.
            is_nested_fn: Callable to detect if an element represents a List type.
        """
        if not self.is_valid:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        api_count = len(api_param_names)
        case_count = len(case_values) if case_values else 0
        if case_count == 0 and api_count == 0:
            return
        if case_count != api_count:
            self.is_valid = False
            self.fail_reason = f"{kind}_COUNT_MISMATCH"
            logging.error(f"L2 interface [{self.api_name}] has {api_count} {kind.lower()} parameters, "
                          f"but testcase configured {case_count}.")
            return
        single_type = f"acl{kind}*"
        list_type = f"acl{kind}List*"
        for idx, element in enumerate(case_values):
            name = api_param_names[idx]
            typ = op_api_info.params[name]["type"]
            if is_nested_fn(element):
                if typ != list_type:
                    self.is_valid = False
                    self.fail_reason = "PARAM_TYPE_MISMATCH"
                    logging.error(f"Parameter [{name}] type is [{typ}]. "
                                  f"But got {kind}List.")
                    return
            elif element is not None:
                if typ != single_type:
                    self.is_valid = False
                    self.fail_reason = "PARAM_TYPE_MISMATCH"
                    logging.error(f"Parameter [{name}] type is [{typ}]. "
                                  f"But got {kind}.")
                    return

    @staticmethod
    def _is_tensor_list_element(element):
        """Check if element represents a TensorList (tuple of shape tuples)."""
        return isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list))

    @staticmethod
    def _is_scalar_list_element(element):
        """Check if element represents a ScalarList (tuple of dtype strings, not a single string)."""
        return isinstance(element, (tuple, list)) and len(element) > 0 and not isinstance(element[0], str)

    def _check_tensor_list_configuration(self):
        """Validate tensor parameters match API definition in count and type (Tensor/TensorList)."""
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        self._check_param_configuration(
            self.tensor_view_shapes, op_api_info.tensors, "Tensor",
            self._is_tensor_list_element)

    def _check_scalar_list_configuration(self):
        """Validate scalar parameters match API definition in count and type (Scalar/ScalarList).

        Uses API info to determine which positions are aclScalarList*,
        since scalar_dtypes nesting alone cannot distinguish ScalarList from multiple Scalars.
        """
        if not self.is_valid:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        api_scalars = op_api_info.scalars
        case_count = len(self.scalar_dtypes) if self.scalar_dtypes else 0
        api_count = len(api_scalars)
        if case_count == 0 and api_count == 0:
            return
        if case_count != api_count:
            self.is_valid = False
            self.fail_reason = "Scalar_COUNT_MISMATCH"
            logging.error(f"L2 interface [{self.api_name}] has {api_count} scalar parameters, "
                          f"but testcase configured {case_count}.")
            return
        for idx, name in enumerate(api_scalars):
            typ = op_api_info.params[name]["type"]
            element = self.scalar_dtypes[idx]
            is_list_type = (typ == "aclScalarList*")
            is_nested = isinstance(element, (tuple, list)) and len(element) > 0
            if is_list_type and not is_nested:
                self.is_valid = False
                self.fail_reason = "PARAM_TYPE_MISMATCH"
                logging.error(f"Parameter [{name}] type is [{typ}]. "
                              f"But scalar_dtypes element is not a tuple/list (expected ScalarList). "
                              f"Got: {element}")
                return
            if not is_list_type and is_nested and isinstance(element[0], (tuple, list)):
                self.is_valid = False
                self.fail_reason = "PARAM_TYPE_MISMATCH"
                logging.error(f"Parameter [{name}] type is [{typ}]. "
                              f"But got ScalarList.")
                return

    def _auto_fill_output_inplace_indices(self):
        """
        automatic fill output_inplace_indexes
        as per parameter's name (ends with `Ref`)
        """
        if not self.is_valid:
            return
        if self.output_inplace_indexes:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        ref_lst = [n for n in op_api_info.tensors if n.endswith('Ref')]
        ref_cnt = len(ref_lst)
        if ref_cnt == 0:
            return
        logging.warning(f"L2 interface [{self.api_name}] has some inplace tensors: {ref_lst}. "
                        f"But `output_inplace_indexes` is not configured. "
                        f"Try to fill it automatically.")
        inplace_indices = []
        for idx, element in enumerate(self.tensor_view_shapes):
            param_name = op_api_info.tensors[idx]
            is_nested = isinstance(element, (tuple, list)) and len(element) > 0 and isinstance(element[0], (tuple, list))
            if param_name in ref_lst:
                if element is None:
                    raise RuntimeError(f"Parameter [{param_name}] is INPLACE type."
                                       f" But got `None` (consider as invalid).")
                inplace_indices.append(idx)
        self.output_inplace_indexes = tuple(inplace_indices)

    def _check_output_configuration(self):
        """
        1. if output_tensor_indexes is not set,
           configure it as (-1,) if not inplace input tensor,
           otherwise tuple of indexes of inplace input tensor.
        2. output_inplace_indexes should be included in output_indexes
        """
        if not self.is_valid:
            return
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self.api_name)
        ref_indexes = tuple([idx for idx, n in enumerate(op_api_info.tensors)
                             if n.endswith('Ref')])
        if not self.output_tensor_indexes:
            # try to fill it automatically...
            self.output_tensor_indexes = ref_indexes or (-1,)
        # correct negative indexes to positive ones
        self.output_tensor_indexes = tuple([idx + len(self.tensor_view_shapes) if idx < 0
                                            else idx for idx in self.output_tensor_indexes])
        out_of_ranges = [idx for idx in self.output_tensor_indexes
                         if idx < 0 or idx >= len(self.tensor_view_shapes)]
        if out_of_ranges:
            self.is_valid = False
            self.fail_reason = "OUTPUT_INDEX_INVALID"
            logging.error(f"Indexes in `output_tensor_indexes` {self.output_tensor_indexes} "
                          f"out of range of `tensor_view_shapes`'s indexes: "
                          f"[0, {len(self.tensor_view_shapes)})")
            return
        if not self.output_inplace_indexes:
            return
        # correct negative indexes to positive ones
        self.output_inplace_indexes = tuple([idx + len(self.tensor_view_shapes) if idx < 0
                                             else idx for idx in self.output_inplace_indexes])
        out_of_ranges = [idx for idx in self.output_inplace_indexes
                         if idx < 0 or idx >= len(self.tensor_view_shapes)]
        if out_of_ranges:
            self.is_valid = False
            self.fail_reason = "OUTPUT_INPLACE_INDEX_INVALID"
            logging.error(f"Indexes in `output_inplace_indexes` {self.output_inplace_indexes} "
                          f"out of range of `tensor_view_shapes`'s indexes: "
                          f"[0, {len(self.tensor_view_shapes)})")
            return
        for ii in self.output_inplace_indexes:
            if ii in self.output_tensor_indexes:
                continue
            self.is_valid = False
            self.fail_reason = "OUTPUT_INDEX_MISMATCH"
            logging.error(f"Indexes in `output_tensor_indexes` and "
                          f"`output_inplace_indexes` mismatch: "
                          f"{self.output_tensor_indexes} vs {self.output_inplace_indexes}")
            return

    def _check_params_count(self):
        """Validate total parameter count (tensor + scalar + attribute) matches API definition."""
        if not self.is_valid:
            return
        op_api_info = OpApiInfoKeeper().info_of(self.api_name)
        tensor_count = len(self.tensor_view_shapes or ())
        scalar_count = len(self.scalar_dtypes or ())
        attrs = self.pure_attrs
        attr_count = len(attrs.keys())
        case_param_count = tensor_count + scalar_count + attr_count
        if case_param_count != len(op_api_info.params.keys()):
            self.is_valid = False
            self.fail_reason = "PARAM_COUNT_MISMATCH"
            logging.error(f"OpApi [{self.api_name}] L2 interface has "
                          f"{len(op_api_info.params.keys())} parameters: "
                          f"{op_api_info.params.keys()}. "
                          f"But testcase configured {case_param_count}: "
                          f"Tensor: {tensor_count}, Scalar: {scalar_count}, "
                          f"Attribute: {attr_count}")
