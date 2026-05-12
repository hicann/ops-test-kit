#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
testcase base class
"""


__all__ = ["TestcaseBase"]


from abc import ABCMeta
from typing import Dict, Optional, Tuple, Any
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

from .field_types import FIELD_TYPES
from .field_parser import *
from ...utilities import get_global_storage


type_processing_func: Dict[FIELD_TYPES, Callable] = {
    FIELD_TYPES.STRING: process_string,
    FIELD_TYPES.SHAPELIKE_DYN: process_dynamic_shapelike,
    FIELD_TYPES.SHAPELIKE_DYN_EX: process_dynamic_inferable_shapelike,
    FIELD_TYPES.RANGELIKE: rangelike,
    FIELD_TYPES.SHAPELIKE_STC: shapelike_stc,
    FIELD_TYPES.SHAPELIKE_STC_EX: shapelike_stc_ex,
    FIELD_TYPES.SHAPELIKE_FLOAT: shapelike_float,
    FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED: shapelike_float_signed,
    FIELD_TYPES.STRING_CONTAINER: string_container,
    FIELD_TYPES.INT_CONTAINER: int_container,
    FIELD_TYPES.INT: process_int,
    FIELD_TYPES.FLOAT: process_float,
    FIELD_TYPES.BOOL: process_bool,
    FIELD_TYPES.DICT: process_dict,
    FIELD_TYPES.FREE_EVAL: process_eval,
    FIELD_TYPES.SHAPE_STRIDE: shape_stride,
    FIELD_TYPES.SHAPELIKE_STC_NESTED: shapelike_stc_nested,
    FIELD_TYPES.SHAPELIKE_STC_EX_NESTED: shapelike_stc_ex_nested,
    FIELD_TYPES.STRING_CONTAINER_NESTED: string_container_nested,
    FIELD_TYPES.INT_CONTAINER_NESTED: int_container_nested,
    FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED: shapelike_float_signed_nested,
    FIELD_TYPES.SHAPELIKE_FLOAT_NESTED: shapelike_float_nested,
    FIELD_TYPES.FLOAT_OR_CONTAINER_NESTED: float_or_container_nested,
}


class TestcaseBase(metaclass=ABCMeta):
    __slots__ = (
                 # === testcase valid configurations === #
                 "testcase_name",
                 "network_name",
                 "input_data_ranges",
                 "is_enabled",
                 "priority",
                 # precision
                 "precision_tolerances",
                 "absolute_precision",
                 # testcase remark
                 "remark",
                 # soc series (short soc version) to specify to run or disable.
                 "soc_series",
                 # Temp
                 "original_line",
                 "original_dict",
                 # === Runtime parameters below === #
                 "device_id",
                 "is_valid",
                 "fail_reason",
                 # private
                 "_actual_input_data_ranges",
                 )

    identity_headers: Dict[str, tuple] = {
        "testcase_name": (FIELD_TYPES.STRING, None),
        "network_name": (FIELD_TYPES.STRING, None, None),
    }
    special_property_headers: Dict[str, tuple] = {
        "input_data_ranges": (FIELD_TYPES.SHAPELIKE_FLOAT_SIGNED_NESTED, None, ((None, None),)),
        "precision_tolerances": (FIELD_TYPES.SHAPELIKE_FLOAT_NESTED, None, None),
        "absolute_precision": (FIELD_TYPES.FLOAT_OR_CONTAINER_NESTED, None, 1e-8),
    }
    option_headers: Dict[str, tuple] = {
        # Manually controlled property
        "is_enabled": (FIELD_TYPES.BOOL, None, True),
        "remark": (FIELD_TYPES.STRING, None, None),
        "soc_series": (FIELD_TYPES.STRING_CONTAINER, None, None),
        "priority": (FIELD_TYPES.INT, None, 0),
    }
    complete_headers: Dict[str, tuple] = {**identity_headers, **special_property_headers, **option_headers}

    def __init__(self):
        super().__init__()
        self.testcase_name: Optional[str] = None
        self.network_name: Optional[str] = None
        self.input_data_ranges = None
        # precision
        self.precision_tolerances = None
        self.absolute_precision: float = 1e-8
        # control
        self.is_enabled: bool = True
        self.priority: int = 0  # testcase priority.
        self.remark = None
        self.soc_series = None
        # End of testcase valid configurations
        # Temp
        self.original_line: Optional[tuple] = None
        self.original_dict: Optional[dict] = None
        # === Runtime parameters below === #
        self.device_id = None
        self.is_valid: Optional[bool] = True
        self.fail_reason: Optional[str] = None
        # private
        self._actual_input_data_ranges = None

    def __hash__(self):
        return hash(self.testcase_name)

    @property
    def actual_input_data_ranges(self):
        if self._actual_input_data_ranges is None:
            self._actual_input_data_ranges = self.input_data_ranges
        return self._actual_input_data_ranges

    @actual_input_data_ranges.setter
    def actual_input_data_ranges(self, value):
        self._actual_input_data_ranges = value

    @classmethod
    def is_legit_header(cls, header_name: str) -> bool:
        return header_name in cls.complete_headers

    @classmethod
    def has_equivalent_header(cls, header_name: str) -> bool:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return True if cls.complete_headers[header_name][1] else False

    @classmethod
    def get_equivalent_headers(cls, header_name: str) -> Optional[tuple]:
        if cls.has_equivalent_header(header_name):
            return cls.complete_headers[header_name][1]
        return ()

    @classmethod
    def get_all_legit_headers(cls) -> Tuple[str]:
        return tuple(cls.complete_headers.keys())

    @classmethod
    def has_default_value(cls, header_name: str) -> bool:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return len(cls.complete_headers[header_name]) >= 3

    @classmethod
    def get_default_value(cls, header_name: str) -> Any:
        if cls.has_default_value(header_name):
            return cls.complete_headers[header_name][2]
        raise ValueError(f"Could not find default value for header {header_name}, required field missing!")

    @classmethod
    def get_header_type(cls, header_name: str) -> FIELD_TYPES:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return cls.complete_headers[header_name][0]

    @classmethod
    def get_header_func(cls, header_name: str) -> Callable:
        if not cls.is_legit_header(header_name):
            raise ValueError(f"Could not find header {header_name}")
        return type_processing_func[cls.get_header_type(header_name)]

    @classmethod
    def get_all_visible_headers(cls) -> Tuple[str]:
        """All visible headers as csv titles"""
        return cls.get_all_legit_headers()

    @staticmethod
    def supported_rerun_title() -> tuple:
        return ()

    def ready_for_profile(self) -> bool:
        """Ready for profile"""
        return True

    def pick_data(self, titles: Tuple[str]) -> tuple:
        """ Pick testcase input data via titles """
        data = []
        legit_headers = self.get_all_legit_headers()
        for t in titles:
            original_input = self.original_dict.get(t, '')
            if get_global_storage().preserve_original_csv:
                data.append(original_input)
            else:
                data.append(getattr(self, t) if t in legit_headers else original_input)
        return tuple(data)

    def validate(self):
        self._normalize_input_data_range()

    def _normalize_input_data_range(self):
        """if only low is given, fix it to inf. Supports nested structures."""
        if not self.input_data_ranges:
            self.input_data_ranges = ((None, None),)
            return
        self.input_data_ranges = self._normalize_range_recursive(self.input_data_ranges)

    @staticmethod
    def _normalize_range_recursive(ranges):
        result = []
        for r in ranges:
            if r is None:
                result.append(None)
            elif isinstance(r, (tuple, list)) and len(r) > 0 and isinstance(r[0], (tuple, list)):
                result.append(TestcaseBase._normalize_range_recursive(r))
            else:
                r = tuple(r)
                if len(r) >= 2:
                    result.append(r)
                elif len(r) == 0:
                    result.append((None, None))
                else:
                    result.append((r[0], r[0]))
        return tuple(result)
