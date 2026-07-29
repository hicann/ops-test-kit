#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""
GeirTestcase — GEIR 模式的用例结构，继承自 TestcaseOp（共享 Golden 生成逻辑）
"""

__all__ = ["GeirTestcase"]

from typing import Dict, Optional

from ttk.core_modules.testcase_manager.field_types import FIELD_TYPES
from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp


class GeirTestcase(TestcaseOp):
    __slots__ = ("dyn_input_shapes",)

    non_platform_static_property_headers: Dict[str, tuple] = {
        **TestcaseOp.non_platform_static_property_headers,
        "dyn_input_shapes": (FIELD_TYPES.SHAPELIKE_DYN_NESTED, None, None),
    }
    property_headers: Dict[str, tuple] = {
        **non_platform_static_property_headers,
        **TestcaseOp.static_property_headers,
        **TestcaseOp.special_property_headers,
    }
    complete_headers: Dict[str, tuple] = {
        **TestcaseOp.identity_headers,
        **property_headers,
        **TestcaseOp.option_headers,
    }

    def __init__(self):
        super().__init__()
        self.dyn_input_shapes: Optional[tuple] = None
