# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""container_utils 测试：嵌套 tensor list 的分布推断与展平。"""

import pytest

from ttk.utilities.container_utils import (
    flatten_nested_sequence,
    infer_list_distribution_from_nesting,
)

# -- infer_list_distribution_from_nesting ------------------------------------


@pytest.mark.parametrize(
    "shapes, expected",
    [
        # 两个 flat tensor → (0, 0)
        (((3, 3), (3, 5)), (0, 0)),
        # 两个 tensor_list → (2, 3)
        ((((1,), (2,)), ((3,), (4,), (5,))), (2, 3)),
    ],
    ids=["flat", "list+list"],
)
def test_infer_distribution_core_patterns(shapes, expected):
    """推断嵌套结构的 tensor list 分布：flat=0，list=元素数，None=0。"""
    assert infer_list_distribution_from_nesting(shapes) == expected


# -- flatten_nested_sequence -------------------------------------------------

@pytest.mark.parametrize(
    "shapes, expected",
    [
        # flat 不变
        (((3, 3), (3, 5)), ((3, 3), (3, 5))),
        # 全嵌套展平
        ((((1,), (2,)), ((3,), (4,))), ((1,), (2,), (3,), (4,))),
    ],
    ids=["flat", "all-nested"],
)
def test_flatten_core_patterns(shapes, expected):
    """展平嵌套序列：flat 不变，一层/全嵌套展平，None 保留。"""
    assert flatten_nested_sequence(shapes) == expected
