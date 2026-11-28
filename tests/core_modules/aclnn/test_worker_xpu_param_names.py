# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""XPU input_names / param_order 纯输出过滤测试。

TensorList 展平后 flat 索引与顶层槽位是两套编号，名单须按
pure_output_slot_indexes 整槽过滤并与 _aclnn_xpu_inputs 对应。
覆盖 TensorList 纯输出、中置输出、inplace 声明与无分布恒等形态。
"""

from collections import OrderedDict
from unittest.mock import patch

import numpy

from ttk.core_modules.aclnn.info_keeper import OpApiInfo
from ttk.core_modules.aclnn.testcase import TestcaseAclnn
from ttk.core_modules.aclnn.worker import _aclnn_param_order, _aclnn_xpu_input_names, _aclnn_xpu_inputs


def _make_case(view_shapes, dtypes, storages, outputs, inplace=()):
    case = TestcaseAclnn()
    case.testcase_name = "test_xpu_param_names"
    case.api_name = "aclnnForeachDummy"
    case.tensor_view_shapes = view_shapes
    case.tensor_dtypes = dtypes
    case.np_storages = list(storages)
    case.output_tensor_indexes = outputs
    if inplace:
        case.output_inplace_indexes = inplace
    return case


def _make_op_api_info(params):
    """真实 OpApiInfo：__post_init__ 产出 tensors/scalars 名单，MagicMock 无法复现。"""
    return OpApiInfo(params=OrderedDict(params))


def _zeros(flat_shapes):
    return [numpy.zeros(s, dtype=numpy.float32) for s in flat_shapes]


@patch("ttk.core_modules.aclnn.worker.OpApiInfoKeeper")
def test_tensorlist_pure_output_whole_slot_filtered(mock_keeper):
    """TensorList 输入 + 尾置纯输出：'out' 整槽剔除，names 与 inputs 对应。"""
    mock_keeper.return_value.info_of.return_value = _make_op_api_info(
        [("x", {"type": "aclTensorList*"}), ("out", {"type": "aclTensorList*"})]
    )
    shapes = (((1,), (2,), (3,), (4,)), ((1,), (2,), (3,), (4,)))
    dtypes = (("float32",) * 4, ("float32",) * 4)
    case = _make_case(shapes, dtypes, _zeros([(1,)] * 8), outputs=(1,))
    assert case.pure_output_slot_indexes == frozenset({1})
    assert _aclnn_xpu_input_names(case) == ["x"]
    assert _aclnn_param_order(case) == ["x"]
    assert len(_aclnn_xpu_inputs(case)) == len(_aclnn_xpu_input_names(case))


@patch("ttk.core_modules.aclnn.worker.OpApiInfoKeeper")
def test_middle_output_trailing_input_kept(mock_keeper):
    """中置纯输出：其后输入槽保留。"""
    mock_keeper.return_value.info_of.return_value = _make_op_api_info(
        [
            ("x", {"type": "aclTensorList*"}),
            ("out", {"type": "aclTensorList*"}),
            ("bias", {"type": "aclTensor*"}),
        ]
    )
    shapes = (((1,), (2,)), ((1,), (2,), (3,), (4,)), (5,))
    dtypes = (("float32", "float32"), ("float32",) * 4, "float32")
    case = _make_case(shapes, dtypes, _zeros([(1,), (2,), (1,), (2,), (3,), (4,), (5,)]), outputs=(1,))
    assert _aclnn_xpu_input_names(case) == ["x", "bias"]
    assert _aclnn_param_order(case) == ["x", "bias"]
    assert len(_aclnn_xpu_inputs(case)) == 2


@patch("ttk.core_modules.aclnn.worker.OpApiInfoKeeper")
def test_inplace_declaration_decides_whole_slot(mock_keeper):
    """声明 inplace：x 按输入参与；缺声明：x 整槽判纯输出，names 与 inputs 同步剔除。"""
    mock_keeper.return_value.info_of.return_value = _make_op_api_info([("x", {"type": "aclTensorList*"})])
    shapes = (((1,), (2,), (3,), (4,)),)
    dtypes = (("float32",) * 4,)
    storages = _zeros([(1,), (2,), (3,), (4,)])

    declared = _make_case(shapes, dtypes, storages, outputs=(0,), inplace=(0,))
    assert declared.pure_output_slot_indexes == frozenset()
    assert _aclnn_xpu_input_names(declared) == ["x"]
    assert _aclnn_param_order(declared) == ["x"]
    assert len(_aclnn_xpu_inputs(declared)) == 1

    missing = _make_case(shapes, dtypes, storages, outputs=(0,))
    assert missing.pure_output_slot_indexes == frozenset({0})
    assert _aclnn_xpu_input_names(missing) == []
    assert _aclnn_param_order(missing) == []
    assert _aclnn_xpu_inputs(missing) == []


def test_slot_mapping_no_dist_identity():
    """无 TensorList：flat 与顶层槽位恒等。"""
    case = _make_case(((1,), (2,), (3,)), ("float32",) * 3, _zeros([(1,), (2,), (3,)]), outputs=(2,))
    assert case.pure_output_slot_indexes == frozenset({2})
