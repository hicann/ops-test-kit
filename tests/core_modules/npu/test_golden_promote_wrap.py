#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Regression test for the golden Promote-context wrap lift.

Before the fix, `__golden_mode` (the dtype-promote context for
`--golden-mode=promote`) only wrapped the numpy/torch **builtin** and **class**
golden paths inside KERNEL `__invoke_golden`. The custom + torch/tf-adapter
paths ran UNGUARDED, so under `golden_mode=Promote`, bfloat16/float16 inputs
were NOT promoted before those goldens ran → inaccurate "true value".

The fix lifts the `__golden_mode` wrap to the OUTER dispatch so ALL forms
(class / numpy / torch / tf / custom) are guarded by a single wrap point.

float16 is used (also in DTYPE_PROMOTE_MAP → promotes to float32) because it is
simpler to construct than bfloat16; the promote path under test is identical.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.npu.op import output_generation as _mod
from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp

_generate_golden = getattr(_mod, "__generate_golden")


def _make_custom_golden(seen_dtypes):
    """构造 custom golden 函数，捕获实际接收的 dtype。"""

    def golden(x, **kwargs):
        if isinstance(x, np.ndarray):
            seen_dtypes.append(x.dtype)
        elif isinstance(x, (list, tuple)):
            for t in x:
                if isinstance(t, np.ndarray):
                    seen_dtypes.append(t.dtype)
        return x

    return golden


def _make_testcase(
    op_name="custom_promote_op",
    input_shapes=((4,),),
    input_dtypes=("float16",),
    output_shapes=((4,),),
    output_dtypes=("float16",),
):
    case = TestcaseOp()
    case.testcase_name = f"test_{op_name}_promote_wrap"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = input_shapes
    case.output_ori_shapes = output_shapes
    n_in = len(input_shapes)
    n_out = len(output_shapes or ())
    case.input_formats = ("ND",) * n_in
    case.input_ori_formats = ("ND",) * n_in
    case.output_formats = ("ND",) * n_out
    case.output_ori_formats = ("ND",) * n_out
    case.input_data_ranges = (None,) * n_in
    case.attributes = {}
    case.input_arrays = tuple(np.ones(s, dtype=d) for s, d in zip(input_shapes, input_dtypes))
    case.original_input_arrays = None
    return case


def _mock_switches_promote():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Promote"  # <-- the mode under test
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch("ttk.core_modules.npu.op.output_generation.OpInfoKeeper")
@patch("ttk.core_modules.npu.op.output_generation.get_global_storage")
@patch("ttk.core_modules.npu.op.output_generation.get_plugin_function")
class TestKernelPromoteWrapCoversAllForms:
    """Under golden_mode=Promote, ALL dispatch forms must see promoted inputs."""

    def test_golden_receives_promoted_float32(self, mock_get_plugin, mock_sw, mock_op_info):
        """Promote 模式下，custom golden 路径应收到提升后的 float32 输入。"""
        seen_dtypes = []

        mock_get_plugin.return_value = _make_custom_golden(seen_dtypes)
        mock_sw.return_value = _mock_switches_promote()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(op_name="custom_promote_op", input_dtypes=("float16",), output_dtypes=("float16",))

        _generate_golden(case, ["float16"])

        assert len(seen_dtypes) >= 1, "golden was not invoked"
        for d in seen_dtypes:
            # float16 ∈ DTYPE_PROMOTE_MAP → must be promoted to float32
            assert d == np.dtype("float32"), (
                f"golden received UN-promoted dtype {d!r}; expected float32 (promoted from float16 under Promote mode)"
            )
