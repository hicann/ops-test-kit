#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
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

These tests prove:
  * a **custom** golden receives a *promoted* float32 input (not the original
    float16) when golden_mode=Promote;
  * a **class** golden receives a *promoted* float32 input likewise.

float16 is used (also in DTYPE_PROMOTE_MAP → promotes to float32) because it is
simpler to construct than bfloat16; the promote path under test is identical.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp
from ttk.core_modules.npu.op import output_generation as _mod

_generate_golden = getattr(_mod, '__generate_golden')


def _make_testcase(op_name="custom_promote_op", input_shapes=((4,),),
                   input_dtypes=("float16",),
                   output_shapes=((4,),),
                   output_dtypes=("float16",)):
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
    sw.golden_mode = "Promote"     # <-- the mode under test
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
@patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
@patch('ttk.core_modules.npu.op.output_generation.get_plugin_function')
class TestKernelPromoteWrapCoversAllForms:
    """Under golden_mode=Promote, ALL dispatch forms must see promoted inputs."""

    def test_custom_golden_receives_promoted_float32(self, mock_get_plugin, mock_sw, mock_op_info):
        """Custom golden path: golden(x, **kw) must get float32, not float16."""
        seen_dtypes = []

        def golden(x, **kwargs):
            # Capture the dtype the golden ACTUALLY received.
            if isinstance(x, np.ndarray):
                seen_dtypes.append(x.dtype)
            elif isinstance(x, (list, tuple)):
                for t in x:
                    if isinstance(t, np.ndarray):
                        seen_dtypes.append(t.dtype)
            return x

        mock_get_plugin.return_value = golden          # plugin → custom golden
        mock_sw.return_value = _mock_switches_promote()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(op_name="custom_promote_op",
                              input_dtypes=("float16",),
                              output_dtypes=("float16",))

        _generate_golden(case, ["float16"])

        assert len(seen_dtypes) >= 1, "custom golden was not invoked"
        for d in seen_dtypes:
            # float16 ∈ DTYPE_PROMOTE_MAP → must be promoted to float32
            assert d == np.dtype("float32"), \
                f"custom golden received UN-promoted dtype {d!r}; " \
                f"expected float32 (promoted from float16 under Promote mode)"

    def test_class_golden_receives_promoted_float32(self, mock_get_plugin, mock_sw, mock_op_info):
        """Class golden path: class-inst(x) must get float32, not float16."""
        seen_dtypes = []

        class _ClassGolden:
            def __call__(self, x, **kwargs):
                if isinstance(x, np.ndarray):
                    seen_dtypes.append(x.dtype)
                return x

        mock_get_plugin.return_value = _ClassGolden      # plugin → class golden
        mock_sw.return_value = _mock_switches_promote()
        # bind_by_name needs the input array pooled under the param name "x"
        mock_op_info.return_value.info_of.return_value = {"inputs": [{"name": "x"}]}

        case = _make_testcase(op_name="class_promote_op",
                              input_dtypes=("float16",),
                              output_dtypes=("float16",))

        _generate_golden(case, ["float16"])

        assert len(seen_dtypes) >= 1, "class golden was not invoked"
        for d in seen_dtypes:
            assert d == np.dtype("float32"), \
                f"class golden received UN-promoted dtype {d!r}; " \
                f"expected float32 (promoted from float16 under Promote mode)"

    def test_promote_is_noop_when_mode_not_promote(self, mock_get_plugin, mock_sw, mock_op_info):
        """Sanity: when golden_mode != Promote, input is NOT promoted (stays float16)."""
        seen_dtypes = []

        def golden(x, **kwargs):
            if isinstance(x, np.ndarray):
                seen_dtypes.append(x.dtype)
            return x

        mock_get_plugin.return_value = golden
        sw = _mock_switches_promote()
        sw.golden_mode = "Enable"     # not Promote
        mock_sw.return_value = sw
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(op_name="custom_noop_op",
                              input_dtypes=("float16",),
                              output_dtypes=("float16",))

        _generate_golden(case, ["float16"])

        assert len(seen_dtypes) >= 1, "custom golden was not invoked"
        for d in seen_dtypes:
            assert d == np.dtype("float16"), \
                f"non-Promote mode should NOT promote; got {d!r}"
