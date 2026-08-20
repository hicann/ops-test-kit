#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Integration coverage for the ACLNN **class-form** golden dispatch in
``ttk.core_modules.npu.op_api.golden_generation.GoldenGenerator``. The full
wiring under test is::

    GoldenGenerator(case).gen()
      -> _generate_golden
           -> _import_golden_funcs: get_plugin_function(api, "golden", "aclnn")
                -> returns a CLASS
           -> _invoke_golden: isinstance(golden_func, type) is True
           -> _invoke_class
                pool = _named_values()   # OpApiInfoKeeper names + tensors + scalars + attrs
                cls.__init__ is object.__init__ ? cls() : bind_by_name(__init__, pool)
                bind_by_name(inst.__call__, pool) -> inst(*ca, **ck)
      -> numeric numpy arrays

ACLNN ``_invoke_class`` differs from the KERNEL one in pool construction
(``_named_values`` derives tensor names from a real ``OpApiInfo`` and merges
scalars + attributes), so this exercises the ACLNN-specific path that the
KERNEL test cannot reach. ``bind_by_name`` is unit-tested elsewhere; this
covers the *integration* (context -> OpApiInfo pool build -> __init__/__call__
bind -> result) that was previously uncovered for the class form.
"""

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfo
from ttk.core_modules.npu.op_api.golden_generation import GoldenGenerator
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn


# --------------------------------------------------------------------------- #
# Golden classes — emulate what a real aclnn plugin author would ship.
# --------------------------------------------------------------------------- #
class _AclnnNegScale:
    """Partial split: ``__init__`` takes a config attr, ``__call__`` takes the input tensor."""
    def __init__(self, *, scale):
        self.scale = scale

    def __call__(self, x):                       # noqa: N803 (short param mirrors real plugins)
        return [x * self.scale]


class _AclnnDouble:
    """No custom ``__init__`` — exercises the ``cls()`` guard in ``_invoke_class``."""
    def __call__(self, x):                       # noqa: N803
        return [x * 2]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Enable"      # not Promote → _golden_mode is a no-op
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


def _make_op_api_info(tensor_names, scalar_names=()):
    """Build a real OpApiInfo: tensor names → aclTensor*, scalar names → aclScalar*.

    A real dataclass is needed because ``_named_values`` reads ``.tensors`` and
    ``.scalars`` attributes produced by ``OpApiInfo.__post_init__``; a MagicMock
    could not faithfully reproduce the tensor/scalar split.
    """
    params = OrderedDict()
    for n in tensor_names:
        params[n] = {"type": "aclTensor*"}
    for n in scalar_names:
        params[n] = {"type": "aclScalar*"}
    return OpApiInfo(params=params)


def _make_testcase(api_name, tensors, attributes, scalars=None):
    case = TestcaseAclnn()
    case.testcase_name = f"test_{api_name}_class_golden"
    case.api_name = api_name
    case.tensors = list(tensors)
    case.scalars = scalars
    case._flat_scalars = scalars
    case.attributes = attributes
    # Short-circuit the property computation (avoid touching distribution/out-index plumbing).
    case.output_tensor_indexes = ()
    case._pure_output_indexes = ()
    case.manual_golden_binaries = None
    return case


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch('ttk.core_modules.npu.op_api.golden_generation.OpApiInfoKeeper')
@patch('ttk.core_modules.npu.op_api.golden_generation.get_global_storage')
@patch('ttk.core_modules.npu.op_api.golden_generation.get_plugin_function')
class TestAclnnClassGoldenDispatch:
    """class-form golden (isinstance(golden_func, type)) → _invoke_class end-to-end."""

    def test_partial_split_init_takes_attr_call_takes_input(self, mock_get_plugin, mock_sw,
                                                            mock_op_info):
        """``_AclnnNegScale(scale=3)`` instantiated; ``__call__(x)`` returns ``x*3``.

        ``scale`` here flows in as a *scalar* (aclnn convention), proving the
        ``_named_values`` scalar-name merge feeds ``__init__``'s keyword-only ``scale``.
        """
        mock_get_plugin.return_value = _AclnnNegScale    # → isinstance(type) branch
        mock_sw.return_value = _mock_switches()
        # OpApiInfo: one input tensor named "x", one scalar named "scale".
        mock_op_info.return_value.info_of.return_value = _make_op_api_info(
            tensor_names=["x"], scalar_names=["scale"])

        inp = np.array([1.0, 2.0, 3.0], dtype="float32")
        case = _make_testcase("aclnnNegScale",
                              tensors=[inp],
                              attributes={},
                              scalars=[3])

        golden = GoldenGenerator(case)._generate_golden()

        assert len(golden) >= 1
        assert isinstance(golden[0], np.ndarray), \
            f"expected ndarray, got sentinel/string: {golden[0]!r}"
        np.testing.assert_array_equal(golden[0], np.array([3.0, 6.0, 9.0], dtype="float32"))
        mock_get_plugin.assert_called_once()

    def test_dispatch_routes_class_not_callable_path(self, mock_get_plugin, mock_sw,
                                                     mock_op_info):
        """A class is ``Callable`` but MUST route to ``_invoke_class`` (not the custom-callable
        branch that would do ``golden_func(*args, **kwargs)`` via the param plan, which needs a
        real OpApiInfo + tensor plan). Guards the ``isinstance(type)`` ordering in
        ``_invoke_golden``: a class leaking to the custom branch would fail building the plan."""
        mock_get_plugin.return_value = _AclnnNegScale
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = _make_op_api_info(
            tensor_names=["x"], scalar_names=["scale"])

        inp = np.array([1.0, 2.0, 3.0], dtype="float32")
        case = _make_testcase("aclnnNegScale",
                              tensors=[inp],
                              attributes={},
                              scalars=[4])

        golden = GoldenGenerator(case)._generate_golden()

        # If mis-routed to the custom-callable branch, golden[0] would be the
        # string "GOLDEN_FAILURE"; the class path yields a numeric ndarray.
        assert isinstance(golden[0], np.ndarray), \
            f"class golden must hit _invoke_class, not the custom-callable branch: {golden[0]!r}"
        np.testing.assert_array_equal(golden[0], np.array([4.0, 8.0, 12.0], dtype="float32"))
