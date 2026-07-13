#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Integration coverage for the KERNEL **class-form** golden dispatch —
the refactor's headline feature. The full wiring under test is::

    __generate_golden
      -> get_plugin_function(op, "golden", "kernel")  -> returns a CLASS
      -> __invoke_golden: isinstance(golden_func, type) is True
      -> __invoke_class
           pool = {**inputs_named(from OpInfoKeeper), **context.attributes}
           cls.__init__ is object.__init__ ? cls() : bind_by_name(__init__, pool) -> cls(*a, **k)
           bind_by_name(inst.__call__, pool) -> inst(*ca, **ck)
      -> numeric numpy arrays

`bind_by_name` itself is unit-tested elsewhere; these tests prove the
*integration* (context -> OpInfoKeeper pool build -> __init__ bind ->
__call__ bind -> result) that was previously uncovered. Two shapes:

1. **Partial split with custom ``__init__``** (``_NegScale``): the config
   attribute flows through ``__init__`` and the input through ``__call__``.
2. **No custom ``__init__``** (``_Double``): exercises the
   ``cls.__init__ is object.__init__`` guard so ``cls()`` is called.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp
from ttk.core_modules.npu.op import output_generation as _mod

# Access module-level private functions via getattr to dodge class-style name mangling.
_generate_golden = getattr(_mod, '__generate_golden')


# --------------------------------------------------------------------------- #
# Golden classes — emulate what a real plugin author would ship.
# --------------------------------------------------------------------------- #
class _NegScale:
    """Partial split: ``__init__`` takes a config attr, ``__call__`` takes the input."""
    def __init__(self, *, scale):
        self.scale = scale

    def __call__(self, x):                 # noqa: N803 (short param mirrors real plugins)
        return [x * self.scale]


class _Double:
    """No custom ``__init__`` — must hit the ``cls()`` guard in ``__invoke_class``."""
    def __call__(self, x):                 # noqa: N803
        return [x * 2]


# --------------------------------------------------------------------------- #
# Helpers — mirror tests/test_kernel_builtin_golden.py::_make_testcase.
# --------------------------------------------------------------------------- #
def _make_testcase(op_name, input_shapes, input_dtypes, output_shapes, output_dtypes,
                   attributes):
    case = TestcaseOp()
    case.testcase_name = f"test_{op_name}_class_golden"
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
    case.attributes = attributes
    case.input_arrays = tuple(np.ones(s, dtype=d) for s, d in zip(input_shapes, input_dtypes))
    case.original_input_arrays = None
    return case


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Enable"      # not Promote → __golden_mode is a no-op
    sw.plugin_path = None
    sw.overflow_mode = 0
    return sw


def _mock_op_info(input_names):
    """OpInfoKeeper mock whose info_of() yields inputs named per ``input_names``."""
    mock = MagicMock()
    mock.return_value.info_of.return_value = {"inputs": [{"name": n} for n in input_names]}
    return mock


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    # Avoid real ASCEND toolkit lookups during construction/validate.
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
@patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
@patch('ttk.core_modules.npu.op.output_generation.get_plugin_function')
class TestKernelClassGoldenDispatch:
    """class-form golden (isinstance(golden_func, type)) → __invoke_class end-to-end."""

    def test_partial_split_init_takes_attr_call_takes_input(self, mock_get_plugin, mock_sw,
                                                            mock_op_info):
        """``_NegScale(scale=3)`` instantiated via ``__init__``; ``__call__(x)`` returns ``x*3``."""
        mock_get_plugin.return_value = _NegScale      # → isinstance(type) branch
        mock_sw.return_value = _mock_switches()
        # Pool build: inputs_named uses these names; attributes merged on top.
        mock_op_info.return_value.info_of.return_value = {"inputs": [{"name": "x"}]}

        inp = np.array([1.0, 2.0, 3.0], dtype="float32")
        case = _make_testcase(op_name="neg_scale",
                              input_shapes=((3,),), input_dtypes=("float32",),
                              output_shapes=((3,),), output_dtypes=("float32",),
                              attributes={"scale": 3})
        case.input_arrays = (inp,)

        golden = _generate_golden(case, ["float32"])

        assert len(golden) >= 1
        assert isinstance(golden[0], np.ndarray), \
            f"expected ndarray, got sentinel/string: {golden[0]!r}"
        np.testing.assert_array_equal(golden[0], np.array([3.0, 6.0, 9.0], dtype="float32"))
        mock_get_plugin.assert_called_once()

    def test_no_custom_init_hits_cls_guard(self, mock_get_plugin, mock_sw, mock_op_info):
        """``_Double`` has no ``__init__`` → ``cls()`` guard path; ``__call__(x)`` → ``x*2``."""
        mock_get_plugin.return_value = _Double
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": [{"name": "x"}]}

        inp = np.array([1.0, 2.0, 3.0], dtype="float32")
        case = _make_testcase(op_name="double",
                              input_shapes=((3,),), input_dtypes=("float32",),
                              output_shapes=((3,),), output_dtypes=("float32",),
                              attributes={})
        case.input_arrays = (inp,)

        golden = _generate_golden(case, ["float32"])

        assert len(golden) >= 1
        assert isinstance(golden[0], np.ndarray), \
            f"expected ndarray, got sentinel/string: {golden[0]!r}"
        np.testing.assert_array_equal(golden[0], np.array([2.0, 4.0, 6.0], dtype="float32"))
        mock_get_plugin.assert_called_once()

    def test_dispatch_routes_class_to_invoke_class_not_callable_path(self, mock_get_plugin,
                                                                     mock_sw, mock_op_info):
        """A class is a ``Callable`` but MUST route to ``__invoke_class`` (not the custom-callable
        path that would do ``golden_func(*input_arrays)``). Guards the ``isinstance(type)`` check
        ordering in ``__invoke_golden``: if a class leaked to the custom branch it would be called
        as ``_NegScale(array)`` → TypeError → ``['GOLDEN_FAILURE']``."""
        mock_get_plugin.return_value = _NegScale
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": [{"name": "x"}]}

        inp = np.array([1.0, 2.0, 3.0], dtype="float32")
        case = _make_testcase(op_name="neg_scale",
                              input_shapes=((3,),), input_dtypes=("float32",),
                              output_shapes=((3,),), output_dtypes=("float32",),
                              attributes={"scale": 5})
        case.input_arrays = (inp,)

        golden = _generate_golden(case, ["float32"])

        # If mis-routed to the custom-callable branch, golden[0] would be the
        # string "GOLDEN_FAILURE"; the class path yields a numeric ndarray.
        assert isinstance(golden[0], np.ndarray), \
            f"class golden must hit __invoke_class, not the custom-callable branch: {golden[0]!r}"
        np.testing.assert_array_equal(golden[0], np.array([5.0, 10.0, 15.0], dtype="float32"))
