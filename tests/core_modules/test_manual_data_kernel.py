from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.manual_data import ManualDataStore
from ttk.core_modules.npu.op import input_generation, output_generation, profiling
from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp
from ttk.utilities.classes import SWITCHES


def _kernel_case(name="kernel_manual_data"):
    case = TestcaseOp()
    case.testcase_name = name
    case.op_name = "add"
    case.input_shapes = ((2,), None)
    case.input_dtypes = ("float32", "float32")
    case.input_formats = ("ND", "ND")
    case.input_ori_shapes = case.input_shapes
    case.input_ori_formats = case.input_formats
    case.output_shapes = ((2,),)
    case.output_dtypes = ("float32",)
    case.output_formats = ("ND",)
    case.output_ori_shapes = case.output_shapes
    case.output_ori_formats = case.output_formats
    case.output_inplace_indexes = ()
    case.output_shape_unknown_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), None)
    case.precision_tolerances = ((0.001, 0.001),)
    case.absolute_precision = (0.0001,)
    case._input_distribution = (0, 0)
    case._output_distribution = (0,)
    case.is_valid = True
    case.fail_reason = None
    return case


def _process_context():
    return SimpleNamespace(
        notify_status=lambda *_: None,
        change_name=lambda *_: None,
    )


def test_kernel_stored_input_skips_random_generation_and_plugin(monkeypatch):
    case = _kernel_case("kernel_stored_input")
    switches = SimpleNamespace(golden_mode="Enable", plugin_path=())
    plugin = MagicMock(side_effect=AssertionError("input plugin must not run"))
    monkeypatch.setattr(input_generation, "get_global_storage", lambda: switches)
    monkeypatch.setattr(input_generation, "get_plugin_function", plugin)
    stored = [np.array([1.0, 2.0], np.float32), None]

    input_generation.__gen_input(case, stored_inputs=stored)

    assert not plugin.called
    np.testing.assert_array_equal(case.input_arrays[0], stored[0])
    assert case.input_arrays[1] is None
    np.testing.assert_array_equal(case.original_input_arrays[0], stored[0])


def test_kernel_stored_golden_skips_golden_plugin(monkeypatch):
    case = _kernel_case("kernel_stored_golden")
    case.input_arrays = (np.array([1.0, 2.0], np.float32), None)
    switches = SimpleNamespace(
        golden_mode="Enable",
        overflow_mode=0,
        plugin_path=(),
        short_soc_version="Ascend910_93",
    )
    plugin = MagicMock(side_effect=AssertionError("golden plugin must not run"))
    monkeypatch.setattr(output_generation, "get_global_storage", lambda: switches)
    monkeypatch.setattr(output_generation, "get_plugin_function", plugin)
    golden = np.array([3.0, 4.0], np.float32)

    output_generation.__gen_output(case, stored_goldens=[golden])

    assert not plugin.called
    np.testing.assert_array_equal(case.golden_arrays[0], golden)
    assert case.output_arrays[0].shape == golden.shape


def test_kernel_prepare_snapshots_input_and_stops_before_device(monkeypatch, tmp_path):
    case = _kernel_case("kernel_prepare")
    switches = SWITCHES()
    switches.manual_data_mode = "prepare"
    switches.manual_data_dirs = (str(tmp_path),)
    switches.no_memory_check = True
    switches.dump_config.enable_input()
    switches.dump_config.enable_golden()
    device_execution = MagicMock(side_effect=AssertionError("Kernel must not execute"))
    prepared = object()
    original_input = np.array([1.0, 2.0], np.float32)

    def generate_input(context, stored_inputs=None):
        assert stored_inputs is None
        context.input_arrays = (original_input, None)

    def generate_output(context, stored_goldens=None):
        assert stored_goldens is None
        original_input[:] = -1
        context.golden_arrays = [np.array([3.0, 4.0], np.float32)]
        context.output_arrays = (np.ones(2, np.float32),)

    monkeypatch.setattr(profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(profiling, "get_process_context", _process_context)
    monkeypatch.setattr(profiling, "__parse_manual_params", lambda *_: None)
    monkeypatch.setattr(profiling, "__parse_dynamic_tiling_data", lambda *_: None)
    monkeypatch.setattr(profiling, "__parse_binary_tiling_data", lambda *_: None)
    monkeypatch.setattr(TestcaseOp, "compile_failed", lambda *_: False)
    monkeypatch.setattr(profiling, "__gen_input", generate_input)
    monkeypatch.setattr(profiling, "__gen_output", generate_output)
    monkeypatch.setattr(profiling, "get_spec_attr", lambda *_: None)
    monkeypatch.setattr(profiling, "do_profiling", device_execution)
    monkeypatch.setattr(profiling, "_manual_data_prepared_end", lambda *_: prepared)

    result = profiling.profile_process(case, {}, {}, 0)

    assert result is prepared
    assert not device_execution.called
    loaded = ManualDataStore(tmp_path).load_case(case, "kernel")
    np.testing.assert_array_equal(
        loaded.inputs[0], np.array([1.0, 2.0], np.float32)
    )
    np.testing.assert_array_equal(
        loaded.load_goldens(
            shapes=case.flat_output_shapes,
            dtypes=case.flat_output_dtypes,
        )[0],
        np.array([3.0, 4.0], np.float32),
    )
