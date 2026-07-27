from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ttk.core_modules.framework_api import profiling as e2e_profiling
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure
from ttk.core_modules.manual_data import (
    ManualDataStore,
    register_manual_data_directory_provider,
    unregister_manual_data_directory_provider,
)
from ttk.core_modules.npu.op_api import profiling as aclnn_profiling
from ttk.core_modules.npu.op_api.profiling_structure import (
    ApiComparisonResult,
    ApiProfilingResult,
)
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
from ttk.utilities.classes import SWITCHES


def _process_context():
    return SimpleNamespace(notify_status=lambda *_: None, change_name=lambda *_: None)


def _e2e_case(name):
    case = TestcaseE2e()
    case.testcase_name = name
    case.api_name = "torch.add"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.golden_api = ""
    case._tensor_list_dist = (0, 0)
    case._pure_output_indexes = [1]
    case._param_plan_cache = object()
    return case


def _aclnn_case(name):
    case = TestcaseAclnn()
    case.testcase_name = name
    case.api_name = "aclnnAdd"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.output_inplace_indexes = ()
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.scalar_dtypes = ("float32",)
    case.scalar_data_ranges = ((0, 1),)
    case._tensor_list_dist = (0, 0)
    case._scalar_list_dist = (0,)
    case._pure_output_indexes = [1]
    case._param_plan_cache = object()
    return case


def _switches(tmp_path, mode):
    switches = SWITCHES()
    switches.manual_data_mode = mode
    switches.manual_data_dirs = (str(tmp_path),)
    switches.no_memory_check = True
    switches.cst_switches.enabled = False
    switches.dyn_switches.enabled = False
    switches.dump_config.enable_input()
    switches.dump_config.enable_golden()
    return switches


def test_e2e_prepare_stops_before_api_resolution_and_device_execution(monkeypatch, tmp_path):
    case = _e2e_case("e2e_prepare")
    switches = _switches(tmp_path, "prepare")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    resolve_api = MagicMock(side_effect=AssertionError("main API must not be resolved"))
    execute = MagicMock(side_effect=AssertionError("main API must not execute"))

    def generate(testcase, *_):
        testcase.np_storages = inputs
        return inputs

    monkeypatch.setattr(e2e_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", generate)
    def generate_golden(*_args, **_kwargs):
        inputs[0][:] = -1
        return golden

    monkeypatch.setattr(e2e_profiling, "_generate_golden_data", generate_golden)
    monkeypatch.setattr(e2e_profiling, "resolve_api", resolve_api)
    monkeypatch.setattr(e2e_profiling, "_execute_eager", execute)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    result = FrameworkApiReturnStructure()

    e2e_profiling._do_profile(
        case, SimpleNamespace(alias=lambda: "npu"), {}, {}, 0, switches, result
    )

    assert result.precision_status == "PASS"
    assert result.eager_precision == "MANUAL_DATA_PREPARED"
    assert not resolve_api.called
    assert not execute.called
    store = ManualDataStore(tmp_path)
    assert (store.case_dir(case.testcase_name) / "input_0_float32.bin").is_file()
    assert not (store.case_dir(case.testcase_name) / "manifest.json").exists()
    np.testing.assert_array_equal(
        store.load_case(case, "e2e").inputs[0],
        np.array([1.0, 2.0], np.float32),
    )


def test_e2e_failed_reprepare_invalidates_previous_case(monkeypatch, tmp_path):
    case = _e2e_case("e2e_failed_reprepare")
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    store = ManualDataStore(tmp_path)
    store.write_case(case, "e2e", inputs, [np.ones(2, np.float32)])
    switches = _switches(tmp_path, "prepare")
    monkeypatch.setattr(e2e_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(
        e2e_profiling, "generate_inputs", MagicMock(side_effect=RuntimeError("input failure"))
    )

    result = FrameworkApiReturnStructure()
    e2e_profiling._do_profile(
        case, SimpleNamespace(alias=lambda: "npu"), {}, {}, 0, switches, result
    )

    assert result.precision_status == "FAIL"
    assert not store.case_dir(case.testcase_name).exists()


def test_e2e_replay_skips_input_and_golden_generation(monkeypatch, tmp_path):
    case = _e2e_case("e2e_replay")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "e2e", inputs, golden)
    switches = _switches(tmp_path, "replay")
    generated = MagicMock()
    evaluated = MagicMock()

    def restore(testcase, _switches, _backend, _plan, stored_inputs=None):
        assert stored_inputs is not None
        testcase.np_storages = stored_inputs
        return stored_inputs

    generated.side_effect = restore
    monkeypatch.setattr(e2e_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", generated)
    monkeypatch.setattr(e2e_profiling, "_generate_golden_data",
                        MagicMock(side_effect=AssertionError("golden generation must be skipped")))
    monkeypatch.setattr(e2e_profiling, "resolve_api", lambda *_: (lambda *_: None, False))
    monkeypatch.setattr(e2e_profiling, "_execute_eager", lambda *_: (golden, None))
    monkeypatch.setattr(e2e_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(e2e_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_dump_inputs", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_dump_goldens", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_dump_outputs", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_apply_pre_compare", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_evaluate_eager_precision", evaluated)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    backend = SimpleNamespace(alias=lambda: "npu", use_device=lambda: False)

    e2e_profiling._do_profile(
        case, backend, {}, {}, 0, switches, FrameworkApiReturnStructure()
    )

    assert generated.call_count == 1
    np.testing.assert_array_equal(evaluated.call_args.args[3][0], golden[0])


def test_e2e_provider_automatically_selects_replay(monkeypatch, tmp_path):
    case = _e2e_case("e2e_provider_replay")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(case, "e2e", inputs, golden)
    switches = _switches(tmp_path, None)
    switches.manual_data_dirs = ()
    restored = MagicMock()

    def restore(testcase, _switches, _backend, _plan, stored_inputs=None):
        restored(stored_inputs)
        testcase.np_storages = stored_inputs
        return stored_inputs

    def provider(testcase, case_type, current_switches):
        assert testcase is case
        assert case_type == "e2e"
        assert current_switches is switches
        return tmp_path

    monkeypatch.setattr(e2e_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(e2e_profiling, "generate_inputs", restore)
    monkeypatch.setattr(e2e_profiling, "_generate_golden_data",
                        MagicMock(side_effect=AssertionError("golden generation must be skipped")))
    monkeypatch.setattr(e2e_profiling, "resolve_api", lambda *_: (lambda *_: None, False))
    monkeypatch.setattr(e2e_profiling, "_execute_eager", lambda *_: (golden, None))
    monkeypatch.setattr(e2e_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(e2e_profiling, "_profiling_print", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_dump_inputs", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_dump_goldens", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_dump_outputs", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_apply_pre_compare", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_evaluate_eager_precision", lambda *_: None)
    monkeypatch.setattr(e2e_profiling, "_profiling_end_print", lambda *_args, **_kwargs: None)
    backend = SimpleNamespace(alias=lambda: "npu", use_device=lambda: False)

    register_manual_data_directory_provider(provider)
    try:
        e2e_profiling._do_profile(
            case, backend, {}, {}, 0, switches, FrameworkApiReturnStructure()
        )
    finally:
        unregister_manual_data_directory_provider(provider)

    assert restored.call_count == 1


def test_aclnn_prepare_stops_before_device_execution(monkeypatch, tmp_path):
    case = _aclnn_case("aclnn_prepare")
    switches = _switches(tmp_path, "prepare")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    scalar = np.array(0.5, np.float32)
    golden = [np.array([3.0, 4.0], np.float32)]
    execute = MagicMock(side_effect=AssertionError("ACLNN must not execute"))

    class Inputs:
        def __init__(self, context):
            self.context = context

        def gen(self):
            self.context.np_storages = inputs
            self.context.tensors = inputs
            self.context.scalars = (scalar,)

    class Golden:
        def __init__(self, context):
            self.context = context

        def gen(self):
            self.context.np_storages[0][:] = -1
            self.context.scalars[0][...] = 9
            self.context.golden_tensors = golden

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(aclnn_profiling, "OpApiInfoKeeper",
                        lambda: SimpleNamespace(has_api=lambda *_: True))
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)
    monkeypatch.setattr(aclnn_profiling, "GoldenGenerator", Golden)
    monkeypatch.setattr(aclnn_profiling, "do_profiling", execute)
    monkeypatch.setattr(aclnn_profiling, "__profiling_end_print", lambda *_: None)

    result = aclnn_profiling.profile_process(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert result.precision == "MANUAL_DATA_PREPARED"
    assert not execute.called
    loaded = ManualDataStore(tmp_path).load_case(case, "aclnn")
    np.testing.assert_array_equal(
        loaded.inputs[0],
        np.array([1.0, 2.0], np.float32),
    )
    assert loaded.scalars[0] == np.array(0.5, np.float32)


def test_aclnn_failed_reprepare_invalidates_previous_case(monkeypatch, tmp_path):
    case = _aclnn_case("aclnn_failed_reprepare")
    inputs = [np.ones(2, np.float32), np.zeros(2, np.float32)]
    scalar = [np.array(0.5, np.float32)]
    store = ManualDataStore(tmp_path)
    store.write_case(
        case, "aclnn", inputs, [np.ones(2, np.float32)], scalars=scalar
    )
    switches = _switches(tmp_path, "prepare")

    class Inputs:
        def __init__(self, _context):
            pass

        def gen(self):
            raise RuntimeError("input failure")

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(
        aclnn_profiling, "OpApiInfoKeeper",
        lambda: SimpleNamespace(has_api=lambda *_: True),
    )
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)

    result = aclnn_profiling.profile_process(case, {}, {}, 0)

    assert result.precision_status == "INPUT_GEN_FAILURE"
    assert not store.case_dir(case.testcase_name).exists()


def test_aclnn_replay_skips_input_and_golden_plugins(monkeypatch, tmp_path):
    case = _aclnn_case("aclnn_replay")
    inputs = [np.array([1.0, 2.0], np.float32), np.zeros(2, np.float32)]
    scalar = [np.array(0.5, np.float32)]
    golden = [np.array([3.0, 4.0], np.float32)]
    ManualDataStore(tmp_path).write_case(
        case, "aclnn", inputs, golden, scalars=scalar
    )
    switches = _switches(tmp_path, "replay")
    restore = MagicMock()

    class Inputs:
        def __init__(self, context):
            self.context = context

        def gen(self, stored_inputs=None, stored_scalars=None):
            restore(stored_inputs, stored_scalars)
            self.context.np_storages = stored_inputs
            self.context.tensors = stored_inputs
            self.context.scalars = stored_scalars

    class Comparator:
        def __init__(self, context):
            self.context = context

        def compare(self):
            np.testing.assert_array_equal(self.context.golden_tensors[0], golden[0])
            return ApiComparisonResult(None).set("100%", "PASS")

    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)
    monkeypatch.setattr(aclnn_profiling, "get_process_context", _process_context)
    monkeypatch.setattr(aclnn_profiling, "OpApiInfoKeeper",
                        lambda: SimpleNamespace(has_api=lambda *_: True))
    monkeypatch.setattr(aclnn_profiling, "InputGenerator", Inputs)
    monkeypatch.setattr(aclnn_profiling, "GoldenGenerator",
                        MagicMock(side_effect=AssertionError("golden plugin must be skipped")))
    monkeypatch.setattr(aclnn_profiling, "Comparator", Comparator)
    monkeypatch.setattr(aclnn_profiling, "DeviceLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(aclnn_profiling, "__profiling_print", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "__profiling_end_print", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "__dump_input", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "__dump_golden", lambda *_: None)
    monkeypatch.setattr(aclnn_profiling, "__dump_output", lambda *_: None)
    monkeypatch.setattr(
        aclnn_profiling,
        "do_profiling",
        lambda *_: ApiProfilingResult(
            True, output_bytes=[np.zeros(2, np.float32).tobytes()],
            output_view_shapes=[(2,)],
        ),
    )

    result = aclnn_profiling.profile_process(case, {}, {}, 0)

    assert result.precision_status == "PASS"
    assert restore.call_count == 1


def test_loaded_golden_still_uses_custom_compare(monkeypatch):
    case = _e2e_case("custom_compare")
    switches = SWITCHES()
    output = np.array([1.0, 2.0], np.float32)
    loaded_golden = np.array([1.0, 2.0], np.float32)
    custom_compare = MagicMock(return_value={"pass": True, "precision": "CUSTOM_PASS"})

    def spec_attr(_api_name, attribute, _plugin_path):
        return custom_compare if attribute == "compare" else None

    monkeypatch.setattr(e2e_profiling, "get_spec_attr", spec_attr)
    result = FrameworkApiReturnStructure()

    e2e_profiling._evaluate_eager_precision(
        case, [], [output], [loaded_golden], switches, None, result
    )

    assert result.precision_status == "PASS"
    assert result.eager_precision == "CUSTOM_PASS"
    np.testing.assert_array_equal(custom_compare.call_args.args[1], loaded_golden)


def test_loaded_golden_still_uses_current_csv_close_tolerance(monkeypatch):
    case = _e2e_case("csv_tolerance")
    case.precision_tolerances = ((0.0, 0.5),)
    case.absolute_precision = (0.0,)
    switches = SWITCHES()
    switches.compare_method = "close"
    monkeypatch.setattr(e2e_profiling, "get_spec_attr", lambda *_: None)
    result = FrameworkApiReturnStructure()

    e2e_profiling._evaluate_eager_precision(
        case,
        [],
        [np.array([0.0, 1.0], np.float32)],
        [np.array([0.0, 0.0], np.float32)],
        switches,
        None,
        result,
    )

    assert result.precision_status == "PASS"
    assert result.eager_precision == "50.0%"
