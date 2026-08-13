import os
import pathlib
from collections import OrderedDict
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import pytest

from ttk.core_modules import pre_npu
from ttk.core_modules.manual_data import case_directory_name
from ttk.core_modules.npu.op_api import pre_npu as aclnn_pre_npu
from ttk.test_spec import PreNpuResult, RuntimeProfile, TtkContext
from ttk.utilities.classes import SWITCHES


def _case(name="case/with spaces"):
    return SimpleNamespace(
        api_name="torch.ops.vendor.main",
        testcase_name=name,
        tensors=(np.zeros(2, np.float32),),
        scalars=(),
        attributes={"alpha": 1},
        original_dict={"remark": "context"},
    )


def test_prepare_context_exposes_testcase_data_directory(tmp_path):
    switches = SWITCHES()
    switches.manual_data_mode = "prepare"
    switches.manual_data_dirs = (str(tmp_path),)

    context = pre_npu.build_ttk_context(_case(), switches, "e2e")

    case_name = case_directory_name("case/with spaces")
    assert context.manual_case_dir == tmp_path / case_name
    assert not context.manual_case_dir.exists()
    assert context.manual_data_dirs == (tmp_path,)
    assert context.attributes == {"alpha": 1}
    with pytest.raises(TypeError):
        context.attributes["alpha"] = 2


def test_context_freezes_nested_csv_data():
    case = _case()
    case.attributes = {"nested": {"values": [1, 2]}}
    case.original_dict = {"raw": {"labels": ["a"]}}

    context = pre_npu.build_ttk_context(case, SWITCHES(), "e2e")

    assert context.attributes["nested"]["values"] == (1, 2)
    assert context.csv_fields["raw"]["labels"] == ("a",)
    with pytest.raises(TypeError):
        context.attributes["nested"]["values"] = ()
    with pytest.raises(TypeError):
        context.csv_fields["raw"]["labels"][0] = "b"


def test_context_freezes_object_with_single_string_slot():
    class Option:
        __slots__ = "payload"

        def __init__(self):
            self.payload = [1, 2]

    case = _case()
    case.attributes = {"option": Option()}

    context = pre_npu.build_ttk_context(case, SWITCHES(), "e2e")

    assert context.attributes["option"] == {"payload": (1, 2)}


def test_replay_context_exposes_selected_manual_case(tmp_path):
    case_dir = tmp_path / "stored-case"
    manual_case = SimpleNamespace(case_dir=case_dir, file_format="bin")

    context = pre_npu.build_ttk_context(
        _case("stored-case"), SWITCHES(), "aclnn", manual_case=manual_case
    )

    assert context.manual_case_dir == case_dir.resolve()
    assert context.manual_data_format == "bin"


def test_manual_case_directories_do_not_collide_after_sanitizing(tmp_path):
    switches = SWITCHES()
    switches.manual_data_dirs = (str(tmp_path),)

    slash = pre_npu.build_ttk_context(_case("case/a"), switches, "e2e")
    colon = pre_npu.build_ttk_context(_case("case:a"), switches, "e2e")

    assert slash.manual_case_dir != colon.manual_case_dir
    assert len(slash.manual_case_dir.name) <= 120
    assert len(colon.manual_case_dir.name) <= 120


def test_manual_case_directory_depends_on_testcase_identity(tmp_path):
    switches = SWITCHES()
    switches.manual_data_dirs = (str(tmp_path),)
    e2e = pre_npu.build_ttk_context(_case("case"), switches, "e2e")
    aclnn = pre_npu.build_ttk_context(_case("case"), switches, "aclnn")
    other_api = _case("case")
    other_api.api_name = "torch.ops.vendor.other"
    other = pre_npu.build_ttk_context(other_api, switches, "e2e")

    expected = tmp_path.resolve() / "case"
    assert e2e.manual_case_dir == aclnn.manual_case_dir == other.manual_case_dir
    assert e2e.manual_case_dir == expected


def test_context_does_not_guess_a_case_directory_from_multiple_roots(tmp_path):
    switches = SWITCHES()
    switches.manual_data_dirs = (
        str(tmp_path / "first"),
        str(tmp_path / "second"),
    )

    context = pre_npu.build_ttk_context(_case("case"), switches, "e2e")

    assert context.manual_data_dirs == (
        (tmp_path / "first").resolve(),
        (tmp_path / "second").resolve(),
    )
    assert context.manual_case_dir is None


def test_runtime_context_requires_explicit_parameter():
    context = pre_npu.build_ttk_context(_case(), SWITCHES(), "e2e")

    def opted_in(*, context: TtkContext = None):
        return context

    def business_parameter(*, context=None):
        return context

    def legacy(**kwargs):
        return kwargs

    def legacy_named_context(**context):
        return context

    opted_kwargs = {}
    legacy_kwargs = {}
    named_kwargs = {}
    business_kwargs = {"context": "csv-value"}
    pre_npu.add_context_if_declared(opted_in, opted_kwargs, context)
    pre_npu.add_context_if_declared(legacy, legacy_kwargs, context)
    pre_npu.add_context_if_declared(
        legacy_named_context, named_kwargs, context
    )
    pre_npu.add_context_if_declared(
        business_parameter, business_kwargs, context
    )

    assert opted_kwargs == {"context": context}
    assert legacy_kwargs == {}
    assert named_kwargs == {}
    assert business_kwargs == {"context": "csv-value"}


@pytest.mark.parametrize(
    "annotation",
    ("TtkContext | None", Optional["TtkContext"]),
)
def test_runtime_context_accepts_explicit_optional_annotations(annotation):
    context = pre_npu.build_ttk_context(_case(), SWITCHES(), "e2e")

    def opted_in(*, context=None):
        return context

    opted_in.__annotations__["context"] = annotation
    kwargs = {}

    pre_npu.add_context_if_declared(opted_in, kwargs, context)

    assert kwargs == {"context": context}


@pytest.mark.parametrize(
    "annotation",
    (list[TtkContext], dict[str, TtkContext], Union[TtkContext, str]),
)
def test_runtime_context_rejects_container_and_business_union_annotations(annotation):
    context = pre_npu.build_ttk_context(_case(), SWITCHES(), "e2e")

    def business_parameter(*, context=None):
        return context

    business_parameter.__annotations__["context"] = annotation
    kwargs = {"context": "csv-value"}

    pre_npu.add_context_if_declared(business_parameter, kwargs, context)

    assert kwargs == {"context": "csv-value"}


def test_runtime_context_does_not_overwrite_a_csv_attribute():
    context = pre_npu.build_ttk_context(_case(), SWITCHES(), "e2e")

    def opted_in(*, context: TtkContext = None):
        return context

    with pytest.raises(ValueError, match="reserved TestSpec hook parameter"):
        pre_npu.add_context_if_declared(
            opted_in, {"context": "csv-value"}, context
        )


def test_legacy_pre_npu_hook_keeps_regular_attribute_kwargs():
    case = _case()
    case.attributes = {
        "alpha": 3,
        "stage_only": False,
    }
    context = pre_npu.build_ttk_context(case, SWITCHES(), "e2e")
    received = {}

    def legacy(alpha, stage_only):
        received["alpha"] = alpha
        received["stop"] = stage_only

    assert pre_npu.execute_pre_npu(
        case, SWITCHES(), context, pre_npu_func=legacy
    ) == PreNpuResult()
    assert received == {"alpha": 3, "stop": False}


def test_pre_npu_build_args_supports_e2e_and_aclnn_plan_signatures():
    case = _case()
    received = []

    class E2ePlan:
        def build_args(self, tensors):
            return list(tensors), {"planned": "e2e"}, {"extra": 1}

    case.get_param_plan = lambda: E2ePlan()

    def e2e_hook(*args, planned, extra):
        received.append((args, planned, extra))

    pre_npu.execute_pre_npu(
        case,
        SWITCHES(),
        pre_npu.build_ttk_context(case, SWITCHES(), "e2e"),
        pre_npu_func=e2e_hook,
    )

    class AclnnPlan:
        def build_args(self, tensors, scalars, attributes):
            assert tensors is case.tensors
            assert scalars == case.scalars
            assert attributes is case.attributes
            return list(tensors), {"extra": 2}

    case.get_param_plan = lambda: AclnnPlan()

    def aclnn_hook(*args, extra):
        received.append((args, "aclnn", extra))

    pre_npu.execute_pre_npu(
        case,
        SWITCHES(),
        pre_npu.build_ttk_context(case, SWITCHES(), "aclnn"),
        pre_npu_func=aclnn_hook,
    )

    assert received == [((case.tensors[0],), "e2e", 1), ((case.tensors[0],), "aclnn", 2)]


def test_pre_npu_does_not_mask_type_error_from_three_argument_build_args():
    case = _case()

    class BrokenPlan:
        def build_args(self, tensors, scalars, attributes):
            raise TypeError("invalid tensor payload")

    case.get_param_plan = lambda: BrokenPlan()

    with pytest.raises(TypeError, match="invalid tensor payload"):
        pre_npu.execute_pre_npu(
            case,
            SWITCHES(),
            pre_npu.build_ttk_context(case, SWITCHES(), "aclnn"),
            pre_npu_func=lambda: None,
        )


def test_runtime_context_exposes_common_cli_options(tmp_path):
    switches = SWITCHES()
    switches.manual_data_dirs = (str(tmp_path),)
    switches.TASK_PROFILING = True
    switches.output_file_name = str(tmp_path / "main_result.csv")
    context = pre_npu.build_ttk_context(_case(), switches, "e2e")

    assert context.manual_data_dirs == (tmp_path.resolve(),)
    assert context.options["manual_data_dirs"] == (tmp_path.resolve(),)
    assert context.options["task_prof"] is True
    assert context.options["output_file_name"] == str(tmp_path / "main_result.csv")
    with pytest.raises(TypeError):
        context.options["task_prof"] = False


def test_runtime_context_exposes_property_backed_runtime_options():
    switches = SWITCHES()
    switches.run_time = 7
    switches.compile_only = True

    context = pre_npu.build_ttk_context(_case(), switches, "e2e")

    assert context.options["run_time"] == 7
    assert context.options["compile_only"] is True


def test_stage_normalizes_continue_and_operator_stop(monkeypatch):
    case = _case()
    switches = SWITCHES()
    runtime_context = pre_npu.build_ttk_context(case, switches, "e2e")

    def continues(*, context: TtkContext):
        context.state["visited"] = True

    assert pre_npu.execute_pre_npu(
        case, switches, runtime_context, pre_npu_func=continues
    ) == PreNpuResult()
    assert runtime_context.state == {"visited": True}

    def stops(*, context: TtkContext):
        assert context is runtime_context
        return PreNpuResult(stop=True, reason="custom-stage-only")

    assert pre_npu.execute_pre_npu(
        case, switches, runtime_context, pre_npu_func=stops
    ) == PreNpuResult(stop=True, reason="custom-stage-only")


def test_stage_rejects_invalid_result_contract():
    case = _case()
    switches = SWITCHES()
    context = pre_npu.build_ttk_context(case, switches, "e2e")

    with pytest.raises(TypeError, match="must return None or"):
        pre_npu.execute_pre_npu(
            case, switches, context, pre_npu_func=lambda: True
        )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"stop": "false"}, "stop must be a boolean"),
        ({"reason": None}, "reason must be a string"),
    ],
)
def test_result_rejects_ambiguous_stop_contract(kwargs, message):
    with pytest.raises(TypeError, match=message):
        PreNpuResult(**kwargs)


def test_stage_clears_temporary_aclnn_runner_when_operator_hook_raises():
    case = _case()
    switches = SWITCHES()
    runtime_context = pre_npu.build_ttk_context(case, switches, "aclnn")
    runner = object()

    def fails(*, context: TtkContext):
        assert context._aclnn_runner is runner
        raise RuntimeError("operator failure")

    with pytest.raises(RuntimeError, match="operator failure"):
        pre_npu.execute_pre_npu(
            case,
            switches,
            runtime_context,
            aclnn_runner=runner,
            pre_npu_func=fails,
        )
    assert runtime_context._aclnn_runner is None


def test_stage_exposes_and_clears_generic_profile_runner():
    case = _case()
    switches = SWITCHES()
    runtime_context = pre_npu.build_ttk_context(case, switches, "e2e")
    calls = []

    def profile_runner(stage_name, operation):
        calls.append(stage_name)
        operation()
        return RuntimeProfile(False, 1, 0.0, None)

    def hook(*, context: TtkContext):
        profile = context.run_profiled(
            "operator_stage", lambda: calls.append("operation")
        )
        assert profile.enabled is False

    pre_npu.execute_pre_npu(
        case,
        switches,
        runtime_context,
        profile_runner=profile_runner,
        pre_npu_func=hook,
    )

    assert calls == ["operator_stage", "operation"]
    assert runtime_context._profile_runner is None
    with pytest.raises(RuntimeError, match="only available"):
        runtime_context.run_profiled("late", lambda: None)


def test_pre_npu_profile_runner_disabled_executes_operation_once(tmp_path):
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    switches.run_time = 7
    switches.warmup = True
    calls = []
    synchronizations = []
    runner = pre_npu.build_pre_npu_profile_runner(
        _case(),
        switches,
        "e2e",
        synchronize=lambda: synchronizations.append(True),
    )

    profile = runner("custom_action", lambda: calls.append(True))

    assert calls == [True]
    assert synchronizations == [True]
    assert profile == RuntimeProfile(False, 1, 0.0, None)
    assert not (tmp_path / "msprof").exists()


def test_pre_npu_profile_runner_uses_fixed_path_and_returns_kernel_summary(
    monkeypatch, tmp_path
):
    from ttk.core_modules.framework_api import profiler as framework_profiler

    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = True
    switches.run_time = 2
    switches.warmup = False
    calls = []

    class FakeProfiler:
        def __init__(self, _backend, result_path):
            self.result_path = result_path

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def result(self, _backend, repeat_count):
            assert repeat_count == 2
            kernel = SimpleNamespace(
                name="CustomStageKernel",
                device_us=12.0,
                calls=2,
                avg_us=6.0,
                max_us=7.0,
                min_us=5.0,
            )
            return SimpleNamespace(
                elapsed_us=6.0,
                kernel_details=SimpleNamespace(kernels=[kernel]),
            )

    monkeypatch.setattr(framework_profiler, "NpuProfiler", FakeProfiler)
    runner = pre_npu.build_pre_npu_profile_runner(_case(), switches, "e2e")

    profile = runner("custom/action", lambda: calls.append(True))

    assert calls == [True, True]
    assert profile.enabled is True
    assert profile.repeat_count == 2
    assert profile.elapsed_us == 6.0
    assert profile.kernels[0].name == "CustomStageKernel"
    assert str(profile.result_path).startswith(str(tmp_path / "msprof" / "pre_npu"))
    assert "/" not in profile.result_path.name

    profile.result_path.mkdir(parents=True)
    marker = profile.result_path / "previous.csv"
    marker.write_text("first run", encoding="utf-8")
    second_profile = runner("custom/action", lambda: calls.append(True))

    assert calls == [True, True, True, True]
    assert second_profile.result_path == profile.result_path


def test_aclnn_pre_npu_profile_runner_uses_native_profiler(monkeypatch, tmp_path):
    from ttk.core_modules import msprof

    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = True
    switches.run_time = 2
    switches.warmup = False
    calls = []
    profiler_steps = []

    class FakeMsProfiler:
        def __init__(
            self, device_id, result_path, profile_type, start_step, **_kwargs
        ):
            assert device_id == 3
            assert profile_type == msprof.TtkMsProfType.API
            assert start_step == 0
            self.result_path = pathlib.Path(result_path)

        def __enter__(self):
            return self

        def step(self):
            profiler_steps.append(True)

        def __exit__(self, *_args):
            self.result_path.mkdir(parents=True)
            (self.result_path / "op_statistic_test.csv").write_text(
                "Device_id,OP Type,Core Type,Count,Total Time(us),"
                "Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n"
                "0,CustomStage,AICPU,2,20.0,9.0,10.0,11.0,100\n",
                encoding="utf-8",
            )
            return False

    monkeypatch.setattr(msprof, "MsProfiler", FakeMsProfiler)
    runner = pre_npu.build_pre_npu_profile_runner(
        _case(), switches, "aclnn", device_id=3
    )

    profile = runner("custom_action", lambda: calls.append(True))

    assert calls == [True, True]
    assert profiler_steps == [True, True]
    assert profile.enabled is True
    assert profile.elapsed_us == 10.0
    assert profile.kernels[0].name == "CustomStage_AiCpu"
    assert profile.kernels[0].calls == 2


def test_aclnn_profile_summary_uses_latest_file_by_mtime(tmp_path):
    header = (
        "Device_id,OP Type,Core Type,Count,Total Time(us),"
        "Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n"
    )
    older = tmp_path / "op_statistic_z.csv"
    newer = tmp_path / "op_statistic_a.csv"
    older.write_text(
        header + "0,Older,AICORE,1,100.0,100.0,100.0,100.0,100\n",
        encoding="utf-8",
    )
    newer.write_text(
        header + "0,Newer,AICPU,2,20.0,9.0,10.0,11.0,100\n",
        encoding="utf-8",
    )
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    profile = pre_npu._read_aclnn_profile(tmp_path, repeat_count=2)

    assert profile.elapsed_us == 10.0
    assert [kernel.name for kernel in profile.kernels] == ["Newer_AiCpu"]


def test_run_aclnn_is_only_available_during_stage():
    context = pre_npu.build_ttk_context(_case(), SWITCHES(), "aclnn")

    with pytest.raises(RuntimeError, match="only available"):
        context.run_aclnn(
            "aclnnAux",
            tensors={},
            attributes={},
            output_names=("output",),
        )


def test_framework_aclnn_runner_builds_header_order_and_copies_output(monkeypatch):
    info = SimpleNamespace(params=OrderedDict([
        ("input", {"type": "aclTensor*", "default": None}),
        ("axis", {"type": "int64_t", "default": None}),
        ("output", {"type": "aclTensor*", "default": None}),
    ]))
    monkeypatch.setattr(
        aclnn_pre_npu,
        "OpApiInfoKeeper",
        lambda: SimpleNamespace(info_of=lambda name: info if name == "aclnnAux" else None),
    )

    class Device:
        def __init__(self):
            self.created = []
            self.workspace_params = None
            self.freed = 0

        def create_acl_tensor(self, value, fmt, storage_shape=None):
            ptr = f"ptr-{len(self.created)}"
            self.created.append((ptr, value, fmt, storage_shape))
            return ptr

        def acl_get_workspace(self, api_name, params):
            self.workspace_params = (api_name, params)
            return 0, "executor"

        def acl_execute(self, api_name, workspace_size, executor, stream):
            assert (api_name, workspace_size, executor, stream) == (
                "aclnnAux", 0, "executor", "stream"
            )
            return "OK"

        def get_view_shape(self, ptr):
            assert ptr == "ptr-1"
            return (2,)

        def get_device_mem_addr(self, ptr):
            return ptr

        def get_data_from_hbm(self, ptr, byte_size):
            assert ptr == "ptr-1"
            assert byte_size == 8
            return np.array([7.0, 8.0], np.float32).tobytes()

        def free_all_memory(self):
            self.freed += 1

    device = Device()
    output = np.zeros(2, np.float32)
    runner = aclnn_pre_npu.PreNpuAclnnRunner(device, "stream")
    runner(
        "aclnnAux",
        tensors={"input": np.ones(2, np.float32), "output": output},
        attributes={"axis": 3},
        output_names=("output",),
    )

    api_name, params = device.workspace_params
    assert api_name == "aclnnAux"
    assert params[0] == "ptr-0"
    assert params[1].value == 3
    assert params[2] == "ptr-1"
    np.testing.assert_array_equal(output, np.array([7.0, 8.0], np.float32))
    assert device.freed == 1


def test_framework_aclnn_runner_rejects_tensor_list_output_before_execution(
    monkeypatch,
):
    info = SimpleNamespace(params=OrderedDict([
        ("outputs", {"type": "aclTensorList*", "default": None}),
    ]))
    monkeypatch.setattr(
        aclnn_pre_npu,
        "OpApiInfoKeeper",
        lambda: SimpleNamespace(info_of=lambda _name: info),
    )

    with pytest.raises(ValueError, match=r"must be an aclTensor\* parameter"):
        aclnn_pre_npu.PreNpuAclnnRunner(object())(
            "aclnnAux",
            tensors={"outputs": [np.zeros(1, np.float32)]},
            attributes={},
            output_names=("outputs",),
        )


def test_framework_aclnn_runner_frees_allocations_after_execute_failure(monkeypatch):
    info = SimpleNamespace(params=OrderedDict([
        ("input", {"type": "aclTensor*", "default": None}),
        ("output", {"type": "aclTensor*", "default": None}),
    ]))
    monkeypatch.setattr(
        aclnn_pre_npu,
        "OpApiInfoKeeper",
        lambda: SimpleNamespace(info_of=lambda _name: info),
    )

    class Device:
        def __init__(self):
            self.freed = 0

        def create_acl_tensor(self, value, _fmt, _storage_shape=None):
            return f"ptr-{id(value)}"

        def acl_get_workspace(self, _api_name, _params):
            return 0, "executor"

        def acl_execute(self, *_args):
            return "ACLNN_EXECUTE_FAILED"

        def free_all_memory(self):
            self.freed += 1

    device = Device()
    runner = aclnn_pre_npu.PreNpuAclnnRunner(device)

    with pytest.raises(RuntimeError, match="execution failed"):
        runner(
            "aclnnAux",
            tensors={
                "input": np.ones(1, np.float32),
                "output": np.zeros(1, np.float32),
            },
            attributes={},
            output_names=("output",),
        )
    assert device.freed == 1


def test_framework_aclnn_runner_copies_numpy_output_view_with_storage_offset(
    monkeypatch,
):
    info = SimpleNamespace(params=OrderedDict([
        ("output", {"type": "aclTensor*", "default": None}),
    ]))
    monkeypatch.setattr(
        aclnn_pre_npu,
        "OpApiInfoKeeper",
        lambda: SimpleNamespace(info_of=lambda _name: info),
    )

    class Device:
        def create_acl_tensor(self, value, _fmt, storage_shape=None):
            assert tuple(value.shape) == (2,)
            assert tuple(storage_shape) == (4,)
            return "output-ptr"

        def acl_get_workspace(self, _api_name, _params):
            return 0, "executor"

        def acl_execute(self, *_args):
            return "OK"

        def get_view_shape(self, _ptr):
            return (2,)

        def get_device_mem_addr(self, ptr):
            return ptr

        def get_data_from_hbm(self, _ptr, byte_size):
            assert byte_size == 16
            return np.array([10.0, 20.0, 30.0, 40.0], np.float32).tobytes()

        def free_all_memory(self):
            pass

    storage = np.zeros(4, np.float32)
    output_view = storage[1:3]
    runner = aclnn_pre_npu.PreNpuAclnnRunner(Device())

    runner(
        "aclnnAux",
        tensors={"output": output_view},
        attributes={},
        output_names=("output",),
        storage_shapes={"output": (4,)},
    )

    np.testing.assert_array_equal(storage, np.array([0.0, 20.0, 30.0, 0.0]))
