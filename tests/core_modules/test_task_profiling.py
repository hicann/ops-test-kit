from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from ttk.core_modules.framework_api.graph_execution import _run_compiled
from ttk.core_modules.framework_api.profiling import _execute_eager
from ttk.core_modules.msprof import MsProfiler, TtkMsProfType
from ttk.core_modules.npu.op_api import profiling as aclnn_profiling
from ttk.utilities.classes import SWITCHES
from ttk.utilities.string_utils import stable_path_component


class _Backend:
    def synchronize(self, _dev_id):
        pass

    def to_numpy(self, value):
        return value.detach().cpu().numpy().copy()


def test_e2e_task_prof_false_runs_main_api_once(monkeypatch, tmp_path):
    calls = []

    def api(value):
        calls.append(value)
        return value + 1

    from ttk.core_modules.framework_api import profiling as e2e_profiling

    monkeypatch.setattr(
        e2e_profiling,
        "prepare_device_args",
        lambda *_args: ([torch.tensor([1.0])], {}),
    )
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    switches.warmup = True
    case = SimpleNamespace(
        api_name="torch.ops.vendor.main",
        testcase_name="precision/case",
        inplace_input_indexes=(),
    )

    outputs, perf = _execute_eager(
        case,
        _Backend(),
        0,
        switches,
        object(),
        api,
        False,
        False,
        [],
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(outputs[0], np.array([2.0], np.float32))
    assert perf.elapsed_us == 0.0
    assert not (tmp_path / "msprof").exists()


def test_graph_task_prof_false_runs_only_the_required_compile_execution(tmp_path):
    calls = []

    def compiled(value):
        calls.append(value)
        return value + 1

    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    switches.warmup = True

    outputs, perf = _run_compiled(
        compiled,
        [torch.tensor([1.0])],
        {},
        _Backend(),
        0,
        switches,
        False,
        None,
        "torch.ops.vendor.main",
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(outputs[0], np.array([2.0], np.float32))
    assert perf.elapsed_us == 0.0
    assert not (tmp_path / "msprof").exists()


def test_graph_task_prof_false_restores_custom_graph_inplace_kwargs(tmp_path):
    calls = []
    original = torch.tensor([1.0])
    backup = original.clone()

    def compiled(value):
        calls.append(value)
        value.add_(1)
        return value

    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    switches.warmup = True

    outputs, perf = _run_compiled(
        compiled,
        [],
        {"value": original},
        _Backend(),
        0,
        switches,
        True,
        None,
        "torch.ops.vendor.inplace",
        inplace_backups={0: backup},
        inplace_kwargs_keys={0: "value"},
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(outputs[0], np.array([2.0], np.float32))
    np.testing.assert_array_equal(original.numpy(), np.array([1.0], np.float32))
    assert perf.elapsed_us == 0.0


def test_aclnn_task_prof_false_uses_one_execution_without_determinism(monkeypatch, tmp_path):
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)

    executor = aclnn_profiling.AclOpExecutor(
        SimpleNamespace(testcase_name="case"), object()
    )

    assert executor._run_time == 1
    assert executor._prof_type == TtkMsProfType.NONE


def test_aclnn_task_prof_false_skips_helper_warmup(monkeypatch, tmp_path):
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    switches.warmup = True
    device = SimpleNamespace(warmup=MagicMock(), is_model=lambda: False)
    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)

    executor = aclnn_profiling.AclOpExecutor(
        SimpleNamespace(testcase_name="case"), device
    )
    executor.rts_context = lambda: nullcontext()
    executor.rts_stream = lambda: nullcontext("stream")
    executor._acl_sequence = MagicMock(return_value=([], [], True, None))
    executor._process_total_cycles = MagicMock(return_value=("UNKNOWN", "UNKNOWN"))

    result = executor.do()

    device.warmup.assert_not_called()
    executor._acl_sequence.assert_called_once_with("stream")
    executor._process_total_cycles.assert_not_called()
    assert result.success is True
    assert result.api_prof == "UNKNOWN"
    assert result.op_prof == "UNKNOWN"


def test_aclnn_task_prof_true_honors_warmup_switch(monkeypatch, tmp_path):
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = True
    switches.warmup = True
    device = SimpleNamespace(warmup=MagicMock(), is_model=lambda: False)
    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)

    executor = aclnn_profiling.AclOpExecutor(
        SimpleNamespace(testcase_name="case"), device
    )
    executor.rts_context = lambda: nullcontext()
    executor.rts_stream = lambda: nullcontext("stream")
    executor._acl_sequence = MagicMock(return_value=([], [], True, None))
    executor._process_total_cycles = MagicMock(return_value=("UNKNOWN", "UNKNOWN"))

    executor.do()

    device.warmup.assert_called_once_with(switches)


def test_aclnn_task_prof_false_keeps_deterministic_repetitions(monkeypatch, tmp_path):
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = False
    switches.deterministic_level = 1
    switches.run_time = 4
    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)

    executor = aclnn_profiling.AclOpExecutor(
        SimpleNamespace(testcase_name="case"), object()
    )

    assert executor._run_time == 4
    assert executor._prof_type == TtkMsProfType.NONE


def test_disabled_msprof_does_not_delete_existing_profile(tmp_path):
    result_path = tmp_path / "msprof" / "op_api" / "case"
    result_path.mkdir(parents=True)
    marker = result_path / "previous.csv"
    marker.write_text("profile")

    MsProfiler(0, str(result_path), TtkMsProfType.NONE)

    assert marker.is_file()


def test_profile_path_component_is_stable_and_collision_resistant():
    slash = stable_path_component("case/name", "testcase")
    colon = stable_path_component("case:name", "testcase")

    assert slash != colon
    assert "/" not in slash
    assert ":" not in colon
    assert slash == stable_path_component("case/name", "testcase")


def test_aclnn_profile_path_cannot_escape_or_collide(monkeypatch, tmp_path):
    switches = SWITCHES()
    switches.root_path = str(tmp_path)
    switches.TASK_PROFILING = True
    monkeypatch.setattr(aclnn_profiling, "get_global_storage", lambda: switches)

    slash = aclnn_profiling.AclOpExecutor(
        SimpleNamespace(testcase_name="../case/name"), object()
    )
    colon = aclnn_profiling.AclOpExecutor(
        SimpleNamespace(testcase_name="../case:name"), object()
    )

    root = (tmp_path / "msprof" / "op_api").resolve()
    assert str(slash._prof_result_path).startswith(str(root))
    assert str(colon._prof_result_path).startswith(str(root))
    assert slash._prof_result_path != colon._prof_result_path
