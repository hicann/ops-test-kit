from types import SimpleNamespace
from unittest.mock import MagicMock

from ttk.core_modules.npu.instance_refactor import NpuInstance


def _instance():
    instance = NpuInstance.__new__(NpuInstance)
    instance.switches = SimpleNamespace(
        manual_data_mode="prepare",
        device_count=-1,
        dev_plat="Ascend950",
        short_soc_version=None,
        mode=SimpleNamespace(is_model=lambda: False, is_online_board=lambda: True),
        compile_only=False,
        validate_only=False,
    )
    return instance


def test_prepare_uses_one_logical_worker_without_querying_devices(monkeypatch):
    instance = _instance()
    dsmi = MagicMock(side_effect=AssertionError("DSMI must not be queried"))
    monkeypatch.setattr(
        "ttk.core_modules.npu.instance_refactor.DSMIInterface", dsmi
    )

    instance.get_device_count()

    assert instance.switches.device_count == 1
    assert not dsmi.called


def test_prepare_device_summary_does_not_query_hardware():
    instance = _instance()

    assert instance.device_info(0) == "manual-data:0 Ascend950"


def test_prepare_setup_skips_helper_kernel_compilation(monkeypatch):
    instance = _instance()
    instance.task_keeper = object()
    instance.mp_context = object()
    instance.case_original_headers = ["api_name"]
    compile_helpers = MagicMock(side_effect=AssertionError("helper kernels must not compile"))
    profile_object = object()
    profile_object_factory = MagicMock(return_value=profile_object)
    instance._compile_help_kernels = compile_helpers
    monkeypatch.setattr(
        "ttk.core_modules.npu.op_api.ApiProfileObject", profile_object_factory
    )

    instance.setup_profile_object()

    assert instance.profile_object is profile_object
    assert not compile_helpers.called
