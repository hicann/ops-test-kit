import argparse
import pytest
from types import SimpleNamespace

from ttk.utilities.classes import SWITCHES


def test_switches_xpu_perf_default_false():
    sw = SWITCHES()
    assert sw.xpu_perf is False


def test_switches_xpu_perf_settable():
    sw = SWITCHES()
    sw.xpu_perf = True
    assert sw.xpu_perf is True


def test_kernel_parser_has_xpu_perf_flag():
    from ttk.cli.kernel import _add_kernel_args
    p = argparse.ArgumentParser()
    _add_kernel_args(p)
    on = p.parse_args(["--xpu-perf"])
    assert on.xpu_perf is True
    off = p.parse_args([])
    assert off.xpu_perf is False


def test_args_to_switches_maps_xpu_perf():
    from ttk.cli.bridge import args_to_switches
    sw_on = args_to_switches(SimpleNamespace(input="x.csv", output="o.csv", xpu_perf=True))
    assert sw_on.xpu_perf is True
    sw_off = args_to_switches(SimpleNamespace(input="x.csv", output="o.csv", xpu_perf=False))
    assert sw_off.xpu_perf is False


def test_xpu_inputs_prefers_original_input_arrays():
    """KERNEL XPU feeds torch/tf the logical ori-shape arrays, not the NPU
    run-format arrays (e.g. NC1HWC0)."""
    from ttk.core_modules.npu.op.profiling import _xpu_inputs
    ctx = SimpleNamespace(input_arrays=("NPU_FMT",), original_input_arrays=("ORI",))
    assert _xpu_inputs(ctx) == ("ORI",)


def test_xpu_inputs_falls_back_when_original_unset():
    """original_input_arrays defaults to None (manual input / golden off) ->
    fall back to input_arrays so XPU still has data."""
    from ttk.core_modules.npu.op.profiling import _xpu_inputs
    ctx = SimpleNamespace(input_arrays=("NPU_FMT",), original_input_arrays=None)
    assert _xpu_inputs(ctx) == ("NPU_FMT",)


def test_xpu_mode_bitwise_or(monkeypatch):
    """_xpu_mode: xpu_perf→PERF, need_data→DATA, bitwise OR. Returns 0 when neither."""
    from ttk.core_modules.npu.op import profiling as prof
    from ttk.remote import PERF, DATA
    from ttk.utilities.classes import SWITCHES

    sw_off = SWITCHES()
    sw_off.xpu_perf = False
    assert prof._xpu_mode(sw_off, need_data=False) == 0
    assert prof._xpu_mode(sw_off, need_data=True) == DATA

    sw_on = SWITCHES()
    sw_on.xpu_perf = True
    assert prof._xpu_mode(sw_on, need_data=False) == PERF
    assert prof._xpu_mode(sw_on, need_data=True) == (DATA | PERF)


def test_validate_xpu_perf_raises_when_no_remote(monkeypatch):
    from ttk.cli import kernel as kernel_mod
    from ttk.utilities.classes import SWITCHES
    monkeypatch.setattr(kernel_mod, "is_remote_configured", lambda: False)
    sw = SWITCHES()
    sw.xpu_perf = True
    with pytest.raises(RuntimeError, match="xpu-perf"):
        kernel_mod._validate_xpu_perf_precondition(sw)


def test_validate_xpu_perf_ok_when_remote(monkeypatch):
    from ttk.cli import kernel as kernel_mod
    from ttk.utilities.classes import SWITCHES
    monkeypatch.setattr(kernel_mod, "is_remote_configured", lambda: True)
    sw = SWITCHES()
    sw.xpu_perf = True
    kernel_mod._validate_xpu_perf_precondition(sw)  # 不抛


def test_validate_xpu_perf_skip_when_not_enabled(monkeypatch):
    from ttk.cli import kernel as kernel_mod
    from ttk.utilities.classes import SWITCHES
    monkeypatch.setattr(kernel_mod, "is_remote_configured", lambda: False)
    sw = SWITCHES()
    sw.xpu_perf = False
    kernel_mod._validate_xpu_perf_precondition(sw)  # 没开 → 不校验、不抛


def test_extract_third_party_fail_closed():
    """_extract_third_party: priority provider PASS+outputs → outputs; else None."""
    from ttk.core_modules.npu.op import profiling as prof

    # no results / no priority → None
    assert prof._extract_third_party(None, "torch") is None
    assert prof._extract_third_party({}, "torch") is None
    assert prof._extract_third_party({"torch": {"status": "PASS", "outputs": [1]}}, None) is None
    # non-PASS / no outputs key → None
    assert prof._extract_third_party({"torch": {"status": "FAIL", "outputs": [1]}}, "torch") is None
    assert prof._extract_third_party({"torch": {"status": "PASS"}}, "torch") is None
    # happy path
    outs = [1, 2]
    assert prof._extract_third_party({"torch": {"status": "PASS", "outputs": outs}}, "torch") is outs


# Sentinel raised by the fake _do_xpu_profiling to short-circuit profile_process
# once the gate has opened — proves (and halts) the call.
class _XpuCalled(Exception):
    pass


def test_profile_process_gate_uses_xpu_mode(monkeypatch):
    """Regression guard for the XPU trigger gate in profile_process.

    The gate is now `xpu_mode = _xpu_mode(sw, need_3party_outputs);
    if xpu_mode: _do_xpu_profiling(context, xpu_mode)`. _xpu_mode is bitwise OR
    of xpu_perf(PERF) and need_3party_outputs(DATA). This test fails if the gate
    is reverted to `is_remote_configured()` (which would open unconditionally).

    Approach: monkeypatch the upstream module-global steps (clear_error_manager,
    get_global_storage, get_process_context, __parse_*, __gen_input,
    __gen_output) + the resolve path (get_spec_attr returns None tolerance,
    resolve_tolerance returns [] -> need_3party_outputs=False so gate depends
    SOLELY on xpu_perf). _do_xpu_profiling records the (context, xpu_mode) call
    and raises to short-circuit the OPEN branch; the SHUT branch (xpu_perf=False)
    proceeds past the gate to __gen_workspaces whose arg
    (context.dyn_compile_result) raises AttributeError on our stub — proof the
    gate stayed shut.
    """
    from ttk.core_modules.npu.op import profiling as prof
    from ttk.utilities.classes import SWITCHES

    sw = SWITCHES()
    sw.no_memory_check = True  # skip waiting_for_memory

    context = SimpleNamespace(
        testcase_name="t_gate",
        op_name="DummyOp",
        is_valid=True,
        compile_failed=lambda: False,
        input_bytes=0,
        output_bytes=0,
        flat_precision_tolerances=(),
        flat_absolute_precision=(),
        flat_output_dtypes=(),
    )

    def _fake_process_ctx():
        return SimpleNamespace(
            change_name=lambda _name: None,
            notify_status=lambda _s: None,
        )

    monkeypatch.setattr(prof, "get_global_storage", lambda: sw)
    monkeypatch.setattr(prof, "get_process_context", _fake_process_ctx)
    monkeypatch.setattr(prof, "__parse_manual_params", lambda _ctx: None)
    monkeypatch.setattr(prof, "__parse_dynamic_tiling_data", lambda _ctx: None)
    monkeypatch.setattr(prof, "__parse_binary_tiling_data", lambda _ctx: None)
    monkeypatch.setattr(prof, "__gen_input", lambda _ctx: None)
    monkeypatch.setattr(prof, "__gen_output", lambda _ctx: None)
    # resolve path: tolerance None -> resolve returns [] -> need_3party=False.
    monkeypatch.setattr(prof, "get_spec_attr", lambda *_a, **_k: None)
    import ttk.core_modules.comparison.resolve as resolve_mod
    monkeypatch.setattr(resolve_mod, "resolve_tolerance", lambda *_a, **_k: [])
    # clear_error_manager is imported lazily inside profile_process.
    import ttk.core_modules.npu.error_cleaner as ec
    monkeypatch.setattr(ec, "clear_error_manager", lambda: None)

    # _do_xpu_profiling(context, xpu_mode): record and (OPEN case) raise to halt.
    xpu_calls = []

    def _fake_do_xpu(_ctx, _mode):
        xpu_calls.append((_ctx, _mode))
        raise _XpuCalled()

    monkeypatch.setattr(prof, "_do_xpu_profiling", _fake_do_xpu)

    # --- gate OPEN (xpu_perf=True -> _xpu_mode returns PERF) ---
    xpu_calls.clear()
    sw.xpu_perf = True
    with pytest.raises(_XpuCalled):
        prof.profile_process(context, {}, {}, 0)
    assert len(xpu_calls) == 1, "gate open must invoke _do_xpu_profiling"

    # --- gate SHUT (xpu_perf=False, need_3party=False -> _xpu_mode returns 0) ---
    # _do_xpu_profiling must NOT be called; profile_process proceeds one line
    # past the gate to __gen_workspaces, whose arg evaluation
    # (context.dyn_compile_result) raises AttributeError on our stub.
    xpu_calls.clear()
    sw.xpu_perf = False
    with pytest.raises(AttributeError, match="dyn_compile_result"):
        prof.profile_process(context, {}, {}, 0)
    assert len(xpu_calls) == 0, (
        "gate shut (xpu_mode==0) must NOT call _do_xpu_profiling"
    )
