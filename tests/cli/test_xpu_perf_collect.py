# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""XPU 性能采集测试：--xpu-perf 参数解析、_xpu_mode 位运算 gate、
validate_xpu_perf_precondition 前置校验、_extract_third_party fail-closed、
profile_process 中 XPU gate 的 OPEN/SHUT 回归看护。"""

import argparse
from types import SimpleNamespace

import pytest

from ttk.utilities.classes import SWITCHES

# -- xpu_perf slot + CLI 参数解析 --------------------------------------------


def test_xpu_perf_slot_default_and_assignable():
    """xpu_perf slot 默认 False，可赋值 True。"""
    sw = SWITCHES()
    assert sw.xpu_perf is False
    sw.xpu_perf = True
    assert sw.xpu_perf is True


def test_xpu_perf_flag_registered_on_kernel_parser():
    """--xpu-perf 在 kernel 子命令注册；不传时默认 False。"""
    from ttk.cli.kernel import _add_kernel_args

    p = argparse.ArgumentParser()
    _add_kernel_args(p)
    assert p.parse_args(["--xpu-perf"]).xpu_perf is True
    assert p.parse_args([]).xpu_perf is False


def test_xpu_perf_mapped_to_switches():
    """args.xpu_perf 经 args_to_switches 映射到 sw.xpu_perf。"""
    from ttk.cli.bridge import args_to_switches

    sw_on = args_to_switches(SimpleNamespace(input="x.csv", output="o.csv", xpu_perf=True))
    assert sw_on.xpu_perf is True
    sw_off = args_to_switches(SimpleNamespace(input="x.csv", output="o.csv", xpu_perf=False))
    assert sw_off.xpu_perf is False


# -- _xpu_inputs 数据选择 ----------------------------------------------------


def test_xpu_inputs_prefers_original_shape_arrays():
    """XPU 优先喂 logical ori-shape 数组，而非 NPU run-format 数组（如 NC1HWC0）。"""
    from ttk.core_modules.npu.op.profiling import _xpu_inputs

    ctx = SimpleNamespace(input_arrays=("NPU_FMT",), original_input_arrays=("ORI",))
    assert _xpu_inputs(ctx) == ("ORI",)


def test_xpu_inputs_falls_back_to_input_arrays():
    """original_input_arrays 为 None 时 fallback 到 input_arrays（manual input / golden off）。"""
    from ttk.core_modules.npu.op.profiling import _xpu_inputs

    ctx = SimpleNamespace(input_arrays=("NPU_FMT",), original_input_arrays=None)
    assert _xpu_inputs(ctx) == ("NPU_FMT",)


# -- _xpu_mode 位运算 --------------------------------------------------------


def test_xpu_mode_bitwise_or():
    """_xpu_mode = xpu_perf(PERF) | need_data(DATA)；全 False 返回 0。"""
    from ttk.core_modules.npu.op import profiling as prof
    from ttk.remote import DATA, PERF

    sw_off = SWITCHES()
    sw_off.xpu_perf = False
    assert prof._xpu_mode(sw_off, need_data=False) == 0
    assert prof._xpu_mode(sw_off, need_data=True) == DATA

    sw_on = SWITCHES()
    sw_on.xpu_perf = True
    assert prof._xpu_mode(sw_on, need_data=False) == PERF
    assert prof._xpu_mode(sw_on, need_data=True) == (DATA | PERF)


# -- validate_xpu_perf_precondition 前置校验 ---------------------------------


def test_validate_xpu_perf_precondition_three_branches(monkeypatch):
    """前置校验三分支：无远端+开启→抛错；有远端+开启→不抛；未开启→跳过。"""
    from ttk.cli import common as common_mod

    sw = SWITCHES()
    sw.xpu_perf = True

    # 无远端配置 + 开启 → RuntimeError
    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: False)
    with pytest.raises(RuntimeError, match="xpu-perf"):
        common_mod.validate_xpu_perf_precondition(sw)

    # 有远端配置 + 开启 → 不抛
    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: True)
    common_mod.validate_xpu_perf_precondition(sw)

    # 未开启 → 跳过（即使无远端也不校验）
    sw.xpu_perf = False
    monkeypatch.setattr(common_mod, "is_remote_configured", lambda: False)
    common_mod.validate_xpu_perf_precondition(sw)


# -- _extract_third_party fail-closed ---------------------------------------


def test_extract_third_party_fail_closed():
    """_extract_third_party fail-closed：非 PASS / 无 outputs / 无 priority → None。"""
    from ttk.core_modules.npu.op import profiling as prof

    # 无 results / 无 priority → None
    assert prof._extract_third_party(None, "torch") is None
    assert prof._extract_third_party({}, "torch") is None
    assert prof._extract_third_party({"torch": {"status": "PASS", "outputs": [1]}}, None) is None
    # 非 PASS / 无 outputs key → None
    assert prof._extract_third_party({"torch": {"status": "FAIL", "outputs": [1]}}, "torch") is None
    assert prof._extract_third_party({"torch": {"status": "PASS"}}, "torch") is None
    # happy path：PASS + outputs → 返回 outputs 引用
    outs = [1, 2]
    assert prof._extract_third_party({"torch": {"status": "PASS", "outputs": outs}}, "torch") is outs


# -- profile_process XPU gate 回归看护 ---------------------------------------


# _fake_do_xpu 被调用时抛此异常，短路 profile_process 证明 gate 已开。
class _XpuCalled(Exception):
    pass


def test_profile_process_gate_open_and_shut(monkeypatch):
    """profile_process 的 XPU gate 回归看护。

    gate = _xpu_mode(sw, need_3party_outputs)；若 gate != 0 则调 _do_xpu_profiling。
    本测试隔离 gate 上游所有依赖（parse/gen_input/gen_output/resolve 等），
    使 need_3party=False，gate 完全由 xpu_perf 决定：

    - gate OPEN（xpu_perf=True）：_do_xpu_profiling 被调用 1 次，短路抛 _XpuCalled。
    - gate SHUT（xpu_perf=False）：_do_xpu_profiling 调用 0 次，profile_process
      越过 gate 到 __gen_workspaces，因 context 无 dyn_compile_result 抛 AttributeError。

    若 gate 被回退为 is_remote_configured()（无条件开启），SHUT 分支会失败。
    """
    from ttk.core_modules.npu.op import profiling as prof

    sw = SWITCHES()
    sw.no_memory_check = True  # 跳过 waiting_for_memory

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
        flat_input_dtypes=(),
    )

    # 隔离 gate 上游：所有 parse/gen 步骤为 no-op
    monkeypatch.setattr(prof, "get_global_storage", lambda: sw)
    monkeypatch.setattr(
        prof,
        "get_process_context",
        lambda: SimpleNamespace(
            change_name=lambda _name: None,
            notify_status=lambda _s: None,
        ),
    )
    monkeypatch.setattr(prof, "__parse_manual_params", lambda _ctx: None)
    monkeypatch.setattr(prof, "__parse_dynamic_tiling_data", lambda _ctx: None)
    monkeypatch.setattr(prof, "__parse_binary_tiling_data", lambda _ctx: None)
    monkeypatch.setattr(prof, "__gen_input", lambda _ctx: None)
    monkeypatch.setattr(prof, "__gen_output", lambda _ctx: None)
    # resolve 路径：tolerance None → resolve 返回 [] → need_3party=False
    monkeypatch.setattr(prof, "get_spec_attr", lambda *_a, **_k: None)
    import ttk.core_modules.comparison.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "resolve_tolerance", lambda *_a, **_k: [])
    # clear_error_manager 在 profile_process 内 lazy import
    import ttk.core_modules.npu.error_cleaner as ec

    monkeypatch.setattr(ec, "clear_error_manager", lambda: None)

    # 记录 _do_xpu_profiling 调用；OPEN 时抛异常短路
    xpu_calls = []

    def _fake_do_xpu(_ctx, _mode):
        xpu_calls.append((_ctx, _mode))
        raise _XpuCalled()

    monkeypatch.setattr(prof, "_do_xpu_profiling", _fake_do_xpu)

    # --- gate OPEN：xpu_perf=True → _xpu_mode 返回 PERF → 调 _do_xpu_profiling ---
    xpu_calls.clear()
    sw.xpu_perf = True
    with pytest.raises(_XpuCalled):
        prof.profile_process(context, {}, {}, 0)
    assert len(xpu_calls) == 1, "gate open must invoke _do_xpu_profiling"

    # --- gate SHUT：xpu_perf=False, need_3party=False → _xpu_mode 返回 0 ---
    # _do_xpu_profiling 不应被调用；profile_process 越过 gate 到 __gen_workspaces，
    # 因 context 无 dyn_compile_result 而抛 AttributeError（证明 gate 未开）。
    xpu_calls.clear()
    sw.xpu_perf = False
    with pytest.raises(AttributeError, match="dyn_compile_result"):
        prof.profile_process(context, {}, {}, 0)
    assert len(xpu_calls) == 0, "gate shut must NOT call _do_xpu_profiling"
