"""T6: _handle_run 执行隔离 + provider 归一化 单测。

测试 _build_device_opts 纯函数（只返回 device 三件：device_id/docker_args/env，
不构造完整 kwargs——避免 BaseHTTPRequestHandler mock）+ get_framework provider 归一化。
spec §4.3.1 / §7。

Step 6b 追加：穿透链 e2e（load→detect→handler profile→executor）+ 异常 6 场景串测试
（spec §7 / §4.6 双层职责——status 三分 ok→info/≥500→error/<500→warning）。
"""
import logging
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ttk.remote.server import xpu_server
from ttk.remote.server.config import detect_hardware, get_framework, load_server_config
from ttk.remote.server.xpu_server import XpuRequestHandler, _build_device_opts


def _h(**ov):
    """造一个类 handler 的 SimpleNamespace（_build_device_opts 只读 3 属性）。"""
    d = dict(use_device=True, sandbox="none",
             profile={"torch_lib": "cuda", "torch_profiler": {"activities": ["CPU"]}})
    d.update(ov)
    return SimpleNamespace(**d)


# ---- _build_device_opts（spec §4.3.1 执行隔离 + §7 执行隔离单测①-⑤）----

def test_device_id_zero():
    """非cpu 分支 device_id 固定 0（容器内域，executor 见 cuda:0）。"""
    assert _build_device_opts(_h(), n=0)["device_id"] == 0


def test_visible_env_cuda():
    """sandbox=none 默认推导 {torch_lib.upper()}_VISIBLE_DEVICES（cuda→CUDA）。"""
    assert _build_device_opts(_h(), n=0)["env"] == {"CUDA_VISIBLE_DEVICES": "0"}


def test_visible_env_mlu():
    """推导 mlu→MLU（spec §4.1 visible_env 默认推导）。"""
    h = _h(profile={"torch_lib": "mlu", "torch_profiler": {"activities": ["CPU"]}})
    assert _build_device_opts(h, n=0)["env"] == {"MLU_VISIBLE_DEVICES": "0"}


def test_visible_env_override():
    """visible_env yaml 显式覆盖推导（特例 npu/rocm）。"""
    h = _h(profile={"torch_lib": "xpu", "visible_env": "ASCEND_RT_VISIBLE_DEVICES",
                    "torch_profiler": {"activities": ["CPU"]}})
    assert _build_device_opts(h, n=0)["env"] == {"ASCEND_RT_VISIBLE_DEVICES": "0"}


def test_docker_args_render():
    """docker 分支 render {device_id} 占位→2-token list；docker 分支不设 env。"""
    h = _h(sandbox="docker", profile={"torch_lib": "cuda", "torch_profiler": {"activities": ["CPU"]},
                                     "docker_args": ["--gpus", "device={device_id}"]})
    opts = _build_device_opts(h, n=0)
    assert opts["docker_args"] == ["--gpus", "device=0"]
    assert "env" not in opts


def test_docker_fail_fast():
    """sandbox=docker 但 profile 缺 docker_args → fail-fast（http_status=500，render 前检查）。"""
    h = _h(sandbox="docker", profile={"torch_lib": "cuda", "torch_profiler": {"activities": ["CPU"]}})
    opts = _build_device_opts(h, n=0)
    assert opts.get("ok") is False and opts.get("http_status") == 500


def test_cpu_device_id():
    """cpu 分支 n 被忽略（None），device_id="cpu"（对齐 §7 device_id 双语义）。"""
    h = _h(use_device=False)
    assert _build_device_opts(h, n=None)["device_id"] == "cpu"


# ---- provider 归一化（spec §4.3.1 第 1 点 caller + §7 provider 归一化单测）----

def test_get_framework_passthrough():
    """无 override：framework = provider。"""
    assert get_framework("torch") == "torch"


def test_get_framework_override():
    """flash_attn + {flash_attn: torch} → torch（激活零消费函数）。"""
    assert get_framework("flash_attn", {"flash_attn": "torch"}) == "torch"


def test_get_framework_unknown_keeps_raw():
    """未知 provider + override dict → 原值（不报错）。"""
    assert get_framework("unknown", {"flash_attn": "torch"}) == "unknown"


# ============================================================================
# Step 6b: 穿透链 e2e + 异常 6 场景串测试（spec §7 / §4.6 双层职责）
# ============================================================================

_FAKE_IN = "/tmp/ttk_t6_fake_in"


def _setup_handler(h, *, use_device=True, sandbox="none", profile=None,
                   hardware="gpu", device_ids=None, provider_framework=None,
                   docker_images=None):
    """造最小可调 _handle_run 的 handler（不经 run_server）。

    复用 test_handle_run_device_lock 的 _setup_handler 形态，补 T6 新字段：
    provider_framework（provider 归一化）/ sandbox / docker_images。_get_header
    stub 让 tenant 合法、provider 默认 torch。
    """
    h.dry_run = False
    h.use_device = use_device
    h.device_ids = device_ids if device_ids is not None else [0]
    h.hardware = hardware
    h.profile = profile if profile is not None else {"torch_lib": "cuda"}
    h.provider_framework = provider_framework if provider_framework is not None else {}
    h.docker_images = docker_images if docker_images is not None else {}
    h.docker_memory = "8g"
    h.docker_network = "none"
    h._device_rr_counter = 0
    h._device_rr_lock = threading.Lock()
    xpu_server._device_locks = {d: threading.Lock() for d in h.device_ids if d != "cpu"}
    h.data_gate = None
    h.tmp_root = "/tmp"
    h.sync_base_dir = "/tmp"
    h.run_deadline_s = 30
    h.sandbox = sandbox
    h._device_rr_counter = 0
    # _get_header: tenant 合法 + provider 默认 torch + 空 body 相关 header
    def _get_header(k, d=""):
        if k == "X-Tenant-ID":
            return "t6_tenant"
        if k == "X-Provider":
            return d          # 默认 "torch"
        return d
    h._get_header = _get_header
    h._send_run_ok = MagicMock()
    h._send_json = MagicMock()
    h.rfile = MagicMock()
    h.headers = MagicMock()


def _patch_run_env(monkeypatch, envelope=None, capture=None):
    """Stub body receive / subprocess / rmtree。

    envelope: _run_in_subprocess 返回的信封（None=ok 200）。
    capture: 若给定 dict，把 kwargs 写入 capture['kwargs']（穿透链断言用）。
    """
    env = envelope if envelope is not None else {
        "ok": True, "http_status": 200, "output_path": None,
        "output_count": 0, "shapes": [], "schema": [], "perf": None, "api": None}
    def _fake_subproc(kwargs, deadline):
        if capture is not None:
            capture["kwargs"] = kwargs
        return dict(env)
    monkeypatch.setattr("ttk.remote.server.xpu_server._run_in_subprocess", _fake_subproc)
    monkeypatch.setattr("ttk.remote.server.xpu_server._receive_body_to_file",
                        lambda handler, dir=None: _FAKE_IN)
    monkeypatch.setattr("ttk.remote.server.xpu_server.shutil.rmtree",
                        lambda *a, **kw: None)


@pytest.fixture
def _fake_body():
    """提供 _FAKE_IN 文件（_receive_body_to_file stub 后 getsize 读它）。"""
    with open(_FAKE_IN, "wb"):
        pass
    yield
    if os.path.exists(_FAKE_IN):
        os.remove(_FAKE_IN)


# ---- 穿透链 e2e：load→detect→handler profile→executor（spec §7）----

def test_profile_penetration_chain_gpu(monkeypatch, _fake_body):
    """穿透链：yaml 段 → detect role → handler.profile → kwargs.profile 同一 dict。

    load_server_config 读 hardware 段 → detect_hardware 选 role（gpu，mock /dev）→
    handler.profile=hw_config[role] → _handle_run kwargs 透传同一 profile。
    断言全链 profile 一致（torch_lib/profiler 不丢）。spec §7 穿透链 e2e。
    """
    import ttk.remote.server.config as cfgmod
    monkeypatch.setattr(cfgmod.os, "listdir", lambda p: ["nvidia0", "tty"])
    cfg = load_server_config()                      # 读 repo xpu_server.yaml（gpu/mlu 段）
    role, device_ids = detect_hardware(cfg["hardware_config"])
    assert role == "gpu"                            # mock /dev/nvidia0 命中 gpu 段
    profile = cfg["hardware_config"][role]          # handler.run_server 设的 profile

    capture = {}
    _patch_run_env(monkeypatch, capture=capture)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, use_device=True, profile=profile, device_ids=device_ids)
    h._handle_run()

    kw = capture["kwargs"]
    assert kw["profile"] is profile                 # 同一对象穿透（未拷贝/未丢）
    assert kw["profile"]["torch_lib"] == "cuda"
    assert kw["device_id"] == 0                     # 容器内固定（executor 见 {lib}:0）


def test_profile_penetration_chain_cpu_fallback(monkeypatch, _fake_body):
    """cpu 兜底穿透链：detect 全未命中 → device_ids=["cpu"] → kwargs device_id="cpu"。"""
    import ttk.remote.server.config as cfgmod
    monkeypatch.setattr(cfgmod.os, "listdir", lambda p: [])   # /dev 空 → cpu 兜底
    cfg = load_server_config()
    role, device_ids = detect_hardware(cfg["hardware_config"])
    assert role == "cpu" and device_ids == ["cpu"]

    capture = {}
    _patch_run_env(monkeypatch, capture=capture)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, use_device=False, hardware="cpu", profile={},
                   device_ids=["cpu"])
    h._handle_run()

    kw = capture["kwargs"]
    assert kw["device_id"] == "cpu"                 # cpu 分支 device_id="cpu"
    assert kw["profile"] == {}                      # cpu 兜底 profile {}


def test_provider_normalization_in_handle_run(monkeypatch, _fake_body):
    """_handle_run :raw_provider→get_framework 归一化（flash_attn→torch）。"""
    capture = {}
    _patch_run_env(monkeypatch, capture=capture)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, profile={"torch_lib": "cuda"},
                   provider_framework={"flash_attn": "torch"})
    # X-Provider=flash_attn → 归一化 torch
    h._get_header = lambda k, d="": {
        "X-Tenant-ID": "t6_tenant", "X-Provider": "flash_attn"}.get(k, d)
    h._handle_run()

    assert capture["kwargs"]["provider"] == "torch"  # 归一化后


# ---- 异常 6 场景串测试（spec §4.6 status 三分）----
#
# executor 层异常（format_device/activities/torch_lib/executor）经 _run_in_subprocess
# 返回 envelope（http_status 400/500）；handler 层异常（docker_args/image）在 _handle_run
# 内捕获。断言 _send_json status + logging 级别（≥500→error / <500→warning）。

def _run_and_get_send_json(h):
    """提取 _handle_run 调用后 _send_json 的 (status, body) 参数。"""
    h._send_json.assert_called_once()
    args, _ = h._send_json.call_args
    return args[0], args[1]      # status, body


def test_exc_format_device_tf_no_device_type_400(monkeypatch, _fake_body, caplog):
    """format_device tf 缺 tf_device_type → 400（客户端错，warning，无堆栈）。"""
    envelope = {"ok": False, "http_status": 400,
                "error": "hardware has no tf_device_type", "api": "tf_api"}
    _patch_run_env(monkeypatch, envelope=envelope)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, profile={"torch_lib": "cuda"})
    with caplog.at_level(logging.WARNING):
        h._handle_run()
    status, body = _run_and_get_send_json(h)
    assert status == 400
    assert "tf_device_type" in body["error"]
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_exc_activities_runtime_error_500(monkeypatch, _fake_body, caplog):
    """activities 缺/枚举名错 → RuntimeError → 500（服务端错，error）。"""
    envelope = {"ok": False, "http_status": 500,
                "error": "profile missing torch_profiler/activities", "api": "x"}
    _patch_run_env(monkeypatch, envelope=envelope)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, profile={"torch_lib": "cuda"})
    with caplog.at_level(logging.ERROR):
        h._handle_run()
    status, body = _run_and_get_send_json(h)
    assert status == 500
    assert "activities" in body["error"]
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_exc_torch_lib_keyerror_500(monkeypatch, _fake_body, caplog):
    """torch_lib KeyError（profile={} 误进 device 路径）→ 500。"""
    envelope = {"ok": False, "http_status": 500,
                "error": "'torch_lib'", "api": "x"}
    _patch_run_env(monkeypatch, envelope=envelope)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, profile={"torch_lib": "cuda"})
    with caplog.at_level(logging.ERROR):
        h._handle_run()
    status, _ = _run_and_get_send_json(h)
    assert status == 500
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_exc_docker_args_fail_fast_500(monkeypatch, _fake_body, caplog):
    """sandbox=docker + profile 缺 docker_args → handler fail-fast 500（render 前检查）。

    不进 executor（_build_device_opts 直接返 ok=False）；_run_in_container 不被调。
    """
    run_called = []
    monkeypatch.setattr("ttk.remote.server.xpu_server._run_in_container",
                        lambda *a, **kw: run_called.append(1) or {"ok": True})
    monkeypatch.setattr("ttk.remote.server.xpu_server._receive_body_to_file",
                        lambda handler, dir=None: _FAKE_IN)
    monkeypatch.setattr("ttk.remote.server.xpu_server.shutil.rmtree",
                        lambda *a, **kw: None)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, sandbox="docker", profile={"torch_lib": "cuda"},
                   docker_images={"torch": "img"})
    with caplog.at_level(logging.ERROR):
        h._handle_run()
    status, body = _run_and_get_send_json(h)
    assert status == 500
    assert "docker_args" in body["error"]
    assert run_called == []                       # fail-fast：未进 _run_in_container
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_exc_executor_failure_500(monkeypatch, _fake_body, caplog):
    """executor 失败 → 500（envelope http_status=500）。"""
    envelope = {"ok": False, "http_status": 500,
                "error": "import failed: No module", "api": "x"}
    _patch_run_env(monkeypatch, envelope=envelope)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, profile={"torch_lib": "cuda"})
    with caplog.at_level(logging.ERROR):
        h._handle_run()
    status, body = _run_and_get_send_json(h)
    assert status == 500
    assert "import failed" in body["error"]
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_exc_docker_image_missing_500(monkeypatch, _fake_body, caplog):
    """sandbox=docker + provider 无镜像 → 500（image 查 fail-fast）。"""
    monkeypatch.setattr("ttk.remote.server.xpu_server._run_in_container",
                        lambda *a, **kw: {"ok": True})
    monkeypatch.setattr("ttk.remote.server.xpu_server._receive_body_to_file",
                        lambda handler, dir=None: _FAKE_IN)
    monkeypatch.setattr("ttk.remote.server.xpu_server.shutil.rmtree",
                        lambda *a, **kw: None)
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    # docker_args 给齐（过 _build_device_opts fail-fast），但 docker_images 空
    _setup_handler(h, sandbox="docker",
                   profile={"torch_lib": "cuda",
                            "docker_args": ["--gpus", "device={device_id}"]},
                   docker_images={})              # 无 torch 镜像
    with caplog.at_level(logging.ERROR):
        h._handle_run()
    status, body = _run_and_get_send_json(h)
    assert status == 500
    assert "no docker image" in body["error"]
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_ok_path_logs_info(monkeypatch, _fake_body, caplog):
    """ok 路径 → logging.info（_send_run_ok，非 _send_json）。spec §4.6 ok→info。"""
    _patch_run_env(monkeypatch, envelope={"ok": True, "http_status": 200,
                                          "output_path": None, "output_count": 0,
                                          "shapes": [], "schema": [], "perf": None, "api": None})
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    _setup_handler(h, profile={"torch_lib": "cuda"})
    with caplog.at_level(logging.INFO):
        h._handle_run()
    h._send_run_ok.assert_called_once()
    h._send_json.assert_not_called()
    assert any("_handle_run ok" in r.getMessage() and r.levelno == logging.INFO
               for r in caplog.records)
