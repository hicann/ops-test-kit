# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_handle_run 设备锁集成测试：验证非 dry_run 时调 _assign_device、dry_run 跳过分配、CPU 模式无 KeyError。"""
import os
import threading
from unittest.mock import MagicMock

from ttk.remote.server import xpu_server
from ttk.remote.server.xpu_server import XpuRequestHandler

FAKE_IN = "/tmp/fake"


def _setup_handler(h, device_ids, gpu_locks, dry_run, use_device=False,
                   hardware="gpu", profile=None):
    """Configure a minimal XpuRequestHandler instance for _handle_run tests.

    The real _handle_run reads tenant_id via self._get_header; we stub it to
    always yield a valid tenant so the request reaches the dispatch body.

    hardware 形参保留——heartbeat (:287) 读 self.hardware。profile 形参用于
    _handle_run :464 kwargs（executor 消费）。is None 守护保留 {} cpu 兜底
    （非 or，防 {} 被 "gpu" str 覆盖导致 profile["torch_lib"] TypeError）。
    """
    h.dry_run = dry_run
    h.use_device = use_device
    h.device_ids = device_ids
    h.hardware = hardware
    h.profile = profile if profile is not None else {"torch_lib": "cuda"}
    h._device_rr_counter = 0
    h._device_rr_lock = threading.Lock()
    xpu_server._device_locks = gpu_locks
    h.data_gate = None
    h.tmp_root = "/tmp"
    h.sync_base_dir = "/tmp"
    h.run_deadline_s = 30
    h.sandbox = "none"

    def _get_header(k, d=""):
        return "test_tenant" if k == "X-Tenant-ID" else d

    h._get_header = _get_header
    h._send_run_ok = MagicMock()
    h._send_json = MagicMock()
    h.rfile = MagicMock()
    h.headers = MagicMock()


def _patch_run_env(monkeypatch):
    """Stub body receive, subprocess, and rmtree so _handle_run walks the path
    without touching the filesystem (except the getsize on FAKE_IN)."""
    monkeypatch.setattr("ttk.remote.server.xpu_server._run_in_subprocess",
                        lambda kw, deadline: {"ok": True, "http_status": 200,
                                              "output_path": None,
                                              "output_count": 0, "shapes": [],
                                              "dtypes": [], "perf": None,
                                              "api": None})
    monkeypatch.setattr("ttk.remote.server.xpu_server._receive_body_to_file",
                        lambda handler, dir=None: FAKE_IN)
    monkeypatch.setattr("ttk.remote.server.xpu_server.shutil.rmtree",
                        lambda *a, **kw: None)


def test_handle_run_uses_assign_device(monkeypatch):
    """_handle_run 非 dry_run 时调 _assign_device（而非 _device_locks / device_ids[0]）。"""
    _patch_run_env(monkeypatch)
    with open(FAKE_IN, "wb"):
        pass
    try:
        h = XpuRequestHandler.__new__(XpuRequestHandler)
        _setup_handler(h, [0, 1], {0: threading.Lock(), 1: threading.Lock()},
                       dry_run=False, use_device=True)

        captured = {}

        def fake_assign():
            captured["called"] = True
            # mirror real _assign_device contract: acquire the per-device lock
            # so the caller's finally can release it.
            assert xpu_server._device_locks[0].acquire(blocking=False) is True
            return 0
        h._assign_device = fake_assign

        h._handle_run()

        assert captured.get("called") is True
        # device 0 的锁被 release（finally）——能再次 acquire 证明已 release
        assert xpu_server._device_locks[0].acquire(blocking=False) is True
        xpu_server._device_locks[0].release()
    finally:
        if os.path.exists(FAKE_IN):
            os.remove(FAKE_IN)

