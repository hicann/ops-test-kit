# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""xpu_server 集成测试：heartbeat/tenant/run 端点。"""

import json
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope="module")
def xpu_server():
    """启动 dry-run xpu_server 子进程（端口 19090），等待 heartbeat 就绪。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server", "--port", "19090", "--dry-run"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to become ready (framework detection may be slow)
    import http.client

    for _ in range(30):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 19090, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                break
        except (ConnectionRefusedError, OSError):
            pass
        time.sleep(0.5)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def http_conn():
    """提供到 19090 端口的 HTTP 连接。"""
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", 19090, timeout=5)
    yield conn
    conn.close()


class TestV1HeartbeatEndpoint:
    """Merged /v1/heartbeat: old /health + /v1/detect + /heartbeat in one."""

    def test_v1_heartbeat_merges_health_detect_register(self, xpu_server, http_conn):
        """heartbeat 合并 health/detect/register：返 status + providers + hardware。"""
        conn = http_conn
        conn.request("GET", "/v1/heartbeat?tenant_id=t1")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        assert resp.status == 200
        assert body["status"] == "ok"
        assert isinstance(body["providers"], list)
        assert "hardware" in body and "device_count" in body

    def test_old_endpoints_are_gone(self, xpu_server, http_conn):
        """旧端点 /health /v1/detect /heartbeat 已移除（404）。"""
        for path in ("/health", "/v1/detect", "/heartbeat"):
            conn = http_conn
            conn.request("GET", path)
            resp = conn.getresponse()
            resp.read()
            assert resp.status == 404


class TestTenantCleanup:
    """租户清理端点。"""

    def test_delete_tenant(self, xpu_server, http_conn):
        """DELETE /v1/tenant/{id} → 200 + cleaned。"""
        http_conn.request("DELETE", "/v1/tenant/test_001")
        resp = http_conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "cleaned" in data
