import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import pytest


class _HealthHandler(BaseHTTPRequestHandler):
    """Mini HTTP server that returns 200 on /v1/heartbeat (merged endpoint)."""
    def do_GET(self):
        if self.path.startswith("/v1/heartbeat"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'{"status": "ok", "hardware": "Ascend910", '
                b'"device_count": 1, "providers": ["torch"]}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence logs


@pytest.fixture
def healthy_server():
    """Start a mock server that responds 200 to /v1/heartbeat."""
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    yield {"host": "127.0.0.1", "port": port}
    server.shutdown()


@pytest.fixture
def dead_server_port():
    """Return a port nobody is listening on — simulates unreachable endpoint."""
    # Just pick a port and don't bind anything
    return 19999


class TestHeartbeatLoopHealthFile:
    def test_writes_health_file_with_alive_true_on_success(self, tmp_path, healthy_server):
        from ttk.remote.heartbeat import heartbeat_loop
        from ttk.remote.config import Endpoint
        health_path = str(tmp_path / "health.json")
        ep = Endpoint(host=healthy_server["host"], port=healthy_server["port"])

        # Run one cycle via stop_event after brief sleep
        import threading
        stop = threading.Event()
        stop.set()  # will exit immediately, but first writes once

        # Actually we need to run heartbeat_loop and read the file
        # Use a thread that we can control
        t = Thread(
            target=heartbeat_loop,
            kwargs={
                "endpoints": [ep],
                "tenant_id": "test",
                "health_path": health_path,
            },
            daemon=True,
        )
        t.start()
        # Wait for at least one cycle to complete
        deadline = time.time() + 5
        while time.time() < deadline:
            if os.path.isfile(health_path):
                with open(health_path) as f:
                    data = json.load(f)
                endpoint_key = f"{ep.host}:{ep.port}"
                if endpoint_key in data.get("endpoints", {}):
                    ep_state = data["endpoints"][endpoint_key]
                    assert ep_state["alive"] is True
                    assert ep_state["last_seen"] > 0
                    # parsed from /v1/heartbeat response body
                    assert ep_state["providers"] == ["torch"]
                    assert ep_state["hardware"] == "Ascend910"
                    return
            time.sleep(0.1)
        pytest.fail(f"Health file not written with endpoint data within 5s: "
                    f"contents={open(health_path).read() if os.path.isfile(health_path) else 'missing'}")

    def test_writes_alive_false_on_connection_refused(self, tmp_path, dead_server_port):
        """When endpoint is unreachable, alive=False is written."""
        from ttk.remote.heartbeat import heartbeat_loop
        from ttk.remote.config import Endpoint
        health_path = str(tmp_path / "health.json")
        ep = Endpoint(host="127.0.0.1", port=dead_server_port)

        t = Thread(
            target=heartbeat_loop,
            kwargs={
                "endpoints": [ep],
                "tenant_id": "test",
                "health_path": health_path,
            },
            daemon=True,
        )
        t.start()
        # Wait for one cycle — single GET /v1/heartbeat fails with connection refused
        # (daemon thread, no per-probe join timeout overhead on refused). Expected
        # ~one heartbeat cycle (11s interval + 5s probe timeout ≈ 16s); 25s ceiling.
        deadline = time.time() + 25
        while time.time() < deadline:
            if os.path.isfile(health_path):
                with open(health_path) as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        time.sleep(0.1)
                        continue
                endpoint_key = f"127.0.0.1:{dead_server_port}"
                if endpoint_key in data.get("endpoints", {}):
                    assert data["endpoints"][endpoint_key]["alive"] is False
                    return
            time.sleep(0.1)
        pytest.fail(f"Health file not written with alive=False within 25s")


def test_daemon_probe_threads_do_not_accumulate_across_cycles(monkeypatch, tmp_path, healthy_server):
    """Probe threads must be joined each cycle — no thread accumulation (spec §13).

    Each heartbeat cycle spawns one daemon Thread per endpoint to probe
    GET /v1/heartbeat concurrently, then JOINs them before sleeping. If those
    threads leaked (not joined, or a stuck probe kept them alive), thread count
    would grow monotonically cycle over cycle. Against an instant-responding mock
    server, probe threads finish + are joined within HEARTBEAT_TIMEOUT_S.

    Sampling strategy: we patch time.sleep (called once per cycle, right
    AFTER the probe threads are joined and the health file is written) to record
    threading.active_count() at that precise between-cycles point. At that moment
    only the loop thread itself should be live over baseline. We assert the
    per-cycle sample does not grow across cycles and stays at baseline+1.
    """
    import threading
    from ttk.remote import heartbeat as hb_mod
    from ttk.remote.heartbeat import heartbeat_loop
    from ttk.remote.config import Endpoint

    health_path = str(tmp_path / "health.json")
    ep = Endpoint(host=healthy_server["host"], port=healthy_server["port"])

    baseline = threading.active_count()
    samples = []
    TARGET_CYCLES = 4

    _real_sleep = time.sleep  # capture before patching

    def _sleep_and_sample(seconds):
        # Called once per cycle AFTER probes are joined + health file written.
        # This is the deterministic between-cycles point: probe threads are gone.
        samples.append(threading.active_count())
        if len(samples) >= TARGET_CYCLES:
            raise SystemExit  # unwind heartbeat_loop's while True, end the thread
        _real_sleep(0.02)  # tiny real sleep so next cycle's probes get CPU

    # heartbeat_loop now calls time.sleep directly (no _interruptible_sleep);
    # patch time.sleep on the heartbeat module to intercept the between-cycles point.
    monkeypatch.setattr(hb_mod.time, "sleep", _sleep_and_sample)

    def _run_with_stop():
        # Contain the SystemExit we raise to end the loop so it does not surface
        # as a PytestUnhandledThreadExceptionWarning.
        try:
            heartbeat_loop(endpoints=[ep], tenant_id="leak-test",
                           health_path=health_path)
        except BaseException:
            pass

    loop_t = Thread(target=_run_with_stop, daemon=True)
    loop_t.start()
    loop_t.join(timeout=10)

    assert len(samples) == TARGET_CYCLES, (
        f"only captured {len(samples)} cycles (expected {TARGET_CYCLES}); "
        f"health_path exists={os.path.isfile(health_path)}")
    # At each between-cycles sample only the loop thread is live over baseline.
    expected = baseline + 1
    assert all(s == expected for s in samples), (
        f"thread count did not stay at baseline+1 across cycles (leak): "
        f"baseline={baseline} expected={expected} samples={samples}")


def test_parent_death_triggers_cleanup(monkeypatch):
    """heartbeat_loop detects ppid change (parent death) and calls _cleanup_all."""
    import ttk.remote.heartbeat as hb

    # Simulate: first getppid() = 1000 (captured as original_ppid),
    # all subsequent = 9999 (reparented → parent death detected).
    _state = {"first": True}
    def _mock_getppid():
        if _state["first"]:
            _state["first"] = False
            return 1000
        return 9999
    monkeypatch.setattr(hb.os, "getppid", _mock_getppid)

    cleanup_called = []
    monkeypatch.setattr(hb, "_cleanup_all",
                        lambda *a, **kw: cleanup_called.append(True))

    hb.heartbeat_loop(endpoints=[], tenant_id="test",
                       health_path="/tmp/_hb_ppid_test.json", tls=None)

    assert cleanup_called, "parent death (ppid change) should trigger _cleanup_all"

