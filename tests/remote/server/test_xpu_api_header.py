"""X-API response header (Task 3): server writes the resolved api back on both
the success branch (_send_run_ok) and the error branch (_send_json).

This prevents the client from guessing whether the server actually resolved the
api it was asked to run — it just reads the X-API header. Covers:
  - success: env has api -> X-API header written
  - backward-compat: env api=None -> no X-API header (old envelopes still work)
  - error: _send_json writes X-API from env (spec §4.2 C4)
"""
from unittest.mock import MagicMock

from ttk.remote.server.xpu_server import XpuRequestHandler


def _make_handler():
    """A bare XpuRequestHandler with mocked HTTP I/O (no socket / __init__)."""
    handler = XpuRequestHandler.__new__(XpuRequestHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()
    return handler


def test_send_run_ok_writes_x_api():
    handler = _make_handler()
    env = {"output_path": None, "output_count": 0, "shapes": [], "dtypes": [],
           "perf": None, "api": "torch.add"}
    handler._send_run_ok(env)
    calls = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
    assert calls.get("X-API") == "torch.add"


def test_send_run_ok_writes_x_output_schema_not_dtypes(tmp_path):
    """With a body, _send_run_ok writes X-Output-Schema (from env['schema']) and
    NO X-Output-Dtypes (dropped in favor of the schema header)."""
    import json
    body_file = tmp_path / "out.npz"
    body_file.write_bytes(b"\x00\x01")       # non-empty -> has_body=True
    handler = _make_handler()
    schema = [{"index": 0, "dtype": "float32"}]
    env = {"output_path": str(body_file), "output_count": 1,
           "shapes": [[2]], "schema": schema, "perf": None, "api": "torch.add"}
    handler._send_run_ok(env)
    calls = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
    assert calls.get("X-Output-Schema") == json.dumps(schema)
    assert "X-Output-Dtypes" not in calls    # dropped, replaced by X-Output-Schema


def test_send_run_ok_no_api_omits_header():
    handler = _make_handler()
    env = {"output_path": None, "output_count": 0, "shapes": [], "dtypes": [],
           "perf": None, "api": None}
    handler._send_run_ok(env)
    names = [c.args[0] for c in handler.send_header.call_args_list]
    assert "X-API" not in names   # api=None -> no X-API (old envelope compat)


def test_send_json_writes_x_api_on_error():
    """Error branch (_send_json) also returns X-API (spec §4.2 C4)."""
    handler = _make_handler()
    env = {"ok": False, "http_status": 500, "error": "boom", "api": "torch.add"}
    handler._send_json(500, {"error": "boom"}, env=env)
    calls = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
    assert calls.get("X-API") == "torch.add"


def test_early_validation_echoes_x_api():
    """Early-validation branches (bad tenant_id / bad schema / 503 gate busy)
    echo back the client's X-API header even though they fail before reading it
    into the kwargs envelope (spec §4.2 C4 — any non-crash scenario echoes back).

    Drives _handle_run via the bare handler; mocks body receipt (no socket).
    """
    handler = _make_handler()
    handler.data_gate = None           # bypass 503-gate path for tenant test
    handler.tenant_manager = MagicMock()
    # headers: bad tenant_id (empty) + X-API present
    headers = {"X-Tenant-ID": "", "X-API": "torch.add"}
    handler.headers = MagicMock()
    handler.headers.get = lambda name, default="": headers.get(name, default)

    handler._handle_run()

    calls = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
    assert calls.get("X-API") == "torch.add"    # echoed despite 400


def test_early_validation_x_api_falls_back_to_spec_class():
    """Legacy clients send X-Spec-Class instead of X-API; the early-validation
    echo falls back to it (req_api hoist covers both)."""
    handler = _make_handler()
    handler.data_gate = None
    handler.tenant_manager = MagicMock()
    headers = {"X-Tenant-ID": "../escape", "X-Spec-Class": "MySpec"}
    handler.headers = MagicMock()
    handler.headers.get = lambda name, default="": headers.get(name, default)

    handler._handle_run()

    calls = {c.args[0]: c.args[1] for c in handler.send_header.call_args_list}
    assert calls.get("X-API") == "MySpec"
