"""Task 4 — Feature 1 client: parse X-API header + api_label uses server response.

Covers:
* `_parse_api_header` (raw string, NOT json): string / None / empty.
* `RemoteResult.api` field (default None).
* collector success path prefers `r.api` (server response) over spec.api.
* collector failure path falls back to pre-computed `spec.api or "custom"`.
"""
from ttk.remote.dispatcher import _parse_api_header, RemoteResult


def test_parse_api_header_string():
    assert _parse_api_header("torch.add") == "torch.add"


def test_parse_api_header_none():
    assert _parse_api_header(None) is None


def test_parse_api_header_empty():
    assert _parse_api_header("") is None


def test_remote_result_has_api_field():
    r = RemoteResult(outputs=[], perf=None, api="torch.add")
    assert r.api == "torch.add"


def test_remote_result_api_default_none():
    r = RemoteResult(outputs=[])
    assert r.api is None


def test_collector_success_uses_server_api(monkeypatch):
    """成功：r.api（server 回传）优先——Feature 1 client 核心 acceptance"""
    from ttk.utilities.singleton import Singleton
    from ttk.remote.xpu_collector import collect_xpu_results
    from ttk.remote import ExecutionSpec, PERF
    from unittest.mock import MagicMock

    Singleton._instances.clear()
    monkeypatch.setattr("ttk.remote.endpoint_view.EndpointView", MagicMock())
    monkeypatch.setattr("ttk.remote.xpu_collector.dispatch_to_remote",
                        lambda **kw: RemoteResult(outputs=[], perf=None, api="torch.add"))
    spec = ExecutionSpec(provider="torch", type="api", api=None)
    results = collect_xpu_results([spec], inputs=[], input_names=[], mode=PERF,
                                   tenant_id="t", op_name="add")
    assert results["torch"]["api"] == "torch.add"   # server 回传，不是 spec.api(None)


def test_collector_fail_uses_spec_api(monkeypatch):
    """异常（dispatch 抛错）：fall back pre-computed spec.api"""
    from ttk.utilities.singleton import Singleton
    from ttk.remote.xpu_collector import collect_xpu_results
    from ttk.remote import ExecutionSpec, PERF
    from ttk.remote.dispatcher import RemoteExecutionError
    from unittest.mock import MagicMock

    Singleton._instances.clear()
    monkeypatch.setattr("ttk.remote.endpoint_view.EndpointView", MagicMock())

    def boom(**kw):
        raise RemoteExecutionError("conn error")
    monkeypatch.setattr("ttk.remote.xpu_collector.dispatch_to_remote", boom)
    spec = ExecutionSpec(provider="torch", type="api", api="torch.custom_api")
    results = collect_xpu_results([spec], inputs=[], input_names=[], mode=PERF,
                                   tenant_id="t", op_name="add")
    assert results["torch"]["api"] == "torch.custom_api"   # spec.api 兜底
    assert results["torch"]["status"] == "FAIL"
