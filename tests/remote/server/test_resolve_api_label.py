from ttk.remote.server.executor import _resolve_3party_api, _aclnn_resolve


def test_resolve_3party_returns_name():
    """_resolve_3party_api 返回 (callable, name)，name=命中搜索入参（非 __name__）。"""
    f, name = _resolve_3party_api("add", "Add", "torch")
    assert name == "add"            # 命中 op_name，不是 __name__
    assert callable(f)


def test_resolve_3party_op_type_fallback():
    """op_name miss → op_type 命中，name=op_type"""
    f, name = _resolve_3party_api("nonexistent_op", "add", "torch")
    assert name == "add"


def test_resolve_3party_miss_returns_none_tuple():
    f, name = _resolve_3party_api("zzz_no_such", "ZzzNoSuch", "torch")
    assert f is None and name is None


def test_aclnn_returns_op_name(monkeypatch):
    """aclnn name = 请求原 op_name（aclnnXxx），非 strip+snake 后的内部 key"""
    import ttk.remote.server.executor as ex
    # mock _auto_import_from_torch 避免真 torch 搜索（monkeypatch auto-undoes
    # at teardown so it doesn't leak into other test files' _aclnn_resolve calls）
    monkeypatch.setattr(ex, "_auto_import_from_torch",
                        lambda s: (lambda *a, **k: None))
    f, name = _aclnn_resolve("aclnnAdd")
    assert name == "add"            # strip+snake 后的实际搜索 key（X-API 回传 torch.add）
    assert callable(f)


def test_ok_envelope_carries_api():
    from ttk.remote.server.executor import _ok
    env = _ok("/tmp/out.npz", 1, [[2, 3]], ["float32"], perf={"device_us": 1000.0}, api="torch.add")
    assert env["api"] == "torch.add"


def test_err_envelope_carries_api():
    from ttk.remote.server.executor import _err
    env = _err(400, "bad", api="torch.add")
    assert env["api"] == "torch.add"
