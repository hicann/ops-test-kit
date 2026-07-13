"""make_execution_specs removed — ExecutionSpec is now built app-side via profiling._build_spec."""


def test_make_execution_specs_removed():
    import ttk.remote as r
    assert not hasattr(r, "make_execution_specs")
    assert not hasattr(r, "_infer_default_api")
    # _derive_provider_from_api + ExecutionSpec + DATA/PERF retained
    assert hasattr(r, "_derive_provider_from_api")
    assert hasattr(r, "ExecutionSpec")
    assert hasattr(r, "DATA") and hasattr(r, "PERF")


def test_execution_spec_construction():
    from ttk.remote import ExecutionSpec
    s = ExecutionSpec(provider="torch", type="api", api="torch.add")
    assert s.provider == "torch" and s.api == "torch.add"


# ---- _derive_provider_from_api: direct unit tests ----

from ttk.remote import _derive_provider_from_api  # noqa: E402

_FB = "torch"  # arbitrary fallback used across cases


def test_derive_known_prefixes():
    assert _derive_provider_from_api("torch.add", _FB) == "torch"
    assert _derive_provider_from_api("tf.raw_ops.Add", _FB) == "tf"
    assert _derive_provider_from_api("tensorflow.raw_ops.Add", _FB) == "tf"  # normalize
    assert _derive_provider_from_api("numpy.something", _FB) == "numpy"
    assert _derive_provider_from_api("np.add", _FB) == "numpy"  # alias normalize


def test_derive_unknown_prefix_passthrough():  # the fix
    assert _derive_provider_from_api("mindspore.ops.Add", _FB) == "mindspore"
    assert _derive_provider_from_api("jax.nn.linear", _FB) == "jax"
    assert _derive_provider_from_api("my_module.foo", _FB) == "my_module"
    assert _derive_provider_from_api("torhc.add", _FB) == "torhc"  # typo, not silently torch


def test_derive_fallback_when_no_prefix():
    assert _derive_provider_from_api("my_custom_op", _FB) == _FB  # no dot
    assert _derive_provider_from_api("", _FB) == _FB
    assert _derive_provider_from_api(None, _FB) == _FB
