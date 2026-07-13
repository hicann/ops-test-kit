# Unit tests for ttk.core_modules.npu.op.profiling app-layer spec resolution.
# Covers _build_spec (Task 7) + _extract_spec_providers (Task 9). No server fixture.
from ttk.core_modules.npu.op import profiling as prof
from ttk.remote import ExecutionSpec


# ---- _build_spec (signature: provider, tp, spec_file, spec_class, op_name, op_type) ----

def test_build_spec_dict_api_string():
    spec = prof._build_spec("torch", {"torch": "torch.add", "tf": "tf.raw.ops.Add"},
                            spec_file=None, spec_class=None,
                            op_name="add", op_type="Add")
    assert spec.provider == "torch" and spec.type == "api" and spec.api == "torch.add"


def test_build_spec_no_spec_marks_server_derive():
    # no third_party -> api=None (server _resolve_3party_api derives)
    spec = prof._build_spec("torch", None,
                            spec_file=None, spec_class=None,
                            op_name="add", op_type="Add")
    assert spec.provider == "torch" and spec.api is None


def test_build_spec_dict_impl_class_marks_spec_mode():
    class _Dummy:
        pass

    spec = prof._build_spec("torch", {"torch": _Dummy},
                            spec_file="/tmp/s.py", spec_class="_SpecCls",
                            op_name="add", op_type="Add")
    assert spec.provider == "torch" and spec.type == "spec"
    assert spec.spec_file == "/tmp/s.py" and spec.spec_class == "_SpecCls"


def test_build_spec_str_uses_tp_as_api():
    spec = prof._build_spec("torch", "torch.add",
                            spec_file=None, spec_class=None,
                            op_name="add", op_type="Add")
    assert spec.provider == "torch" and spec.type == "api" and spec.api == "torch.add"


# ---- _extract_spec_providers (replaces _resolve_candidate_providers) ----

def test_extract_spec_providers_dict_order_is_priority():
    # third_party dict insertion order preserved = priority order
    tp = {"torch": "torch.add", "tf": "tf.raw.ops.Add"}
    assert prof._extract_spec_providers(tp) == ["torch", "tf"]
