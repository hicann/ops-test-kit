"""third_party style: None — spec exists but no third_party.
XPU falls back to _resolve_3party_api(op_name, op_type)."""
import numpy


class AddNoneSpec:
    golden = "torch.add"
    third_party = None


__test_spec__ = {"add": AddNoneSpec}
