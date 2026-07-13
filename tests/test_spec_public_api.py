import textwrap
import pytest
import ttk.test_spec as ts
from ttk.test_spec import get_spec_attr, get_spec_class_meta


def test_test_spec_manager_not_exposed():
    assert hasattr(ts, "get_spec_attr") and hasattr(ts, "get_spec_class_meta")
    assert not hasattr(ts, "TestSpecManager")   # 公开面不再暴露


@pytest.mark.parametrize("paths", [None, ()])
def test_get_spec_class_meta_none_when_no_plugin(paths):
    assert get_spec_class_meta("any_op", paths) is None


def test_get_spec_class_meta_returns_dict(tmp_path):
    spec_file = tmp_path / "myop.py"
    spec_file.write_text(textwrap.dedent('''
        class MyopTestSpec:
            third_party = {"torch": "torch.abs"}
    '''), encoding="utf-8")
    meta = get_spec_class_meta("myop", str(tmp_path))
    assert meta is not None
    assert meta["class_name"] == "MyopTestSpec"
    assert meta["spec_file"] is not None
