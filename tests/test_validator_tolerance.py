import pytest
from ttk.test_spec import InvalidSpecError, validate


def _spec(**kw):
    class S: pass
    for k, v in kw.items():
        setattr(S, k, v)
    return S


@pytest.mark.parametrize("tok", ["stat_rel_err", "binary_equal", "cross_check", "quant"])
def test_2_1_standard_accepted(tok):
    validate(_spec(tolerance={"float32": {"standard": tok}}))


@pytest.mark.parametrize("tok", ["isclose", "close", "cosine", "bin", "binary", "requant"])
def test_framework_token_rejected_with_cli_hint(tok):
    with pytest.raises(InvalidSpecError) as ei:
        validate(_spec(tolerance={"float32": {"standard": tok}}))
    assert "--compare" in str(ei.value)


def test_legacy_camelcase_rejected_as_unknown():
    # 不维护 legacy 映射；CamelCase 走 generic unknown
    with pytest.raises(InvalidSpecError) as ei:
        validate(_spec(tolerance={"float32": {"standard": "BinaryCompareStandard"}}))
    assert "expected one of" in str(ei.value)


def test_unknown_token_rejected():
    with pytest.raises(InvalidSpecError):
        validate(_spec(tolerance={"float32": {"standard": "nope"}}))


def test_non_dict_cfg_rejected():
    with pytest.raises(InvalidSpecError):
        validate(_spec(tolerance={"float32": "stat_rel_err"}))


def test_non_dict_tolerance_rejected():
    with pytest.raises(InvalidSpecError):
        validate(_spec(tolerance=[("float32",)]))
