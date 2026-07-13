"""UT for compare() 接口（comparison.py 核心分发逻辑）。"""
import numpy as np
from unittest.mock import MagicMock

from ttk.core_modules.comparison.comparison import compare, _filter_fake_fail


def _stds(n):
    return [MagicMock() for _ in range(n)]


class TestEmptyOutputs:
    def test_empty_returns_unknown(self):
        r = compare([], [], (), standards=[])
        assert r[0] == "UNKNOWN"
        assert r[2] is False


class TestOutputSentinels:
    def test_dyn_off_passes(self):
        r = compare(["DYN_OFF"], [np.array([1.0])], ("float32",), standards=_stds(1))
        assert "DYN_OFF" in r[0]
        assert r[2] is True

    def test_bin_off_passes(self):
        r = compare(["BIN_OFF"], [np.array([1.0])], ("float32",), standards=_stds(1))
        assert r[2] is True

    def test_unsupported_passes(self):
        r = compare(["DYN_UNSUPPORTED"], [np.array([1.0])], ("float32",), standards=_stds(1))
        assert r[2] is True

    def test_operator_not_found_passes(self):
        r = compare(["DYN_OPERATOR_NOT_FOUND"], [np.array([1.0])], ("float32",), standards=_stds(1))
        assert r[2] is True

    def test_none_output_fails(self):
        r = compare([None], [np.array([1.0])], ("float32",), standards=_stds(1))
        assert "NO_OUTPUT" in r[0]
        assert r[2] is False

    def test_non_fake_string_fails(self):
        r = compare(["SOMETHING_ELSE"], [np.array([1.0])], ("float32",), standards=_stds(1))
        assert r[2] is False


class TestGoldenSentinels:
    def test_none_golden_suppressed(self):
        r = compare([np.array([1.0])], [None], ("float32",), standards=_stds(1))
        assert "SUPPRESSED" in r[0]
        assert r[2] is True

    def test_string_golden_passes_if_fake(self):
        r = compare([np.array([1.0])], ["SUPPRESSED"], ("float32",), standards=_stds(1))
        assert r[2] is True


class TestMultiOutput:
    def test_mix_dyn_off_and_none_fails(self):
        r = compare(["DYN_OFF", None], [np.array([1.0]), np.array([2.0])],
                    ("float32", "float32"), standards=_stds(2))
        assert r[2] is False

    def test_all_dyn_off_passes(self):
        r = compare(["DYN_OFF", "DYN_OFF"], [np.array([1.0]), np.array([2.0])],
                    ("float32", "float32"), standards=_stds(2))
        assert r[2] is True


class TestThirdPartyCount:
    def test_dyn_off_skips_mismatch(self):
        r = compare(["DYN_OFF", "DYN_OFF"], [np.array([1.0]), np.array([2.0])],
                    ("float32", "float32"), standards=_stds(2),
                    third_parties=[np.array([1.0])])
        assert r[2] is True

    def test_real_outputs_less_third_parties_fails(self):
        r = compare([np.array([1.0]), np.array([2.0])],
                    [np.array([1.0]), np.array([2.0])],
                    ("float32", "float32"), standards=_stds(2),
                    third_parties=[np.array([1.0])])
        assert r[0] == "COMPARE_FAILURE"
        assert r[2] is False

    def test_none_third_parties_ok(self):
        r = compare(["DYN_OFF"], [np.array([1.0])],
                    ("float32",), standards=_stds(1))
        assert r[2] is True

    def test_empty_third_parties_ok(self):
        r = compare(["DYN_OFF", "DYN_OFF"], [np.array([1.0]), np.array([2.0])],
                    ("float32", "float32"), standards=_stds(2),
                    third_parties=[])
        assert r[2] is True


class TestFilterFakeFail:
    def test_all_fake_passes(self):
        for token in ("DYN_OFF", "CST_OFF", "BIN_OFF", "STC_OFF",
                      "DYN_UNSUPPORTED", "SUPPRESSED", "DYN_INPUT_MISSING"):
            assert _filter_fake_fail(token) is True

    def test_non_fake_fails(self):
        for token in ("PASS", "FAIL", "SOMETHING", ""):
            assert _filter_fake_fail(token) is False
