"""DATA/PERF mode-bitmask constants + helpers (spec §4.1)."""
import pytest


def test_mode_constants_are_bits():
    from ttk.remote import DATA, PERF
    assert DATA == 0b01
    assert PERF == 0b10
    assert DATA & PERF == 0  # orthogonal


def test_none_and_combos():
    from ttk.remote import DATA, PERF
    NONE = 0b00
    assert NONE == DATA & NONE == 0  # NONE has neither DATA nor PERF
    both = DATA | PERF
    assert both == 0b11


def test_has_data_has_perf():
    from ttk.remote import has_data, has_perf, DATA, PERF
    assert has_data(DATA) is True and has_perf(DATA) is False
    assert has_data(PERF) is False and has_perf(PERF) is True
    assert has_data(DATA | PERF) is True and has_perf(DATA | PERF) is True
    assert has_data(0) is False and has_perf(0) is False
