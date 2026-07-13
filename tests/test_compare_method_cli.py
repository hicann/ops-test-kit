import argparse
import pytest

from ttk.cli.common import add_common_args
from ttk.utilities.classes import SWITCHES


def _parse(argv):
    p = argparse.ArgumentParser()
    add_common_args(p)
    # add_common_args requires -i/--input; inject a dummy so --compare can be exercised.
    return p.parse_args(["-i", "dummy.csv"] + list(argv))


def test_default_compare_method_none():           # SWITCHES 单元默认
    assert SWITCHES().compare_method is None


def test_cli_no_flag_defaults_none():              # 不传 --compare → None
    assert _parse([]).compare is None


def test_cli_accepts_stat_rel_err():               # 新 token 过 choices
    assert _parse(["--compare", "stat_rel_err"]).compare == "stat_rel_err"


def test_cli_keeps_legacy_binary():                # 原 token 仍在
    assert _parse(["--compare", "binary"]).compare == "binary"


def test_cli_rejects_alias_not_in_choices():       # 别名 bitwise 不在 choices → argparse 拒
    p = argparse.ArgumentParser(); add_common_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--compare", "bitwise"])


def test_cli_accepts_cross_check():                # cross_check 过 choices（看护）
    assert _parse(["--compare", "cross_check"]).compare == "cross_check"
