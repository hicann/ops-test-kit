import argparse
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.cli.aclnn import register_aclnn_command
from ttk.cli.bridge import (
    _log_manual_data_configuration,
    _parse_clean_val,
    configure_manual_data,
)
from ttk.cli.e2e import register_e2e_command
from ttk.cli.kernel import register_kernel_command
from ttk.utilities.classes import SWITCHES


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_e2e_command(subparsers)
    register_aclnn_command(subparsers)
    register_kernel_command(subparsers)
    return parser


def _args(**overrides):
    values = {
        "manual_data_dirs": None,
        "no_prof": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepare_switches(file_format="bin"):
    switches = SWITCHES()
    switches.dyn_switches.enabled = False
    switches.dump_config.enable_input()
    switches.dump_config.enable_golden()
    switches.dump_config.file_format = file_format
    return switches


def test_two_stage_arguments_are_exposed_on_supported_commands():
    parser = _parser()
    e2e = parser.parse_args([
        "e2e", "-i", "case.csv", "--no-prof", "--dump", "in,golden",
        "--manual-data-dirs", "prepared",
    ])
    aclnn = parser.parse_args([
        "aclnn", "-i", "case.csv", "--manual-data-dirs", "first", "second",
    ])
    kernel = parser.parse_args([
        "kernel", "-i", "case.csv", "--manual-data-dirs", "prepared",
    ])

    assert e2e.no_prof is True
    assert e2e.manual_data_dirs == ["prepared"]
    assert aclnn.manual_data_dirs == ["first", "second"]
    assert kernel.manual_data_dirs == ["prepared"]


def test_prepare_defaults_to_first_plugin_manual_data_dir(tmp_path):
    switches = _prepare_switches()
    switches.plugin_path = (tmp_path / "assets",)

    configure_manual_data(switches, _args(no_prof=True), "e2e")

    assert switches.manual_data_mode == "prepare"
    assert switches.manual_data_dirs == (str((tmp_path / "assets" / "manual_data").resolve()),)


def test_prepare_without_plugin_defaults_to_working_directory(tmp_path, monkeypatch, caplog):
    switches = _prepare_switches()
    monkeypatch.chdir(tmp_path)

    configure_manual_data(switches, _args(no_prof=True), "e2e")
    with caplog.at_level(logging.INFO):
        _log_manual_data_configuration(switches)

    assert switches.manual_data_dirs == (str((tmp_path / "manual_data").resolve()),)
    assert "using current-directory manual-data output" in caplog.text


def test_prepare_with_multiple_plugins_requires_explicit_output(tmp_path):
    switches = _prepare_switches()
    switches.plugin_path = (tmp_path / "first", tmp_path / "second")

    with pytest.raises(ValueError, match="multiple --plugin paths"):
        configure_manual_data(switches, _args(no_prof=True), "e2e")


def test_prepare_accepts_explicit_output_directory(tmp_path):
    switches = _prepare_switches("pt")
    switches.plugin_path = (tmp_path / "first", tmp_path / "second")

    configure_manual_data(
        switches,
        _args(no_prof=True, manual_data_dirs=[str(tmp_path / "data")]),
        "aclnn",
    )

    assert switches.manual_data_mode == "prepare"
    assert switches.manual_data_dirs == (str((tmp_path / "data").resolve()),)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda sw: setattr(sw.dump_config, "mode", 0), "exactly --dump in,golden"),
        (lambda sw: sw.dump_config.enable_output(), "exactly --dump in,golden"),
        (lambda sw: setattr(sw.dump_config, "file_format", "print"), "not restorable"),
        (lambda sw: setattr(sw.dump_config, "dump_on_fail", True), "--dump-on-fail"),
        (lambda sw: setattr(sw, "golden_mode", "Disable"), "CPU golden"),
        (lambda sw: setattr(sw, "validate_only", True), "--validate"),
    ],
)
def test_prepare_rejects_incompatible_combinations(mutate, message):
    switches = _prepare_switches()
    mutate(switches)

    with pytest.raises(ValueError, match=message):
        configure_manual_data(switches, _args(no_prof=True), "e2e")


def test_replay_uses_ordered_search_directories(tmp_path):
    switches = SWITCHES()
    directories = [str(tmp_path / "one"), str(tmp_path / "two")]

    configure_manual_data(
        switches, _args(manual_data_dirs=directories), "aclnn"
    )

    assert switches.manual_data_mode == "replay"
    assert switches.manual_data_dirs == tuple(str((tmp_path / name).resolve())
                                                for name in ("one", "two"))


def test_e2e_replay_rejects_cpu_backend(tmp_path):
    switches = SWITCHES()
    switches.force_cpu = True

    with pytest.raises(ValueError, match="device stage"):
        configure_manual_data(
            switches, _args(manual_data_dirs=[str(tmp_path)]), "e2e"
        )


def test_kernel_no_prof_without_prepare_dump_keeps_legacy_semantics():
    switches = SWITCHES()

    configure_manual_data(switches, _args(no_prof=True), "kernel")

    assert switches.manual_data_mode is None
    assert switches.manual_data_dirs == ()


def test_kernel_prepare_uses_exact_dump_pair(tmp_path):
    switches = _prepare_switches("npy")

    configure_manual_data(
        switches,
        _args(no_prof=True, manual_data_dirs=[str(tmp_path)]),
        "kernel",
    )

    assert switches.manual_data_mode == "prepare"
    assert switches.manual_data_dirs == (str(tmp_path.resolve()),)


def test_kernel_replay_accepts_ordered_directories(tmp_path):
    switches = SWITCHES()
    directories = [str(tmp_path / "bin"), str(tmp_path / "fallback")]

    configure_manual_data(
        switches,
        _args(manual_data_dirs=directories),
        "kernel",
    )

    assert switches.manual_data_mode == "replay"
    assert switches.manual_data_dirs == tuple(
        str(path.resolve()) for path in (tmp_path / "bin", tmp_path / "fallback")
    )


def test_kernel_prepare_directory_without_dump_pair_is_rejected(tmp_path):
    switches = SWITCHES()

    with pytest.raises(ValueError, match="exactly --no-prof --dump in,golden"):
        configure_manual_data(
            switches,
            _args(no_prof=True, manual_data_dirs=[str(tmp_path)]),
            "kernel",
        )


@pytest.mark.parametrize("mode", ["prepare", "replay"])
def test_kernel_manual_data_rejects_compile_only(tmp_path, mode):
    switches = _prepare_switches() if mode == "prepare" else SWITCHES()
    switches.compile_only = True
    args = _args(
        no_prof=mode == "prepare",
        manual_data_dirs=[str(tmp_path)],
    )

    with pytest.raises(ValueError, match="compile-only"):
        configure_manual_data(switches, args, "kernel")


def test_manual_data_switches_survive_worker_pickle(tmp_path):
    switches = SWITCHES()
    switches.manual_data_mode = "replay"
    switches.manual_data_dirs = (str(tmp_path),)

    restored = pickle.loads(pickle.dumps(switches))

    assert restored.manual_data_mode == "replay"
    assert restored.manual_data_dirs == (str(tmp_path),)


@pytest.mark.parametrize(
    "value, expected_type, expected",
    [
        ("7", np.int32, 7),
        ("0xff", np.int32, 255),
        ("-1.25", np.float32, -1.25),
        ("float16(1.5)", np.float16, 1.5),
        ("uint8(0xff)", np.uint8, 255),
    ],
)
def test_clear_value_parser_accepts_numeric_literals(value, expected_type, expected):
    parsed = _parse_clean_val("UB", value)

    assert isinstance(parsed, expected_type)
    assert parsed == expected


def test_clear_value_parser_accepts_special_float_values():
    assert np.isinf(_parse_clean_val("L1", "float32(inf)"))
    assert np.isnan(_parse_clean_val("L1", "nan"))


def test_clear_value_parser_rejects_code_like_input():
    with pytest.raises(ValueError, match="Cannot parse UB clean value"):
        _parse_clean_val("UB", "float32(__import__('os').getcwd())")


def test_clear_value_parser_rejects_non_numeric_dtype():
    with pytest.raises(ValueError, match="Unsupported L1 clean value dtype"):
        _parse_clean_val("L1", "object(1)")
