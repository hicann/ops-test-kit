from ttk.cli.bridge import (
    apply_aclnn_args,
    args_to_switches,
    configure_manual_data,
    run_with_switches,
)
from ttk.cli.common import add_common_args, validate_xpu_perf_precondition
from ttk.cli.device import add_device_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_aclnn_command(subparsers):
    parser = subparsers.add_parser("aclnn", help="ACLNN API mode: aclnn* C API execute + compare")
    add_common_args(parser)
    add_device_args(parser)
    _add_aclnn_args(parser)
    add_sim_args(parser)
    parser.set_defaults(handler=_handle_aclnn)


def _add_aclnn_args(parser):
    parser.add_argument(
        "--no-prof", action="store_true", help="Prepare input and optional CPU golden data without running ACLNN"
    )
    parser.add_argument(
        "--xpu-perf",
        dest="xpu_perf",
        action="store_true",
        help="Collect 3rd-party (XPU) performance per case. "
        "Requires remote XPU config (ttk.conf.yaml or --config). PERF-only.",
    )


def _handle_aclnn(args):
    sw = args_to_switches(args)
    sw.test_mode = "aclnn"
    apply_aclnn_args(sw, args)
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "aclnn")
    validate_xpu_perf_precondition(sw)
    run_with_switches(sw)
