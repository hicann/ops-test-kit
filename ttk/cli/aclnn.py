from ttk.cli.common import add_common_args
from ttk.cli.device import add_device_args
from ttk.cli.bridge import args_to_switches, apply_aclnn_args, run_with_switches


def register_aclnn_command(subparsers):
    parser = subparsers.add_parser(
        "aclnn",
        help="ACLNN API mode: aclnn* C API execute + compare"
    )
    add_common_args(parser)
    add_device_args(parser)
    _add_aclnn_args(parser)
    parser.set_defaults(handler=_handle_aclnn)


def _add_aclnn_args(parser):
    pass


def _handle_aclnn(args):
    sw = args_to_switches(args)
    sw.test_mode = "aclnn"
    apply_aclnn_args(sw, args)
    run_with_switches(sw)
