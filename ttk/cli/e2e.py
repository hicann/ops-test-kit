from ttk.cli.common import add_common_args
from ttk.cli.device import add_device_args
from ttk.cli.bridge import args_to_switches, apply_e2e_args, run_with_switches


def register_e2e_command(subparsers):
    parser = subparsers.add_parser(
        "e2e",
        help="Framework API mode: torch_npu/tf end-to-end test"
    )
    add_common_args(parser)
    add_device_args(parser)
    _add_e2e_args(parser)
    parser.set_defaults(handler=_handle_e2e)


def _add_e2e_args(parser):
    parser.add_argument("--backend",
                        help="Hardware backend: npu, gpu, cpu (auto-detect if not specified)")
    parser.add_argument("--validate", dest="validate_only", action="store_true",
                        help="Validate CSV cases only, skip device execution")


def _handle_e2e(args):
    sw = args_to_switches(args)
    sw.test_mode = "framework-api"
    apply_e2e_args(sw, args)
    run_with_switches(sw)
