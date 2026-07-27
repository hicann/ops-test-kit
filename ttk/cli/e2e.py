from ttk.cli.bridge import (
    apply_e2e_args,
    args_to_switches,
    configure_manual_data,
    run_with_switches,
)
from ttk.cli.common import add_common_args
from ttk.cli.device import add_device_args


def register_e2e_command(subparsers):
    parser = subparsers.add_parser(
        "e2e",
        help="Framework API mode: torch_npu end-to-end test"
    )
    add_common_args(parser)
    add_device_args(parser)
    _add_e2e_args(parser)
    parser.set_defaults(handler=_handle_e2e)


def _add_e2e_args(parser):
    parser.add_argument("--no-prof", action="store_true",
                        help="Prepare input and CPU golden data without running the main API")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU backend")
    parser.add_argument("-d", "--dynamic", nargs="?", const=True, default=None,
                        help="Enable dynamic shape graph test (default: disabled); use -d=false to disable")
    parser.add_argument("-c", "--const", nargs="?", const=True, default=None,
                        help="Enable static shape graph test (default: disabled); use -c=false to disable")
    parser.add_argument("--aclgraph", action="store_true", default=False,
                        help="Enable aclgraph mode (reduce-overhead) via torch.compile")
    parser.add_argument("--fullgraph", dest="fullgraph", default=0, type=int,
                        help="Capture full graph in torch.compile (0=off, 1=on; default: 0)")


def _handle_e2e(args):
    sw = args_to_switches(args)
    sw.test_mode = "framework-api"
    sw.dyn_switches.enabled = False
    apply_e2e_args(sw, args)
    configure_manual_data(sw, args, "e2e")
    run_with_switches(sw)
