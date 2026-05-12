from ttk.cli.common import add_common_args
from ttk.cli.device import add_device_args
from ttk.cli.bridge import args_to_switches, apply_kernel_args, run_with_switches


def register_kernel_command(subparsers):
    parser = subparsers.add_parser(
        "kernel",
        help="Op mode: TBE/Davinci kernel compile + execute + compare"
    )
    add_common_args(parser)
    add_device_args(parser)
    _add_kernel_args(parser)
    parser.set_defaults(handler=_handle_kernel)


def _add_kernel_args(parser):
    parser.add_argument("-d", "--dynamic", nargs="?", const=True, default=None,
                        help="Enable dynamic shape test (default: enabled); use -d=false to disable")
    parser.add_argument("-s", "--static", nargs="?", const=True, default=None,
                        help="Enable static shape test; use -s=false to disable")
    parser.add_argument("-c", "--const", nargs="?", const=True, default=None,
                        help="Enable const shape test; use -c=false to disable")
    parser.add_argument("-b", "--binary", nargs="?", const=True, default=None,
                        help="Enable binary test; --binary=release for released kernel")
    parser.add_argument("--cce", nargs="?", const=True, default=None,
                        help="Compile CCE file, e.g. --cce or --cce=d,s")
    parser.add_argument("--co", "--compile-only", dest="compile_only", action="store_true",
                        help="Compile only, skip profiling")
    parser.add_argument("--no-prof", action="store_true", help="Disable profiling (same as --co)")
    parser.add_argument("--tr", "--tiling-run", dest="tiling_run", type=int, default=None,
                        help="Tiling function run times (default: 3)")

    parser.add_argument("--compile-opts", dest="compile_opts",
                        help="Kernel compile options, e.g. --compile-opts='-g,-O0,oom'")
    parser.add_argument("--impl-mode", dest="impl_mode",
                        help="Operator implement mode")
    parser.add_argument("--ct", "--core-type", dest="core_type",
                        help="Core type: VectorCore or AiCore")
    parser.add_argument("--npu-timeout", dest="npu_timeout", type=int, default=0,
                        help="NPU execution timeout in ms (default: 0)")
    parser.add_argument("--reuse-hbm", dest="reuse_hbm", action="store_true",
                        help="Reuse input/output HBM")
    parser.add_argument("--reserve-hbm", dest="reserve_hbm", type=int, default=None,
                        help="Reserve N MB of HBM")
    parser.add_argument("--clear-atomic", dest="clear_atomic", nargs="?", const=True,
                        help="Force clear atomic, e.g. --clear-atomic=d,s")
    parser.add_argument("--clear-ub", dest="clear_ub", default=None,
                        help="Clear UB to value, e.g. --clear-ub=0 or --clear-ub=1.0")
    parser.add_argument("--clear-l1", dest="clear_l1", default=None,
                        help="Clear L1 to value")

    parser.add_argument("--simt-ub", dest="simt_ub", default=None,
                        help="Force SIMT UB size in bytes")
    parser.add_argument("--simt-stack-dcu", dest="simt_stack_dcu", type=int, default=None,
                        help="SIMT DCU stack size in bytes")
    parser.add_argument("--simt-stack-dvg", dest="simt_stack_dvg", type=int, default=None,
                        help="SIMT DVG stack size in bytes")
    parser.add_argument("--force-block-dim", dest="force_block_dim", default=None,
                        help="Force block dim, e.g. --force-block-dim=2")


def _handle_kernel(args):
    sw = args_to_switches(args)
    sw.test_mode = "op"
    apply_kernel_args(sw, args)
    run_with_switches(sw)
