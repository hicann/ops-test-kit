import argparse
import os

from ttk.cli.aclnn import register_aclnn_command
from ttk.cli.e2e import register_e2e_command
from ttk.cli.geir import register_geir_command
from ttk.cli.info import register_info_command
from ttk.cli.kernel import register_kernel_command
from ttk.cli.list_cmd import register_list_command


def main():
    os.environ["TTK_PARENT_PID"] = str(os.getpid())

    from ttk._env import setup_env

    setup_env()

    try:
        _cli_main()
    finally:
        try:
            from ttk.core_modules.tbe_multiprocessing import SimpleCommandProcess

            for proc in SimpleCommandProcess.all_processes:
                proc.close()
        except Exception:
            pass


def _print_version():
    from ttk._version import __version__, get_build

    print(f"TTK v{__version__}, build {get_build()}")


def _version_callback(value):
    if value:
        _print_version()
        raise SystemExit(0)


def _cli_main():
    parser = argparse.ArgumentParser(prog="ttk", description="TTK — Ascend NPU Single Operator Test Framework")
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version and build info",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    register_kernel_command(subparsers)
    register_aclnn_command(subparsers)
    register_e2e_command(subparsers)
    register_geir_command(subparsers)
    register_info_command(subparsers)
    register_list_command(subparsers)

    args = parser.parse_args()

    if args.version:
        _print_version()
        return

    if not args.command:
        parser.print_help()
        return

    args.handler(args)
