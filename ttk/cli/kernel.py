# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from ttk.cli.bridge import (
    apply_kernel_args,
    args_to_switches,
    configure_manual_data,
    run_with_switches,
)
from ttk.cli.common import add_common_args, validate_xpu_perf_precondition
from ttk.cli.device import add_device_args
from ttk.cli.sim_args import add_sim_args, apply_sim_args


def register_kernel_command(subparsers):
    parser = subparsers.add_parser("kernel", help="Op mode: TBE/Davinci kernel compile + execute + compare")
    add_common_args(parser)
    add_device_args(parser)
    _add_kernel_args(parser)
    add_sim_args(parser)
    parser.set_defaults(handler=_handle_kernel)


def _add_kernel_args(parser):
    parser.add_argument(
        "-d",
        "--dynamic",
        nargs="?",
        const=True,
        default=None,
        help="Enable dynamic shape test (default: enabled); use -d=false to disable",
    )
    parser.add_argument(
        "-c", "--const", nargs="?", const=True, default=None, help="Enable const shape test; use -c=false to disable"
    )
    parser.add_argument(
        "-b",
        "--binary",
        nargs="?",
        const=True,
        default=None,
        help="Binary mode; use -b=release to reuse pre-compiled release kernel",
    )
    parser.add_argument(
        "--co", "--compile-only", dest="compile_only", action="store_true", help="Compile only, skip profiling"
    )
    parser.add_argument(
        "--no-prof",
        action="store_true",
        help="Generate input/golden but suppress dynamic, const, and binary kernel execution",
    )
    parser.add_argument(
        "--tr", "--tiling-run", dest="tiling_run", type=int, default=None, help="Tiling function run times (default: 3)"
    )

    parser.add_argument(
        "--compile-opts",
        dest="compile_opts",
        action="append",
        metavar="KEY=VALUE",
        help="Compile option as key=value. Can be specified multiple times. "
        "e.g. --compile-opts op_debug_config=oom,dump_cce "
        "--compile-opts enable_deterministic_mode=1",
    )
    parser.add_argument(
        "--npu-timeout", dest="npu_timeout", type=int, default=0, help="NPU execution timeout in ms (default: 0)"
    )
    parser.add_argument("--reuse-hbm", dest="reuse_hbm", action="store_true", help="Reuse input/output HBM")
    parser.add_argument("--reserve-hbm", dest="reserve_hbm", type=int, default=None, help="Reserve N MB of HBM")
    parser.add_argument(
        "--clear-atomic",
        dest="clear_atomic",
        nargs="?",
        const=True,
        help="Force clear output and workspace memory before execution",
    )
    parser.add_argument(
        "--clear-ub", dest="clear_ub", default=None, help="Clear UB to specified value before execution (default: 0)"
    )
    parser.add_argument(
        "--clear-l1", dest="clear_l1", default=None, help="Clear L1 to specified value before execution (default: 0)"
    )
    parser.add_argument(
        "--clear-l0",
        dest="clear_l0",
        default=None,
        help="Clear L0A/L0B/L0C to specified value before execution (default: 0). "
        "L0A/L0B are filled with the value; L0C is the matmul result (0 when value is 0).",
    )

    parser.add_argument("--simt-ub", dest="simt_ub", default=None, help="Force SIMT UB size in bytes")
    parser.add_argument(
        "--simt-stack-dcu", dest="simt_stack_dcu", type=int, default=None, help="SIMT DCU stack size in bytes"
    )
    parser.add_argument(
        "--simt-stack-dvg", dest="simt_stack_dvg", type=int, default=None, help="SIMT DVG stack size in bytes"
    )
    parser.add_argument(
        "--force-block-dim", dest="force_block_dim", default=None, help="Force block dim, e.g. --force-block-dim=2"
    )
    parser.add_argument(
        "--xpu-perf",
        dest="xpu_perf",
        action="store_true",
        help="Collect 3rd-party (XPU) performance per case. "
        "Requires remote XPU config (ttk.conf.yaml or --config). PERF-only.",
    )


def _handle_kernel(args):
    sw = args_to_switches(args)
    sw.test_mode = "op"
    apply_kernel_args(sw, args)
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "kernel")

    validate_xpu_perf_precondition(sw)
    run_with_switches(sw)
