#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ttk.cli.bridge import args_to_switches, run_with_switches
from ttk.cli.common import add_common_args, validate_xpu_perf_precondition
from ttk.cli.device import add_device_args


def register_geir_command(subparsers):
    parser = subparsers.add_parser("geir", help="GEIR mode: GE graph compile + execute + compare")
    add_common_args(parser)
    add_device_args(parser)
    _add_geir_args(parser)
    parser.set_defaults(handler=_handle_geir)


def _add_geir_args(parser):
    parser.add_argument(
        "-c",
        "--const",
        nargs="?",
        const="true",
        default=None,
        help="Enable const (static) shape test. Default: enabled. -c=false to disable.",
    )
    parser.add_argument(
        "-d",
        "--dynamic",
        nargs="?",
        const="true",
        default=None,
        help="Enable dynamic shape test. Default: disabled. -d or -d=true to enable.",
    )
    parser.add_argument(
        "-b",
        "--binary",
        nargs="?",
        const="release",
        default=None,
        help="Binary reuse: ge.jit_compile=0; use -b=release to reuse compiled kernel",
    )
    parser.add_argument(
        "--xpu-perf",
        dest="xpu_perf",
        action="store_true",
        help="Collect 3rd-party (XPU) performance per case. "
        "Requires remote XPU config (ttk.conf.yaml or --config). PERF-only.",
    )
    parser.add_argument(
        "--geir-serve",
        dest="geir_serve",
        action="store_true",
        help="Keep one long-lived GEIR executor process per device instead of spawning one per case, "
        "so GEInitialize (measured ~8.5s) runs once per batch. Falls back to one-shot on any error.",
    )
    parser.add_argument(
        "--xpu-zero-upload",
        dest="xpu_zero_upload",
        action="store_true",
        help="Send input generation recipes + per-tensor sha256 instead of the input data itself; "
        "the server regenerates and verifies each digest. Falls back to full upload on mismatch.",
    )


def _handle_geir(args):
    sw = args_to_switches(args)
    sw.test_mode = "geir"
    # Default: const on, dynamic off
    sw.cst_switches.enabled = True
    sw.dyn_switches.enabled = False
    if args.const is not None:
        sw.cst_switches.enabled = args.const.lower() not in ("false", "0", "no", "off")
    if args.dynamic is not None:
        sw.dyn_switches.enabled = args.dynamic.lower() not in ("false", "0", "no", "off")
    if args.binary is not None:
        val = args.binary
        if isinstance(val, str):
            val = val.lower() not in ("false", "0", "no", "off")
        sw.geir_binary = val
    else:
        sw.geir_binary = False
    validate_xpu_perf_precondition(sw)
    run_with_switches(sw)
