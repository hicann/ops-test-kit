# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
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
        "--no-prof", action="store_true", help="Prepare input and CPU golden data without running ACLNN"
    )
    parser.add_argument(
        "--xpu-perf",
        dest="xpu_perf",
        action="store_true",
        help="Collect 3rd-party (XPU) performance per case. "
        "Requires remote XPU config (ttk.conf.yaml or --config). PERF-only.",
    )
    parser.add_argument(
        "--xpu-zero-upload",
        dest="xpu_zero_upload",
        action="store_true",
        help="Send input generation recipes + per-tensor sha256 instead of the input data itself; "
        "the server regenerates and verifies each digest. Saves the upload over slow links. "
        "Falls back to full upload automatically when a digest does not match.",
    )


def _handle_aclnn(args):
    sw = args_to_switches(args)
    sw.test_mode = "aclnn"
    apply_aclnn_args(sw, args)
    apply_sim_args(sw, args)
    configure_manual_data(sw, args, "aclnn")
    validate_xpu_perf_precondition(sw)
    run_with_switches(sw)
