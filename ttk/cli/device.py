#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


def add_device_args(parser):
    parser.add_argument("--dev", "--device", dest="device", type=int, help="Number of devices to use (default: all)")
    parser.add_argument(
        "--device-blacklist", dest="device_blacklist", help="Device blacklist, e.g. --device-blacklist=1,2,3"
    )
    parser.add_argument(
        "--device-whitelist", dest="device_whitelist", help="Device whitelist, e.g. --device-whitelist=0,1"
    )
    parser.add_argument("--pc", "--process-count", dest="process_count", type=int, help="Process count per device")
    parser.add_argument("--plat", "--platform", dest="platform", help="SoC version, e.g. --plat=Ascend910B2")
    parser.add_argument(
        "--proc-timeout",
        dest="proc_timeout",
        type=int,
        default=3600,
        help="Per-testcase timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--warmup", default=True, type=lambda x: x.lower() != "false", help="Warmup before profiling (default: true)"
    )
    parser.add_argument("--run", type=int, help="Execution count (default: 3)")
    parser.add_argument(
        "-l", "--limit", type=int, default=None, help="HBM memory limit per testcase in GB (default: 30)"
    )
    parser.add_argument(
        "--deterministic-level",
        "--dl",
        dest="deterministic_level",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Deterministic level: 0=off (default), 1=deterministic "
        "compute (MD5 consistent across NPU runs), "
        "2=strong consistency, "
        "3=deterministic execution; complete batch fields enable phase-two slice compare",
    )
