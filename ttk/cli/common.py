#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from ttk.remote import is_remote_configured


def _add_io_args(parser):
    parser.add_argument("-i", "--input", required=True, help="Test case file (csv/xlsx)")
    parser.add_argument(
        "--sheet", default=None, help="Excel worksheet name (default: first worksheet); ignored for csv"
    )
    parser.add_argument(
        "--config", default=None, help="Path to ttk config YAML (overrides ~/.config/ttk/ and ./ttk.conf.yaml)"
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-o", "--output", help="Output CSV file (overwrite existing)")
    output_group.add_argument(
        "-a",
        "--append",
        dest="append_file",
        help="Append results to existing CSV file; overwrites if file header does not match",
    )


def _add_case_filter_args(parser):
    parser.add_argument("-t", "--testcase", help="Specify testcase name(s), comma-separated")
    parser.add_argument(
        "--ti", "--testcase-index", dest="testcase_index", help="Specify testcase indexes, e.g. --ti=1,3,6 or --ti=1-5"
    )
    parser.add_argument("--tc", "--testcase-count", type=int, dest="testcase_count", help="Randomly pick N test cases")
    parser.add_argument("--priority", help="Filter by priority, e.g. --priority=1,3 or --priority=1-3")
    parser.add_argument("--op", "--operator", dest="operator", help="Filter by operator name(s)")
    parser.add_argument("--no-op", "--exclude-operator", dest="exclude_operator", help="Exclude operator name(s)")
    parser.add_argument("--seed", "--random-seed", type=int, dest="random_seed", help="Random seed")
    parser.add_argument(
        "--input-dist",
        dest="input_dist",
        choices=["uniform", "normal"],
        default="uniform",
        help="Input data distribution (default: uniform)",
    )


def _add_precision_args(parser):
    parser.add_argument(
        "--compare",
        default=None,
        choices=["close", "stat_rel_err", "mixed", "cosine", "binary", "requant", "cross_check"],
        help="Comparison method. Default: per-output routing via Spec.tolerance "
        "(requires --plugin), else mixed (mix_tolerance).",
    )
    parser.add_argument(
        "--golden-mode",
        dest="golden_mode",
        default="Enable",
        choices=["Enable", "Disable", "Promote"],
        help="Golden generation mode (default: Enable)",
    )
    parser.add_argument("--dump", nargs="?", const="full", help="Dump data: full, in, out, golden (comma-separated)")
    parser.add_argument(
        "--dump-format",
        dest="dump_format",
        default="bin",
        choices=["bin", "npy", "pt", "print"],
        help="Dump file format (default: bin)",
    )
    parser.add_argument(
        "--dump-on-fail", dest="dump_on_fail", action="store_true", help="Dump all data when precision check fails"
    )
    parser.add_argument(
        "--manual-data-dirs",
        dest="manual_data_dirs",
        nargs="+",
        help="Prepared input/golden data directories. Without --no-prof, "
        "load matching testcase data and run device comparison",
    )


def _add_run_args(parser):
    parser.add_argument("--plugin", help="External plugin path for customized golden/inputs")
    parser.add_argument("--rerun", help="Rerun failed cases, e.g. --rerun=precision_status")
    parser.add_argument(
        "--validate", dest="validate_only", action="store_true", help="Validate CSV cases only, skip device execution"
    )
    parser.add_argument(
        "--proc-no-reuse", dest="proc_no_reuse", action="store_true", help="Create new process for each case"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Which providers to TEST (e.g. torch,tf). "
        "Test filter — narrows dispatch, does NOT override remote config. "
        "If not set, uses the first available key from spec's third_party.",
    )


def _add_output_args(parser):
    parser.add_argument(
        "--title", "--titles", dest="title", help="Custom output columns, e.g. --title=testcase_name,dyn_perf_us"
    )
    parser.add_argument(
        "--csv-preserve", dest="csv_preserve", action="store_true", help="Preserve original CSV headers"
    )
    parser.add_argument(
        "--single-log", dest="single_log", action="store_true", help="Each test case gets its own log file"
    )
    parser.add_argument(
        "--print",
        dest="summary_print",
        default=True,
        type=lambda x: x.lower() != "false",
        help="Print summary info periodically (default: true)",
    )
    parser.add_argument("--no-memory-check", dest="no_memory_check", action="store_true", help="Skip host memory check")
    parser.add_argument(
        "--task-prof",
        dest="task_prof",
        default=True,
        type=lambda x: x.lower() != "false",
        help="Task-level profiling (msprof) switch (default: true)",
    )
    parser.add_argument("--po", "--progress-output", dest="progress_output", help="Progress output file")


def add_common_args(parser):
    _add_io_args(parser)
    _add_case_filter_args(parser)
    _add_precision_args(parser)
    _add_run_args(parser)
    _add_output_args(parser)


def validate_xpu_perf_precondition(sw):
    """--xpu-perf requires remote XPU config (ttk.conf.yaml or --config)."""
    if sw.xpu_perf and not is_remote_configured():
        raise RuntimeError("--xpu-perf requires remote XPU config (ttk.conf.yaml or --config), but none is configured.")
