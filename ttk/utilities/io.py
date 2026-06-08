#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Precious Utility Functions
"""
# Standard Packages
import sys
import logging
import pathlib
from functools import wraps
from typing import Sequence, Dict, Any

# Third-party Packages
import ttk
from .classes import SWITCHES

param_map = {}
param_help = {}


def __add_option(param_names: Sequence[str], wrap_func, param_help_message: str):
    if param_names[0].startswith("--"):
        param_help["    " + ", ".join(param_names)] = param_help_message
    else:
        param_help[", ".join(param_names)] = param_help_message

    for param_name in param_names:
        if param_name in param_map:
            logging.warning("param function of %s has already been registered!" % param_name)
        param_map[param_name] = wrap_func


def register_param(param_names: Sequence[str], param_help_message: str = None,
                   activate: bool = True):
    """Register command options parse function"""

    def __inner_param_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if activate:
            __add_option(param_names, __wrapper, param_help_message)
        return __wrapper

    return __inner_param_registry


def parse_params(switches: SWITCHES, params: Sequence):
    """
    :param switches:
    :param params:
    :return:
    """
    param_invalid = False
    for param in params:
        param = param.strip()
        main_param: str = param.split("=")[0]
        main_param: str = main_param.lower()
        if main_param in param_map:
            secondary_param = None
            if "=" in param:
                secondary_param = "=".join(param.split("=")[1:])
            param_map[main_param](switches, secondary_param)
        elif not param.startswith("-"):
            if switches.input_files is None:
                switches.input_files = param.split(',') if param is not None else []
            elif switches.output_file_name is None:
                switches.output_file_name = param
            else:
                logging.error(f"Invalid param: {param}")
                param_invalid = True
        else:
            logging.error(f"Invalid param: {param}")
            param_invalid = True
    if param_invalid:
        sys.exit(1)


def parse_complex_options(secondary_param: str) -> Dict[str, Any]:
    result = {}
    sections = secondary_param.split(';')
    for section in sections:
        if not section.strip():
            continue
        if '=' in section:
            k, v = section.split('=', 1)
            k = k.strip()
            v = v.strip()
            if ',' in v:
                result[k] = [x.strip() for x in v.split(',')]
            else:
                result[k] = v
        else:
            result[section] = True
    return result


def print_help():
    print("Welcome to use ttk v%s" % ttk.utilities.VERSION)
    print("Usage: ./run.sh case_file [result_file] [option(s)]")
    print(" The options are:")
    for param in param_help:
        help_msg = str(param_help[param]).split('\n')
        first_line = f"  {param.ljust(28)} : {help_msg[0]}"
        colon = first_line.index(':')
        print(f"  {param.ljust(28)} : {help_msg[0]}")
        for i in range(1, len(help_msg)):
            print(" " * (colon + 2) + help_msg[i])


@register_param(["-d", "--dynamic"], "Enable or disable dynamic shape test. Default is enabled.\n"
                                     "-d=false to disable. -d or -d=true to enable.")
def __set_dynamic(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.dyn_switches.enabled = True
    elif secondary_param.lower() == "false":
        switches.dyn_switches.enabled = False
    else:
        raise RuntimeError("Invalid dynamic shape mode: %s" % secondary_param)



@register_param(["-c", "--const"], "Enable or disable const shape test. Default is disabled. Usage same as -d")
def __set_const(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        switches.cst_switches.enabled = True
    elif secondary_param.lower() == "false":
        switches.cst_switches.enabled = False
    else:
        raise RuntimeError("Invalid const shape mode: %s" % secondary_param)


@register_param(["-b", "--binary"], "Enable binary test with released kernel.\n"
                                    "-b=release to match the released kernel installed skipping online compile.\n"
                                    "Note: -b or -b=true (online compile) is not supported currently.")
def __set_binary(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() == "true":
        raise RuntimeError("Binary online compile (-b or -b=true) is not supported. Use -b=release for released kernel.")
    elif secondary_param.lower() == "false":
        switches.bin_switches.enabled = False
    elif secondary_param.lower() == "release":
        switches.bin_switches.enabled = True
        switches.bin_switches.realtime = "release"
    else:
        raise RuntimeError("Invalid binary mode: %s" % secondary_param)


@register_param(["-l", "--limit"], "Limit the maximum HBM memory per testcase. Unit: GB. Default is 30.")
def __set_hbm_size_limit(switches: SWITCHES, secondary_param: str):
    switches.DAVINCI_HBM_SIZE_LIMIT = int(secondary_param)


@register_param(["-t", "--testcase"], "Specify testcase name to test separated by commas")
def __set_testcase(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify name for testcase")
    else:
        switches.selected_testcases = secondary_param.split(",")


@register_param(["--ti", "--testcase-index"], "Specify index of cases to test, separated by commas or dash.\n"
                                              "Example: --ti=1,3,6 means cases with index 1,3,6 (count from 0)\n"
                                              "--ti=1-5 means cases with index 1,2,3,4 except 5")
def __set_testcase_indexes(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify indexes for testcases")
    else:
        selected_indexes = []
        indexes = secondary_param.split(",")
        for _i in indexes:
            if '-' in _i:
                lower = int(_i.split('-')[0])
                higher = int(_i.split('-')[1])
                selected_indexes += list(range(lower, higher))
            else:
                selected_indexes.append(int(_i))
        switches.selected_testcase_indexes = tuple(selected_indexes)


@register_param(["--tc", "--testcase-count"], "Specify maximum case count to test. Cases will pick randomly")
def __set_testcase_count(switches: SWITCHES, secondary_param: str):
    if secondary_param is None:
        raise RuntimeError("Please specify testcase count")
    else:
        switches.selected_testcase_count = int(secondary_param)


@register_param(["--priority"], "Specifies the priority of testcases to run.")
def __set_testcase_priority(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or not secondary_param.strip():
        raise RuntimeError("Please specify a value for testcase priority")
    secondary_param = secondary_param.strip()
    priorities = []
    p = secondary_param.split(",")
    for _i in p:
        if '-' in _i:
            r = []
            lower = _i.split('-')[0]
            higher = _i.split('-')[1]
            r.append(0 if not lower else int(lower))
            r.append(float('inf') if not higher else int(higher))
            priorities.append(tuple(r))
        else:
            priorities.append((int(_i), int(_i)))
    switches.priorities = tuple(priorities)


@register_param(["--simt-stack-dcu"], "Specify SIMT DCU stack size. Unit: Bytes.")
def __set_simt_dcu_stack_size(switches: SWITCHES, secondary_param: str):
    if not secondary_param:
        raise RuntimeError(f"--simt-stack-dcu must specify a int value.")
    else:
        try:
            switches.simt_cfg.dcu_stack = int(secondary_param)
        except ValueError as e:
            raise RuntimeError(f"Invalid integer for --simt-stack-dcu option: {secondary_param}")
        if switches.simt_cfg.dcu_stack < 0:
            raise RuntimeError(f"Value for --simt-stack-dcu option should not be negative: {secondary_param}")


@register_param(["--simt-stack-dvg"], "Specify SIMT DVG stack size. Unit: Bytes.")
def __set_simt_dvg_stack_size(switches: SWITCHES, secondary_param: str):
    if not secondary_param:
        raise RuntimeError(f"--simt-stack-dvg must specify a int value.")
    else:
        try:
            switches.simt_cfg.dvg_stack = int(secondary_param)
        except ValueError as e:
            raise RuntimeError(f"Invalid integer for --simt-stack-dvg option: {secondary_param}")
        if switches.simt_cfg.dvg_stack < 0:
            raise RuntimeError(f"Value for --simt-stack-dvg option should not be negative: {secondary_param}")


@register_param(["--proc-no-reuse"], "Create a process for each case. Default false")
def __proc_no_reuse(switches: SWITCHES, _: str):
    switches.proc_no_reuse = True


@register_param(["--input-dist"], "Specify input data distribution.\n"
                                  "uniform: generate input data from a uniform distribution. (default)\n"
                                  "normal: generate input data from a normal distribution.")
def __input_distribution(switches: SWITCHES, secondary_param: str):
    if not secondary_param:
        return
    elif secondary_param.strip().lower() not in ("uniform", "normal"):
        raise RuntimeError(f"Invalid option [{secondary_param}] for --input-dist")
    else:
        switches.input_distribution = secondary_param.strip().lower()


@register_param(["--golden-mode"], "Golden generation mode. \n"
                                   "true/enable: use same inputs to feed golden function as npu. \n"
                                   "false/disable: golden function will not be invoked.\n"
                                   "promote: promote dtype of inputs to feed golden function.")
def __golden_mode(switches: SWITCHES, secondary_param: str):
    if secondary_param is None or secondary_param.lower() in ("true", "enable"):
        switches.golden_mode = "Enable"
    elif secondary_param.lower() in ("false", "disable"):
        switches.golden_mode = "Disable"
    elif secondary_param.lower() == "promote":
        switches.golden_mode = "Promote"
    else:
        raise RuntimeError("Invalid Precision Test Mode: %s" % secondary_param)


@register_param(["--compare"], "Method to compare npu output with golden. Options:\n"
                               "close: use numpy.isclose method to evaluate. (default)\n"
                               "cosine: use cosine similarity to evaluate, usually for quantize operators.\n"
                               "binary: compare npu output to golden in binary mode.\n"
                               "requant: used for re-quantize operators.")
def __compare_method(switches: SWITCHES, secondary_param: str):
    if not secondary_param:
        switches.compare_method = "close"
    else:
        s = secondary_param.lower()
        if s not in ("close", "cosine", "bin", "binary", "requant"):
            raise RuntimeError("Invalid secondary parameter for option --compare.")
        else:
            switches.compare_method = s


@register_param(["--pr", "--precision-report"], "Output precision detail report. Eg: --pd or --pd=precision_report.\n"
                                                "Note: report file name default as precision_report if not specified.")
def __precision_report(switches: SWITCHES, secondary_param: str):
    if not secondary_param:
        switches.precision_report = "precision_report"
    else:
        switches.precision_report = secondary_param.strip()


@register_param(["--plugin"], "Specify external plugin path for customized golden/inputs.")
def __plugin_path(switches: SWITCHES, secondary_param: str):
    if secondary_param:
        switches.plugin_path = tuple([pathlib.Path(p.strip()).resolve()
                                    for p in secondary_param.split(',')
                                    if p.strip()])


@register_param(["--mode"], "Test mode: op (default), aclnn, framework-api")
def __test_mode(switches: SWITCHES, secondary_param: str):
    valid_modes = ("op", "aclnn", "framework-api")
    if secondary_param and secondary_param.lower() in valid_modes:
        switches.test_mode = secondary_param.lower()
    else:
        logging.error(f"Invalid --mode value: {secondary_param}. Must be one of {valid_modes}")
        sys.exit(1)


@register_param(["--backend"], "Hardware backend for framework-api mode: npu, gpu, cpu. Auto-detect if not specified.")
def __backend(switches: SWITCHES, secondary_param: str):
    valid_backends = ("npu", "gpu", "cpu")
    if secondary_param and secondary_param.lower() in valid_backends:
        switches.backend_name = secondary_param.lower()
    else:
        logging.error(f"Invalid --backend value: {secondary_param}. Must be one of {valid_backends}")
        sys.exit(1)


@register_param(["-h", "--help"], "Display this information")
def __help(switches: SWITCHES, _: str):
    switches.print_help = True
