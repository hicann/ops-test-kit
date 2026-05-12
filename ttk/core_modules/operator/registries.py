#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Interfaces to access gen_simplifiedKey_func
Functions above are registered in libregister.so from liboptiling.so/libopmaster_rt2.0.so
"""


__all__ = ["customize_gen_simplified_key"]


# Standard Packages
import ctypes
import json
import logging
import math
import os
from typing import Union
# Third-Party Packages
from ...utilities import get_ascend_scene_info, get_builtin_opp_path, get_custom_opp_paths


def _load_tiling_from_opp_impl(opp_impl_path, scene_os, scene_arch):
    """Load tiling .so files from a given opp impl path."""
    op_tiling_so_dir = os.path.join(opp_impl_path, "ai_core", "tbe", "op_tiling")

    if scene_os:
        op_tiling_so_dir = os.path.join(op_tiling_so_dir, "lib")
    op_tiling_so_dir = os.path.join(op_tiling_so_dir, scene_os, scene_arch)

    op_tiling_so = os.path.join(op_tiling_so_dir, "liboptiling.so")
    master_rt_so = os.path.join(op_tiling_so_dir, "libopmaster_rt.so")
    master_rt2_so = os.path.join(op_tiling_so_dir, "libopmaster_rt2.0.so")
    # also check for custom operator naming: libcust_opmaster_rt2.0.so
    cust_master_rt2_so = os.path.join(op_tiling_so_dir, "libcust_opmaster_rt2.0.so")
    op_host_dir = os.path.join(opp_impl_path, "ai_core", "tbe", "op_host", "lib", scene_os, scene_arch)

    if os.path.isfile(op_tiling_so):
        ctypes.CDLL(op_tiling_so)

    if os.path.isfile(master_rt_so):
        dll = ctypes.CDLL(master_rt_so)
        dll.TbeLoadSoAndSaveToRegistry(str(master_rt_so).encode('utf_8'))
    elif os.path.isfile(master_rt2_so):
        dll = ctypes.CDLL(master_rt2_so)
        dll.TbeLoadSoAndSaveToRegistry(str(master_rt2_so).encode('utf_8'))
    elif os.path.isfile(cust_master_rt2_so):
        dll = ctypes.CDLL(cust_master_rt2_so)
        dll.TbeLoadSoAndSaveToRegistry(str(cust_master_rt2_so).encode('utf_8'))
    elif os.path.isdir(op_host_dir):
        for root, _, files in os.walk(op_host_dir, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                if os.path.isfile(file_path) and str(file_path).endswith(".so"):
                    dll = ctypes.CDLL(file_path)
                    dll.TbeLoadSoAndSaveToRegistry(str(file_path).encode('utf_8'))


def load_op_registries():
    try:
        opp_path = get_builtin_opp_path()
        new_built_in = os.path.abspath(os.path.join(opp_path, "built-in", "op_impl"))
        old_built_in = os.path.abspath(os.path.join(opp_path, "op_impl", "built-in"))
        opp_impl_path = new_built_in if os.path.exists(new_built_in) else old_built_in

        scene_os, scene_arch = get_ascend_scene_info(opp_path)

        # Load built-in tiling registries
        _load_tiling_from_opp_impl(opp_impl_path, scene_os, scene_arch)

        # Load custom tiling registries from ASCEND_CUSTOM_OPP_PATH
        for custom_path in get_custom_opp_paths():
            custom_impl = os.path.join(custom_path, "op_impl")
            if os.path.isdir(custom_impl):
                logging.info(f"Loading custom tiling registries from: {custom_impl}")
                _load_tiling_from_opp_impl(custom_impl, scene_os, scene_arch)
    except BaseException as e:
        logging.critical(f"TilingInterface init failed: {e}")
        raise


def customize_gen_simplified_key(simplified_key, op_type, inputs, outputs, attrs=None) -> str:
    """ invoke customize simplified key generator.
    simplified_key = f"{op_type}/d={deterministic},p={impl_mode}/xxxxx"
    """
    load_op_registries()
    registry_accessor = _load_registry_accessor()
    if not hasattr(registry_accessor, "GenerateCustomizeSimplifiedKey"):
        raise RuntimeError(f"Interface [GenerateCustomizeSimplifiedKey] is not found.")

    deterministic = int(simplified_key.split('/')[1].split(',')[0].split('=')[1])
    extra_params = {"op_name": op_type, "deterministic": deterministic}
    _inputs_pre_process(inputs)
    _attrs_pre_process(attrs)
    op_type_c = op_type.encode('utf_8')
    inputs_c = json.dumps(inputs).encode('utf_8')
    outputs_c = json.dumps(outputs).encode('utf_8')
    extra_params_c = json.dumps(extra_params).encode('utf_8')
    if not attrs:
        attrs_c = ctypes.c_void_p()
    else:
        attrs_c = json.dumps(attrs).encode('utf_8')

    res_buf = ctypes.create_string_buffer(simplified_key.encode('utf_8'), 256)

    func = getattr(registry_accessor, "GenerateCustomizeSimplifiedKey")
    func.restype = ctypes.c_int
    ret = func(op_type_c, inputs_c, outputs_c, attrs_c, extra_params_c, res_buf)
    if ret != 0:
        msg = _parse_c_return_code(ret)
        raise RuntimeError(f"invoke customized simplified key generator for op [{op_type}] failed: {msg}. "
                           f"Please check `plog` for more details.")
    return res_buf.value.decode('utf-8')


def _parse_c_return_code(ret):
    CODE_MAP = {
        1: "Customized simplified key generator function is not found",
        2: "Parse input/output/attr/extra_info failed",
        3: "Invoke customized simplified key generator function failed"
    }
    return CODE_MAP.get(ret, f"Unknown return code: [{ret}]")


def _load_registry_accessor():
    try:
        return ctypes.CDLL("libregistry_accessor.so")
    except BaseException as e:
        logging.critical(f"Load libregistry_accessor.so in libs failed: {e}")
        raise e


def _inputs_pre_process(inputs: Union[list, tuple]):
    if not isinstance(inputs, (list, tuple)):
        return
    for ipt in inputs:
        if not isinstance(ipt, dict):
            continue
        const_value = ipt.get("const_value")
        if not isinstance(const_value, (list, tuple)):
            continue
        const_value_list = list(const_value)
        const_value_null_desc = _gen_null_desc(const_value_list)
        if const_value_null_desc is not None:
            ipt["const_value"] = const_value_list
            ipt["const_value_null_desc"] = const_value_null_desc


def _attrs_pre_process(attrs):
    if not isinstance(attrs, (list, tuple)):
        return
    for single_attr in attrs:
        if not isinstance(single_attr, dict):
            continue
        attr_dtype = single_attr.get("dtype")
        if attr_dtype not in ("float", "float32", "list_float", "list_float32"):
            continue
        attr_value = single_attr.get("value")
        if attr_value is None:
            continue
        is_single_element = False
        if not isinstance(attr_value, (list, tuple)):
            is_single_element = True
            attr_value = [attr_value]
        attr_value_list = list(attr_value)
        attr_null_desc = _gen_null_desc(attr_value_list)
        if attr_null_desc is not None:
            if is_single_element:
                single_attr["value_null_desc"] = attr_null_desc[0]
                single_attr["value"] = attr_value_list[0]
            else:
                single_attr["value_null_desc"] = attr_null_desc
                single_attr["value"] = attr_value_list


def _gen_null_desc(value_list):
    if not isinstance(value_list, list):
        return None
    value_null_desc = []
    is_exist_null = False
    for idx, value in enumerate(value_list):
        if not isinstance(value, float):
            continue
        if value == float("inf"):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("inf")
        elif value == float("-inf"):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("-inf")
        elif math.isnan(value):
            is_exist_null = True
            value_list[idx] = None
            value_null_desc.append("nan")
        else:
            value_null_desc.append(None)

    return value_null_desc if is_exist_null else None
