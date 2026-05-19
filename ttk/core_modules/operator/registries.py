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
import logging
import os
from typing import Union
# Third-Party Packages
from ...utilities import get_ascend_scene_info
from ...utilities.platform import get_op_impl_paths


def _load_custom_vendor_tiling(opp_impl_path):
    """Load custom/vendor tiling SO: flat path, only liboptiling.so + TbeLoadSoAndSaveToRegistry.

    Follows official CANN op_tiling.py: custom/vendor SO is at
    {impl}/ai_core/tbe/op_tiling/liboptiling.so (no lib/{os}/{arch} subdirectory).
    OSError is silently ignored (SO may not exist).
    """
    so_path = os.path.join(opp_impl_path, "ai_core", "tbe", "op_tiling", "liboptiling.so")
    if not os.path.isfile(so_path):
        return
    try:
        dll = ctypes.CDLL(so_path)
        dll.TbeLoadSoAndSaveToRegistry(str(so_path).encode('utf_8'))
    except OSError:
        pass


def _load_builtin_tiling(opp_impl_path):
    """Load builtin tiling SOs with full if-else chain.

    Follows official CANN op_tiling.py:
      1. liboptiling.so (1.0 registration, CDLL only)
      2. libopmaster_rt.so → libopmaster_rt2.0.so → libcust_opmaster_rt2.0.so → op_host/*.so
         (2.0 registration via TbeLoadSoAndSaveToRegistry, mutually exclusive)
    """
    scene_os, scene_arch = get_ascend_scene_info()

    op_tiling_so_dir = os.path.join(opp_impl_path, "ai_core", "tbe", "op_tiling")

    if scene_os:
        op_tiling_so_dir = os.path.join(op_tiling_so_dir, "lib")
    op_tiling_so_dir = os.path.join(op_tiling_so_dir, scene_os, scene_arch)

    op_tiling_so = os.path.join(op_tiling_so_dir, "liboptiling.so")
    master_rt_so = os.path.join(op_tiling_so_dir, "libopmaster_rt.so")
    master_rt2_so = os.path.join(op_tiling_so_dir, "libopmaster_rt2.0.so")
    cust_master_rt2_so = os.path.join(op_tiling_so_dir, "libcust_opmaster_rt2.0.so")
    op_host_dir = os.path.join(opp_impl_path, "ai_core", "tbe", "op_host", "lib", scene_os, scene_arch)

    # 1. builtin optiling 1.0 registration
    if os.path.isfile(op_tiling_so):
        ctypes.CDLL(op_tiling_so)

    # 2. builtin optiling 2.0 registration (mutually exclusive)
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
        non_legacy = []
        legacy = []
        for root, _, files in os.walk(op_host_dir, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                if os.path.isfile(file_path) and file_path.endswith(".so"):
                    if 'legacy' in f.lower():
                        legacy.append(file_path)
                    else:
                        non_legacy.append(file_path)
        for fp in non_legacy:
            dll = ctypes.CDLL(fp)
            dll.TbeLoadSoAndSaveToRegistry(str(fp).encode('utf_8'))
        for fp in legacy:
            dll = ctypes.CDLL(fp)
            dll.TbeLoadSoAndSaveToRegistry(str(fp).encode('utf_8'))


def load_op_registries():
    # Load in priority order: custom → vendors → built-in (first-registered-wins)
    for source in ("custom", "vendor", "builtin"):
        try:
            for impl_path in get_op_impl_paths(source):
                if os.path.isdir(impl_path):
                    logging.info(f"Loading {source} tiling registries from: {impl_path}")
                    if source in ("custom", "vendor"):
                        _load_custom_vendor_tiling(impl_path)
                    else:
                        _load_builtin_tiling(impl_path)
        except BaseException as e:
            logging.critical(f"TilingInterface init failed for {source}: {e}")
            raise


_GERT_DTYPE_MAP = {
    'float32': 1, 'float': 1,
    'float16': 2,
    'int8': 3,
    'int16': 4,
    'int32': 5, 'int': 5,
    'int64': 9,
    'uint8': 7,
    'uint16': 8,
    'uint32': 9,
    'uint64': 10,
    'bool': 11,
    'double': 12, 'float64': 12,
    'bf16': 13, 'bfloat16': 13,
}

_GERT_FORMAT_MAP = {
    'ND': 0, 'NCHW': 1, 'NHWC': 2, 'NC1HWC0': 5,
    'FRACTAL_Z': 4, 'FRACTAL_NZ': 12, 'NZ': 12,
    'NDC1HWC0': 33, 'FRACTAL_Z_G': 34,
    'ND_RNN_BIAS': 29, 'FRACTAL_ZN_LSTM': 30,
    'NCDHW': 0, 'NDHWC': 0, 'DHWCN': 0,
}


class _TensorDesc(ctypes.Structure):
    _fields_ = [("dtype", ctypes.c_int32), ("format", ctypes.c_int32)]


class _AttrDesc(ctypes.Structure):
    _fields_ = [("dtype", ctypes.c_char_p), ("value", ctypes.c_void_p),
                ("value_type", ctypes.c_int32), ("value_count", ctypes.c_int32)]


def _build_tensor_descs(tensors):
    if not tensors:
        return None, 0
    descs = []
    for t in tensors:
        if t is None:
            descs.append(_TensorDesc(0, 0))
            continue
        dtype_str = t.get("dtype", "float32")
        fmt_str = t.get("format", "ND")
        if isinstance(dtype_str, (list, tuple)):
            dtype_str = dtype_str[0] if dtype_str else "float32"
        if isinstance(fmt_str, (list, tuple)):
            fmt_str = fmt_str[0] if fmt_str else "ND"
        dtype_str = dtype_str.lower().replace("ge_", "")
        gert_dtype = _GERT_DTYPE_MAP.get(dtype_str, 0)
        gert_fmt = _GERT_FORMAT_MAP.get(fmt_str, 0)
        descs.append(_TensorDesc(gert_dtype, gert_fmt))
    arr = (_TensorDesc * len(descs))(*descs)
    return arr, len(descs)


_ATTR_VALUE_TYPE_INT64 = 0
_ATTR_VALUE_TYPE_FLOAT = 1
_ATTR_VALUE_TYPE_BOOL = 2
_ATTR_VALUE_TYPE_STR = 3
_ATTR_VALUE_TYPE_LIST_INT64 = 4
_ATTR_VALUE_TYPE_LIST_FLOAT = 5
_ATTR_VALUE_TYPE_LIST_BOOL = 6
_ATTR_VALUE_TYPE_LIST_STR = 7


def _build_attr_descs(attrs):
    if not attrs:
        return None, 0
    descs = []
    _holders = []
    for a in attrs:
        if not isinstance(a, dict):
            continue
        dtype_str = a.get("dtype", "int")
        val = a.get("value")
        if val is None:
            descs.append(_AttrDesc(dtype_str.encode(), ctypes.c_void_p(0), 0, 0))
            continue
        is_list = dtype_str.startswith("list_")
        base_dtype = dtype_str.replace("list_", "")
        if base_dtype in ("int", "int64", "int32"):
            if is_list:
                vals = val if isinstance(val, (list, tuple)) else [val]
                buf = (ctypes.c_int64 * len(vals))(*[int(v) for v in vals])
                _holders.append(buf)
                descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(buf, ctypes.c_void_p),
                                       _ATTR_VALUE_TYPE_LIST_INT64, len(vals)))
            else:
                buf = ctypes.c_int64(int(val))
                _holders.append(buf)
                descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(ctypes.pointer(buf), ctypes.c_void_p),
                                       _ATTR_VALUE_TYPE_INT64, 1))
        elif base_dtype in ("float", "float32"):
            if is_list:
                vals = val if isinstance(val, (list, tuple)) else [val]
                buf = (ctypes.c_float * len(vals))(*[float(v) for v in vals])
                _holders.append(buf)
                descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(buf, ctypes.c_void_p),
                                       _ATTR_VALUE_TYPE_LIST_FLOAT, len(vals)))
            else:
                buf = ctypes.c_double(float(val))
                _holders.append(buf)
                descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(ctypes.pointer(buf), ctypes.c_void_p),
                                       _ATTR_VALUE_TYPE_FLOAT, 1))
        elif base_dtype == "bool":
            if is_list:
                vals = val if isinstance(val, (list, tuple)) else [val]
                buf = (ctypes.c_bool * len(vals))(*[bool(v) for v in vals])
                _holders.append(buf)
                descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(buf, ctypes.c_void_p),
                                       _ATTR_VALUE_TYPE_LIST_BOOL, len(vals)))
            else:
                buf = ctypes.c_bool(bool(val))
                _holders.append(buf)
                descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(ctypes.pointer(buf), ctypes.c_void_p),
                                       _ATTR_VALUE_TYPE_BOOL, 1))
        elif base_dtype in ("str", "string"):
            encoded = str(val).encode('utf_8')
            _holders.append(encoded)
            descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(ctypes.c_char_p(encoded), ctypes.c_void_p),
                                   _ATTR_VALUE_TYPE_STR, 1))
        else:
            buf = ctypes.c_int64(int(val) if val is not None else 0)
            _holders.append(buf)
            descs.append(_AttrDesc(dtype_str.encode(), ctypes.cast(ctypes.pointer(buf), ctypes.c_void_p),
                                   _ATTR_VALUE_TYPE_INT64, 1))
    if not descs:
        return None, 0
    arr = (_AttrDesc * len(descs))(*descs)
    return arr, len(descs)


_registries_loaded = False


def _ensure_registries_loaded():
    global _registries_loaded
    if not _registries_loaded:
        load_op_registries()
        _registries_loaded = True


_registry_accessor = None


def _get_registry_accessor():
    global _registry_accessor
    if _registry_accessor is None:
        _registry_accessor = _load_registry_accessor()
    return _registry_accessor


def customize_gen_simplified_key(simplified_key, op_type, inputs, outputs, attrs=None) -> str:
    """ invoke customize simplified key generator.
    simplified_key = f"{op_type}/d={deterministic},p={impl_mode}/xxxxx"
    """
    _ensure_registries_loaded()
    registry_accessor = _get_registry_accessor()
    if not hasattr(registry_accessor, "GenerateCustomizeSimplifiedKey"):
        raise RuntimeError(f"Interface [GenerateCustomizeSimplifiedKey] is not found.")

    _inputs_pre_process(inputs)
    _attrs_pre_process(attrs)

    op_type_c = op_type.encode('utf_8')
    inputs_arr, num_inputs = _build_tensor_descs(inputs)
    outputs_arr, num_outputs = _build_tensor_descs(outputs)
    attrs_arr, num_attrs = _build_attr_descs(attrs)

    res_buf = ctypes.create_string_buffer(simplified_key.encode('utf_8'), 256)

    func = getattr(registry_accessor, "GenerateCustomizeSimplifiedKey")
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.c_char_p,
                     ctypes.POINTER(_TensorDesc), ctypes.c_int,
                     ctypes.POINTER(_TensorDesc), ctypes.c_int,
                     ctypes.POINTER(_AttrDesc), ctypes.c_int,
                     ctypes.c_char_p]
    ret = func(op_type_c,
               inputs_arr, num_inputs,
               outputs_arr, num_outputs,
               attrs_arr, num_attrs,
               res_buf)
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
