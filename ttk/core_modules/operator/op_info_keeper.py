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
op info keeper.
"""

__all__ = ["OpInfoKeeper"]


# Standard Packages
from collections import defaultdict
from configparser import ConfigParser
import json
import logging
import os
from typing import Dict, Union, Callable, List, Any

# Third-Party Packages
from .tbe_interface import Opc
from ...utilities import Singleton, camel_to_snake, is_main_process, get_custom_opp_paths


class OpInfoKeeper(metaclass=Singleton):
    """
    Singleton Class to keep ops info
    """
    def __init__(self):
        self._opc = Opc()
        self._ops_info = None

    @property
    def ops_info(self) -> dict:
        if self._ops_info is None:
            self._ops_info = self._load_op_info_cfg()
        return self._ops_info

    @property
    def short_soc_version(self):
        return self._opc.soc_info["short_soc_version"]

    def has_op(self, op_name: str) -> bool:
        return op_name in self.ops_info

    def info_of(self, op_name: str) -> dict:
        return self.ops_info.get(op_name, None)

    def op_file_of(self, op_name: str) -> str:
        return "" if not self.has_op(op_name) else self.ops_info[op_name]["opFile.value"]

    def op_output_defined(self, op_name: str):
        return not self.has_op(op_name) or self.ops_info[op_name]["outputs"]

    def _load_op_info_cfg(self) -> dict:
        keys = ("coreType.value", "opInterface.value", "opFile.value", "op.pattern",
                "dynamicShapeSupport.flag", "dynamicCompileStatic.flag", "dynamicRankSupport.flag",
                "needCheckSupport.flag", "dynamicFormat.flag")
        soc_lower = (self._opc.soc_info.get("short_soc_version") or "").lower()

        # Collect all op info config files: built-in first, then custom (custom overrides built-in)
        all_op_info_paths = []

        # 1. built-in op info
        builtin_op_info_path = self._find_builtin_op_info_paths(soc_lower)
        if builtin_op_info_path:
            all_op_info_paths.extend(builtin_op_info_path)

        # 2. custom op info from ASCEND_CUSTOM_OPP_PATH (higher priority, loaded last to override)
        custom_op_info_paths = self._find_custom_op_info_paths(soc_lower)
        all_op_info_paths.extend(custom_op_info_paths)

        if not all_op_info_paths:
            logging.warning("No op info config found (built-in or custom).")
            return {}

        # tikc op file list
        op_info_snake: Dict[str, dict] = defaultdict(dict)

        for f in all_op_info_paths:
            if f.endswith(".ini"):
                op_info = _get_op_info_from_ini_file(f, keys)
            else:
                op_info = _get_op_info_from_json_file(f, keys)

            for op in op_info.keys():
                op_interface = op_info[op]["opInterface.value"] \
                    if op_info[op]["opInterface.value"] else camel_to_snake(op)
                if not op_info[op]["coreType.value"]:
                    op_info[op]["coreType.value"] = "VectorCore" if self._opc.soc_info.get("cv_split") else "AiCore"
                elif len(op_info[op]["coreType.value"].split(',')) > 1:
                    #  AiCore,VectorCore or AiCore,AIV
                    op_info[op]["coreType.value"] = "AiCore"  # do not know exactly, consider it as AiCore
                if op_interface.replace('_', '').lower() != op.lower():
                    # Add & AddV2 use add as interface...
                    continue
                if op_info[op].get("opFile.value") is None:
                    op_info[op]["opFile.value"] = camel_to_snake(op)
                op_info_snake[op_interface] = op_info[op]
                op_info_snake[op_interface].update({"op_type": op})
                self._eval_attr_str(op_info_snake[op_interface]["attr"], op)
        return op_info_snake

    def _find_builtin_op_info_paths(self, soc_lower: str) -> List[str]:
        """Find built-in op info config files."""
        # ini
        op_info_path = os.path.abspath(
            os.path.join(self._opc.impl_path, "..", "op_info_cfg", "ai_core", soc_lower,
                         "aic-" + soc_lower + "-ops-info.ini")
        )
        if os.path.exists(op_info_path):
            return [op_info_path]
        # json
        op_info_path = os.path.abspath(
            os.path.join(self._opc.impl_path, "..", "config", soc_lower,
                         "aic-" + soc_lower + "-ops-info.json")
        )
        if os.path.exists(op_info_path):
            return [op_info_path]
        # all json files in config dir
        json_lst_path = os.path.abspath(
            os.path.join(self._opc.impl_path, "..", "config", soc_lower)
        )
        if os.path.isdir(json_lst_path):
            paths = [os.path.join(json_lst_path, f) for f in os.listdir(json_lst_path)
                     if f.endswith(".json") and os.path.isfile(os.path.join(json_lst_path, f))]
            if paths:
                return paths
        logging.warning("Built-in op info config path does not exist for soc: %s" % soc_lower)
        return []

    @staticmethod
    def _find_custom_op_info_paths(soc_lower: str) -> List[str]:
        """Find op info config files from ASCEND_CUSTOM_OPP_PATH directories."""
        result = []
        for custom_path in get_custom_opp_paths():
            config_dir = os.path.join(custom_path, "op_impl", "ai_core", "tbe", "config", soc_lower)
            if not os.path.isdir(config_dir):
                continue
            for f in os.listdir(config_dir):
                full = os.path.join(config_dir, f)
                if f.endswith(".json") and os.path.isfile(full):
                    logging.info(f"Loading custom op info from: {full}")
                    result.append(full)
        return result

    def _eval_attr_str(self, attrs: List[dict], op_name: str):
        for attr in attrs:
            attr_type: str = attr["type"]
            attr_default: Any = attr["defaultValue"]
            if attr_default is None:
                continue
            try:
                if attr_type.startswith("list"):
                    if attr_type.endswith("Bool"):
                        attr_default = attr_default.replace("true", "True").replace("false", "False")
                    attr_default = eval(attr_default)
                elif attr_type == "bool":
                    attr_default = True if attr_default.lower() == "true" else False
                else:
                    typ = self._attr_dtype_str_to_type(attr_type)
                    attr_default = typ(attr_default)
            except Exception as e:
                # some operator may configure wrong value,
                # like  attr_spatial_scale  in RoiExtractor. SHIT...
                if is_main_process():
                    logging.warning(f"Try to parse [defaultValue: {attr_default}] "
                                    f"of attribute [{attr['name']}] "
                                    f"for operator [{op_name}] failed: {e}")
            finally:
                attr["defaultValue"] = attr_default

    @staticmethod
    def _attr_dtype_str_to_type(s):
        return list if s.startswith("list") else eval(s)


def _get_op_info_from_ini_file(file_path: str, keys: Union[list, tuple] = ("coreType.value",)):
    def _get_val(_cfp, _op, key, default=None):
        if _cfp.has_section(_op) and key.lower() in _cfp.options(_op):
            return _cfp.get(_op, key.lower())
        else:
            return default

    cfp = ConfigParser()
    cfp.read(file_path, encoding="UTF-8")
    ops = cfp.sections()
    return _extract_op_info(cfp, ops, keys, _get_val)


def _get_op_info_from_json_file(file_path: str, keys: Union[list, tuple] = ("coreType.value",)):
    def _get_val(full_json, _op, key, default=None):
        val = full_json
        key_section = key.split(".")
        key_section.insert(0, _op)
        for ks in key_section:
            if ks in val:
                val = val[ks]
            else:
                val = default
                break
        return val

    with open(file_path, encoding="UTF-8") as f:
        json_data: dict = json.load(f)
    ops = json_data.keys()
    return _extract_op_info(json_data, ops, keys, _get_val)


def _pre_normalize_dtype(xput: dict):
    if xput['dtype']:
        xput['dtype'] = xput['dtype'].split(',')
        xput['dtype'] = ["float32" if d == "float" else d for d in xput['dtype']]


def _pre_normalize_format(xpt: dict):
    if not xpt['unknownshape_format']:
        xpt['unknownshape_format'] = xpt['format']
    if xpt['unknownshape_format']:
        xpt['unknownshape_format'] = xpt['unknownshape_format'].split(',')
    else:
        xpt['unknownshape_format'] = []
    if xpt['format']:
        xpt['format'] = xpt['format'].split(',')
    else:
        xpt['format'] = []


def _extract_op_info(all_info, ops, keys: Union[list, tuple], get_val_func: Callable) -> dict:
    op_info: Dict[str, dict] = defaultdict(dict)
    xput_key_defaults = {"paramType": "required", "valueDepend": "ignore", "dtype": None,
                         "unknownshape_format": None, "format": None}
    attr_key_defaults = {"type": None, "value": None, "defaultValue": None}
    for op in ops:
        for k in keys:
            op_info[op][k] = get_val_func(all_info, op, k)
        # load inputs & outputs
        for ele in ("input", "output"):
            idx = 0
            op_info[op][f"{ele}s"] = []
            while True:
                xput_name_key = f"{ele}{idx}.name"
                xput_name = get_val_func(all_info, op, xput_name_key)
                if not xput_name:
                    break
                xput: dict = {"name": xput_name}
                for k, v in xput_key_defaults.items():
                    fk = f"{ele}{idx}.{k}"
                    xput[k] = get_val_func(all_info, op, fk, v)
                _pre_normalize_dtype(xput)
                _pre_normalize_format(xput)
                op_info[op][f"{ele}s"].append(xput)
                idx = idx + 1
        # load attrs
        attrs = get_val_func(all_info, op, "attr.list")
        op_info[op]["attr"] = []
        if attrs:
            attr_list = attrs.split(",")
            for attr in attr_list:
                attr_dict: dict = {"name": attr}
                for k, v in attr_key_defaults.items():
                    fk = f"attr_{attr}.{k}"
                    attr_dict[k] = get_val_func(all_info, op, fk, v)
                op_info[op]["attr"].append(attr_dict)
    return op_info
