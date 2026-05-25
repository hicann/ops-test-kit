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
Platform Related Utilities
"""


__all__ = ["get_ascend_lib64_path",
           "get_ascend_scene_info",
           "get_opp_paths",
           "get_op_impl_paths",
           "get_impl_base_paths",
           "get_op_info_paths",
           "get_ascend_full_soc_version",
           "get_npu_hw_info",
           "get_npu_available_device_ids",
           "PLATFORM_BEFORE_DAVID"]


# Standard Packages
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Tuple


PLATFORM_BEFORE_DAVID = ("Ascend031", "Ascend310", "Ascend310P", "Ascend310B",
                         "Ascend610", "Ascend610Lite", "BS9SX1A",
                         "Ascend910B", "Ascend910_93", "Ascend910",
                         "Hi3796CV300CS", "Hi3796CV300ES",
                         "OPTG", "SD3403", "TsnsC", "TsnsE")

_VALID_SOURCES = ("builtin", "vendor", "custom")


@lru_cache(maxsize=None)
def get_opp_paths(source: str) -> list:
    """Return opp root paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        opp_path = os.getenv('ASCEND_OPP_PATH', '')
        if not opp_path:
            raise RuntimeError("Environment `ASCEND_OPP_PATH` is not set.")
        return [opp_path]
    elif source == "vendor":
        opp_path = os.getenv('ASCEND_OPP_PATH', '')
        if not opp_path:
            return []
        config_file = os.path.join(opp_path, "vendors", "config.ini")
        if not os.path.isfile(config_file):
            # fallback: no vendors config → treat as custom path
            custom_path = os.path.join(opp_path, "op_impl", "custom")
            return [custom_path] if os.path.isdir(custom_path) else []
        vendors = []
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("load_priority="):
                    continue
                for name in line.split("=", 1)[1].split(","):
                    name = name.strip()
                    if not name:
                        continue
                    p = os.path.join(opp_path, "vendors", name)
                    if os.path.isdir(p):
                        vendors.append(p)
                break
        return vendors
    elif source == "custom":
        env_val = os.getenv('ASCEND_CUSTOM_OPP_PATH', '')
        if not env_val:
            return []
        return [p for p in env_val.split(':') if p and os.path.isdir(p)]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


def _builtin_op_impl_rel():
    """Return builtin op_impl relative path, compatible with both layouts:
    - vendors/ exists → built-in/op_impl (newer CANN)
    - vendors/ absent → op_impl/built-in (older CANN)
    """
    opp_path = get_opp_paths("builtin")[0]
    vendors_dir = os.path.join(opp_path, "vendors")
    return os.path.join("built-in", "op_impl") if os.path.isdir(vendors_dir) else os.path.join("op_impl", "built-in")


@lru_cache(maxsize=None)
def get_op_impl_paths(source: str) -> list:
    """Return {opp}/op_impl paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        return [os.path.join(get_opp_paths("builtin")[0], _builtin_op_impl_rel())]
    elif source == "vendor":
        return [os.path.join(p, "op_impl") for p in get_opp_paths("vendor")]
    elif source == "custom":
        return [os.path.join(p, "op_impl") for p in get_opp_paths("custom")]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


@lru_cache(maxsize=None)
def get_impl_base_paths(source: str) -> list:
    """Return {opp}/op_impl/ai_core/tbe paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        return [os.path.join(get_opp_paths("builtin")[0], _builtin_op_impl_rel(), "ai_core", "tbe")]
    elif source == "vendor":
        return [os.path.join(p, "op_impl", "ai_core", "tbe") for p in get_opp_paths("vendor")]
    elif source == "custom":
        return [os.path.join(p, "op_impl", "ai_core", "tbe") for p in get_opp_paths("custom")]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


@lru_cache(maxsize=None)
def get_ascend_scene_info() -> Tuple[str, str]:
    opp_path = get_opp_paths("builtin")[0]
    scene_file = os.path.abspath(os.path.join(opp_path, "scene.info"))
    scene_os, scene_arch = "", ""
    if os.path.isfile(scene_file):
        with open(scene_file) as f:
            scene_info = list(map(lambda x: x.strip(), f.readlines()))
            for item_info in scene_info:
                if "os=" in item_info:
                    scene_os = item_info.split("=")[-1]
                elif "arch=" in item_info:
                    scene_arch = item_info.split("=")[-1]
    return scene_os, scene_arch


@lru_cache(maxsize=None)
def get_op_info_paths(source: str, soc_lower: str) -> list:
    """Return {impl_base}/config/{soc_lower} paths. source: 'builtin' | 'vendor' | 'custom'. Always returns list."""
    if source == "builtin":
        return [os.path.join(get_impl_base_paths("builtin")[0], "config", soc_lower)]
    elif source == "vendor":
        return [os.path.join(p, "op_impl", "ai_core", "tbe", "config", soc_lower)
                for p in get_opp_paths("vendor")]
    elif source == "custom":
        return [os.path.join(p, "op_impl", "ai_core", "tbe", "config", soc_lower)
                for p in get_opp_paths("custom")]
    raise ValueError(f"Unknown source: {source}, must be one of {_VALID_SOURCES}")


@lru_cache(maxsize=None)
def get_ascend_lib64_path():
    opp_path = get_opp_paths("builtin")[0]
    scene_os, scene_arch = get_ascend_scene_info()
    return os.path.abspath(
        os.path.join(opp_path, "..", f"{scene_arch}-{scene_os}", "lib64")
    )


@lru_cache(maxsize=None)
def get_ascend_full_soc_version():
    """Get full SoC version string (e.g. 'Ascend910B3') via asys info -r hardware."""
    if not shutil.which("asys"):
        raise FileNotFoundError("Command 'asys' not found in PATH")
    try:
        output = subprocess.check_output(
            ["asys", "info", "-r", "hardware"],
            stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"'asys info -r hardware' failed: {e}") from e
    for line in output.splitlines():
        if "Chip Info" not in line:
            continue
        m = re.search(r'(Ascend)\s*(\S+?)(?:\s+V?\d.*)?$', line.split("|")[-2].strip())
        if m:
            return m.group(1) + m.group(2)
    raise RuntimeError("Failed to parse Chip Info from 'asys info -r hardware' output")


@lru_cache(maxsize=None)
def get_npu_hw_info(full_soc_version):
    """Parse hardware info from platform_config ini file.

    Args:
        full_soc_version: e.g. 'Ascend910B3'

    Returns:
        dict with keys: short_soc_version, ccec_aic_version, npu_arch,
        ai_core_cnt, cube_core_cnt, vector_core_cnt, cv_split,
        core_type_list, support_bf16, support_inf_nan

    Raises:
        FileNotFoundError: if ini file not found
        RuntimeError: if required fields missing
    """
    import configparser

    opp_path = get_opp_paths("builtin")[0]
    scene_os, scene_arch = get_ascend_scene_info()
    config_dir = os.path.join(opp_path, "..", f"{scene_arch}-{scene_os}",
                              "data", "platform_config")
    ini_path = os.path.join(config_dir, f"{full_soc_version}.ini")
    if not os.path.isfile(ini_path):
        raise FileNotFoundError(f"Platform config not found: {ini_path}")

    cfg = configparser.ConfigParser()
    cfg.read(ini_path)

    raw_fields = {
        "short_soc_version": ("version", "Short_SoC_version"),
        "ccec_aic_version": ("version", "CCEC_AIC_version"),
        "ai_core_cnt": ("SoCInfo", "ai_core_cnt"),
        "cube_core_cnt": ("SoCInfo", "cube_core_cnt"),
        "vector_core_cnt": ("SoCInfo", "vector_core_cnt"),
        "cv_split": ("SoCInfo", "cube_vector_combine"),
        "core_type_list": ("SoCInfo", "core_type_list"),
        "support_bf16": ("SoCInfo", "support_bf16"),
        "support_inf_nan": ("SoCInfo", "support_inf_nan"),
    }

    int_fields = {"ai_core_cnt", "cube_core_cnt", "vector_core_cnt"}
    bool_fields = {"support_bf16", "support_inf_nan"}
    int_defaults = {"cube_core_cnt": 0, "vector_core_cnt": 0}

    result = {}
    missing = []
    for key, (section, option) in raw_fields.items():
        if cfg.has_option(section, option):
            val = cfg.get(section, option)
            if key in bool_fields:
                val = bool(int(val))
            elif key in int_fields:
                val = int(val)
            result[key] = val
        elif key in int_defaults:
            result[key] = int_defaults[key]
        elif key in bool_fields:
            result[key] = False
        else:
            missing.append(option)
    if missing:
        raise RuntimeError(f"[{full_soc_version}] missing fields: {missing} in {ini_path}")

    ccec = result.get("ccec_aic_version", "")
    if ccec.endswith("cube"):
        result["ccec_aic_version"] = ccec[:-4] + "$core_type"

    cv_split_val = result.pop("cv_split", None)
    if cv_split_val is None:
        cv_split_val = cfg.get("SoCInfo", "CUBE_VECTOR_SPLIT", fallback=None)
    if cv_split_val is not None:
        if not isinstance(cv_split_val, (list, tuple)):
            cv_split_val = str(cv_split_val).split(",")
        cv_split = bool("split" in cv_split_val)
    else:
        cv_split = False
    short_soc = result.get("short_soc_version", "")
    if short_soc in ("Ascend310B",):
        cv_split = False
    result["cv_split"] = cv_split

    result["full_soc_version"] = full_soc_version

    return result


@lru_cache(maxsize=None)
def get_npu_available_device_ids():
    """Get list of available Ascend device IDs.

    Returns:
        list of int: available device IDs, e.g. [0, 1, 2, 3]

    Raises:
        FileNotFoundError: if 'asys' command not found in PATH
        RuntimeError: if 'asys health' execution fails
    """
    if not shutil.which("asys"):
        raise FileNotFoundError("Command 'asys' not found in PATH")
    try:
        output = subprocess.check_output(
            ["asys", "health"],
            stderr=subprocess.DEVNULL, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"'asys health' failed: {e}") from e

    available = []
    for line in output.splitlines():
        if "Device ID:" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        for i, p in enumerate(parts):
            if p.startswith("Device ID:"):
                dev_id = int(p.split(":")[-1].strip())
                available.append(dev_id)
                break
    return available
