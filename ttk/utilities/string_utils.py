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
String Related Utilities
"""
# Standard Packages
import hashlib
import logging
import re
from typing import Any, List

_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def stable_path_component(value: Any, fallback: str = "item") -> str:
    """Return one collision-resistant, portable directory-name component."""
    original = str(value)
    safe = _SAFE_PATH_COMPONENT.sub("_", original).strip("._") or fallback
    if safe == original and len(safe) <= 120:
        return safe
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:96]}-{digest}"


def tostr(value: Any) -> str:
    """
    Convert objects to meaningful string
    :param value: Anything
    :return:
    """
    result = ""
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return '_'.join(tuple(tostr((key, tostr(value[key]))) for key in value))
    if isinstance(value, (tuple, list)):
        for sub_value in value:
            if isinstance(sub_value, str):
                result += sub_value.replace("-", "neg").replace(".", "p")
            elif isinstance(sub_value, int):
                result += str(sub_value).replace("-", "neg").replace(".", "p")
            elif isinstance(sub_value, (tuple, list)):
                first_process = '__' + '_'.join(tuple(map(str, sub_value))).replace("-", "neg").replace(".", "p")
                result += first_process.replace('(', '').replace(')', '').replace(' ', '').replace(',', '_')
            else:
                raise TypeError('Invalid type %s of %s for string conversion!' % (type(sub_value), str(value)))
    else:
        raise TypeError('Invalid type %s of %s for string conversion!' % (type(value), str(value)))
    return result


def process_kernel_string(value: str) -> str:
    """
    String for kernel_name
    :param value:
    :return:
    """
    if not isinstance(value, str):
        raise TypeError("Testcase name must be a string.")
    if len(value) == 0:
        raise RuntimeError("Testcase name must not be empty.")
    valid_char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    for idx in range(len(value)):
        if value[idx] not in valid_char:
            logging.warning("%s is not a valid kernel name, all invalid character will be converted to _" % value)
            break
    value_container = [v if v in valid_char else "_" if v != "-" else "neg" for v in value]
    result = "".join(value_container)
    if len(result) > 120:
        hash_code = hash(result)
        result = result[:80] + str(hash_code).replace("-", "neg")
    return result


def camel_to_snake(camel_name: str):
    """
    Operator Registered Camel name convert to snake name
    """
    snake_name = ""
    sub_head = False
    name_list = list(camel_name)
    for _idx, _char in enumerate(name_list):
        if _char.islower():
            sub_head = False
        if _char.isdigit():
            sub_head = True
        if _char.isupper() and _idx != 0:
            if not sub_head:
                snake_name += "_"
                sub_head = True
            else:
                _idx_next = _idx + 1
                if _idx_next < len(name_list):
                    if name_list[_idx_next].islower():
                        snake_name += "_"
        snake_name += _char

    return snake_name.lower()


def longest_match(match_str: str, match_list: List[str]) -> str:
    def lcp(str1, str2):
        length, index = min(len(str1), len(str2)), 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]

    if not match_list:
        return ""

    prefix = ""
    for m in match_list:
        candidate = lcp(match_str, m)
        if len(candidate) > len(prefix):
            prefix = candidate
    return prefix if prefix in match_list else ""
