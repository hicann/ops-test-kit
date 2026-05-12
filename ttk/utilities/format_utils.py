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
shape utils
"""


__all__ = ["FORMAT_DICT",
           "PRIVATE_FORMATS",
           "is_5hd", "is_6hd",
           "is_nchw_like", "is_ndhwc_like"]


# Standard Packages


PRIVATE_FORMATS = ["FRACTAL_NZ", "NZ"]


FORMAT_DICT = {
    'NCHW': 0,
    'NHWC': 1,
    'ND': 2,
    'NC1HWC0': 3,
    'FRACTAL_Z': 4,
    'NC1C0HWPAD': 5,
    'NHWC1C0': 6,
    'FSR_NCHW': 7,
    'FRACTAL_DECONV': 8,
    'C1HWNC0': 9,
    'FRACTAL_DECONV_TRANSPOSE': 10,
    'FRACTAL_DECONV_SP_STRIDE_TRANS': 11,
    'NC1HWC0_C04': 12,
    'FRACTAL_Z_C04': 13,
    'CHWN': 14,
    'FRACTAL_DECONV_SP_STRIDE8_TRANS': 15,
    'HWCN': 16,
    'NC1KHKWHWC0': 17,
    'BN_WEIGHT': 18,
    'FILTER_HWCK': 19,
    'HASHTABLE_LOOKUP_LOOKUPS': 20,
    'HASHTABLE_LOOKUP_KEYS': 21,
    'HASHTABLE_LOOKUP_VALUE': 22,
    'HASHTABLE_LOOKUP_OUTPUT': 23,
    'HASHTABLE_LOOKUP_HITS': 24,
    'C1HWNCoC0': 25,
    'MD': 26,
    'NDHWC': 27,
    'FRACTAL_ZZ': 28,
    'FRACTAL_NZ': 29,
    'NZ': 29,
    'NCDHW': 30,
    'DHWCN': 31,
    'NDC1HWC0': 32,
    'FRACTAL_Z_3D': 33,
    'CN': 34,
    'NC': 35,
    'DHWNC': 36,
    'FRACTAL_Z_3D_TRANSPOSE': 37,
    'FRACTAL_ZN_LSTM': 38,
    'FRACTAL_Z_G': 39,
    'RESERVED': 40,
    'ALL': 41,
    'NULL': 42,
    'ND_RNN_BIAS': 43,
    'FRACTAL_ZN_RNN': 44,
    'NYUV': 45,
    'NYUV_A': 46,
    'NCL': 47,
}


def is_nchw_like(format_: str) -> bool:
    """
    check format is NCHW-like or not
    """
    if len(set(format_)) != 4:
        return False
    return all([c in format_ for c in "NCHW"])


def is_ndhwc_like(format_: str) -> bool:
    """
    check format is NDHWC-like or not
    """
    if len(set(format_)) != 5:
        return False
    return all([c in format_ for c in "NDCHW"])


def is_5hd(format_: str) -> bool:
    """
    check format is NC1HWC0 or not
    """
    if any([c in format_ for c in "PQ"]):
        return False
    format_ = format_.replace("C1", "P")
    format_ = format_.replace("C0", "Q")
    if len(set(format_)) != 5:
        return False
    return all([c in format_ for c in "NPHWQ"])


def is_6hd(format_: str) -> bool:
    """
    check format is NDC1HWC0 or not
    """
    if any([c in format_ for c in "PQ"]):
        return False
    format_ = format_.replace("C1", "P")
    format_ = format_.replace("C0", "Q")
    if len(set(format_)) != 6:
        return False
    return all([c in format_ for c in "NDPHWQ"])

