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
dtype utils
"""

import copy
import re
import struct
from typing import Optional, Union

import numpy

BFP16_NEEDS_FP32_FOR_NPY: Optional[bool] = None

# Numpy 4-bit dtypes (unpacked, 1 byte/element) → torch packed dtype names.
# torch stores 2 values per byte (x2 suffix); numpy en_dtypes stores 1 per byte.
_NUMPY_TO_TORCH_4BIT_DTYPE = {
    "float4_e2m1": "float4_e2m1fn_x2",
    "float4_e1m2": "float4_e1m2fn_x2",
    "int4": "int4",
}


# Shared dtype promotion map for golden "Promote" mode.
#
# Used by BOTH flows:
#   - KERNEL (output_generation.__promote_dtype) — numpy arrays
#   - ACLNN   (golden_generation._promote_dtype)  — torch tensors
#
# Promoting to a wider dtype yields the "true value" of the op computed at
# higher precision, against which the low-precision NPU output is compared.
# Kept here (not in either flow module) to avoid a cross-flow import dependency
# and duplication.
DTYPE_PROMOTE_MAP: dict = {
    "float16": "float32",
    "bfloat16": "float32",
    "float32": "float64",
    "complex32": "complex64",
    "complex64": "complex128",
    "float8_e5m2": "float32",
    "float8_e4m3fn": "float32",
    "float8_e8m0": "float32",
    "hifloat8": "float32",
}


dtype_width_map = {
    "complex32": 4,
    "complex64": 8,
    "complex128": 16,
    "double": 8,
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "uint64": 8,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "int4": 0.5,
    "uint1": 1,
    "bool": 1,
    "float4_e2m1": 0.5,
    "float4_e1m2": 0.5,
    "float8_e8m0": 1,
    "float8_e5m2": 1,
    "float8_e4m3fn": 1,
    "hifloat8": 1,
    None: 0,
}

dtype_map = {
    "c32": "complex32",
    "c64": "complex64",
    "complex32": "complex32",
    "complex64": "complex64",
    "c128": "complex128",
    "complex128": "complex128",
    "f64": "float64",
    "fp64": "float64",
    "float64": "float64",
    "double": "double",
    "f32": "float32",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
    "f16": "float16",
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfp16": "bfloat16",
    "bfloat16": "bfloat16",
    "s64": "int64",
    "int64": "int64",
    "u64": "uint64",
    "uint64": "uint64",
    "s32": "int32",
    "int32": "int32",
    "u32": "uint32",
    "uint32": "uint32",
    "s16": "int16",
    "int16": "int16",
    "u16": "uint16",
    "uint16": "uint16",
    "s8": "int8",
    "int8": "int8",
    "u8": "uint8",
    "uint8": "uint8",
    "s4": "int4",
    "int4": "int4",
    "u1": "uint1",
    "uint1": "uint1",
    "s1": "uint1",
    "int1": "uint1",
    "bool": "bool",
    "f4_e2m1": "float4_e2m1",
    "fp4_e2m1": "float4_e2m1",
    "float4_e2m1": "float4_e2m1",
    "float4_e2m1fn": "float4_e2m1",
    "fp4_e2m1fn": "float4_e2m1",
    "f4_e2m1fn": "float4_e2m1",
    "f4_e1m2": "float4_e1m2",
    "fp4_e1m2": "float4_e1m2",
    "float4_e1m2": "float4_e1m2",
    "float4_e1m2fn": "float4_e1m2",
    "fp4_e1m2fn": "float4_e1m2",
    "f4_e1m2fn": "float4_e1m2",
    "f8_e8m0": "float8_e8m0",
    "fp8_e8m0": "float8_e8m0",
    "float8_e8m0": "float8_e8m0",
    "f8_e5m2": "float8_e5m2",
    "fp8_e5m2": "float8_e5m2",
    "float8_e5m2": "float8_e5m2",
    "f8_e4m3fn": "float8_e4m3fn",
    "fp8_e4m3fn": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
    "hif8": "hifloat8",
    "hifp8": "hifloat8",
    "hifloat8": "hifloat8",
    "hifloat4": "hifloat4",
    None: None,
}


DATA_TYPE_DICT = {
    "float32": 0,
    "float16": 1,
    "int8": 2,
    "int32": 3,
    "uint8": 4,
    "int16": 6,
    "uint16": 7,
    "uint32": 8,
    "int64": 9,
    "uint64": 10,
    "double": 11,
    "float64": 11,
    "bool": 12,
    "complex64": 16,
    "complex128": 17,
    "qint8": 18,
    "qint16": 19,
    "qint32": 20,
    "quint8": 21,
    "quint16": 22,
    "resource": 23,
    "dual": 25,
    "variant": 26,
    "bf16": 27,
    "bfloat16": 27,
    "int4": 29,
    "uint1": 30,
    "int2": 31,
    "uint2": 32,
    "complex32": 33,
    "hifloat8": 34,
    "float8_e5m2": 35,
    "float8_e4m3fn": 36,
    "float8_e8m0": 37,
    "float8_e8m0fnu": 37,
    "float4_e2m1": 40,
    "float4_e1m2": 41,
    "hifloat4": 42,
}


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    18: "qint8",
    19: "qint16",
    20: "qint32",
    21: "quint8",
    22: "quint16",
    23: "resource",
    25: "dual",
    26: "variant",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    31: "int2",
    32: "uint2",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}


def str_to_torch_dtype(dtype_str: str):
    """Convert dtype string to torch.dtype/torch_npu.dtype object.

    Args:
        dtype_str: string like 'float16', 'int8', 'fp16', 'bf16', etc.

    Returns:
        torch.dtype/torch_npu object, or original value if not recognized.
    """
    if not isinstance(dtype_str, str):
        return dtype_str
    splited = dtype_str.split(".")
    if len(splited) > 2:
        return None
    module = splited[0]
    canonical = splited[-1]
    if len(splited) == 1:
        if is_torch_native_dtype(canonical):
            import torch

            return getattr(torch, canonical, None)
        else:
            import torch_npu

            npu_attr_name = "float8_e8m0fnu" if canonical == "float8_e8m0" else canonical
            return getattr(torch_npu, npu_attr_name, None)
    else:
        if module not in ("torch", "torch_npu"):
            return None
        try:
            if module == "torch":
                import torch
            else:
                import torch_npu
            return eval(dtype_str)
        except Exception:
            return None


def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = numpy.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def is_4bit_dtype(dtype) -> bool:
    """
    True if dtype is a packed 4-bit type (int4 / float4_e2m1 / float4_e1m2 / hifloat4).

    Accepts str, numpy.dtype, torch.dtype, or any object whose str() contains
    the dtype name.
    """
    s = str(dtype)
    return "int4" in s or "float4" in s


def get_dtype_width(dt: str) -> Union[int, float]:
    """
    Dtype string to byte width mapping
    """
    str_dtype = ""
    if isinstance(dt, str):
        str_dtype = dt
    elif hasattr(dt, "name"):
        str_dtype = dt.name
    if is_4bit_dtype(str_dtype):
        return 0.5
    try:
        dtype = numpy.dtype(dt)
        return dtype.itemsize
    except TypeError:
        return dtype_width_map[dt]


def parse_dtype(_dtype: str):
    """
    Dtype string to valid dtype mapping
    """

    return dtype_map[_dtype]


def numpy_int4():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import int4

        return int4
    except ModuleNotFoundError:
        raise RuntimeError(
            "ml_dtypes is needed to support int4 dtype!!! Please install with `pip3 install ml-dtypes`"
        ) from None


def numpy_bfloat16():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            # noinspection PyUnresolvedReferences
            import tensorflow

            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                "Please install with `pip3 install ml-dtypes` "
                "or `pip3 install tensorflow`"
            ) from None
    # some older TF version (v1.15.0) bfp16 needs to convert to fp32 to calculate in numpy
    global BFP16_NEEDS_FP32_FOR_NPY
    if BFP16_NEEDS_FP32_FOR_NPY is None:
        test_array = numpy.array([1, 2, 3], dtype=bfloat16)
        try:
            numpy.max(test_array)
        except TypeError:
            # TypeError: No loop matching the specified signature
            # and casting was found for ufunc maximum
            BFP16_NEEDS_FP32_FOR_NPY = True
        else:
            BFP16_NEEDS_FP32_FOR_NPY = False
    return bfloat16


def numpy_float8_e5m2():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    except ModuleNotFoundError:
        raise RuntimeError(
            "ml_dtypes is needed to support float8_e5m2 dtype!!! Please install with `pip3 install ml-dtypes`"
        ) from None


def numpy_float8_e4m3fn():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError(
            "ml_dtypes is needed to support float8_e4m3fn dtype!!! Please install with `pip3 install ml-dtypes`"
        ) from None


def ensure_en_dtypes_version(version):
    import en_dtypes

    cur_ver = list(map(int, en_dtypes.__version__.split("."))) + [0, 0]
    min_ver = list(map(int, version.split("."))) + [0, 0]
    if (cur_ver[0], cur_ver[1], cur_ver[2]) >= (min_ver[0], min_ver[1], min_ver[2]):
        return
    else:
        raise RuntimeError(f"Please upgrade en-dtypes to at least {version}")


def numpy_float8_e8m0():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float8_e8m0

        ensure_en_dtypes_version("0.0.4")
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float8_e8m0 dtype!!! Please install with `pip3 install en-dtypes`"
        ) from None


def numpy_float4_e2m1():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float4_e2m1

        ensure_en_dtypes_version("0.0.4")
        return float4_e2m1
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float4_e2m1 dtype!!! Please install with `pip3 install en-dtypes`"
        ) from None


def numpy_float4_e1m2():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float4_e1m2

        ensure_en_dtypes_version("0.0.4")
        return float4_e1m2
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float4_e1m2 dtype!!! Please install with `pip3 install en-dtypes`"
        ) from None


def numpy_hifloat4():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import float4_e1m2

        ensure_en_dtypes_version("0.0.4")
        return float4_e1m2
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support hifloat4 dtype!!! Please install with `pip3 install en-dtypes`"
        ) from None


def IsRoundOne(sign, man, truncLen):
    roundingTruncLen = 64
    if truncLen >= roundingTruncLen:
        mask0 = 0
    else:
        mask0 = 0x1 << truncLen
    if truncLen > roundingTruncLen:
        mask1 = 0
    else:
        mask1 = 0x1 << (truncLen - 1)

    mask2 = mask1 - 1

    # ROUND_TO_NEAREST
    lastBit = (man & mask0) > 0  # Last bit after conversion
    truncHighBit = (man & mask1) > 0  # Highest bit in the truncated part
    truncLeft = (man & mask2) > 0  # Truncated left part (except for the highest bit)
    return truncHighBit and (truncLeft or lastBit)


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def cvt_bfloat16_to_fp4_e2m1(x):
    import math

    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    x = eval(float_to_hex(x_abs))
    x = x >> 16

    ef = (x >> 7) & 0xFF
    mf = x & 0x7F
    mLenDelta = 7 - 1  #
    maxExp = 3  # max E encoding value of e2m1 is 3
    expBias = 1  # Exponent Bias value of e2m1/e1m2 is 1
    eRet = 0
    mRet = 0
    eNorm = 0
    if ef == 0 and mf != 0:
        eNorm = ef - 127 + 1  # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127  # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 1)):
        return (sRet << 3) | 0b111
    elif eNorm <= -(expBias):
        eRet = 0
        mf = mf | 0x80
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta)  # determine if need to carry
        mRet = mf >> mLenDelta
        if needRound:
            mRet += 1
    else:
        eRet = eNorm + expBias
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = mf >> mLenDelta
        if needRound:
            mRet += 1

        if ((mRet & 0b10) != 0) and (needRound):
            eRet += 1
            mRet = 0

    if eRet >= 3:
        eRet = 3
    elif eRet == 0 and mRet == 0b10:
        eRet += 1
        mRet = 0

    return ((sRet) << 3) | ((eRet) << 1) | ((mRet) & 1)


def cvt_bfloat16_to_fp4_e1m2(x):
    import math

    sRet = 0
    if x < 0.0:
        sRet = 1

    x_abs = math.fabs(x)
    x = eval(float_to_hex(x_abs))
    x = x >> 16

    ef = x >> 7 & 0xFF
    mf = x & 0x7F
    mLenDelta = 7 - 2  #
    maxExp = 1  # max E encoding value of e1m2 is 3
    expBias = 1  # Exponent Bias value of e2m1/e1m2 is 1

    eRet = 0
    mRet = 0
    eNorm = 0
    if ef == 0 and mf != 0:
        eNorm = ef - 127 + 1  # the exp bias of subnormal bf16 is 126
    else:
        eNorm = ef - 127  # the exp bias of bf16 is 127

    if (eNorm > (maxExp - expBias)) or ((eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)):
        return (sRet << 3) | 0b111
    elif eNorm <= -(expBias):
        eRet = 0
        mf = mf | 0x80
        mLenDelta -= eNorm + expBias - 1
        needRound = IsRoundOne(sRet, mf, mLenDelta)  # determine if need to carry
        mRet = mf >> mLenDelta
        if needRound:
            mRet += 1
    else:
        eRet = eNorm + expBias
        needRound = IsRoundOne(sRet, mf, mLenDelta)
        mRet = mf >> mLenDelta
        if needRound:
            mRet += 1
        if ((mRet & 0b100) != 0) and (needRound):
            eRet += 1
            mRet = 0

    if eRet >= 1:
        eRet = 1
    elif eRet == 0 and mRet == 0b100:
        eRet += 1
        mRet = 0

    return ((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3)


def trans_np_bfloat16_tensor_to_fp4_e2m1(in_tensor):
    import numpy as np

    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.uint8)
    in_tensor = in_tensor.reshape(multi_shape)

    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e2m1(in_tensor[i])

    out_tensor = out_tensor.astype(np.uint8)

    # 每两个fp4拼成一个uint8保存
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape // 2).astype(np.uint8)
    for i in range(multi_shape // 2):
        # fp4_tensor[i] = (out_tensor[i*2] << 4) | out_tensor[i*2+1]      # 按常规顺序保存b4
        fp4_tensor[i] = (out_tensor[i * 2 + 1] << 4) | out_tensor[
            i * 2
        ]  # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100

    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor


def trans_np_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    import numpy as np

    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.uint8)
    in_tensor = in_tensor.reshape(multi_shape)

    for i in range(multi_shape):
        out_tensor[i] = cvt_bfloat16_to_fp4_e1m2(in_tensor[i])

    out_tensor = out_tensor.astype(np.uint8)

    # 每两个fp4拼成一个uint8保存
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape // 2).astype(np.uint8)
    for i in range(multi_shape // 2):
        # fp4_tensor[i] = (out_tensor[i*2] << 4) | out_tensor[i*2+1] # 按常规顺序保存b4
        fp4_tensor[i] = (out_tensor[i * 2 + 1] << 4) | out_tensor[
            i * 2
        ]  # 按两两交叉顺序保存b4，比如b4两个数：0100 0010 存为b8后为0010 0100
    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor


def cvt_fp4_e2m1_to_bfloat16(x):
    Fp4e2m1ToBf16 = {
        "0": 0x0,
        "1": 0x3F00,
        "2": 0x3F80,
        "3": 0x3FC0,
        "4": 0x4000,
        "5": 0x4040,
        "6": 0x4080,
        "7": 0x40C0,
        "8": 0x8000,
        "9": 0xBF00,
        "10": 0xBF80,
        "11": 0xBFC0,
        "12": 0xC000,
        "13": 0xC040,
        "14": 0xC080,
        "15": 0xC0C0,
    }

    x = int(x)
    first_fp4val = x & 0x0F
    second_fp4val = (x >> 4) & 0x0F
    first_fp4str = str(first_fp4val)
    second_fp4str = str(second_fp4val)

    return Fp4e2m1ToBf16[first_fp4str], Fp4e2m1ToBf16[second_fp4str]


def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {
        "0": 0x0,
        "1": 0x3E80,
        "2": 0x3F00,
        "3": 0x3F40,
        "4": 0x3F80,
        "5": 0x3FA0,
        "6": 0x3FC0,
        "7": 0x3FE0,
        "8": 0x8000,
        "9": 0xBE80,
        "10": 0xBF00,
        "11": 0xBF40,
        "12": 0xBF80,
        "13": 0xBFA0,
        "14": 0xBFC0,
        "15": 0xBFE0,
    }

    x = int(x)
    first_fp4val = x & 0x0F
    second_fp4val = (x >> 4) & 0x0F
    first_fp4str = str(first_fp4val)
    second_fp4str = str(second_fp4val)

    return Fp4e1m2ToBf16[first_fp4str], Fp4e1m2ToBf16[second_fp4str]


def trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    import numpy as np

    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)

    # 1个uint8包含两个fp4, 先拆成两个uint8
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i * 2], bfloat16_tensor[i * 2 + 1] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
        fp32_tensor[i * 2] = struct.unpack("!f", struct.pack("!I", bfloat16_tensor[i * 2] << 16))[0]
        fp32_tensor[i * 2 + 1] = struct.unpack("!f", struct.pack("!I", bfloat16_tensor[i * 2 + 1] << 16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


def trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    import numpy as np

    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)

    # 1个uint8包含两个fp4, 先拆成两个uint8
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)

    for i in range(multi_shape):
        bfloat16_tensor[i * 2], bfloat16_tensor[i * 2 + 1] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
        fp32_tensor[i * 2] = struct.unpack("!f", struct.pack("!I", bfloat16_tensor[i * 2] << 16))[0]
        fp32_tensor[i * 2 + 1] = struct.unpack("!f", struct.pack("!I", bfloat16_tensor[i * 2 + 1] << 16))[0]

    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


def numpy_hifloat8():
    try:
        # noinspection PyUnresolvedReferences
        from en_dtypes import hifloat8

        ensure_en_dtypes_version("0.0.4")
        return hifloat8
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support hifloat8 dtype!!! Please install with `pip3 install en-dtypes`"
        ) from None
    except ImportError:
        raise RuntimeError(
            "Please upgrade en_dtypes to v0.0.4 at least to support hifloat8 dtype!!! "
            "Command is `pip3 install --upgrade en-dtypes`"
        ) from None


def resolve_custom_numpy_dtypes(container):
    """
    Convert custom numpy dtype strings (bfloat16/int4/fp8/fp4/hifloat) to numpy dtype objects.
    Supports nested structures (tuples within tuples).
    """
    if not container:
        return container
    special_dtypes = (
        "bfloat16",
        "int4",
        "float8_e5m2",
        "float8_e4m3fn",
        "float8_e8m0",
        "float4_e2m1",
        "float4_e1m2",
        "hifloat8",
        "hifloat4",
    )

    def _convert(item):
        if isinstance(item, (tuple, list)):
            converted = [_convert(e) for e in item]
            return tuple(converted) if isinstance(item, tuple) else list(converted)
        if isinstance(item, str):
            for sd in special_dtypes:
                if sd == item:
                    return eval(f"numpy_{sd}()")
        return item

    return _convert(container)


def pack_4bits(src: numpy.ndarray):
    """
    Pack int4/float4 numpy array (each int4 number stored in one byte actually)
    to be continuous bytes stored in uint8
    """
    if not is_4bit_dtype(src.dtype):
        raise RuntimeError("Dtype of source tensor only support int4/float4")
    pack_size = 2
    shift = numpy.array([0, 4], dtype=numpy.uint8)
    array = src
    if src.size % pack_size != 0:
        array = numpy.pad(src.flatten(), (0, pack_size - src.size % pack_size), mode="constant")
    reshaped = array.reshape([-1, 2])
    # bitwise_and is for arm
    out = numpy.sum(numpy.bitwise_and(reshaped.view(numpy.uint8), 0b00001111) << shift, axis=1, dtype=numpy.uint8)
    return out


def unpack_4bits(src: numpy.ndarray, dst_dtype):
    """
    Unpack uint8 numpy array to int4 array
    """
    if isinstance(dst_dtype, str):
        dst_dtype = eval(f"numpy_{dst_dtype}()")
    shift = numpy.array([0, 4], dtype=numpy.uint8)
    return numpy.bitwise_and(src.reshape([-1, 1]) >> shift, 0b00001111).view(dst_dtype).reshape([-1])


def encode_float8_e8m0(fp_array: numpy.ndarray):
    if not isinstance(fp_array, numpy.ndarray):
        raise NotImplementedError("only support numpy array.")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if "float16" == fp_array.dtype.name:
        fp_array = fp_array.astype("float32")
    if "float32" == fp_array.dtype.name:
        uint_array = fp_array.view(numpy.uint32)
        uint_array = (uint_array << 1) >> 24
    else:  # bfloat16
        uint_array = fp_array.view(numpy.uint16)
        uint_array = (uint_array << 1) >> 8
    return uint_array.astype(numpy_float8_e8m0())


def normalize_to_tf_dtype(np_array: numpy.ndarray):
    import tensorflow as tf

    if np_array.dtype.name == "bfloat16" and np_array.dtype.type != tf.bfloat16.as_numpy_dtype.dtype.type:
        return np_array.view(tf.bfloat16.as_numpy_dtype)
    return np_array


def tf_dtype_revert(tf_tensor):
    if tf_tensor.dtype.name == "bfloat16":
        ttk_bf16 = numpy_bfloat16()
        if ttk_bf16.dtype.type != tf_tensor.dtype.type:
            return tf_tensor.view(ttk_bf16)
    return tf_tensor


def torch_dtype_conversion(container):
    # convert string dtype to torch dtype
    import torch

    return tuple([getattr(torch, c) if isinstance(c, str) else c for c in container])


def acl_to_torch_dtype(container):
    return torch_dtype_conversion([DATA_TYPE_INT_TO_STR[ad] for ad in container])


def numpy_to_torch_tensor(np_array: numpy.ndarray, is_complex32: bool = False):
    import torch

    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=numpy.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    elif is_4bit_dtype(np_dtype):
        if np_dtype == "int4":
            raise RuntimeError(f"Can only transfer numpy.ndarray to torch.Tensor with dtype [{np_dtype}]")
        torch_dtype_name = _NUMPY_TO_TORCH_4BIT_DTYPE.get(np_dtype)
        if torch_dtype_name is None or not hasattr(torch, torch_dtype_name):
            raise RuntimeError(f"Current pytorch version [{torch.__version__}] does not support [{np_dtype}].")
        packed = pack_4bits(np_array)
        return torch.from_numpy(packed).view(getattr(torch, torch_dtype_name))
    elif "float8" in np_dtype:
        if np_dtype not in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0"):
            raise RuntimeError(f"Dtype [{np_dtype}] is not supported to convert to torch.Tensor yet.")
        # numpy float8_e8m0 has no suffix; torch dtype is float8_e8m0fnu
        torch_dtype_name = {
            "float8_e4m3fn": "float8_e4m3fn",
            "float8_e5m2": "float8_e5m2",
            "float8_e8m0": "float8_e8m0fnu",
        }[np_dtype]
        if not hasattr(torch, torch_dtype_name):
            raise RuntimeError(
                f"Current pytorch version [{torch.__version__}] is too old. {torch_dtype_name} is not supported."
            )
        return torch.from_numpy(np_array.view(dtype=numpy.uint8)).view(getattr(torch, torch_dtype_name))
    elif is_complex32:
        if np_dtype != "float16":
            raise RuntimeError(f"Can only transfer numpy.float16 to torch.complex32 rather than {np_dtype}")
        if not hasattr(torch, "complex32"):
            raise RuntimeError(
                f"Current pytorch version [{torch.__version__}] is too old. Please update to at least v1.13.1"
            )
        ret = torch.from_numpy(np_array)
        return ret.view(torch.complex32)
    else:
        return torch.from_numpy(np_array)


def torch_to_numpy_tensor(torch_tensor) -> numpy.ndarray:
    import torch

    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    torch_dtype_str = str(torch_dtype)
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=numpy_bfloat16())
    elif "complex32" in torch_dtype_str:
        t_fp16 = torch_tensor.view(torch.float16)
        return t_fp16.numpy()
    elif "float8" in torch_dtype_str:
        np_func_suffix = torch_dtype_str.split(".")[-1].replace("fnu", "")
        np_dtype = eval(f"numpy_{np_func_suffix}()")
        np_uint8 = torch_tensor.view(torch.uint8).numpy()
        return np_uint8.view(np_dtype)
    elif is_4bit_dtype(torch_dtype_str):
        np_uint8 = torch_tensor.view(torch.uint8).numpy()
        torch_dtype_name = torch_dtype_str.split(".")[-1]
        for np_name, torch_name in _NUMPY_TO_TORCH_4BIT_DTYPE.items():
            if torch_name == torch_dtype_name:
                return unpack_4bits(np_uint8, eval(f"numpy_{np_name}()"))
        raise RuntimeError(f"Unsupported torch 4-bit dtype [{torch_dtype_str}]")
    else:
        return torch_tensor.numpy()


def _mx_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, block_size: int):
    fp_array = numpy.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = numpy.pad(fp_array, pad, "constant")
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple):
    # Undo tile reshaping
    fp_array = fp_array.reshape(padded_shape)
    # Undo padding
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = numpy.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(numpy.log2(max_norm))
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True)
    res = numpy.floor(numpy.log2(fp_abs_max.astype(numpy.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))) - ele_emax
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_nv(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    import numpy

    max_norm = get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True).astype(numpy.float32)
    s_fp32 = fp_abs_max / max_norm
    binary_ints = numpy.array(s_fp32.view(numpy.uint32))
    exponent_mask = numpy.uint32(0x7F800000)  # 二进制：01111111100000000000000000000000
    mantissa_mask = numpy.uint32(0x007FFFFF)  # 二进制：00000000011111111111111111111111
    # 提取指数部分并转换为uint16
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(numpy.int16)
    # 提取尾数部分并转换为float
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    # 2 ** 23 fp32的尾数位值0.5，即：二进制：0 00000000 10000000000000000000000
    condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
    exponents_int16 = numpy.where((condition_1 | condition_2), exponents_int16 + 1, exponents_int16)
    res = (exponents_int16 - 127).astype(numpy.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
    """
    For example:
    fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, -0.0, 0.0, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
    - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
    - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
    - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0,   0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
    - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
    """
    if round_mode in ("rint", "even"):  # tie to even(c language rint)
        fp_array = numpy.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
        sign = numpy.signbit(fp_array)
        rounded_abs = numpy.floor(numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
        fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity(c language floor)
        fp_array = numpy.floor(fp_array)
    elif round_mode == "ceil":  # round to positive infinity(c language ceil)
        fp_array = numpy.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero(c language truncation)
        fp_array = numpy.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(
    fp_array: numpy.ndarray, share_exp: numpy.ndarray, mx_ele_dtype: str, round_mode: str
):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r"e(\d+)m(\d+)", mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2**share_exp)
    private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))).astype(
        fp_array.dtype, copy=False
    )
    # The minimum representable exponent
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2**private_exp) * (2**mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2**mantissa_bits) * (2**private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    max_norm = get_dtype_range(mx_dtype)[1]
    numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def pad_to_even(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
    """
    在指定的 axis 上将 tensor 对齐到偶数长度（即按 2 对齐），不足补 0。

    参数:
        tensor (numpy.ndarray): 输入数组
        axis (int): 要对齐的轴（从 0 开始）

    返回:
        numpy.ndarray: 对齐后的数组
    """
    if not isinstance(tensor, numpy.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

    shape = tensor.shape
    length = shape[axis]

    # 如果已经是偶数，直接返回原数组
    if length % 2 == 0:
        return tensor

    # 构造 pad_width：仅对目标 axis 补一个 0
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)  # 在 axis 维度末尾补一个 0

    padded_tensor = numpy.pad(tensor, pad_width, mode="constant", constant_values=2**-127)
    return padded_tensor


def interleave(tensor: numpy.ndarray, axis: int, n_group: int = 2) -> numpy.ndarray:
    if not isinstance(tensor, numpy.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")
    # 获取目标轴的长度
    length = tensor.shape[axis]
    # 检查是否可整除
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")

    group_length = length // n_group  # 每组长度
    shape = list(tensor.shape)

    # 重塑形状：在目标轴后插入组维度
    new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
    reshaped = tensor.reshape(new_shape)

    # 构建转置顺序：交换组维度和组内维度
    transpose_order = (
        list(range(0, axis + 1))  # 目标轴之前的维度
        + list(range(axis + 2, len(new_shape)))
        + [
            axis + 1,
        ]
    )  # 后续维度

    # 执行转置
    transposed = reshaped.transpose(transpose_order)

    return transposed


def mx_quantize(
    fp_array: numpy.ndarray,
    mx_ele_dtype: str = "float4_e2m1",
    axis: int = -1,
    block_size: int = 32,
    round_mode: str = "rint",
    scale_alg: int = 0,
) -> tuple:
    """
    quantize BFP16/FP16/FP32 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16/FP32
    :parameter mx_ele_dtype: dtype of element in MX dtype.
               support float4_e2m1/float4_e1m2/float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :parameter scale_alg: The calculation method for scale.Support MxFP8(OCP , count 0) or MxFP8(nvidia-cuBLAS , count 1)
    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """
    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)
    # get mx scale exponents
    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    else:
        share_exp = _mx_calculate_share_exp_nv(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
    # share_exp is always float32
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    # Cube only supports even scales. need to pad zero.
    scale_array_pad = pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    scale_array = scale_array_pad.astype(numpy_float8_e8m0(), copy=False)

    return scale_array, ele_array


def _grouped_mx_undo_reshape_to_blocks(
    fp_array: numpy.ndarray, group_index: numpy.ndarray, axis: int, padded_group_index: list, padded_shape: tuple
) -> numpy.ndarray:
    """
    根据 group_index 和 padded_group_index 还原被分组补 Pad 的数组

    Args:
        fp_array: 输入数组（已分块和补 Pad）
        axis: 原始分组轴
        group_index: 原始分组索引（未补 Pad）
        padded_group_index: 补 Pad 后的分组索引
        padded_shape: 补 Pad 后的形状（含 expand_dims）

    Returns:
        restored_array: 还原后的原始数组（未补 Pad）
    """
    # Step 1: Undo reshape to blocks
    print(padded_shape)
    fp_array = fp_array.reshape(padded_shape)

    # Step 2: Remove the expanded dimension (axis+1)
    fp_array = numpy.squeeze(fp_array, axis=axis + 1)

    # Step 3: Split into padded groups
    split_indices = padded_group_index[:-1]  # 获取分割点（排除最后一个总长度）
    groups = numpy.split(fp_array, split_indices, axis=axis)

    # Step 4: Trim padding from each group
    trimmed_groups = []
    prev = 0
    for i, end in enumerate(group_index):
        # 计算原始组长度
        original_length = end - prev
        # 沿 axis 轴截取原始数据（去掉 Pad 部分）
        slices = [slice(None)] * fp_array.ndim
        slices[axis] = slice(0, original_length)
        trimmed_group = groups[i][tuple(slices)]
        trimmed_groups.append(trimmed_group)
        prev = end

    # Step 5: Concatenate trimmed groups
    restored_array = numpy.concatenate(trimmed_groups, axis=axis)
    return restored_array


def _grouped_mx_reshape_to_blocks(fp_array: numpy.ndarray, group_index: numpy.ndarray, axis: int, block_size: int):
    """
    根据 group_index 对 fp_array 按轴分组，每个组内按 block_size 对齐补 Pad

    Args:
        fp_array: 输入数组
        group_index: 分组索引，定义每个组的结束位置
        axis: 需要分组的轴
        block_size: 块大小，每个组内长度需对齐到此值的整数倍

    Returns:
        reshaped_array: 分块后的数组
        padded_group_index: 补 Pad 后的 group_index
        padded_shape: 补 Pad 后的形状（包含 expand_dims 后的维度）
    """
    # Step 1: Split into groups based on group_index
    groups = []
    padded_group_index = []
    prev = 0

    for end in group_index:
        # 沿 axis 轴切片
        slices = [slice(None)] * fp_array.ndim
        slices[axis] = slice(prev, end)
        group = fp_array[tuple(slices)]
        groups.append(group)
        prev = end

    # Step 2: Pad each group to align with block_size
    padded_groups = []
    padded_index = 0
    for group in groups:
        group_len = group.shape[axis]
        pad_size = (block_size - (group_len % block_size)) % block_size
        if pad_size > 0:
            # 构造 Pad 宽度，仅在 axis 轴补 Pad
            pad_width = [(0, 0)] * group.ndim
            pad_width[axis] = (0, pad_size)
            padded_group = numpy.pad(group, pad_width, mode="constant")
        else:
            padded_group = group
        padded_index = padded_index + group_len + pad_size
        padded_groups.append(padded_group)
        padded_group_index.append(padded_index)

    # Step 3: Concatenate all groups along axis
    padded_array = numpy.concatenate(padded_groups, axis=axis)
    # Step 4: Expand dimensions as in original function
    expanded_array = numpy.expand_dims(padded_array, axis=axis + 1)
    # Step 5: Reshape into blocks
    # 总块数 = 总长度（已对齐）// block_size
    total_blocks = padded_array.shape[axis] // block_size
    reshape = list(expanded_array.shape)
    reshape[axis] = total_blocks  # 替换原轴为块数
    reshape.insert(axis + 1, block_size)  # 插入块大小维度

    reshaped_array = expanded_array.reshape(reshape)
    padded_shape = expanded_array.shape  # 补 Pad 后的形状（包含 expand_dims）

    return reshaped_array, padded_group_index, padded_shape


def reshape_scale_array_pad(scale_array: numpy.ndarray, group_index: numpy.ndarray, axis):
    import math

    scale_array = numpy.squeeze(scale_array, 1)
    cur_idx = 0
    pre_element = 0
    for idx, element in enumerate(group_index):
        next_group_start = (element // 64 + idx + 1) * 2  # 下一个group的初始地址
        real_idx = cur_idx + math.ceil((element - pre_element) / 32)  # 实际计算到多少行
        pad_idx = math.ceil(real_idx / 2) * 2  # 需要pad到多少行
        zero_row = numpy.full((1, scale_array.shape[1]), 2**-127)
        one_row = numpy.full((1, scale_array.shape[1]), 1)
        for i in range(real_idx, pad_idx):
            scale_array = numpy.insert(scale_array, i, zero_row, axis=0)
        for i in range(pad_idx, next_group_start):
            scale_array = numpy.insert(scale_array, i, one_row, axis=0)
        pre_element = element  # 前一个element
        cur_idx = next_group_start  # 当前计算到多少行

    scale_array = (
        scale_array.reshape(int(scale_array.shape[0] / 2), 2, scale_array.shape[1])
        .transpose(0, 2, 1)
        .reshape(int(scale_array.shape[0] / 2), scale_array.shape[1], 2)
    )

    return scale_array


def grouped_mx_quantize(
    fp_array: numpy.ndarray,
    group_index: numpy.ndarray,
    mx_ele_dtype: str = "float8_e5m2",
    axis: int = -2,
    block_size: int = 32,
    round_mode: str = "rint",
) -> tuple:
    """
    quantize BFP16/FP16 to MX dtypes
    :parameter fp_array: input numpy array with dtype BFP16/FP16
    :parameter mx_ele_dtype: dtype of element in MX dtype.
               support float8_e4m3fn/float8_e5m2
    :parameter axis: specify the axis across which shared scales/exponents are calculated.
    :parameter block_size: each block_size shares the same mx scale along the axis
    :parameter round_mode: round mode. support rint/floor/round/nearest
    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """
    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16"):
        raise RuntimeError(f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    def is_non_reverse_order(arr):
        if len(arr) <= 1:
            return True  # 空数组或单元素数组视为逆序
        diff = numpy.diff(arr)
        return numpy.all(diff >= 0)

    if not is_non_reverse_order(group_index):
        raise RuntimeError("Input tensor group_index should be non-reverse order.")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    if axis != -2 and axis != 0:
        raise RuntimeError(f"Not support {axis} yet!")

    if group_index[-1] != fp_array.shape[axis]:
        raise RuntimeError("The last element of group_index should match the dimension size of the input x axis.")

    # padding & reshape to block_size
    fp_array, padded_group_index, padded_shape = _grouped_mx_reshape_to_blocks(fp_array, group_index, axis, block_size)
    # get mx scale exponents
    share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    # undo reshape
    ele_array = _grouped_mx_undo_reshape_to_blocks(ele_array, group_index, axis, padded_group_index, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp8 dtype
    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")
    # share_exp is always float32
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array = reshape_scale_array_pad(scale_array, group_index, axis)
    scale_array = scale_array.astype(numpy_float8_e8m0(), copy=False)

    return scale_array, ele_array


def fp32_to_hf32(torch_tensor):
    import torch

    data_hf32 = torch_tensor.numpy().view(numpy.int32)
    data_hf32 = numpy.right_shift(numpy.right_shift(data_hf32, 12) + 1, 1)
    data_hf32 = numpy.left_shift(data_hf32, 13)
    data_hf32 = data_hf32.view(numpy.float32)
    return torch.from_numpy(data_hf32)


def is_torch_native_dtype(dtype_name):
    """Check if a dtype name is natively supported by the current torch version.

    Uses ``hasattr(torch, dtype_name)`` to dynamically detect support,
    avoiding hardcoded blacklists that become stale as torch evolves.

    Note: ``torch.int4`` exists but is non-functional on CPU (cannot create
    valued tensors, no numpy interop), so it is explicitly excluded.

    ``float8_e8m0`` is an alias for ``torch.float8_e8m0fnu``; the canonical
    torch attribute name differs, so the alias is resolved before checking.
    """
    import torch

    name = str(dtype_name)
    if name == "int4":
        return False
    torch_alias = {"float8_e8m0": "float8_e8m0fnu"}.get(name)
    if torch_alias is not None:
        return hasattr(torch, torch_alias)
    return hasattr(torch, name)


_TF_DTYPE_MAP = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float64",
    "double": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
    "complex64": "complex64",
    "complex128": "complex128",
}


def str_to_tf_dtype(dtype_str: str):
    """Convert dtype string to tf.dtype object.

    Args:
        dtype_str: string like 'float16', 'int8', 'fp16', 'bf16', etc.

    Returns:
        tf.dtype object, or None if not recognized.
    """
    if not isinstance(dtype_str, str):
        return dtype_str
    import tensorflow as tf

    canonical = dtype_map.get(dtype_str, dtype_str)
    tf_name = _TF_DTYPE_MAP.get(canonical)
    if tf_name is None:
        return None
    return getattr(tf, tf_name, None)


def is_tf_native_dtype(dtype_name) -> bool:
    """Check if a dtype name is natively supported by TensorFlow."""
    canonical = dtype_map.get(str(dtype_name), str(dtype_name))
    return canonical in _TF_DTYPE_MAP


def np_as_strided_safe(base, shape, strides):
    """numpy as_strided that handles non-native dtypes (e.g. ml_dtypes.float8_e5m2).

    numpy.as_strided internally calls np.asarray which fails with
    ``TypeError: data type '<f1' not understood`` for certain third-party
    dtypes (kind='f' but internal type code conflicts with numpy reserved codes).
    Workaround: view the array as numpy.void with same itemsize, apply as_strided,
    then view back to original dtype.

    Native numpy dtypes (kind in 'biufcmMOSUV') skip the try/except overhead.
    Third-party dtypes with kind='V' work natively and also skip the fallback.
    Only the problematic case (kind='f' but non-standard, like ml_dtypes.float8_e5m2)
    triggers the void-proxy workaround.
    """
    from numpy.lib.stride_tricks import as_strided as _np_as_strided

    dtype = base.dtype
    if dtype.kind != "f" or dtype.char in ("f", "d", "e", "g"):
        return _np_as_strided(base, shape=shape, strides=strides)
    try:
        return _np_as_strided(base, shape=shape, strides=strides)
    except (TypeError, ValueError):
        proxy = numpy.dtype((numpy.void, dtype.itemsize))
        base_proxy = base.view(proxy)
        view_proxy = _np_as_strided(base_proxy, shape=shape, strides=strides)
        return view_proxy.view(dtype)
