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
data generator

Module constraints (do not violate in future changes):
- numpy-only: do NOT import torch/tensorflow (or any other framework) here, not
  even inside functions. TTK must work in environments with only numpy
  installed (e.g. tensorflow-only e2e without torch). A historical torch.rand
  branch broke this and was removed in the fix for issue #169.
- _sample_distinct_positions must stay O(count) in time and memory regardless
  of `total`. Replacing it with numpy.random.choice(total, count,
  replace=False) regresses to O(total) (a full permutation: ~61MB temp for an
  8M-element tensor, ~320MB for 40M) and, combined with a large per_count,
  re-introduces both the memory spike and the collision-driven retry storm.
  Benchmark reference: 8M tensor, 20k positions — choice ~167ms vs
  unique+randint ~4ms; 40M fp32 generate end-to-end 77.8s -> 0.18s.
"""

__all__ = ["RandomData", "fixed_np_array"]


# Standard Packages
import math
from typing import Union

import numpy

# Third-party Packages
from .container_utils import get
from .dtypes import get_dtype_range, resolve_custom_numpy_dtypes
from .math import is_negative_zero, is_positive_zero

DEFAULT_LOW = -2
DEFAULT_HIGH = 2
CHUNK_ELEMS = 4_000_000


def fixed_np_array(dtype, shape, init_value=1):
    """Create a numpy array filled with a fixed value, handling complex32 specially.

    complex32 is stored as float16 with an extra trailing dim [2] (real, imag).
    For complex32: real part = init_value, imag part = 0.
    """
    if "complex32" in str(dtype):
        c_shape = list(shape) + [1]
        real = numpy.full(c_shape, init_value, dtype=numpy.float16)
        imag = numpy.zeros(c_shape, dtype=numpy.float16)
        return numpy.concatenate((real, imag), axis=-1)
    return numpy.full(shape, init_value, dtype=dtype)


def _is_native_integer(dtype) -> bool:
    """是否为 numpy 原生整型(int8..int64 / uint8..uint64)。

    自定义窄类型(int4/uint1 等)不在此列——它们没有原生 RNG，仍走 float64 路径。
    """
    try:
        return numpy.issubdtype(numpy.dtype(dtype), numpy.integer)
    except TypeError:
        return False


class RandomData:
    def __init__(self, dtype: str, shape: Union[list, tuple], data_range: Union[list, tuple]):
        self._dtype = resolve_custom_numpy_dtypes([dtype])[0]
        self._shape = list(shape)
        self._data_range = self._replace_none_in_data_range(data_range)

    @property
    def data_range(self):
        return self._data_range

    def generate(self, distribution: str = "uniform"):
        if self._dtype == "uint1":
            np_uint8 = self._random("uint8", self._shape, distribution=distribution)
            np_bool = np_uint8.astype("bool", copy=False)
            np_array = numpy.packbits(np_bool)
        elif "complex" in str(self._dtype):
            if self._dtype == "complex32":
                shape = list(self._shape) + [1]
                real = self._random("float16", shape, distribution=distribution)
                imag = self._random("float16", shape, distribution=distribution, is_complex_imag=True)
                np_array = numpy.concatenate((real, imag), axis=-1)
            else:  # complex64 / complex128
                bits = int(str(self._dtype)[7:])
                fp_dtype = f"float{bits // 2}"
                real = self._random(fp_dtype, self._shape, distribution=distribution)
                imag = self._random(fp_dtype, self._shape, distribution=distribution, is_complex_imag=True)
                np_array = numpy.array(real + imag * 1j)
        elif "e8m0" in str(self._dtype):
            # float8_e8m0 only represents positive scale powers; generate a positive range.
            low = float(self._data_range[0])
            high = float(self._data_range[1])
            if not numpy.isfinite(low) or low <= 0:
                low = 2**-3
            if not numpy.isfinite(high) or high <= low:
                high = 2**7
            f32 = self._gen_uniform_data(low, high, "float32", self._shape)
            from .dtypes import numpy_float8_e8m0

            np_array = f32.astype(numpy_float8_e8m0())
        else:
            np_array = self._random(self._dtype, self._shape, distribution=distribution)
        return np_array

    @staticmethod
    def _replace_none_in_data_range(data_range: Union[list, tuple]) -> list:
        low, high = get(data_range, 0), get(data_range, 1)
        if low is None:
            low = DEFAULT_LOW
        if high is None:
            high = DEFAULT_HIGH
        return [low, high, *data_range[2:]]

    @staticmethod
    def _digitize_inf_nan(r, dtype: str):
        tmp = numpy.array([r], dtype="float64")
        if numpy.isinf(tmp):
            dtype_range = get_dtype_range(dtype)
            return dtype_range[0] if numpy.isneginf(tmp) else dtype_range[1]
        if numpy.isnan(tmp):
            return 0
        return tmp[0]

    def _get_must_contain_dataset(self, dtype, is_complex_imag: bool) -> tuple:
        # border value will always be included
        replace_list = list(set(self._data_range))
        if _is_native_integer(dtype):
            # 整型端点不经 float64：INT64_MAX 经 float64 会舍入到 2^63 再溢出成 INT64_MIN，
            # 声明的区间端点会拿到符号相反的值。
            info = numpy.iinfo(dtype)
            clamped = [
                min(max(int(x), int(info.min)), int(info.max)) for x in replace_list if numpy.isfinite(numpy.float64(x))
            ]
            replace_np_array = numpy.array(clamped, dtype=dtype)
        else:
            replace_np_array = numpy.array(replace_list, dtype="float64").astype(dtype)
        # remove duplicate 0 in replace_np_array
        replace_list = list(set(replace_np_array))
        if "float" in str(dtype):
            # handle -float(0) & float(0)
            positive_zero = [is_positive_zero(x) for x in self._data_range]
            negative_zero = [is_negative_zero(x) for x in self._data_range]
            if any(positive_zero) and any(negative_zero):
                replace_list[replace_list.index(0)] = 0
                replace_list.append(-float(0))
        if is_complex_imag:
            # remove inf/nan
            replace_list = [x for x in replace_list if numpy.isfinite(x)]
        return tuple(replace_list)

    @staticmethod
    def _is_full_value_range(dtype, low, high) -> bool:
        s = str(dtype)
        if "float" not in s and "double" not in s and "bfloat" not in s:
            return False
        try:
            dtype_min, dtype_max = get_dtype_range(dtype)
        except Exception:
            return False
        max_abs = max(abs(low), abs(high))
        dtype_max_abs = max(abs(dtype_min), abs(dtype_max))
        if dtype_max_abs == 0:
            return False
        return max_abs >= dtype_max_abs * 0.99 and low < 0 < high

    @staticmethod
    def _gen_exponential_data(dtype, shape) -> numpy.ndarray:
        try:
            info = numpy.finfo(dtype)
            max_v = float(info.max)
            tiny = float(info.tiny)
        except (TypeError, ValueError):
            _, max_v = get_dtype_range(dtype)
            s = str(dtype)
            if "float16" in s:
                tiny = float(numpy.finfo(numpy.float16).tiny)
            elif "bfloat" in s:
                tiny = float(numpy.finfo(numpy.float32).tiny)
            else:
                tiny = float(numpy.finfo(numpy.float64).tiny)
        low_exp = int(numpy.log10(tiny)) + 1
        high_exp = int(numpy.log10(max_v)) + 1
        elem_count = int(numpy.prod(shape)) if numpy.ndim(shape) else 1

        # 全尺寸路径只对小张量保留：uniform/randint/power/multiply 四个全尺寸 float64|int64
        # 临时量 = 32 倍元素数，dim0=2^30 的 fp32 用例要 34GB，容器 32G 必 OOM，主进程被
        # 打掉后 heartbeat 报 "Parent died (ppid->1)"、rc 仍是 0，跑批**静默截断**(issue #169
        # 同类问题，当时漏了这条路径)。阈值以下一个元素都不改，既有用例逐位不变。
        if elem_count <= 2 * CHUNK_ELEMS:
            arr = numpy.random.uniform(low=-1.0, high=1.0, size=shape)
            if low_exp >= high_exp:
                return arr.astype(dtype, copy=False)
            arr_exp = numpy.random.randint(low=low_exp, high=high_exp, size=shape)
            arr_pow = numpy.power(10.0, arr_exp.astype(numpy.float64))
            result = numpy.multiply(arr, arr_pow)
            return result.astype(dtype, copy=False)

        # 大张量：分块算进预分配的目标类型缓冲，峰值 = 输出本身 + 一个分块的 float64 临时量。
        # 注意分块后 RandomState 的消费次序由"先全量 uniform 再全量 randint"变成逐块交替，
        # 取到的随机值与全尺寸路径不同(分布与"同 seed 同数据"的确定性都不变)。这条路径原本
        # 100% OOM、一条结果都出不来，不存在需要对齐的历史基线。
        out = numpy.empty(shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, elem_count, CHUNK_ELEMS):
            end = min(start + CHUNK_ELEMS, elem_count)
            chunk = numpy.random.uniform(low=-1.0, high=1.0, size=end - start)
            if low_exp < high_exp:
                chunk_exp = numpy.random.randint(low=low_exp, high=high_exp, size=end - start)
                chunk = numpy.multiply(chunk, numpy.power(10.0, chunk_exp.astype(numpy.float64)))
            flat[start:end] = chunk
        return out

    @staticmethod
    def _gen_normal_data(gen, dtype, shape):
        """Generate truncated-normal samples into a pre-allocated typed buffer chunk by chunk.

        gen.rvs consumes one numpy RandomState stream element by element, so the chunked
        output is bitwise identical to a single full-size rvs(...).astype(dtype) call.
        """
        elem_count = int(numpy.prod(shape)) if numpy.ndim(shape) else 1
        if elem_count <= 2 * CHUNK_ELEMS:
            return gen.rvs(shape).astype(dtype, copy=False)
        out = numpy.empty(shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, elem_count, CHUNK_ELEMS):
            end = min(start + CHUNK_ELEMS, elem_count)
            flat[start:end] = gen.rvs(end - start)
        return out

    @staticmethod
    def _gen_uniform_data(low, high, dtype, shape):
        """Generate uniform samples into a pre-allocated typed buffer chunk by chunk.

        numpy.random.uniform consumes the RandomState stream element by element
        (low + (high-low) * next_double), so the chunked output is bitwise identical
        to a single full-size uniform(...).astype(dtype) call.
        """
        elem_count = int(numpy.prod(shape)) if numpy.ndim(shape) else 1
        if elem_count <= 2 * CHUNK_ELEMS:
            return numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
        out = numpy.empty(shape, dtype=dtype)
        flat = out.ravel()
        for start in range(0, elem_count, CHUNK_ELEMS):
            end = min(start + CHUNK_ELEMS, elem_count)
            flat[start:end] = numpy.random.uniform(low, high, end - start)
        return out

    @staticmethod
    def _gen_integer_data(low, high, dtype, shape):
        """原生整型直接用整型 RNG 生成，不经 float64。

        其余 dtype 仍走 _gen_uniform_data(float64 再 astype)——那条路对 ≤32 位整型无损
        (2^32 < 2^53)、对浮点也无损(往窄了转)，且能覆盖 bfloat16/float8/int4 这些 numpy
        无原生 RNG 的类型。唯独 64 位整型的值域超出 float64 的 53 位尾数：
          * > 2^53 的值只能落在 2 的幂格点上——一个奇数都生成不出来；
          * INT64_MAX 舍入到 2^63，astype(int64) 后**静默溢出**成 INT64_MIN
            (只有一条 RuntimeWarning)，声明的区间端点拿到的是符号相反的值。
        """
        info = numpy.iinfo(dtype)
        lo = max(int(math.floor(low)), int(info.min))
        hi = min(int(math.ceil(high)), int(info.max))
        lo = min(lo, hi)
        generator = numpy.random.default_rng()
        return generator.integers(lo, hi, size=shape, endpoint=True, dtype=dtype)

    def _random(
        self, dtype, shape: Union[list, tuple], is_complex_imag: bool = False, distribution: str = "uniform"
    ) -> numpy.ndarray:
        low, high = self._data_range[:2]
        tmp = numpy.array([low, high], dtype="float64")
        if low == high or all(numpy.isnan(tmp)):
            if dtype == "hifloat4":
                dtype = "float4_e1m2"
            fill_val = 0 if is_complex_imag and not numpy.isfinite(low) else low
            array = numpy.full(shape, fill_value=fill_val, dtype=dtype)
        else:
            low = self._digitize_inf_nan(low, dtype)
            high = self._digitize_inf_nan(high, dtype)
            if distribution == "normal":
                from scipy.stats import truncnorm

                mean = (high + low) / 2
                sigma = (high - mean) / 3
                gen = truncnorm((low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma)
                array = self._gen_normal_data(gen, dtype, shape)
            elif dtype == "hifloat4":
                dtype = "float4_e1m2"
                array = numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
            elif dtype in ("float64", "double"):
                finfo = numpy.finfo(numpy.float64)
                low = max(low, finfo.min / 2)
                high = min(high, finfo.max / 2)
                array = numpy.random.uniform(low, high, shape).astype(dtype, copy=False)
            elif distribution == "uniform" and self._is_full_value_range(dtype, low, high):
                array = self._gen_exponential_data(dtype, shape)
            elif _is_native_integer(dtype):
                array = self._gen_integer_data(low, high, dtype, shape)
            else:
                array = self._gen_uniform_data(low, high, dtype, shape)
        return self._mix_expect_data(array, dtype, shape, is_complex_imag)

    @staticmethod
    def _sample_distinct_positions(total: int, count: int) -> numpy.ndarray:
        """Draw `count` distinct random positions from [0, total) in O(count) time/memory.

        PERFORMANCE CONTRACT — keep this O(count), never O(total):
        do not "simplify" this back to numpy.random.choice(total, count,
        replace=False): choice internally builds a full permutation of `total`
        (~61MB temp for 8M elements, ~320MB for 40M) where this stays at KB
        scale; see the module docstring for the benchmark history (issue #169).

        Collision-driven retries converge in ~count^2/2total expected rounds,
        which is negligible while per_count stays capped (2048) — keep the cap
        if you touch the per_count formula, otherwise the retry loop degenerates
        at high sampling density (k/n of 25% once measured 70s+).
        """
        positions = numpy.unique(numpy.random.randint(0, total, size=count))
        while positions.size < count:
            extra = numpy.random.randint(0, total, size=count - positions.size)
            positions = numpy.unique(numpy.concatenate((positions, extra)))
        # unique returns sorted positions; shuffle so that slicing per value does
        # not concentrate each must-contain value in one region of the tensor.
        return numpy.random.permutation(positions)

    def _mix_expect_data(
        self, np_array: numpy.ndarray, dtype, shape: Union[list, tuple], is_complex_imag: bool
    ) -> numpy.ndarray:
        replace_list = self._get_must_contain_dataset(dtype, is_complex_imag)
        replace_count = len(replace_list)
        if replace_count > 1 and np_array.size > 0:  # not only low == high
            if np_array.size <= replace_count:
                # replace_list 先落成 np_array.dtype：直接 concatenate typed 数组与
                # Python float 列表会被 numpy 提升成 float64（如 bf16 输入 + range(-0,+0)
                # 的小 tensor 被静默改成 double）。
                candidate = numpy.concatenate((np_array.reshape([-1]), numpy.array(replace_list, dtype=np_array.dtype)))
                idx = numpy.random.permutation(candidate.size)
                # use copy() to discard view of `candidate`
                np_array = candidate[idx][: np_array.size].reshape(shape).copy()
            else:
                per_count = (
                    1 if np_array.size <= 4 * replace_count else min(int(0.25 * np_array.size / replace_count), 2048)
                )
                replace_idx = self._sample_distinct_positions(np_array.size, per_count * replace_count)
                for idx, x in enumerate(replace_list):
                    start, end = idx * per_count, (idx + 1) * per_count
                    np_array.flat[replace_idx[start:end]] = x
        return np_array
