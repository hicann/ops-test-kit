# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""RandomData dtype 保持测试：小 tensor + range(-0,+0) 不得被提升成 float64。"""

import numpy as np
import pytest

from ttk.utilities.data import RandomData


def _missing_en_dtypes():
    try:
        import en_dtypes  # noqa: F401

        return False
    except ImportError:
        return True


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_small_tensor_zero_range_keeps_dtype(dtype):
    """size <= replace_count 走 concatenate 混入路径，dtype 必须保持声明值（防 float64 提升）。"""
    arr = RandomData(dtype, (2,), (-0.0, 0.0)).generate()
    assert str(arr.dtype) == dtype


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_small_tensor_mixed_range_keeps_dtype(dtype):
    """range 含多个边界值（replace_count>1）且 size<=replace_count 时 dtype 保持。"""
    arr = RandomData(dtype, (2,), (-1.0, 1.0)).generate()
    assert str(arr.dtype) == dtype


def test_large_tensor_zero_range_keeps_dtype():
    """size > replace_count 走按位赋值路径，dtype 保持（防回归）。"""
    arr = RandomData("bfloat16", (100,), (-0.0, 0.0)).generate()
    assert str(arr.dtype) == "bfloat16"


def test_empty_tensor_keeps_dtype():
    """size=0 由 size>0 守卫跳过混入，dtype 保持（防回归）。"""
    arr = RandomData("bfloat16", (0,), (-0.0, 0.0)).generate()
    assert str(arr.dtype) == "bfloat16"


def test_zero_range_values_all_zero():
    """range(-0,+0) 生成的值必须全为 ±0（混入语义不引入非零值）。"""
    arr = RandomData("float32", (2,), (-0.0, 0.0)).generate()
    assert np.all(arr == 0.0)


@pytest.mark.parametrize(
    ("dtype", "max_val"),
    [("float32", 3.4028234663852886e38), ("float16", 65504.0)],
)
def test_full_value_range_covers_multiple_magnitudes(dtype, max_val):
    """全值域应走指数分布法，覆盖多个量级而非全是极大数。

    回归 issue #135：uniform 在全值域下 97% 的值落在 1e37~1e38，
    小数几乎无法生成。改用指数分布后各量级概率均等。
    """
    np.random.seed(0)
    arr = RandomData(dtype, (5000,), (-max_val, max_val)).generate()
    assert str(arr.dtype) == dtype
    arr64 = arr.astype("float64")
    assert not np.any(np.isinf(arr64))
    assert not np.any(np.isnan(arr64))
    small = np.sum(np.abs(arr64) < 1e10)
    assert small > 500


def test_full_value_range_explicit_max_triggers_exponential():
    """显式写出 bf16 max 区间也应触发指数分布法。"""
    np.random.seed(1)
    rd = RandomData("float32", (3000,), (-3.3895313892515355e38, 3.3895313892515355e38))
    assert rd._is_full_value_range("float32", -3.3895313892515355e38, 3.3895313892515355e38)
    arr = rd.generate()
    arr64 = arr.astype("float64")
    assert not np.any(np.isinf(arr64))
    assert np.sum(np.abs(arr64) < 1e10) > 300


def test_narrow_range_does_not_trigger_exponential():
    """小范围区间不应触发指数分布法，保持 uniform 行为。"""
    assert not RandomData._is_full_value_range("float32", -1.0, 1.0)
    assert not RandomData._is_full_value_range("float32", -100.0, 100.0)
    assert not RandomData._is_full_value_range("int32", -2147483648, 2147483647)


@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "float64", "int32", "int64", "uint8", "bool"],
)
def test_normal_chunked_bitwise_equal(dtype):
    """normal 大 tensor 分块生成与单次全量生成逐位相同（含非整除尾块）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 1024
    rd = RandomData(dtype, (n,), (-1.0, 1.0))
    low, high = rd._digitize_inf_nan(-1.0, rd._dtype), rd._digitize_inf_nan(1.0, rd._dtype)

    from scipy.stats import truncnorm

    mean = (high + low) / 2
    sigma = (high - mean) / 3
    gen = truncnorm((low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma)

    np.random.seed(42)
    full = gen.rvs(n).astype(rd._dtype, copy=False)
    np.random.seed(42)
    chunked = rd._gen_normal_data(gen, rd._dtype, (n,))
    assert chunked.dtype == full.dtype
    assert np.array_equal(chunked, full)


def test_normal_small_tensor_keeps_single_shot_path():
    """elem_count <= 2*CHUNK_ELEMS 时保持单次全量路径（行为与改动前一致）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS
    rd = RandomData("float32", (n,), (-1.0, 1.0))
    low, high = rd._digitize_inf_nan(-1.0, rd._dtype), rd._digitize_inf_nan(1.0, rd._dtype)

    from scipy.stats import truncnorm

    mean = (high + low) / 2
    sigma = (high - mean) / 3
    gen = truncnorm((low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma)

    np.random.seed(42)
    full = gen.rvs(n).astype(rd._dtype, copy=False)
    np.random.seed(42)
    small = rd._gen_normal_data(gen, rd._dtype, (n,))
    assert np.array_equal(small, full)


@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int32",
        "int64",
        "uint8",
        "int4",
        "float8_e5m2",
        "float8_e4m3fn",
        pytest.param(
            "float4_e2m1", marks=pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
        ),
        pytest.param("hifloat8", marks=pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")),
    ],
)
def test_chunked_direct_write_matches_astype(dtype):
    """resolve 后的各 dtype：np.empty 预分配 + 分块直写 cast 与整块 astype 逐位一致。"""
    from ttk.utilities.dtypes import resolve_custom_numpy_dtypes

    resolved = resolve_custom_numpy_dtypes([dtype])[0]
    src = np.random.uniform(-1.0, 1.0, 4096)
    chunk = 1000
    out = np.empty(4096, dtype=resolved)
    for start in range(0, 4096, chunk):
        end = min(start + chunk, 4096)
        out[start:end] = src[start:end]
    assert np.array_equal(out, src.astype(resolved, copy=False))


def test_normal_chunked_generate_dtype_preserved():
    """generate 入口走分块路径后 dtype 保持声明值（防 float64 提升/降级）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 8
    arr = RandomData("bfloat16", (n,), (-1.0, 1.0)).generate(distribution="normal")
    assert str(arr.dtype) == "bfloat16"
