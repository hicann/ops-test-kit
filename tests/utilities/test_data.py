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
import torch

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


@pytest.mark.parametrize(
    "dtype",
    ["float16", "bfloat16", "float32", "float64", "int32", "int64", "uint8", "bool"],
)
def test_uniform_chunked_bitwise_equal(dtype):
    """uniform 大 tensor 分块生成与单次全量生成逐位相同（含非整除尾块）。"""
    from ttk.utilities.data import CHUNK_ELEMS
    from ttk.utilities.dtypes import resolve_custom_numpy_dtypes

    n = 2 * CHUNK_ELEMS + 1024
    resolved = resolve_custom_numpy_dtypes([dtype])[0]

    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, n).astype(resolved, copy=False)
    np.random.seed(42)
    chunked = RandomData._gen_uniform_data(-1.0, 1.0, resolved, (n,))
    assert chunked.dtype == full.dtype
    assert np.array_equal(chunked, full)


def test_uniform_small_tensor_keeps_single_shot_path():
    """elem_count <= 2*CHUNK_ELEMS 时保持单次全量路径（行为与改动前一致）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS

    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, n).astype("float32", copy=False)
    np.random.seed(42)
    small = RandomData._gen_uniform_data(-1.0, 1.0, "float32", (n,))
    assert np.array_equal(small, full)


def test_uniform_chunked_generate_dtype_preserved():
    """generate 入口走 uniform 分块路径后 dtype 保持声明值（防 float64 提升/降级）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 8
    arr = RandomData("float32", (n,), (-1.0, 1.0)).generate()
    assert str(arr.dtype) == "float32"


@pytest.mark.parametrize(
    "tail",
    [
        1,
        4_000_000 - 1,
    ],
)
def test_uniform_chunked_tail_boundaries_bitwise_equal(tail):
    """非 4M 对齐边界：最小分块入口（2xCHUNK+1）与最大尾块（CHUNK-1）均逐位相同。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + tail

    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, n).astype("float32", copy=False)
    np.random.seed(42)
    chunked = RandomData._gen_uniform_data(-1.0, 1.0, "float32", (n,))
    assert np.array_equal(chunked, full)


def test_uniform_chunked_exact_multiple_bitwise_equal():
    """整除边界（n = 3*CHUNK，无尾块）分块生成与单次全量逐位相同。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 3 * CHUNK_ELEMS

    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, n).astype("float32", copy=False)
    np.random.seed(42)
    chunked = RandomData._gen_uniform_data(-1.0, 1.0, "float32", (n,))
    assert np.array_equal(chunked, full)


def test_uniform_chunked_actually_splits(monkeypatch):
    """大 tensor 必须真实走分块路径：uniform 调用次数 >= 2（防阈值被误改后静默退化为单发）。"""
    import numpy

    from ttk.utilities.data import CHUNK_ELEMS

    calls = []
    real_uniform = numpy.random.uniform

    def counting_uniform(low, high, size):
        calls.append(size)
        return real_uniform(low, high, size)

    monkeypatch.setattr("ttk.utilities.data.numpy.random.uniform", counting_uniform)
    arr = RandomData("float32", (2 * CHUNK_ELEMS + 1,), (-1.0, 1.0)).generate()
    assert len(calls) >= 2
    assert arr.dtype == numpy.dtype("float32")  # dtype 保持


@pytest.mark.skipif(_missing_en_dtypes(), reason="en_dtypes is not installed")
def test_hifloat4_uses_packed_uint8_without_a_torch_dtype():
    """HIF4 logical storage remains packed uint8 when torch lacks E1M2."""
    from ttk.utilities.dtypes import numpy_hifloat4, numpy_to_torch_tensor

    logical = np.zeros((2, 4), dtype=numpy_hifloat4())
    logical.view(np.uint8)[:] = np.arange(logical.size, dtype=np.uint8).reshape(logical.shape)
    packed = numpy_to_torch_tensor(logical)
    expected_dtype = getattr(torch, "float4_e1m2fn_x2", None)

    assert tuple(packed.shape) == (2, 2)
    if isinstance(expected_dtype, torch.dtype):
        assert packed.dtype == expected_dtype
    else:
        assert packed.dtype == torch.uint8


def test_uniform_chunked_multidim_bitwise_equal():
    """多维 shape（总元素数非 4M 对齐）分块生成与单次全量逐位相同（ravel 写入顺序一致）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    shape = (3, CHUNK_ELEMS // 2 + 17, 5)  # 总数 = 1.5*CHUNK 非对齐，含多维 stride
    n = int(np.prod(shape))
    assert n > 2 * CHUNK_ELEMS

    np.random.seed(42)
    full = np.random.uniform(-1.0, 1.0, shape).astype("float32", copy=False)
    np.random.seed(42)
    chunked = RandomData._gen_uniform_data(-1.0, 1.0, "float32", shape)
    assert chunked.shape == shape
    assert np.array_equal(chunked, full)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_large_float16_uniform_is_numpy_seeded(dtype):
    """>10M float16/bfloat16 走 numpy 分块路径：同 seed 逐位可复现（torch RNG 不受 numpy seed 控制，必失败）。"""
    n = 10_000_001  # 超过旧 torch 分支阈值
    rd = RandomData(dtype, (n,), (-1.0, 1.0))

    np.random.seed(42)
    first = rd.generate()
    np.random.seed(42)
    second = rd.generate()

    assert str(first.dtype) == dtype
    assert np.array_equal(first, second)
    assert float(first.min()) >= -1.0
    assert float(first.max()) <= 1.0


def test_large_uniform_mix_expect_bitwise_reproducible():
    """>10M 混入走轻量采样路径：同 seed 整个 generate（含混入）逐位一致。"""
    n = 10_000_001
    rd = RandomData("float32", (n,), (-1.0, 1.0))

    np.random.seed(42)
    first = rd.generate()
    np.random.seed(42)
    second = rd.generate()

    assert np.array_equal(first, second)


def test_large_uniform_must_contain_values_present():
    """>10M 混入不再跳过：data_range 端点值与显式边界值在结果中出现（uniform 连续采样打不中端点，出现即混入证明）。"""
    n = 10_000_001
    rd = RandomData("float32", (n,), (-1.0, 1.0, 3.5))
    arr = rd.generate()

    assert (arr == -1.0).any()
    assert (arr == 1.0).any()
    assert (arr == 3.5).any()
    # 每值覆写次数封顶 2048（大 tensor 不再按 0.25% 比例膨胀，防高密度采样
    # 崩溃与边界值淹没正常分布；uniform 本身也可能量化命中端点，故用 >=）
    for v in (-1.0, 1.0, 3.5):
        assert (arr == v).sum() >= 2048


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
def test_integer_reaches_dtype_bounds(dtype):
    """整型能取到自身值域的上下界。

    原实现所有 dtype 统一走 uniform(float64) 再 astype：对 ≤32 位整型无损
    （2^32 < 2^53），但 64 位整型值域超出 float64 的 53 位尾数——INT64_MAX 舍入到
    2^63 后 astype 静默溢出成 INT64_MIN，声明的区间端点拿到的是符号相反的值。
    """
    info = np.iinfo(dtype)
    arr = RandomData(dtype, (512,), (int(info.min), int(info.max))).generate()

    assert arr.dtype == np.dtype(dtype)
    assert (arr == info.max).any()
    assert (arr == info.min).any()


@pytest.mark.parametrize("high", [1 << 56, 1 << 60, 1 << 62, (1 << 63) - 1])
def test_int64_high_range_is_not_grid_quantized(high):
    """int64 高值域能生成任意值，而非落在 2 的幂格点上。

    float64 在 2^60 量级的最小间隔是 2^(60-52)=256，经它生成的值全是 256 的倍数，
    一个奇数都产不出——这会让"索引取大值"的用例实际只覆盖到格点上的少数取值。
    """
    arr = RandomData("int64", (2048,), (0, high)).generate()

    assert (arr % 2 == 1).any(), "高值域生成不出奇数，说明仍经 float64 量化"
    assert (arr == high).any(), "区间上端点未命中"


def test_exponential_small_tensor_keeps_single_shot_path():
    """elem_count <= 2*CHUNK_ELEMS 时保持单次全量路径（逐位对齐改动前的实现）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS
    tiny = float(np.finfo("float32").tiny)
    max_v = float(np.finfo("float32").max)
    low_exp = int(np.log10(tiny)) + 1
    high_exp = int(np.log10(max_v)) + 1

    np.random.seed(42)
    arr = np.random.uniform(low=-1.0, high=1.0, size=(n,))
    arr_exp = np.random.randint(low=low_exp, high=high_exp, size=(n,))
    full = np.multiply(arr, np.power(10.0, arr_exp.astype(np.float64))).astype("float32", copy=False)

    np.random.seed(42)
    chunked = RandomData._gen_exponential_data("float32", (n,))
    assert chunked.dtype == full.dtype
    assert np.array_equal(chunked, full)


def test_exponential_chunked_actually_splits(monkeypatch):
    """大 tensor 必须真实走分块路径：uniform 调用次数 >= 2（防阈值被误改后静默退化为单发）。

    单发路径对每个全值域浮点 tensor 要开 uniform/randint/power/multiply 四个全尺寸
    float64|int64 临时量（32 倍元素数），dim0=2^30 时 34GB —— 容器内必 OOM，主进程被打掉
    后 rc 仍是 0，跑批静默截断。
    """
    import numpy

    from ttk.utilities.data import CHUNK_ELEMS

    calls = []
    real_uniform = numpy.random.uniform

    def counting_uniform(low, high, size):
        calls.append(size)
        return real_uniform(low, high, size)

    monkeypatch.setattr("ttk.utilities.data.numpy.random.uniform", counting_uniform)
    arr = RandomData._gen_exponential_data("float32", (2 * CHUNK_ELEMS + 1,))
    assert len(calls) >= 2
    assert max(calls) <= CHUNK_ELEMS  # 没有任何一次是全尺寸
    assert arr.dtype == numpy.dtype("float32")


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_exponential_chunked_dtype_and_magnitudes(dtype):
    """分块路径的输出 dtype/shape 正确，且仍覆盖多个数量级（不是退化成单一量级或全 0）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 1024
    arr = RandomData._gen_exponential_data(dtype, (n,))
    assert arr.dtype == np.dtype(dtype)
    assert arr.shape == (n,)

    nonzero = np.abs(arr[np.isfinite(arr) & (arr != 0)])
    assert nonzero.size > 0
    magnitudes = np.unique(np.floor(np.log10(nonzero.astype(np.float64))))
    assert magnitudes.size >= 3  # 指数分布应跨多个量级


def test_exponential_chunked_is_seed_reproducible():
    """同 seed 两次生成必须逐位相同（分块只改消费次序，不破坏确定性）。"""
    from ttk.utilities.data import CHUNK_ELEMS

    n = 2 * CHUNK_ELEMS + 7
    np.random.seed(20260917)
    first = RandomData._gen_exponential_data("float32", (n,))
    np.random.seed(20260917)
    second = RandomData._gen_exponential_data("float32", (n,))
    assert np.array_equal(first, second)
