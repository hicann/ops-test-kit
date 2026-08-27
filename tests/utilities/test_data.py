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
