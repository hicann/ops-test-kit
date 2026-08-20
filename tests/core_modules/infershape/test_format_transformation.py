# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.infershape.format_transformation: NPU format data transforms."""

import numpy as np
import pytest

from ttk.core_modules.infershape.format_transformation import (
    align_factor,
    determine_c0,
    fhd2nd,
    gen_axes_for_transpose,
    is_nchw_like,
    is_ndchw_like,
    is_transformable,
    nd_shape2fhd_shape,
    nd_shape2nz_shape,
    nd_to_fractal_nz,
    nz2nd,
    to_NC1HWC0,
    transform,
)


@pytest.mark.parametrize("dtype,expected", [
    ("float16", 16), ("int8", 32), ("int64", 4), ("bogus", 16),
])
def test_align_factor(dtype, expected):
    assert align_factor(dtype) == expected


@pytest.mark.parametrize("dtype,target,expected", [
    ("float16", [1, 2, 8, 8, 24], 24),
    ("int8", None, 32),
], ids=["target-shape", "fallback"])
def test_determine_c0(dtype, target, expected):
    assert determine_c0(dtype, target) == expected


def test_determine_c0_accepts_numpy_dtype():
    assert determine_c0(np.dtype("float16")) == 16


@pytest.mark.parametrize("shape,fmt,expected", [
    ((1, 32, 8, 8), "NCHW", True),
    ((1, 32, 8), "NCHW", False),
], ids=["ok", "bad-rank"])
def test_is_nchw_like(shape, fmt, expected):
    assert is_nchw_like(shape, fmt) is expected


@pytest.mark.parametrize("shape,fmt,expected", [
    ((1, 2, 32, 8, 8), "NCDHW", True),
    ((1, 32, 8, 8), "NCDHW", False),
], ids=["ok", "bad-rank"])
def test_is_ndchw_like(shape, fmt, expected):
    assert is_ndchw_like(shape, fmt) is expected


@pytest.mark.parametrize("shape,fmt,expected", [
    ((1, 32, 8, 8), "NCHW", (1, 2, 8, 8, 16)),
    ((1, 8, 8, 32), "NHWC", (1, 2, 8, 8, 16)),
])
def test_nd_shape2fhd_shape(shape, fmt, expected):
    assert nd_shape2fhd_shape(shape, fmt) == expected


def test_nd_shape2nz_shape():
    assert nd_shape2nz_shape((2, 3)) == (1, 1, 16, 16)


def test_nd_shape2fhd_rejects_non_nchw():
    with pytest.raises(RuntimeError, match="not NCHW-like"):
        nd_shape2fhd_shape((1, 2, 3), "NCH")


def test_nchw_to_nc1hwc0_and_back():
    data = np.arange(1 * 32 * 8 * 8, dtype=np.float16).reshape(1, 32, 8, 8)
    fractal = to_NC1HWC0(data, "NCHW")
    assert fractal.shape == (1, 2, 8, 8, 16)
    np.testing.assert_array_equal(fhd2nd(fractal, (1, 32, 8, 8), "NCHW"), data)


def test_nd_to_fractal_nz_and_back():
    data = np.arange(2 * 3, dtype=np.float16).reshape(2, 3)
    fractal = nd_to_fractal_nz(data)
    assert fractal.shape == (1, 1, 16, 16)
    np.testing.assert_array_equal(nz2nd(fractal, (2, 3)), data)


def test_nd_to_fractal_nz_4d_and_back():
    data = np.arange(2 * 5 * 4 * 7, dtype=np.float16).reshape(2, 5, 4, 7)
    fractal = nd_to_fractal_nz(data)
    np.testing.assert_array_equal(nz2nd(fractal, (2, 5, 4, 7)), data)


@pytest.mark.parametrize("ori,target,expected", [
    ("NCHW", "NC1HWC0", True),
    ("ND", "FRACTAL_NZ", True),
    ("BOGUS", "NC1HWC0", False),
    ("NCHW", "BOGUS", False),
])
def test_is_transformable(ori, target, expected):
    assert is_transformable(ori, target) is expected


def test_transform_returns_none_when_not_transformable():
    assert transform(np.zeros((1, 2, 3, 4), dtype=np.float16), "BOGUS", "NC1HWC0") is None


def test_transform_dispatches_to_NC1HWC0():
    data = np.arange(1 * 16 * 8 * 8, dtype=np.float16).reshape(1, 16, 8, 8)
    assert transform(data, "NCHW", "NC1HWC0").shape == (1, 1, 8, 8, 16)


def test_gen_axes_for_transpose():
    assert gen_axes_for_transpose(2, [2, 0, 1, 3]) == [0, 1, 4, 2, 3, 5]
