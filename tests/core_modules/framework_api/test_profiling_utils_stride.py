#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for non-contiguous stride preservation in profiling_utils.

clone_preserving_stride and _to_device_preserving_stride must keep the
original (possibly non-contiguous) stride of a tensor, unlike .clone() /
.to(device) / .npu() which materialize non-contiguous views into contiguous
tensors.
"""
import os

import numpy as np
import pytest
import torch

from ttk.core_modules.framework_api.profiling_utils import (
    clone_preserving_stride,
    _to_device_preserving_stride,
)


def _make_non_contiguous(shape=(1, 128, 8, 4), stride=(16384, 64, 4, 1),
                         dtype=torch.float32, device="cpu"):
    """Build a non-contiguous view via as_strided over a larger storage."""
    storage = torch.arange(
        max(stride[0] * shape[0], 1) if len(stride) == len(shape) else np.prod(shape),
        dtype=dtype, device=device,
    ).reshape(-1)
    return torch.as_strided(storage, shape, stride, 0)


class TestClonePreservingStride:

    def test_none_returns_none(self):
        assert clone_preserving_stride(None) is None

    def test_contiguous_falls_back_to_clone(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        out = clone_preserving_stride(t)
        assert out is not t
        assert torch.equal(out, t)
        assert out.is_contiguous()

    def test_non_contiguous_stride_preserved(self):
        t = _make_non_contiguous()
        assert not t.is_contiguous()
        out = clone_preserving_stride(t)
        assert out.stride() == t.stride()
        assert not out.is_contiguous()

    def test_non_contiguous_data_equal(self):
        t = _make_non_contiguous()
        out = clone_preserving_stride(t)
        assert torch.equal(out, t)

    def test_non_contiguous_bfloat16_stride_preserved(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.bfloat16)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        out = clone_preserving_stride(t)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()
        assert torch.equal(out, t)

    def test_clone_does_not_flatten(self):
        t = _make_non_contiguous()
        assert not t.is_contiguous()
        out = clone_preserving_stride(t)
        assert out.stride() == t.stride()
        assert out.shape == t.shape
        assert out.dtype == t.dtype

    def test_independent_storage(self):
        t = _make_non_contiguous()
        out = clone_preserving_stride(t)
        out[0, 0, 0, 0] = 999.0
        assert t[0, 0, 0, 0].item() != 999.0


class _FakeBackend:
    """Minimal backend for _to_device_preserving_stride."""

    def __init__(self, torch_lib="npu"):
        self.torch_lib = torch_lib

    def to_device(self, tensor, dev_id=0):
        return getattr(tensor, self.torch_lib)(dev_id)


_npu_ok = False
try:
    import torch_npu
    torch_npu.npu.set_device(0)
    _npu_ok = torch_npu.npu.is_available()
except Exception:
    _npu_ok = False

_ASCEND_ENV_KEYS = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_ROOT")
_ASCEND_ENV_SNAPSHOT = {k: os.environ.get(k) for k in _ASCEND_ENV_KEYS}


@pytest.mark.skipif(not _npu_ok, reason="NPU not available")
class TestToDevicePreservingStrideNPU:
    """NPU: .npu() flattens non-contiguous; _to_device_preserving_stride must keep stride."""

    @pytest.fixture(autouse=True)
    def _restore_ascend_env(self, monkeypatch):
        """conftest deletes ASCEND_* env vars; NPU copy_ needs them. Restore from snapshot."""
        for k, v in _ASCEND_ENV_SNAPSHOT.items():
            if v is not None:
                monkeypatch.setenv(k, v)
        yield

    def test_none_returns_none(self):
        backend = _FakeBackend("npu")
        assert _to_device_preserving_stride(None, backend, 0) is None

    def test_contiguous_uses_npu(self):
        t = torch.arange(12, dtype=torch.float32)
        backend = _FakeBackend("npu")
        out = _to_device_preserving_stride(t, backend, 0)
        assert out.is_contiguous()
        assert torch.equal(out.cpu(), t)
        assert out.device.type == "npu"

    def test_non_contiguous_stride_preserved(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        backend = _FakeBackend("npu")
        out = _to_device_preserving_stride(t, backend, 0)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()
        assert out.device.type == "npu"

    def test_non_contiguous_data_equal(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        backend = _FakeBackend("npu")
        out = _to_device_preserving_stride(t, backend, 0)
        assert torch.equal(out.cpu(), t)

    def test_bfloat16_non_contiguous_stride_preserved(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.bfloat16)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        backend = _FakeBackend("npu")
        out = _to_device_preserving_stride(t, backend, 0)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()

    def test_plain_npu_flattens_non_contiguous(self):
        """Regression guard: plain .npu() DOES flatten — proves why we need the fix."""
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        flat = t.npu()
        assert flat.is_contiguous()
        assert flat.stride() == (4096, 32, 4, 1)
