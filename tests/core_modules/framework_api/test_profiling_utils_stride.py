#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for non-contiguous stride preservation via Backend.clone / to_device.

clone() and to_device(preserve_stride=True) must keep the original (possibly
non-contiguous) stride of a tensor, unlike .clone() / .to(device) / .npu()
which materialize non-contiguous views into contiguous tensors.
"""

import os

import numpy as np
import pytest
import torch

from ttk.core_modules.framework_api.backends.torch_backend import TorchBackend


def _make_non_contiguous(shape=(1, 128, 8, 4), stride=(16384, 64, 4, 1), dtype=torch.float32, device="cpu"):
    """Build a non-contiguous view via as_strided over a larger storage."""
    storage = torch.arange(
        max(stride[0] * shape[0], 1) if len(stride) == len(shape) else np.prod(shape),
        dtype=dtype,
        device=device,
    ).reshape(-1)
    return torch.as_strided(storage, shape, stride, 0)


class TestClonePreservingStride:
    def _backend(self):
        tb = TorchBackend()
        tb.torch_lib = "cpu"
        tb.profile = {}
        return tb

    def test_none_returns_none(self):
        assert self._backend().clone(None) is None

    def test_contiguous_falls_back_to_clone(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        out = self._backend().clone(t)
        assert out is not t
        assert torch.equal(out, t)
        assert out.is_contiguous()

    def test_non_contiguous_stride_preserved(self):
        t = _make_non_contiguous()
        assert not t.is_contiguous()
        out = self._backend().clone(t)
        assert out.stride() == t.stride()
        assert not out.is_contiguous()

    def test_non_contiguous_data_equal(self):
        t = _make_non_contiguous()
        out = self._backend().clone(t)
        assert torch.equal(out, t)

    def test_non_contiguous_bfloat16_stride_preserved(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.bfloat16)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        out = self._backend().clone(t)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()
        assert torch.equal(out, t)

    def test_clone_does_not_flatten(self):
        t = _make_non_contiguous()
        assert not t.is_contiguous()
        out = self._backend().clone(t)
        assert out.stride() == t.stride()
        assert out.shape == t.shape
        assert out.dtype == t.dtype

    def test_independent_storage(self):
        t = _make_non_contiguous()
        out = self._backend().clone(t)
        out[0, 0, 0, 0] = 999.0
        assert t[0, 0, 0, 0].item() != 999.0

    def test_clone_preserves_stride_gap_data(self):
        """Clone must copy the *entire* underlying storage, not just visible elements.

        ``empty_strided`` + ``copy_`` leaves stride-gap memory uninitialized, so
        an operator reading into gaps sees garbage that differs from the original.
        The clone's storage must be the same size as the original's and contain
        identical gap data.
        """
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        out = self._backend().clone(t)
        assert out.untyped_storage().size() == t.untyped_storage().size()
        gap_idx = 8200
        out_flat = torch.as_strided(
            torch.empty(0, dtype=out.dtype, device=out.device).set_(
                out.untyped_storage(), 0, (out.untyped_storage().size() // out.element_size(),), (1,)
            ),
            (out.untyped_storage().size() // out.element_size(),),
            (1,),
            0,
        )
        assert out_flat[gap_idx].item() == storage[gap_idx].item()

    def test_clone_preserves_nonzero_offset(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 128)
        out = self._backend().clone(t)
        assert out.stride() == t.stride()
        assert out.storage_offset() == t.storage_offset()
        assert torch.equal(out, t)


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
    """NPU: .npu() flattens non-contiguous; to_device(preserve_stride=True) must keep stride.

    Uses TorchBackend._to_device_preserving_stride directly (the method that
    NpuTorchBackend inherits) to avoid NpuTorchBackend.to_device's dtype-specific
    fast path which is covered by other integration tests.
    """

    def _move(self, tensor, dev_id=0):
        tb = TorchBackend()
        tb.torch_lib = "npu"
        tb.profile = {}
        return tb._to_device_preserving_stride(tensor, dev_id)

    @pytest.fixture(autouse=True)
    def _restore_ascend_env(self, monkeypatch):
        """conftest deletes ASCEND_* env vars; NPU copy_ needs them. Restore from snapshot."""
        for k, v in _ASCEND_ENV_SNAPSHOT.items():
            if v is not None:
                monkeypatch.setenv(k, v)
        yield

    def test_none_returns_none(self):
        assert self._move(None) is None

    def test_contiguous_uses_npu(self):
        t = torch.arange(12, dtype=torch.float32)
        out = self._move(t)
        assert out.is_contiguous()
        assert torch.equal(out.cpu(), t)
        assert out.device.type == "npu"

    def test_non_contiguous_stride_preserved(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        out = self._move(t)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()
        assert out.device.type == "npu"

    def test_non_contiguous_data_equal(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        out = self._move(t)
        assert torch.equal(out.cpu(), t)

    def test_non_contiguous_gap_data_equal(self):
        """to_device must copy the full storage so stride-gap data matches."""
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        out = self._move(t)
        assert out.untyped_storage().size() == t.untyped_storage().size()
        out_flat = torch.as_strided(
            torch.empty(0, dtype=out.dtype, device=out.device).set_(
                out.untyped_storage(), 0, (out.untyped_storage().size() // out.element_size(),), (1,)
            ),
            (out.untyped_storage().size() // out.element_size(),),
            (1,),
            0,
        )
        assert out_flat[4096].cpu().item() == storage[4096].item()

    def test_bfloat16_non_contiguous_stride_preserved(self):
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.bfloat16)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        assert not t.is_contiguous()
        out = self._move(t)
        assert out.stride() == (16384, 64, 4, 1)
        assert not out.is_contiguous()

    def test_plain_npu_flattens_non_contiguous(self):
        """Regression guard: plain .npu() DOES flatten — proves why we need the fix."""
        storage = torch.arange(2 * 256 * 8 * 4, dtype=torch.float32)
        t = torch.as_strided(storage, (1, 128, 8, 4), (16384, 64, 4, 1), 0)
        flat = t.npu()
        assert flat.is_contiguous()
        assert flat.stride() == (4096, 32, 4, 1)
