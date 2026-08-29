#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for ACL runtime initialization interoperability."""

import ctypes

import numpy
import pytest

from ttk.core_modules.aclnn.acl_interface import (
    ACL_ERROR_REPEAT_INITIALIZE,
    AclInterface,
)
from ttk.core_modules.runtime import RTSInterface
from ttk.utilities.dtypes import np_as_strided_safe


class FakeRtsInterface(RTSInterface):
    @classmethod
    def create(cls, captured):
        instance = cls.__new__(cls)
        instance.captured = captured
        instance.skip_teardown = True
        return instance

    def copy_nparray_to_hbm(self, array):
        self.captured["copied_array"] = array
        self.captured["flattened_array"] = self._flatten_numpy_array(array)
        return ctypes.c_void_p(0x1000)


class NumpyTensorAclInterface(AclInterface):
    @classmethod
    def create(cls, captured):
        instance = cls.__new__(cls)
        instance.captured = captured
        instance._rts_interface = FakeRtsInterface.create(captured)
        instance._acl_tensor_to_device_mem = {}
        instance._acl_tensors = set()
        return instance

    def _opbase_api_call_with_ptr_return(self, api_name, extra_log, *args):
        self.captured["api_name"] = api_name
        self.captured["extra_log"] = extra_log
        self.captured["args"] = args
        return ctypes.c_void_p(0x2000)

    def _on_exit(self):
        self.captured["closed"] = True


def make_numpy_tensor_device():
    captured = {}
    return NumpyTensorAclInterface.create(captured), captured


def _noop():
    pass


def test_parse_error_accepts_repeat_initialization():
    AclInterface.parse_error(
        ACL_ERROR_REPEAT_INITIALIZE,
        "aclInit",
        "runtime already initialized by a custom input callback",
        accepted_errors=(ACL_ERROR_REPEAT_INITIALIZE,),
    )


def test_parse_error_keeps_other_acl_errors_strict():
    with pytest.raises(RuntimeError, match="100003"):
        AclInterface.parse_error(
            100003,
            "aclInit",
            "unexpected initialization failure",
            accepted_errors=(ACL_ERROR_REPEAT_INITIALIZE,),
        )


def test_acl_init_tracks_external_runtime(monkeypatch):
    device = AclInterface.__new__(AclInterface)
    device._acl_inited = False
    device._owns_acl_runtime = False

    def _noop():
        pass

    device._on_exit = _noop
    call = {}

    def record_call(kind, api_name, extra_log, *args, accepted_errors=()):
        call.update(
            {
                "kind": kind,
                "api_name": api_name,
                "extra_log": extra_log,
                "args": args,
                "accepted_errors": accepted_errors,
            }
        )
        return ACL_ERROR_REPEAT_INITIALIZE

    monkeypatch.setattr(device, "_api_call", record_call)

    device._acl_init()

    assert device._acl_inited is True
    assert device._owns_acl_runtime is False
    assert call == {
        "kind": "ACL",
        "api_name": "aclInit",
        "extra_log": None,
        "args": (None,),
        "accepted_errors": (ACL_ERROR_REPEAT_INITIALIZE,),
    }


@pytest.mark.parametrize("owns_runtime, expected_reset_calls", [(False, 0), (True, 1)])
def test_reset_only_resets_owned_runtime_device(monkeypatch, owns_runtime, expected_reset_calls):
    device = AclInterface.__new__(AclInterface)
    device._owns_acl_runtime = owns_runtime
    device._device_id = 0
    device._rts_interface = None
    device._on_exit = _noop
    reset_calls = []

    monkeypatch.setattr(device, "_release_acl_memory", _noop)

    def _record_reset():
        reset_calls.append(True)

    monkeypatch.setattr(device, "_acl_reset_device", _record_reset)

    device.reset()

    assert len(reset_calls) == expected_reset_calls
    if not owns_runtime:
        assert device._device_id is None


@pytest.mark.parametrize("owns_runtime, expected_finalize_calls", [(False, 0), (True, 1)])
def test_finalize_only_finalizes_owned_runtime(monkeypatch, owns_runtime, expected_finalize_calls):
    device = AclInterface.__new__(AclInterface)
    device._acl_inited = True
    device._owns_acl_runtime = owns_runtime
    device._on_exit = _noop
    finalize_calls = []

    monkeypatch.setattr(
        device,
        "_acl_api_call",
        lambda api_name, extra_log, *args: finalize_calls.append(api_name),
    )

    device._acl_finalize()

    assert len(finalize_calls) == expected_finalize_calls
    assert device._acl_inited is False
    assert device._owns_acl_runtime is False


def test_create_acl_tensor_from_numpy_copies_complete_storage_with_offset():
    storage = numpy.arange(64, dtype=numpy.uint8).reshape(4, 16)
    view = np_as_strided_safe(storage.ravel()[3:], shape=(2, 2, 4), strides=(16, 4, 1))
    device, captured = make_numpy_tensor_device()

    device.create_acl_tensor(view, "ND", storage.shape)

    copied = captured["copied_array"]
    assert copied.flags.c_contiguous
    assert copied.nbytes == storage.nbytes
    assert numpy.array_equal(copied.reshape(storage.shape), storage)
    assert tuple(captured["args"][3]) == (16, 4, 1)
    assert captured["args"][4].value == 3


def test_create_acl_tensor_from_numpy_rejects_ambiguous_parent_storage():
    parent = numpy.arange(128, dtype=numpy.uint8)
    storage_shape = (4, 16)
    # The 93-byte tail and 128-byte parent are both larger than the declared storage.
    view = np_as_strided_safe(parent[35:], shape=(2, 2, 4), strides=(16, 4, 1))
    device, _ = make_numpy_tensor_device()

    with pytest.raises(ValueError, match="exact contiguous numpy storage"):
        device.create_acl_tensor(view, "ND", storage_shape)
