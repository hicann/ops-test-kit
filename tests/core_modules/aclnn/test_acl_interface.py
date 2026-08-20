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

import pytest

from ttk.core_modules.aclnn.acl_interface import (
    ACL_ERROR_REPEAT_INITIALIZE,
    AclInterface,
)


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
        call.update({
            "kind": kind,
            "api_name": api_name,
            "extra_log": extra_log,
            "args": args,
            "accepted_errors": accepted_errors,
        })
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
def test_reset_only_resets_owned_runtime_device(
        monkeypatch, owns_runtime, expected_reset_calls):
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
def test_finalize_only_finalizes_owned_runtime(
        monkeypatch, owns_runtime, expected_finalize_calls):
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
