# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""SWITCHES config 相关字段（config_path / provider_filter）的 slot 契约与 pickle 往返。

force_cpu 的 slot/pickle 测试在 tests/cli/test_switches_args.py 中覆盖。
"""

import pickle

from ttk.utilities.classes import SWITCHES


def test_config_path_and_provider_filter_slot():
    """config_path / provider_filter slot 存在、默认 None、可赋值。"""
    sw = SWITCHES()
    assert sw.config_path is None
    assert sw.provider_filter is None
    sw.config_path = "/tmp/x.yaml"
    sw.provider_filter = "torch,tf"
    assert sw.config_path == "/tmp/x.yaml"
    assert sw.provider_filter == "torch,tf"


def test_config_fields_survive_worker_pickle():
    """config_path / provider_filter 经 pickle 往返不丢（forkserver 传 worker 保障）。"""
    sw = SWITCHES()
    sw.config_path = "/tmp/x.yaml"
    sw.provider_filter = "torch"
    revived = pickle.loads(pickle.dumps(sw))
    assert revived.config_path == "/tmp/x.yaml"
    assert revived.provider_filter == "torch"
