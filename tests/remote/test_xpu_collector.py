# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""xpu_collector 测试：_format_xpu_metrics 格式化、pick_endpoint 选择、_select_run_specs 模式分发。"""

import pytest


@pytest.mark.parametrize(
    "raw, expected_status, has_perf",
    [
        (
            {
                "torch": {
                    "status": "PASS",
                    "api": "torch.add",
                    "outputs": [],
                    "perf": {"device_us": 120.0, "peak_memory_mb": 8.5},
                }
            },
            "PASS",
            True,
        ),
        ({"tf": {"status": "FAIL", "api": "tf.raw.ops.Add", "error": "import failed"}}, "FAIL", False),
    ],
    ids=["pass-with-perf", "fail-no-perf"],
)
def test_format_xpu_metrics(raw, expected_status, has_perf):
    """_format_xpu_metrics: PASS 带 perf / FAIL 不带 perf。"""
    from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics

    result = _format_xpu_metrics(raw)
    for provider in raw:
        assert result[provider]["status"] == expected_status
        if has_perf:
            assert "device_us" in result[provider]
        else:
            assert "device_us" not in result[provider]
