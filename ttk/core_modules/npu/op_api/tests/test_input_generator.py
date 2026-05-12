#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Test cases for InputGenerator.is_broadcast
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import torch
from ttk.core_modules.npu.op_api.input_generation import InputGenerator


def test_none_tensor():
    """Test None tensor returns False"""
    result = InputGenerator.is_broadcast(None)
    assert result is False, f"Expected False, got {result}"
    print("✓ None tensor")


def test_scalar_tensor():
    """Test 0-dim scalar tensor returns False"""
    tensor = torch.tensor(1.0)
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ Scalar tensor: shape={tensor.shape}, stride={tensor.stride()}")


def test_empty_tensor():
    """Test empty tensor returns False"""
    tensor = torch.tensor([])
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ Empty tensor: shape={tensor.shape}, stride={tensor.stride()}")


def test_1d_tensor():
    """Test 1D tensor returns False"""
    tensor = torch.zeros(32)
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ 1D tensor: shape={tensor.shape}, stride={tensor.stride()}")


def test_2d_contiguous():
    """Test 2D contiguous tensor returns False"""
    tensor = torch.zeros(16, 32)
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ 2D contiguous: shape={tensor.shape}, stride={tensor.stride()}")


def test_unsqueeze():
    """Test unsqueeze (size=1 dim) returns False"""
    tensor = torch.zeros(32).unsqueeze(0)  # [1, 32]
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ Unsqueeze: shape={tensor.shape}, stride={tensor.stride()}")


def test_squeeze():
    """Test squeeze returns False"""
    tensor = torch.zeros(1, 32).squeeze(0)  # [32]
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ Squeeze: shape={tensor.shape}, stride={tensor.stride()}")


def test_row_broadcast():
    """Test row broadcast [32] -> [16, 32] returns True"""
    storage = torch.zeros(32)
    tensor = storage.unsqueeze(0).expand(16, 32)  # stride=(0, 1)
    result = InputGenerator.is_broadcast(tensor)
    assert result is True, f"Expected True, got {result}"
    print(f"✓ Row broadcast: shape={tensor.shape}, stride={tensor.stride()}")


def test_col_broadcast():
    """Test column broadcast [16,1] -> [16, 32] returns True"""
    storage = torch.zeros(16, 1)
    tensor = storage.expand(16, 32)  # stride=(1, 0)
    result = InputGenerator.is_broadcast(tensor)
    assert result is True, f"Expected True, got {result}"
    print(f"✓ Col broadcast: shape={tensor.shape}, stride={tensor.stride()}")


def test_3d_broadcast():
    """Test 3D broadcast returns True"""
    storage = torch.zeros(1, 4, 8)
    tensor = storage.expand(2, 4, 8)  # stride=(0, 8, 1)
    result = InputGenerator.is_broadcast(tensor)
    assert result is True, f"Expected True, got {result}"
    print(f"✓ 3D broadcast: shape={tensor.shape}, stride={tensor.stride()}")


def test_3d_contiguous():
    """Test 3D contiguous tensor returns False"""
    tensor = torch.zeros(2, 4, 8)
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ 3D contiguous: shape={tensor.shape}, stride={tensor.stride()}")


def test_mixed_broadcast():
    """Test mixed broadcast (multiple dims with stride=0) returns True"""
    storage = torch.zeros(1, 4, 1)
    tensor = storage.expand(2, 4, 8)  # stride=(0, 1, 0)
    result = InputGenerator.is_broadcast(tensor)
    assert result is True, f"Expected True, got {result}"
    print(f"✓ Mixed broadcast: shape={tensor.shape}, stride={tensor.stride()}")


def test_permute():
    """Test permute returns False"""
    tensor = torch.zeros(4, 8).permute(1, 0)  # stride=(1, 8)
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ Permute: shape={tensor.shape}, stride={tensor.stride()}")


def test_slice():
    """Test slice returns False"""
    tensor = torch.zeros(4, 8)[:, ::2]  # stride=(8, 2)
    result = InputGenerator.is_broadcast(tensor)
    assert result is False, f"Expected False, got {result}"
    print(f"✓ Slice: shape={tensor.shape}, stride={tensor.stride()}")


def test_view_operations():
    """Test various view operations"""
    base = torch.zeros(32, 64)

    # reshape
    t1 = base.reshape(32, 64)
    assert InputGenerator.is_broadcast(t1) is False

    # contiguous after non-contiguous
    t2 = base.t().contiguous()
    assert InputGenerator.is_broadcast(t2) is False

    # narrow
    t3 = base.narrow(0, 0, 16)
    assert InputGenerator.is_broadcast(t3) is False

    print("✓ View operations (reshape, contiguous, narrow)")


def run_all_tests():
    """Run all test cases"""
    tests = [
        test_none_tensor,
        test_scalar_tensor,
        test_empty_tensor,
        test_1d_tensor,
        test_2d_contiguous,
        test_unsqueeze,
        test_squeeze,
        test_row_broadcast,
        test_col_broadcast,
        test_3d_broadcast,
        test_3d_contiguous,
        test_mixed_broadcast,
        test_permute,
        test_slice,
        test_view_operations,
    ]

    print("=== Testing InputGenerator.is_broadcast ===\n")

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
