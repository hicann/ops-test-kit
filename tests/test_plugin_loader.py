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
Test cases for plugin loader
"""

import os
import tempfile
import shutil
from pathlib import Path

from ttk.core_modules.plugin_loader import get_plugin_function, get_available_operators


def test_builtin_golden_functions():
    """Test builtin golden functions"""
    print("=== Test builtin golden functions ===")
    
    available = get_available_operators("golden", "kernel")
    print(f"Available builtin golden operators: {sorted(available)}")
    
    test_operators = ["add", "relu", "gelu"]
    
    for op_name in test_operators:
        func, source = get_plugin_function(op_name, "golden", "kernel")
        if func:
            print(f"✓ {op_name}: loaded successfully from {source}")
        else:
            print(f"✗ {op_name}: not found")


def test_builtin_aclnn_functions():
    """Test builtin aclnn functions"""
    print("\n=== Test builtin aclnn functions ===")
    
    available = get_available_operators("golden", "aclnn")
    print(f"Available builtin aclnn operators: {sorted(available)}")
    
    test_operators = ["aclnnAdds", "aclnnMuls"]
    
    for op_name in test_operators:
        func, source = get_plugin_function(op_name, "golden", "aclnn")
        if func:
            print(f"✓ {op_name}: loaded successfully from {source}")
        else:
            print(f"✗ {op_name}: not found")


def test_custom_golden_functions():
    """Test custom golden functions"""
    print("\n=== Test custom golden functions ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_path = Path(tmpdir) / "custom_plugins"
        plugin_path.mkdir()
        
        test_file = plugin_path / "test_custom.py"
        test_file.write_text("""#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np

def test_custom_golden(x, **kwargs):
    return x * 2

__golden__ = {
    "kernel": {
        "test_custom": "test_custom_golden"
    }
}
""")
        
        available = get_available_operators("golden", "kernel", str(plugin_path))
        print(f"Available custom golden operators: {sorted(available)}")
        
        if "test_custom" in available:
            func, source = get_plugin_function("test_custom", "golden", "kernel", str(plugin_path))
            if func:
                print(f"✓ test_custom: loaded successfully from {source}")
            else:
                print(f"✗ test_custom: not found")
        else:
            print(f"✗ test_custom: not in available list")


def test_priority():
    """Test priority: custom > builtin"""
    print("\n=== Test priority: custom > builtin ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_path = Path(tmpdir) / "custom_plugins"
        plugin_path.mkdir()
        
        override_file = plugin_path / "override_relu.py"
        override_file.write_text("""#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np

def custom_relu_golden(x, **kwargs):
    return input_x + 1

__golden__ = {
    "kernel": {
        "relu": "custom_relu_golden"
    }
}
""")

        func, source = get_plugin_function("relu", "golden", "kernel", str(plugin_path))
        if func:
            print(f"✓ Custom relu loaded (overrides builtin) from {source}")
        else:
            print(f"✗ Failed to load custom relu")


def test_invalid_plugin_type():
    """Test invalid plugin type"""
    print("\n=== Test invalid plugin type ===")
    
    try:
        get_available_operators("invalid_type", "kernel")
        print("✗ Should have raised KeyError")
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {e}")


def test_invalid_golden_type():
    """Test invalid golden type"""
    print("\n=== Test invalid golden type ===")
    
    try:
        get_available_operators("golden", "invalid_type")
        print("✗ Should have raised KeyError")
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {e}")


def test_spec_priority_over_custom():
    """测试优先级：spec > custom > builtin"""
    print("\n=== Test priority: spec > custom > builtin ===")

    examples_path = str(Path(__file__).resolve().parent.parent / "ttk" / "test_spec" / "examples")

    func, source = get_plugin_function("add", "golden", "kernel", examples_path)
    assert func is not None, "应找到 add 的 spec golden"
    assert source == "spec", f"期望 source='spec'，实际 source='{source}'"
    print(f"✓ add: 从 {source} 加载（spec 优先）")


def test_spec_input():
    """测试 spec customize_inputs 通过 plugin_type='input'"""
    print("\n=== Test spec customize_inputs ===")

    examples_path = str(Path(__file__).resolve().parent.parent / "ttk" / "test_spec" / "examples")

    func, source = get_plugin_function("histogram_input", "input", "kernel", examples_path)
    assert func is not None, "应找到 histogram_input 的 spec customize_inputs"
    assert source == "spec", f"期望 source='spec'，实际 source='{source}'"
    print(f"✓ histogram_input: customize_inputs 从 {source} 加载")


def test_spec_fallback_to_builtin():
    """测试无 spec 时回退到 builtin"""
    print("\n=== Test spec fallback to builtin ===")

    examples_path = str(Path(__file__).resolve().parent.parent / "ttk" / "test_spec" / "examples")

    func, source = get_plugin_function("relu", "golden", "kernel", examples_path)
    if func is not None:
        assert source != "spec", "relu 不应匹配 spec"
        print(f"✓ relu: 回退到 {source}")
    else:
        print(f"✓ relu: 无 spec 也无 builtin — 返回 None")


if __name__ == "__main__":
    test_builtin_golden_functions()
    test_builtin_aclnn_functions()
    test_custom_golden_functions()
    test_priority()
    test_invalid_plugin_type()
    test_invalid_golden_type()
    
    print("\n=== All tests completed ===")
