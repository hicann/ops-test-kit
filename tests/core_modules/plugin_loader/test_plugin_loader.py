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

import tempfile
from pathlib import Path

from ttk.core_modules.plugin_loader import get_plugin_function


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

        func = get_plugin_function("test_custom", "golden", "kernel", str(plugin_path))
        assert func is not None, "test_custom 应加载成功"
        print("✓ test_custom: loaded successfully")


def test_priority():
    """Test priority: custom plugin loads when provided"""
    print("\n=== Test priority: custom plugin ===")

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

        func = get_plugin_function("relu", "golden", "kernel", str(plugin_path))
        assert func is not None, "custom plugin 未加载"
        assert callable(func)
        assert func.__name__ == "custom_relu_golden"


def test_spec_priority_over_custom():
    """测试优先级：spec > custom"""
    print("\n=== Test priority: spec > custom ===")

    examples_path = str(Path(__file__).resolve().parent.parent.parent.parent / "ttk" / "test_spec" / "examples")

    func = get_plugin_function("add", "golden", "kernel", examples_path)
    assert func is not None, "应找到 add 的 spec golden"
    print("✓ add: spec golden 加载成功（spec 优先）")


def test_spec_input():
    """测试 spec customize_inputs 通过 plugin_type='input'"""
    print("\n=== Test spec customize_inputs ===")

    examples_path = str(Path(__file__).resolve().parent.parent.parent.parent / "ttk" / "test_spec" / "examples")

    func = get_plugin_function("histogram_input", "input", "kernel", examples_path)
    assert func is not None, "应找到 histogram_input 的 spec customize_inputs"
    print("✓ histogram_input: customize_inputs 加载成功")


def test_spec_fallback_to_builtin():
    """测试无 spec 时 plugin_loader 不再 builtin 兜底,返回 None"""
    print("\n=== Test spec fallback (no builtin in plugin_loader) ===")

    examples_path = str(Path(__file__).resolve().parent.parent.parent.parent / "ttk" / "test_spec" / "examples")

    func = get_plugin_function("relu", "golden", "kernel", examples_path)
    assert func is None, "relu 无 spec 时 plugin_loader 应返回 None(builtin 兜底已移到流程层)"


if __name__ == "__main__":
    test_custom_golden_functions()
    test_priority()
    test_spec_priority_over_custom()
    test_spec_input()
    test_spec_fallback_to_builtin()

    print("\n=== All tests completed ===")
