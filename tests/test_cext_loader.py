#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Tests for ttk/utilities/cext_loader.py
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


def _import_cext_loader():
    import importlib
    import sys
    mod_name = "ttk.utilities.cext_loader"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


class TestCextLoaderFindSo(unittest.TestCase):
    """_find_so returns path when .so exists, None otherwise."""

    def test_finds_in_whl_lib_dir(self):
        loader = _import_cext_loader()
        tmpdir = tempfile.mkdtemp()
        lib_dir = os.path.join(tmpdir, "ttk", "lib")
        os.makedirs(lib_dir, exist_ok=True)
        so_path = os.path.join(lib_dir, "libfake.so")
        with open(so_path, "w") as f:
            f.write("fake")

        result = loader._find_so(tmpdir, "libfake.so", "unused")
        self.assertEqual(result, so_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_finds_in_src_build_dir(self):
        loader = _import_cext_loader()
        tmpdir = tempfile.mkdtemp()
        build_dir = os.path.join(tmpdir, "csrc", "my_lib", "build")
        os.makedirs(build_dir, exist_ok=True)
        so_path = os.path.join(build_dir, "libfake.so")
        with open(so_path, "w") as f:
            f.write("fake")

        result = loader._find_so(tmpdir, "libfake.so", "my_lib")
        self.assertEqual(result, so_path)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_returns_none_when_missing(self):
        loader = _import_cext_loader()
        tmpdir = tempfile.mkdtemp()
        result = loader._find_so(tmpdir, "libnonexistent.so", "nope")
        self.assertIsNone(result)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_whl_priority_over_src(self):
        loader = _import_cext_loader()
        tmpdir = tempfile.mkdtemp()
        # Create both locations
        lib_dir = os.path.join(tmpdir, "ttk", "lib")
        os.makedirs(lib_dir, exist_ok=True)
        whl_path = os.path.join(lib_dir, "libfake.so")
        with open(whl_path, "w") as f:
            f.write("whl")

        build_dir = os.path.join(tmpdir, "csrc", "my_lib", "build")
        os.makedirs(build_dir, exist_ok=True)
        src_path = os.path.join(build_dir, "libfake.so")
        with open(src_path, "w") as f:
            f.write("src")

        result = loader._find_so(tmpdir, "libfake.so", "my_lib")
        self.assertEqual(result, whl_path)
        shutil.rmtree(tmpdir, ignore_errors=True)


class TestCextLoaderBuildCext(unittest.TestCase):
    """_build_cext with flock + marker logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.src_dir = os.path.join(self.tmpdir, "my_lib")
        os.makedirs(self.src_dir, exist_ok=True)
        self.so_name = "libfake.so"
        self.build_dir = os.path.join(self.src_dir, "build")
        self.so_path = os.path.join(self.build_dir, self.so_name)
        self.marker_path = os.path.join(self.build_dir, f".build.{self.so_name}.done")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("subprocess.check_call")
    def test_fast_path_skips_build(self, mock_check_call):
        os.makedirs(self.build_dir, exist_ok=True)
        with open(self.so_path, "w") as f:
            f.write("fake")
        with open(self.marker_path, "w") as f:
            f.write("1")

        loader = _import_cext_loader()
        loader._build_cext(self.src_dir, self.so_name)
        mock_check_call.assert_not_called()

    @patch("subprocess.check_call")
    def test_build_creates_marker(self, mock_check_call):
        def fake_build(*args, **kwargs):
            os.makedirs(self.build_dir, exist_ok=True)
            with open(self.so_path, "w") as f:
                f.write("fake")

        mock_check_call.side_effect = fake_build

        loader = _import_cext_loader()
        loader._build_cext(self.src_dir, self.so_name)

        self.assertTrue(os.path.isfile(self.marker_path))
        self.assertTrue(os.path.isfile(self.so_path))

    @patch("subprocess.check_call")
    def test_concurrent_builds_only_one(self, mock_check_call):
        build_count = 0
        lock = threading.Lock()

        def fake_build(*args, **kwargs):
            nonlocal build_count
            with lock:
                build_count += 1
            time.sleep(0.05)
            os.makedirs(self.build_dir, exist_ok=True)
            with open(self.so_path, "w") as f:
                f.write("fake")

        mock_check_call.side_effect = fake_build

        loader = _import_cext_loader()
        errors = []

        def worker():
            try:
                loader._build_cext(self.src_dir, self.so_name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"Worker errors: {errors}")
        self.assertEqual(build_count, 2)
        self.assertTrue(os.path.isfile(self.marker_path))

    @patch("subprocess.check_call")
    def test_stale_marker_removed(self, mock_check_call):
        os.makedirs(self.build_dir, exist_ok=True)
        # Stale marker without so file
        with open(self.marker_path, "w") as f:
            f.write("1")

        def fake_build(*args, **kwargs):
            with open(self.so_path, "w") as f:
                f.write("fake")

        mock_check_call.side_effect = fake_build

        loader = _import_cext_loader()
        loader._build_cext(self.src_dir, self.so_name)

        self.assertTrue(os.path.isfile(self.marker_path))


class TestCextLoaderLoadCext(unittest.TestCase):
    """load_cext raises FileNotFoundError when .so not found and no src."""

    def test_raises_when_not_found(self):
        loader = _import_cext_loader()
        with self.assertRaises(FileNotFoundError):
            loader.load_cext("libnonexistent.so", "nonexistent_dir")


if __name__ == "__main__":
    unittest.main()
