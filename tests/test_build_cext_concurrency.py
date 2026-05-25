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
Tests for _build_cext() flock + marker concurrency protection.

These tests do NOT require CANN environment. They mock subprocess.check_call
to simulate cmake builds and verify the locking/marker logic.
"""
import fcntl
import os
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, MagicMock


def _import_build_cext():
    """Import _build_cext from cext_loader module."""
    import importlib
    import sys
    mod_name = "ttk.utilities.cext_loader"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    return mod._build_cext


class TestBuildCextFastPath(unittest.TestCase):
    """When .so + marker already exist, _build_cext returns immediately."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.build_dir = os.path.join(self.tmpdir, "build")
        os.makedirs(self.build_dir, exist_ok=True)
        self.so_path = os.path.join(self.build_dir, "libttk_op_registry_accessor.so")
        self.marker_path = os.path.join(self.build_dir, ".build.libttk_op_registry_accessor.so.done")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("subprocess.check_call")
    def test_fast_path_skips_build(self, mock_check_call):
        """If .so and marker both exist, cmake is never called."""
        # Create fake .so and marker
        with open(self.so_path, "w") as f:
            f.write("fake")
        with open(self.marker_path, "w") as f:
            f.write("1")

        build_cext = _import_build_cext()
        build_cext(self.tmpdir, "libttk_op_registry_accessor.so")

        mock_check_call.assert_not_called()

    @patch("subprocess.check_call")
    def test_missing_marker_triggers_rebuild(self, mock_check_call):
        """If .so exists but marker is missing, rebuild occurs."""
        with open(self.so_path, "w") as f:
            f.write("fake")
        # No marker file

        build_cext = _import_build_cext()
        build_cext(self.tmpdir, "libttk_op_registry_accessor.so")

        # cmake should be called
        self.assertEqual(mock_check_call.call_count, 2)
        # marker should now exist
        self.assertTrue(os.path.isfile(self.marker_path))


class TestBuildCextMarkerAtomicity(unittest.TestCase):
    """Marker file is written atomically via rename."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.build_dir = os.path.join(self.tmpdir, "build")
        os.makedirs(self.build_dir, exist_ok=True)
        self.so_path = os.path.join(self.build_dir, "libttk_op_registry_accessor.so")
        self.marker_path = os.path.join(self.build_dir, ".build.libttk_op_registry_accessor.so.done")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("subprocess.check_call")
    def test_marker_created_after_build(self, mock_check_call):
        """Successful build creates .build_done marker."""
        def fake_build(*args, **kwargs):
            # Simulate cmake producing the .so
            with open(self.so_path, "w") as f:
                f.write("fake_so")

        mock_check_call.side_effect = fake_build

        build_cext = _import_build_cext()
        build_cext(self.tmpdir, "libttk_op_registry_accessor.so")

        self.assertTrue(os.path.isfile(self.marker_path))

    @patch("subprocess.check_call")
    def test_no_tmp_marker_left_after_success(self, mock_check_call):
        """No .tmp file is left after successful build."""
        def fake_build(*args, **kwargs):
            with open(self.so_path, "w") as f:
                f.write("fake_so")

        mock_check_call.side_effect = fake_build

        build_cext = _import_build_cext()
        build_cext(self.tmpdir, "libttk_op_registry_accessor.so")

        tmp_path = self.marker_path + ".tmp"
        self.assertFalse(os.path.isfile(tmp_path))

    @patch("subprocess.check_call")
    def test_build_fails_no_marker(self, mock_check_call):
        """If cmake fails, no marker is written."""
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "cmake")

        build_cext = _import_build_cext()
        with self.assertRaises(subprocess.CalledProcessError):
            build_cext(self.tmpdir, "libttk_op_registry_accessor.so")

        self.assertFalse(os.path.isfile(self.marker_path))


class TestBuildCextConcurrency(unittest.TestCase):
    """Multiple threads calling _build_cext simultaneously."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.build_dir = os.path.join(self.tmpdir, "build")
        os.makedirs(self.build_dir, exist_ok=True)
        self.so_path = os.path.join(self.build_dir, "libttk_op_registry_accessor.so")
        self.marker_path = os.path.join(self.build_dir, ".build.libttk_op_registry_accessor.so.done")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("subprocess.check_call")
    def test_concurrent_calls_only_one_builds(self, mock_check_call):
        """When N threads call _build_cext concurrently, cmake runs exactly once."""
        build_count = 0
        lock = threading.Lock()

        def fake_build(*args, **kwargs):
            nonlocal build_count
            with lock:
                build_count += 1
            # Simulate cmake producing the .so (with small delay for concurrency)
            time.sleep(0.05)
            with open(self.so_path, "w") as f:
                f.write("fake_so")

        mock_check_call.side_effect = fake_build

        build_cext = _import_build_cext()
        threads = []
        errors = []

        def worker():
            try:
                build_cext(self.tmpdir, "libttk_op_registry_accessor.so")
            except Exception as e:
                errors.append(e)

        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"Worker errors: {errors}")
        # cmake configure + cmake build = 2 calls, but only from 1 thread
        self.assertEqual(build_count, 2, f"Expected 2 cmake calls (configure + build), got {build_count}")
        self.assertTrue(os.path.isfile(self.marker_path))


if __name__ == "__main__":
    unittest.main()
