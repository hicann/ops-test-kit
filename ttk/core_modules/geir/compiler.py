#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os
import subprocess
from typing import Optional


class GeirCompiler:
    def __init__(self, switches, build_dir=None):
        self._switches = switches
        self._build_dir = build_dir or os.path.join(getattr(switches, "root_path", os.getcwd()), "geir")
        os.makedirs(self._build_dir, exist_ok=True)

    @property
    def build_dir(self):
        return self._build_dir

    def compile(self, source_path: str, binary_name: str = "ttk_geir_test") -> Optional[str]:
        from ttk._env import _find_ascend_root

        asc_path = _find_ascend_root()
        if not asc_path:
            raise RuntimeError("Ascend installation not found. Please check Ascend installation.")

        from ttk.utilities.platform import get_ascend_scene_info

        scene_os, scene_arch = get_ascend_scene_info()
        arch_dir = f"{scene_arch}-{scene_os}" if scene_arch and scene_os else "x86_64-linux"

        include_dirs = [
            os.path.join(asc_path, arch_dir, "include"),
            os.path.join(asc_path, arch_dir, "include", "graph"),
            os.path.join(asc_path, arch_dir, "include", "ge"),
            os.path.join(asc_path, arch_dir, "include", "external"),
            os.path.join(asc_path, "opp", "built-in", "op_proto", "inc"),
            os.path.join(asc_path, "opp", "built-in", "op_graph", "inc"),
        ]

        # Add custom and vendor op_proto/inc directories (for *_proto.h headers)
        from ttk.utilities.platform import get_opp_paths

        for copp in get_opp_paths("custom"):
            d = os.path.join(copp, "op_proto", "inc")
            if os.path.isdir(d):
                include_dirs.append(d)
        for vopp in get_opp_paths("vendor"):
            d = os.path.join(vopp, "op_proto", "inc")
            if os.path.isdir(d):
                include_dirs.append(d)

        lib_dirs = [
            os.path.join(asc_path, arch_dir, "lib64"),
            os.path.join(asc_path, arch_dir, "lib64", "stub"),
        ]
        libs = ["graph", "ge_runner", "graph_base", "ge_compiler", "msprofiler"]
        binary_path = os.path.join(self._build_dir, binary_name)

        cmd = ["g++", "-std=c++17", "-O2", "-o", binary_path, source_path]
        for d in include_dirs:
            if os.path.isdir(d):
                cmd.extend(["-I", d])
        for d in lib_dirs:
            if os.path.isdir(d):
                cmd.extend(["-L", d])
        for lib in libs:
            cmd.append(f"-l{lib}")

        logging.info(f"GEIR compile: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=self._build_dir)
            if result.returncode != 0:
                logging.error(f"GEIR compile failed:\n{result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logging.error("GEIR compile timed out (300s)")
            return None

        self._binary_path = binary_path
        return binary_path

    def cleanup(self, input_prefix=None, source_path=None):
        if input_prefix:
            import glob

            for f in glob.glob(f"{input_prefix}_*.bin"):
                try:
                    os.remove(f)
                except OSError:
                    pass
