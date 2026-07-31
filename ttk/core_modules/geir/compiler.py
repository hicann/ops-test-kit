#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os
import re
import subprocess
from typing import Optional


def _resolve_header_mtime(header_name, include_dirs):
    """Locate *header_name* in include_dirs and return its mtime (float seconds).

    Returns None when the header cannot be found (CANN built-in merged headers
    like ops_proto_nn.h live under opp/built-in/op_graph/inc and are resolvable).
    """
    for d in include_dirs:
        p = os.path.join(d, header_name)
        if os.path.isfile(p):
            try:
                return os.path.getmtime(p)
            except OSError:
                return None
    return None


class GeirCompiler:
    def __init__(self, switches, build_dir=None):
        self._switches = switches
        self._build_dir = build_dir or os.path.join(getattr(switches, "root_path", os.getcwd()), "geir")
        os.makedirs(self._build_dir, exist_ok=True)

    @property
    def build_dir(self):
        return self._build_dir

    _PCH_TEMPLATE = (
        '#include "graph.h"\n'
        '#include "graph/operator.h"\n'
        '#include "graph/operator_reg.h"\n'
        '#include "types.h"\n'
        '#include "tensor.h"\n'
        '#include "ge_error_codes.h"\n'
        '#include "ge_api.h"\n'
        '#include "ge_prof.h"\n'
        '#include "{proto_file}"\n'
        "#include <vector>\n"
        "#include <cstdlib>\n"
        "#include <cstdio>\n"
        "#include <cstdint>\n"
        "#include <cstring>\n"
        "#include <map>\n"
        "#include <new>\n"
        "#include <string>\n"
        "#include <fstream>\n"
    )

    def _ensure_pch(self, proto_file, include_dirs):
        """Build (once, process-safe) and return (header_name, pch_dir) or None."""
        if not proto_file:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(proto_file))
        if not safe:
            return None

        pch_dir = os.path.join(getattr(self._switches, "root_path", os.getcwd()), "geir", ".pch")
        os.makedirs(pch_dir, exist_ok=True)
        pch_header = "ttk_pch_" + safe
        pch_src = os.path.join(pch_dir, pch_header)
        pch_gch = pch_src + ".gch"

        if not os.path.isfile(pch_src):
            tmp = pch_src + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(self._PCH_TEMPLATE.format(proto_file=proto_file))
            os.replace(tmp, pch_src)

        proto_mtime = _resolve_header_mtime(proto_file, include_dirs)

        import fcntl

        lock_path = os.path.join(pch_dir, pch_header + ".lock")
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            stamp_path = pch_src + ".stamp"
            if os.path.isfile(pch_gch) and os.path.isfile(stamp_path):
                try:
                    cached_mtime = float(open(stamp_path).read().strip())
                except (ValueError, OSError):
                    cached_mtime = -1.0
                if proto_mtime is not None and cached_mtime == proto_mtime:
                    return (pch_header, pch_dir)
                logging.info(
                    "PCH source changed (proto mtime %s != stamp %.3f), rebuilding",
                    proto_mtime if proto_mtime is not None else "unknown",
                    cached_mtime,
                )
                try:
                    os.remove(pch_gch)
                except OSError:
                    pass
            self._cleanup_stale_pch(pch_dir, pch_header, include_dirs)
            cmd = ["g++", "-std=c++17", "-O0", "-x", "c++-header", "-o", pch_gch, pch_src]
            for d in include_dirs:
                if os.path.isdir(d):
                    cmd.extend(["-I", d])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=pch_dir)
                if result.returncode != 0:
                    try:
                        os.remove(pch_gch)
                    except OSError:
                        pass
                    logging.warning(
                        "PCH build failed for %s, fallback to no-PCH:\n%s", proto_file, result.stderr[-500:]
                    )
                    return None
            except subprocess.TimeoutExpired:
                logging.warning("PCH build timed out for %s, fallback to no-PCH", proto_file)
                return None
            if proto_mtime is not None:
                try:
                    with open(stamp_path, "w") as sf:
                        sf.write(str(proto_mtime))
                except OSError:
                    pass
        return (pch_header, pch_dir)

    @staticmethod
    def _cleanup_stale_pch(pch_dir, current_header, include_dirs):
        """Remove PCH artifacts whose proto header no longer exists in include_dirs."""
        try:
            stale_bases = set()
            for entry in os.listdir(pch_dir):
                if not entry.startswith("ttk_pch_"):
                    continue
                base = entry
                for suffix in (".gch", ".stamp", ".lock"):
                    if base.endswith(suffix):
                        base = base[: -len(suffix)]
                        break
                if base != current_header:
                    stale_bases.add(base)
            for base in stale_bases:
                proto_name = base[len("ttk_pch_") :]
                if proto_name and _resolve_header_mtime(proto_name, include_dirs) is None:
                    for suffix in ("", ".gch", ".stamp", ".lock"):
                        try:
                            os.remove(os.path.join(pch_dir, base + suffix))
                        except OSError:
                            pass
        except OSError:
            pass

    def compile(
        self, source_path: str, binary_name: str = "ttk_geir_test", proto_file: Optional[str] = None
    ) -> Optional[str]:
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

        pch = self._ensure_pch(proto_file, include_dirs)
        cmd = ["g++", "-std=c++17", "-O0", "-o", binary_path, source_path]
        if pch:
            cmd.extend(["-include", pch[0], "-I", pch[1]])
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
