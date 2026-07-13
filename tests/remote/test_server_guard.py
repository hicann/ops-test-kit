"""Static guard: server package deployment constraints.

Runs at every commit (fast, no server). Fails if someone accidentally:
  - adds a ttk.* import to the server (breaks standalone deployment);
  - adds an ml_dtypes import (server must be ml_dtypes-free);
  - adds an ungated import-torch (breaks torch-free for TF scenario).
"""
import pathlib

import pytest


SERVER_DIR = pathlib.Path("ttk/remote/server")


def test_server_pkg_is_ttk_free():
    """No from ttk.* / import ttk.* in the server package."""
    for py in sorted(SERVER_DIR.glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            s = line.lstrip()
            if s.startswith(("from ttk", "import ttk")):
                pytest.fail(f"{py.name}:{i} imports ttk — server must be ttk-free")


def test_server_pkg_is_ml_dtypes_free():
    """No ml_dtypes import in the server package."""
    for py in sorted(SERVER_DIR.glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            s = line.lstrip()
            if "ml_dtypes" in s and s.startswith(("import ", "from ")):
                pytest.fail(f"{py.name}:{i} imports ml_dtypes — server must be ml_dtypes-free")


def test_torch_imports_are_vendor_gated():
    """Every 'import torch' must sit under a provider=='torch' / torch_dev gate.

    This guarantees a TF request path never triggers a torch import.
    Heuristic: look back 5 lines for a gate condition. Recognizes the renamed
    torch_dev flag (was torch_cuda) plus the provider=='torch' pattern.
    """
    for py in sorted(SERVER_DIR.glob("*.py")):
        if py.name == "config.py":
            continue  # startup-only hardware detection, not per-request path
        lines = py.read_text().splitlines()
        for i, line in enumerate(lines):
            if "import torch" not in line.lstrip():
                continue
            context = "\n".join(lines[max(0, i - 5):i])
            gated = (("provider" in context and "torch" in context)
                     or "torch_dev" in context or "torch_cuda" in context)
            assert gated, (
                f"{py.name}:{i+1} 'import torch' without a provider=='torch' "
                f"or torch_dev gate within 5 lines above")
