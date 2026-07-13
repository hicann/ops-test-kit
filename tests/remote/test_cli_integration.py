"""Test simplified _handle_kernel: no --xpu override, no env mutation.

The --xpu CLI override mechanism and set_remote_config were removed (Task 7).
_handle_kernel now only does: args_to_switches → apply_kernel_args →
_validate_xpu_perf_precondition → run_with_switches. It must NOT write
TTK_XPU_ENDPOINTS / TTK_XPU_PROVIDER env vars (those were override-era).
"""
import argparse
from unittest.mock import patch, MagicMock


def test_handle_kernel_no_env_mutation(monkeypatch):
    """_handle_kernel must not write TTK_XPU_* env vars (override mechanism removed)."""
    monkeypatch.delenv("TTK_XPU_ENDPOINTS", raising=False)
    monkeypatch.delenv("TTK_XPU_PROVIDER", raising=False)

    from ttk.cli.kernel import _handle_kernel

    args = argparse.Namespace(
        input="test.csv",
        config=None,
        provider=None,
    )

    with patch("ttk.cli.kernel.args_to_switches") as mock_sw, \
         patch("ttk.cli.kernel.apply_kernel_args"), \
         patch("ttk.cli.kernel.run_with_switches"):
        mock_sw.return_value = MagicMock(xpu_perf=False)
        _handle_kernel(args)

    import os
    # override block removed → no env mutation
    assert "TTK_XPU_ENDPOINTS" not in os.environ
    assert "TTK_XPU_PROVIDER" not in os.environ
