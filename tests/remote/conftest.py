"""Shared test config for tests/remote/.

Launch xpu-server as a STANDALONE package — the `server` package from ttk/remote/,
mirroring production where only the server layer is copied to the XPU box.

Putting ttk/remote on PYTHONPATH lets `python -m server.xpu_server` find the
package WITHOUT importing ttk/__init__.py. Importing the ttk namespace would pull
in ttk.core_modules.tbe_logging (which sets the root logger to NOTSET); forkserver
executor children inherit that NOTSET root, so tensorflow's vlog(1) gradient-
registration floods ~700 lines on first tf use.
"""
import os

_TTK_REMOTE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ttk", "remote"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_TTK_REMOTE, os.environ.get("PYTHONPATH", "")) if p)
