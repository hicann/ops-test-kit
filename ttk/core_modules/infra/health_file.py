"""Backward-compat re-export. Real implementation moved to ttk.remote.health_file.

Used by the heartbeat subprocess (writer) and dispatch workers (reader). The
canonical home is now ``ttk/remote/health_file.py``; this shim keeps existing
importers (``from ttk.core_modules.infra.health_file import ...``) working during
the transition.
"""

from ttk.remote.health_file import (  # noqa: F401
    atomic_write_json,
    read_health_file,
)
