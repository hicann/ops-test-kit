"""xpu-server — standalone-deployable remote XPU execution server.

Deployment boundary (authoritative):
  * This package (``ttk/remote/server/``) deploys STANDALONE on the XPU box and
    is COMPLETELY ttk-free and ml_dtypes-free:
      - no ``from ttk...`` / ``import ttk...`` imports at all;
      - bfloat16 is handled with ``numpy.int16`` ``view`` (bit-preserving) plus
        ``torch`` — never ml_dtypes;
      - only intra-package RELATIVE imports (``from .execution_container ...``,
        ``from . import executor``), so the directory can be copied/renamed and
        run as ``python -m <pkg>.xpu_server`` with no ttk hierarchy on path.
    Runtime deps: Python stdlib + numpy + torch (lazy-imported).

  * The CLIENT side (``ttk/remote/dispatcher.py`` and the rest of ``ttk/``)
    runs on the TTK worker and MAY freely use ml_dtypes and ttk modules — the
    ttk-free constraint applies to THIS server package only.
"""
