# ttk/test_spec/manager.py
from typing import Any, Optional

from .loader import SpecLoader
from .validator import validate as _validate


class TestSpecManager:
    """Operator test spec manager — load/inspect/validate spec classes.

    Usage:
        mgr = TestSpecManager(search_paths=["/path/to/specs"])
        cls = mgr.load("softmax_v2")
        if cls and mgr.has(cls, "golden"):
            golden_fn = mgr.get(cls, "golden")
    """

    def __init__(self, search_paths: list[str]):
        self._loader = SpecLoader(search_paths)

    # -- Loading --

    def load(self, op_name: str) -> Optional[type]:
        """Load spec class. __spec__ dict preferred, naming convention fallback. Cached."""
        return self._loader.load(op_name)

    # -- Inspection --

    def has(self, cls: type, name: str) -> bool:
        """Check if attribute exists (including inherited) and is not None.
        Distinguishes 'not defined' from 'defined as None'.
        Note: traverses MRO via hasattr — inherited attributes return True.
        """
        return hasattr(cls, name) and getattr(cls, name) is not None

    def get(self, cls: type, name: str, default: Any = None) -> Any:
        """Get attribute value. Returns default if missing or None."""
        value = getattr(cls, name, None)
        return value if value is not None else default

    # -- Convenience --

    def list_vendors(self, cls: type) -> list[str]:
        """List vendor keys in third_party dict.

        Returns:
            ["torch", "tf", "flash_attn"] or []
        """
        tp = self.get(cls, "third_party")
        if isinstance(tp, dict):
            return list(tp.keys())
        if isinstance(tp, str):
            return ["torch"]  # default inference: torch
        return []

    # -- Validation --

    def validate(self, cls: type) -> list[str]:
        """Shallow type check, returns warning list. Does not block."""
        return _validate(cls)

    # -- Cache management --

    def clear_cache(self):
        """Clear cache (for testing or hot reload)."""
        self._loader.clear_cache()
