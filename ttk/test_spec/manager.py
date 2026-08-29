# ttk/test_spec/manager.py
from typing import Any, List, Optional, Set

from ..utilities import Singleton
from .loader import SpecLoader
from .validator import validate as _validate


class TestSpecManager(metaclass=Singleton):
    """Operator test spec manager — load/inspect/validate spec classes.

    Usage:
        mgr = TestSpecManager(search_paths=("/path/to/specs",))
        cls = mgr.load("softmax_v2")
        if cls and mgr.has(cls, "golden"):
            golden_fn = mgr.get(cls, "golden")
    """

    def __init__(self, search_paths):
        self._loader = SpecLoader(search_paths)
        self._validated: Set[int] = set()  # spec classes already validated (by id)

    # -- Loading --

    def load(self, op_name: str) -> Optional[type]:
        """Load spec class. __spec__ dict preferred, naming convention fallback. Cached.
        First load of a class triggers validate (raises InvalidSpecError on mismatch)."""
        cls = self._loader.load(op_name)
        if cls is not None and id(cls) not in self._validated:
            _validate(cls)  # raises before marking validated → retry re-validates
            self._validated.add(id(cls))
        return cls

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

    def list_vendors(self, cls: type) -> List[str]:
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

    def validate(self, cls: type) -> None:
        """Type check. Raises InvalidSpecError on mismatch."""
        _validate(cls)

    # -- Cache management --

    def clear_cache(self):
        """Clear cache (for testing or hot reload)."""
        self._loader.clear_cache()
        self._validated.clear()


def get_spec_attr(op_name: str, attr: str, plugin_path) -> Any:
    """Get an attribute from a TestSpec class by operator name.

    Args:
        op_name: operator name (e.g. "softmax_v2")
        attr: attribute name (e.g. "pre_compare", "compare", "tolerance")
        plugin_path: search paths (str/path/tuple/list) or None

    Returns:
        Attribute value (callable, dict, str, type, ...) or None if not found.
    """
    if not plugin_path:
        return None
    paths = tuple(plugin_path) if isinstance(plugin_path, (list, tuple)) else (plugin_path,)
    mgr = TestSpecManager(search_paths=paths)
    cls = mgr.load(op_name)
    if cls is None or not mgr.has(cls, attr):
        return None
    return mgr.get(cls, attr)


def get_spec_class_meta(op_name: str, plugin_path):
    """返回 spec 类元数据（供 XPU/远端 sync+实例化）。无 spec 或无 plugin_path → None。

    Returns:
        {"spec_file": <abs .py path or None>, "class_name": <cls.__name__>} or None
    """
    if not plugin_path:
        return None
    paths = tuple(plugin_path) if isinstance(plugin_path, (list, tuple)) else (plugin_path,)
    cls = TestSpecManager(search_paths=paths).load(op_name)
    if cls is None:
        return None
    return {"spec_file": getattr(cls, "__ttk_spec_file__", None), "class_name": cls.__name__}
