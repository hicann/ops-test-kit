# ttk/test_spec/loader.py
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional, List, Dict


def _snake_to_pascal(name: str) -> str:
    """snake_case -> PascalCase: 'softmax_v2' -> 'SoftmaxV2'

    Limitation: does not handle already-PascalCase names (e.g. 'BatchMatMul'
    would become 'Batchmatmul'). Use __spec__ dict registration for
    those cases.
    """
    return "".join(part.capitalize() for part in name.split("_"))


class SpecLoader:
    """Spec file discovery, import and cache.

    Discovery priority: __spec__ dict > naming convention
    Multiple search paths: first match wins (same as sys.path)
    """

    def __init__(self, search_paths: List[str]):
        self._search_paths = [Path(p).resolve() for p in search_paths]
        self._cache: Dict[str, Optional[type]] = {}
        self._file_cache: Dict[Path, Optional[dict]] = {}  # __spec__ content cache
        self._loaded_modules: Dict[str, object] = {}  # track modules for cleanup

    def load(self, op_name: str) -> Optional[type]:
        """Load spec class. Returns class or None."""

        # 1. Cache hit
        if op_name in self._cache:
            return self._cache[op_name]

        # 2. Iterate files in discovery order
        for py_file in self._iter_py_files():
            cls = self._try_find_in_file(py_file, op_name)
            if cls is not None:
                self._cache[op_name] = cls
                return cls

        # 3. Not found
        self._cache[op_name] = None
        return None

    def _iter_py_files(self):
        """Iterate all .py files in search path order (exclude _-prefixed hidden files).

        Each search path can be a directory (rglob *.py) or a single .py file
        (loaded directly, _-prefix filter not applied since user chose it explicitly).
        """
        seen = set()
        for search_path in self._search_paths:
            if search_path.is_file():
                resolved = search_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved
            elif search_path.is_dir():
                for py_file in search_path.rglob("*.py"):
                    name = py_file.name
                    if name.startswith("_") and name != "__init__.py":
                        continue
                    resolved = py_file.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved

    def _try_find_in_file(self, py_file: Path, op_name: str) -> Optional[type]:
        """Try to find spec class in a single .py file"""

        # Prioritize: __spec__ dict
        reg = self._read_test_spec_dict(py_file)
        if reg is not None and op_name in reg:
            cls = reg[op_name]
            self._mark_source(cls, py_file)
            return cls

        # Check naming convention class
        cls_name = _snake_to_pascal(op_name) + "TestSpec"
        module = self._import_file(py_file)
        if module is None:
            return None

        cls = getattr(module, cls_name, None)
        if cls is not None and isinstance(cls, type):
            self._mark_source(cls, py_file)
            return cls

        return None

    @staticmethod
    def _mark_source(cls: type, py_file: Path) -> None:
        """Record the source file on the class so remote spec-mode can locate it."""
        try:
            setattr(cls, "__ttk_spec_file__", str(py_file))
        except (AttributeError, TypeError):
            pass

    def _read_test_spec_dict(self, py_file: Path) -> Optional[dict]:
        """Read __spec__ dict. Imports the file to get runtime objects.
        Results are cached to avoid repeated imports."""
        if py_file in self._file_cache:
            return self._file_cache[py_file]

        module = self._import_file(py_file)
        if module is None:
            self._file_cache[py_file] = None
            return None

        reg = getattr(module, "__spec__", None)
        if isinstance(reg, dict):
            self._file_cache[py_file] = reg
            return reg

        self._file_cache[py_file] = None
        return None

    def _import_file(self, py_file: Path) -> Optional[object]:
        """Dynamically import .py file as module object.
        Does NOT register in sys.modules — spec modules are ephemeral."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"ttk_test_spec_{py_file.stem}_{hash(str(py_file))}",
                str(py_file),
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            # Do NOT add to sys.modules — avoids memory leak in long sessions
            spec.loader.exec_module(module)
            self._loaded_modules[str(py_file)] = module
            return module
        except Exception:
            return None

    def clear_cache(self):
        """Clear cache (for testing or hot reload)"""
        self._cache.clear()
        self._file_cache.clear()
        self._loaded_modules.clear()
