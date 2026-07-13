import json
import os

from ttk.remote.health_file import (
    atomic_write_json,
    read_health_file,
)


# --- atomic_write_json -------------------------------------------------------

def test_creates_file_with_content(tmp_path):
    path = str(tmp_path / "health.json")
    atomic_write_json(path, {"endpoints": {"a:1": {"alive": True}}})
    with open(path) as f:
        data = json.load(f)
    assert data["endpoints"]["a:1"]["alive"] is True


def test_overwrites_existing_file(tmp_path):
    path = str(tmp_path / "health.json")
    atomic_write_json(path, {"v": 1})
    atomic_write_json(path, {"v": 2})
    with open(path) as f:
        data = json.load(f)
    assert data["v"] == 2


def test_never_leaves_temp_file(tmp_path):
    path = str(tmp_path / "health.json")
    atomic_write_json(path, {"v": 1})
    leftovers = [f for f in os.listdir(str(tmp_path)) if f.startswith(".")]
    assert leftovers == []


def test_creates_parent_directories(tmp_path):
    path = str(tmp_path / "subdir" / "nested" / "health.json")
    atomic_write_json(path, {"ok": True})
    assert os.path.isfile(path)


# --- read_health_file --------------------------------------------------------

def test_atomic_write_then_read(tmp_path):
    path = str(tmp_path / "health.json")
    data = {"endpoints": {"h:9090": {"alive": True, "providers": ["torch"]}}}
    atomic_write_json(path, data)
    result = read_health_file(path)
    assert result["endpoints"]["h:9090"]["alive"] is True


def test_read_missing_file_returns_none():
    assert read_health_file("/nonexistent/path.json") is None


def test_returns_none_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("TTK_XPU_HEALTH_PATH", raising=False)
    assert read_health_file("") is None


def test_returns_none_when_file_corrupted(tmp_path):
    path = str(tmp_path / "health.json")
    with open(path, "w") as f:
        f.write("not valid json {{{")
    assert read_health_file(path) is None


# --- shim re-export still works ---------------------------------------------

def test_legacy_infra_import_path_re_exports():
    # Existing importers use the infra path; the shim must keep them working.
    from ttk.core_modules.infra.health_file import (
        atomic_write_json as legacy_write,
        read_health_file as legacy_read,
    )
    assert legacy_write is atomic_write_json
    assert legacy_read is read_health_file
