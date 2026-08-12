import ctypes
import ctypes.util
import glob
import importlib.util
import logging
import os
import resource
import subprocess
import sys
import time



def setup_env():
    ascend_root = _find_ascend_root()
    if ascend_root:
        _source_setenv_bash(ascend_root)
        _setup_cann_paths(ascend_root)
        _setup_ascend_logging()

    _setup_ulimit()
    _preload_libgomp()
    _setup_runtime_env()
    _cleanup_old_logs()
    _ensure_log_dirs()


def _find_ascend_root():
    candidates = []
    for env_var in ["ASCEND_CUSTOM_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"]:
        val = os.getenv(env_var)
        if val:
            candidates.append(val)

    opp = os.getenv("ASCEND_OPP_PATH")
    if opp:
        candidates.append(opp.rstrip("/opp").rstrip("/"))

    if not candidates:
        user_path = "/usr/local/Ascend" if os.getuid() == 0 else os.path.expanduser("~/Ascend")
        if os.path.isdir(os.path.join(user_path, "cann", "bin")):
            candidates.append(os.path.join(user_path, "cann"))
        else:
            candidates.append(os.path.join(user_path, "latest"))

    for root in candidates:
        root = root.rstrip("/")
        if os.path.isdir(os.path.join(root, "compiler")) and os.path.isdir(os.path.join(root, "opp")):
            return root
        if root.endswith("/latest"):
            parent = root.rsplit("/", 1)[0]
            if os.path.isdir(os.path.join(parent, "compiler")) and os.path.isdir(os.path.join(parent, "opp")):
                return parent
    return None


def _sim_ld_paths():
    """LD_LIBRARY_PATH segments pointing into a camodel/simulator install.

    The NPUSim camodel runtime is injected via ``LD_LIBRARY_PATH`` (by cannsim
    record or the E2E npusim backend). CANN ``setenv.bash`` rebuilds
    ``LD_LIBRARY_PATH`` and would drop those segments, so they are recorded
    before sourcing and restored afterwards.
    """
    return [
        p for p in (os.environ.get("LD_LIBRARY_PATH", "") or "").split(":")
        if p and ("camodel" in p or "/simulator/" in p)
    ]


def _restore_ld_paths(segments):
    """Prepend ``segments`` to LD_LIBRARY_PATH unless already present."""
    if not segments:
        return
    existing = (os.environ.get("LD_LIBRARY_PATH", "") or "").split(":")
    missing = [p for p in segments if p not in existing]
    if missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing + existing)


def _source_setenv_bash(ascend_root):
    setenv = os.path.join(ascend_root, "bin", "setenv.bash")
    if not os.path.isfile(setenv):
        return

    sim_paths = _sim_ld_paths()
    try:
        result = subprocess.run(
            ["bash", "-c", f'source "{setenv}" && env -0'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return

        for entry in result.stdout.split('\0'):
            if '=' in entry:
                key, _, val = entry.partition('=')
                if key and "\n" not in key:
                    os.environ[key] = val
        # setenv.bash rebuilds LD_LIBRARY_PATH; keep the camodel runtime visible.
        _restore_ld_paths(sim_paths)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _get_custom_impl_parent_paths():
    """Get op_impl/ai_core/tbe paths from ASCEND_CUSTOM_OPP_PATH."""
    env_val = os.getenv("ASCEND_CUSTOM_OPP_PATH", "")
    if not env_val:
        return []
    paths = []
    for p in env_val.split(":"):
        if not p:
            continue
        tbe = os.path.join(p, "op_impl", "ai_core", "tbe")
        if os.path.isdir(tbe):
            paths.append(tbe)
    return paths


def _get_vendor_impl_parent_paths(opp_path):
    """Get op_impl/ai_core/tbe paths from vendors/config.ini."""
    config_file = os.path.join(opp_path, "vendors", "config.ini")
    if not os.path.isfile(config_file):
        return []
    paths = []
    with open(config_file) as f:
        for line in f:
            if line.strip().startswith("load_priority="):
                for name in line.split("=", 1)[1].split(","):
                    name = name.strip()
                    if name:
                        tbe = os.path.join(opp_path, "vendors", name,
                                           "op_impl", "ai_core", "tbe")
                        if os.path.isdir(tbe):
                            paths.append(tbe)
                break
    return paths


def _get_builtin_impl_parent_path(opp_path):
    """Get built-in op_impl/ai_core/tbe path."""
    tbe = os.path.join(opp_path, "built-in", "op_impl", "ai_core", "tbe")
    return tbe if os.path.isdir(tbe) else None


def _prepend_to_pythonpath(paths):
    """Prepend paths to PYTHONPATH env var and sys.path."""
    if not paths:
        return
    existing = os.getenv("PYTHONPATH", "")
    new = ":".join(paths)
    os.environ["PYTHONPATH"] = new + ":" + existing if existing else new
    path_set = set(paths)
    sys.path[:] = list(paths) + [p for p in sys.path if p not in path_set]


def _setup_cann_paths(ascend_root):
    drv_info = "/etc/ascend_install.info"
    if os.path.isfile(drv_info):
        with open(drv_info) as f:
            for line in f:
                if line.startswith("Driver_Install_Path_Param="):
                    drv_path = line.strip().split("=", 1)[1]
                    drv_lib = os.path.join(drv_path, "driver", "lib64", "driver")
                    if os.path.isdir(drv_lib):
                        existing = os.getenv("LD_LIBRARY_PATH", "")
                        if drv_lib not in existing:
                            os.environ["LD_LIBRARY_PATH"] = drv_lib + ":" + existing
                    break

    opp_path = os.path.join(ascend_root, "opp")
    if os.path.isdir(opp_path):
        os.environ.setdefault("ASCEND_OPP_PATH", opp_path)

        # Collect tbe paths in priority order: custom > vendors > built-in
        tbe_paths = []
        tbe_paths.extend(_get_custom_impl_parent_paths())
        tbe_paths.extend(_get_vendor_impl_parent_paths(opp_path))
        builtin_tbe = _get_builtin_impl_parent_path(opp_path)
        if builtin_tbe:
            tbe_paths.append(builtin_tbe)
        _prepend_to_pythonpath(tbe_paths)


def _setup_ascend_logging():
    os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    os.environ.setdefault("ASCEND_GLOBAL_EVENT_ENABLE", "0")
    os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")


def _setup_ulimit():
    for res, soft_limit in [
        (resource.RLIMIT_MEMLOCK, 65535),
        (resource.RLIMIT_NOFILE, 655300),
        (resource.RLIMIT_STACK, 81920),
    ]:
        try:
            soft, hard = resource.getrlimit(res)
            resource.setrlimit(res, (min(soft_limit, hard), hard))
        except (ValueError, OSError):
            pass


def _preload_libgomp():
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        torch_dir = os.path.dirname(spec.origin)
        for subdir in ["lib", ".libs"]:
            lib_dir = os.path.join(torch_dir, subdir)
            if os.path.isdir(lib_dir):
                matches = glob.glob(os.path.join(lib_dir, "libgomp*.so*"))
                if matches:
                    ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                    return

    for pkg in ["tensorflow_cpu_aws", "tensorflow", "tensorflow-cpu"]:
        try:
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                pkg_dir = os.path.dirname(spec.origin)
                for suffix in [pkg + ".libs", ".libs"]:
                    lib_dir = os.path.join(pkg_dir, suffix)
                    if os.path.isdir(lib_dir):
                        matches = glob.glob(os.path.join(lib_dir, "libgomp*.so*"))
                        if matches:
                            ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                            return
        except (ModuleNotFoundError, ValueError):
            continue

    path = ctypes.util.find_library("gomp")
    if path:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _setup_runtime_env():
    os.environ["PYTHONHASHSEED"] = "0"


def _cleanup_old_logs():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for f in glob.glob(os.path.join(base_dir, "ttk-*.log")):
        try:
            os.remove(f)
        except OSError:
            pass


def _ensure_log_dirs():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.expanduser("~/ascend/log"), exist_ok=True)

    cutoff = time.time() - 15 * 86400
    for d in [os.path.expanduser("~/ascend/log/plog"), os.path.expanduser("~/ascend/log/debug/plog")]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                try:
                    if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                except OSError:
                    pass
