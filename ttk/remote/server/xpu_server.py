#!/usr/bin/env python3
"""
xpu_server - TTK Remote XPU Execution Server.

Deployment constraint: MUST NOT import outside the ttk.remote.server package
(no ttk.core_modules, no ttk.remote) — the server deploys standalone on the XPU
box, which has no TTK framework installed.

Quick start:
    python -m ttk.remote.server.xpu_server --port 9090

All options:
    python -m ttk.remote.server.xpu_server \\
        --port 9090          # 监听端口（默认：配置或 9090）
        --bind 127.0.0.1     # 绑定地址（默认：127.0.0.1）
        --config xpu_server.yaml  # 配置文件路径
        --devices 0,1        # 设备 ID 列表（默认：0）
        --dry-run            # 空跑模式（返回随机数据）

Full deployment guide (plain HTTP / mTLS / per-process / per-container):
    ttk/remote/server/README.md
"""

import argparse
import base64
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import shutil
import ssl
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import executor
from .config import detect_hardware, get_framework, load_server_config
from .container import _run_in_container
from .execution_container import DATA, PERF

CHUNK_SIZE = 64 * 1024  # 64KB

# Module-level variables (set by run_server from config)
SYNC_BASE_DIR = None
TMP_ROOT = None
HEARTBEAT_TIMEOUT_S = 600
# Per-device locks: one Lock per device_id, allowing concurrent execution on
# distinct devices while serializing access to the same device. Initialized by
# run_server. Kill-safe: parent-held around child -> never orphaned even if the
# child crashes.
_device_locks: dict = {}   # {device_id: threading.Lock()}


def _init_device_locks(device_ids):
    """按 device_ids 初始化 per-device Lock 字典。CPU 跳过。

    先 clear 再填——避免残留旧 device 的锁。run_server 启动时调用。
    """
    _device_locks.clear()
    for dev in device_ids:
        if dev != "cpu":
            _device_locks[dev] = threading.Lock()


def _build_device_opts(handler, n):
    """提取 device 分支选项（device_id/docker_args/env）——纯函数，不构造完整 kwargs。

    避免在单测里 mock BaseHTTPRequestHandler：handler 只读 use_device/sandbox/profile
    三个属性。返回 device 相关三件：
      - device_id：0（容器内固定，executor 见 {torch_lib}:0）/ "cpu"
      - docker_args：sandbox=docker 时 render 后的 2-token list（{device_id}→物理 n）
      - env：sandbox=none 时 {visible_env: str(n)}（forkserver 不继承 os.environ，显式传）

    fail-fast：sandbox=docker + 非cpu 但 profile 缺 docker_args → 返回 ok=False/http_status=500
    （检查在 render 前，防 .format(None) 崩）。执行隔离单测①-⑤覆盖此分支。
    """
    if not handler.use_device:
        return {"device_id": "cpu"}
    opts = {"device_id": 0}            # 容器内固定（executor 见 {torch_lib}:0）
    if handler.sandbox == "docker":
        if not handler.profile.get("docker_args"):
            return {"ok": False, "http_status": 500,
                    "error": "sandbox=docker but profile missing docker_args"}
        opts["docker_args"] = [a.format(device_id=n) for a in handler.profile["docker_args"]]
    else:
        env_name = handler.profile.get("visible_env") or \
            f"{handler.profile['torch_lib'].upper()}_VISIBLE_DEVICES"
        opts["env"] = {env_name: str(n)}
    return opts


def _parse_mode(raw):
    """X-Mode: int bitmask (new protocol) or legacy 'data'/'perf' string."""
    raw = (raw or "").strip()
    if raw.isdigit():
        return int(raw)
    return {"data": DATA, "perf": PERF}.get(raw.lower(), DATA)


def _clamp_runtime(raw: str, default: int = 3, lo: int = 1, hi: int = 100) -> int:
    """Clamp the X-Runtime header to [lo, hi]; non-numeric -> default.

    Extracted from _handle_run so the guard is unit-testable. Boundary
    behavior: out-of-range clamps to nearest bound; non-int-parseable
    (ValueError/TypeError, incl. None) falls back to ``default``.
    """
    try:
        return max(lo, min(int(raw), hi))
    except (ValueError, TypeError):
        return default


def _receive_body_to_file(handler: BaseHTTPRequestHandler, dir=None) -> Optional[str]:
    """Stream request body into a temporary file.

    Returns path to the temp file, or None if body is empty.
    Uses constant memory regardless of body size.
    """
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length == 0:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=dir)
    remaining = content_length
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        chunk = handler.rfile.read(chunk_size)
        if not chunk:
            break
        tmp.write(chunk)
        remaining -= len(chunk)
    tmp.close()
    return tmp.name


def _resolve_class(module: object, dotted_name: str):
    """Resolve a dotted class name within a module.

    Supports nested classes: 'OuterClass.InnerImpl'.
    """
    obj = module
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _run_in_subprocess(kwargs: dict, deadline: float) -> dict:
    """Run execute_request in a FRESH forkserver child process; return envelope.

    One child per request -> fresh sys.modules (cross-tenant isolation), a crash
    kills only that child. A hard crash (segfault/OOM) means the child exits
    with nonzero code and sends nothing -> 500. Timeout -> kill -> 500.
    """
    ctx = multiprocessing.get_context("forkserver")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=executor.child_main, args=(child_conn, kwargs))
    logging.info("_run_in_subprocess: starting child for provider=%s", kwargs.get("provider"))
    proc.start()
    logging.info("_run_in_subprocess: waiting pid=%s deadline=%s", proc.pid, deadline)
    proc.join(timeout=deadline)
    logging.info("_run_in_subprocess: done alive=%s exitcode=%s", proc.is_alive(), proc.exitcode)
    if proc.is_alive():
        proc.kill()
        proc.join(5)
        return executor._err(500, f"request timed out after {deadline}s",
                             api=executor._api_from_kwargs(kwargs))
    envelope = None
    if parent_conn.poll(0):
        try:
            envelope = parent_conn.recv()
        except EOFError:
            envelope = None
    parent_conn.close()
    if envelope is None:
        return executor._err(500, f"child process exited with code {proc.exitcode}",
                             api=executor._api_from_kwargs(kwargs))
    return envelope


class TenantManager:
    """Thread-safe tenant lifecycle management."""

    def __init__(self, sync_base_dir: str):
        self._lock = threading.Lock()
        self._tenants: dict = {}
        self.sync_base_dir = sync_base_dir

    def heartbeat(self, tenant_id: str):
        with self._lock:
            if tenant_id not in self._tenants:
                tenant_path = os.path.join(self.sync_base_dir, tenant_id)
                try:
                    os.makedirs(tenant_path, exist_ok=True)
                except OSError:
                    pass  # May not have write permission; track tenant anyway
                self._tenants[tenant_id] = {
                    "last_heartbeat": time.time(),
                    "path": tenant_path,
                }
            else:
                self._tenants[tenant_id]["last_heartbeat"] = time.time()

    def cleanup(self, tenant_id: str) -> bool:
        with self._lock:
            info = self._tenants.pop(tenant_id, None)
            if info and os.path.isdir(info["path"]):
                shutil.rmtree(info["path"], ignore_errors=True)
                return True
            return False

    def cleanup_expired(self):
        now = time.time()
        with self._lock:
            expired = [
                (tid, info["path"]) for tid, info in self._tenants.items()
                if now - info["last_heartbeat"] > HEARTBEAT_TIMEOUT_S
            ]
            for tid, _ in expired:
                del self._tenants[tid]
        # Clean up files outside the lock to avoid holding it during I/O
        for tid, path in expired:
            logging.info(f"Tenant {tid} expired, cleaning up")
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def _atomic_write_file(abs_path: str, content: bytes) -> None:
    """Write content to abs_path atomically via temp file + rename.

    Concurrent writes to the same path never produce a torn file: a reader
    sees either the previous version or the complete new version, never a
    half-written one. Missing parent directories are created.
    """
    dir_path = os.path.dirname(abs_path)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, prefix=".sync_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp_path, abs_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


class XpuRequestHandler(BaseHTTPRequestHandler):

    tenant_manager: TenantManager
    dry_run: bool = False
    device_count: int = 1
    device_ids = ["cpu"]   # device=[ids]，CPU=["cpu"]；run_server 设置
    _device_rr_counter: int = 0
    _device_rr_lock = threading.Lock()
    use_device: bool = False
    data_gate = None                 # threading.BoundedSemaphore (backpressure)
    provider: str = ""               # set by run_server() from config/detect
    hardware: str = ""               # device_str format key; "" = auto-detect
    profile: dict = {}               # hardware_config[role] segment; {} = cpu 兜底
    hardware_config: dict = {}       # full hardware section from cfg
    provider_framework: dict = {}    # provider→framework overrides
    frameworks: dict = {}
    sync_base_dir: str = ""          # set by run_server() from config
    tmp_root: str = ""               # set by run_server() from config
    gate_wait_s: float = 1.0         # set by run_server() from config
    run_deadline_s: int = 300        # set by run_server() from config

    def log_message(self, format, *args):
        logging.info(f"{self.client_address[0]} - {format % args}")

    def _send_json(self, status: int, data: dict, env=None):
        body = json.dumps(data).encode()
        self.send_response(status)
        if env and env.get("api"):
            self.send_header("X-API", env["api"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _get_header(self, name: str, default: str = "") -> str:
        return self.headers.get(name, default)

    @staticmethod
    def _valid_tenant_id(tid):
        """Reject path-traversal / injection in tenant_id (flows into os.path.join)."""
        import re
        return bool(tid) and bool(re.fullmatch(r"[A-Za-z0-9._-]{1,64}", tid))

    def do_GET(self):
        if self.path.startswith("/v1/heartbeat"):
            # Merge of old /health + /v1/detect + /heartbeat:
            #   - tenant register side-effect (old /heartbeat)
            #   - hardware/device_count (old /health)
            #   - providers, incl. docker sandbox branch (old /v1/detect)
            qs = parse_qs(urlparse(self.path).query)
            tenant_id = qs.get("tenant_id", [""])[0]
            if tenant_id:
                if not self._valid_tenant_id(tenant_id):
                    self._send_json(400, {"error": "invalid tenant_id"})
                    return
                self.tenant_manager.heartbeat(tenant_id)
            # Capability set ONLY — order has NO priority meaning (priority is
            # app-side spec order). providers 为无序集合。
            providers = list(self.frameworks.keys()) if self.frameworks else []
            if getattr(self, "sandbox", "none") == "docker":
                docker_images = getattr(self, "docker_images", {}) or {}
                providers = list(docker_images.keys())
            self._send_json(200, {
                "status": "ok",
                "device_count": self.device_count,
                "hardware": self.hardware,
                "providers": providers,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self):
        if self.path.startswith("/v1/tenant/"):
            tenant_id = self.path.split("/v1/tenant/", 1)[1].split("/")[0]
            if not self._valid_tenant_id(tenant_id):
                self._send_json(400, {"error": "invalid tenant_id"})
                return
            cleaned = self.tenant_manager.cleanup(tenant_id)
            self._send_json(200, {
                "cleaned": cleaned,
                "path": os.path.join(self.sync_base_dir, tenant_id),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/sync":
            self._handle_sync()
        elif self.path == "/v1/run":
            self._handle_run()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_sync(self):
        tenant_id = self._get_header("X-Tenant-ID", "")
        if not self._valid_tenant_id(tenant_id):
            self._send_json(400, {"error": "invalid or missing tenant_id"})
            return

        self.tenant_manager.heartbeat(tenant_id)
        tenant_path = os.path.join(self.sync_base_dir, tenant_id)

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024 * 1024:
            self._send_json(413, {"error": "sync body too large (max 10GB)"})
            return
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        files = data.get("files", {})
        synced = 0
        skipped = 0
        errors = []

        for rel_path, file_info in files.items():
            # Security: path traversal check
            if ".." in rel_path or os.path.isabs(rel_path):
                errors.append({"path": rel_path, "error": "path traversal rejected"})
                continue
            # Security: .py only
            if not rel_path.endswith(".py"):
                errors.append({"path": rel_path, "error": "only .py files allowed"})
                continue

            abs_path = os.path.join(tenant_path, rel_path)
            content_b64 = file_info.get("content", "")
            expected_hash = file_info.get("hash", "")

            try:
                content_bytes = base64.b64decode(content_b64)
            except Exception:
                errors.append({"path": rel_path, "error": "invalid base64"})
                continue

            # Hash check — skip if unchanged
            if expected_hash and os.path.isfile(abs_path):
                with open(abs_path, "rb") as f:
                    existing = f.read()
                existing_hash = hashlib.sha256(existing).hexdigest()
                if existing_hash == expected_hash.replace("sha256:", ""):
                    skipped += 1
                    continue

            try:
                _atomic_write_file(abs_path, content_bytes)
                synced += 1
            except OSError as e:
                errors.append({"path": rel_path, "error": str(e)})

        if errors:
            self._send_json(400, {"synced": synced, "skipped": skipped,
                                  "errors": errors})
        else:
            self._send_json(200, {"synced": synced, "skipped": skipped,
                                  "errors": errors})

    def _device_rr_next(self) -> int:
        """Thread-safe RR counter increment. Returns previous value."""
        with self._device_rr_lock:
            self._device_rr_counter += 1
            return self._device_rr_counter - 1

    def _assign_device(self):
        """RR + try-lock: 从 RR 起点出发找空闲 device，全占则阻塞起点。

        所有请求（PERF + DATA）都 acquire Lock。返回 device_id（int 或 "cpu"）。
        调用方负责 finally release。
        """
        device_ids = [d for d in self.device_ids if d != "cpu"]
        if not device_ids:
            return "cpu"

        start = self._device_rr_next() % len(device_ids)
        for offset in range(len(device_ids)):
            idx = (start + offset) % len(device_ids)
            dev = device_ids[idx]
            if _device_locks[dev].acquire(blocking=False):
                return dev
        # 全占 → 阻塞等起点 device
        _device_locks[device_ids[start]].acquire()
        return device_ids[start]

    def _handle_run(self):
        # Hoist client api BEFORE validation: early-validation branches must
        # also echo X-API (any non-crash scenario echoes back).
        # Prefer X-API; fall back to X-Spec-Class (legacy) if absent.
        req_api = self._get_header("X-API", "") or self._get_header("X-Spec-Class", "") or None

        tenant_id = self._get_header("X-Tenant-ID", "")
        if not self._valid_tenant_id(tenant_id):
            self._send_json(400, {"error": "invalid or missing tenant_id"},
                            env={"api": req_api})
            return
        mode = _parse_mode(self._get_header("X-Mode", "data"))
        exec_type = self._get_header("X-Execution-Type", "api")
        runtime = _clamp_runtime(self._get_header("X-Runtime", "3"))

        schema_raw = self._get_header("X-Input-Schema", "[]")
        try:
            input_schema = json.loads(schema_raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid X-Input-Schema JSON"},
                            env={"api": req_api})
            return
        try:
            input_count = int(self._get_header("X-Input-Count", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid X-Input-Count (not integer)"},
                            env={"api": req_api})
            return
        attrs_raw = self._get_header("X-Attrs", "{}")
        try:
            attrs = json.loads(attrs_raw)
        except json.JSONDecodeError:
            attrs = {}
        param_order_raw = self._get_header("X-Param-Order", "")
        param_order = None
        if param_order_raw:
            try:
                param_order = json.loads(param_order_raw)
            except json.JSONDecodeError:
                param_order = None

        # Gate (max_concurrent backpressure) applies to ALL /v1/run, including
        # dry-run — dry-run must walk the real path (gate → req_dir →
        # _send_run_ok) so it exercises the same control flow, not a shortcut.
        gate = self.data_gate
        gate_held = False
        if gate is not None:
            wait_s = self.gate_wait_s
            if not gate.acquire(timeout=wait_s):
                self._send_json(503, {"error": "server busy, retry"},
                                env={"api": req_api})
                return
            gate_held = True

        req_dir = tempfile.mkdtemp(prefix=f"req_{tenant_id}_", dir=self.tmp_root)
        try:
            tmp_in = _receive_body_to_file(self, dir=req_dir)
            logging.info("_handle_run: dry_run=%s body_size=%s",
                         self.dry_run, os.path.getsize(tmp_in) if tmp_in else 0)
            deadline = self.run_deadline_s
            # n=物理/cpu/None（主进程域，lock+render+visible_env）；provider 兜底 None（dry_run
            # 不进 else，防 logging NameError）；result 兜底（_assign_device 抛异常时 result 已
            # 定义，防 UnboundLocalError）。
            n = None
            provider = None
            result = {"ok": False, "http_status": 500,
                      "error": "internal error", "api": req_api}
            try:
                if self.dry_run:
                    result = _dry_run_env(req_dir)
                else:
                    raw_provider = self._get_header("X-Provider", "torch")
                    provider = get_framework(raw_provider, self.provider_framework)
                    n = "cpu" if not self.use_device else self._assign_device()
                    opts = _build_device_opts(self, n)
                    if opts.get("ok") is False:           # docker fail-fast（render 前检查）
                        result = {**opts, "api": req_api}
                    else:
                        kwargs = dict(
                            tenant_sync_dir=os.path.join(self.sync_base_dir, tenant_id),
                            exec_type=exec_type, provider=provider,
                            profile=self.profile,
                            op_name=self._get_header("X-Op-Name", "") or None,
                            op_type=self._get_header("X-Op-Type", "") or None,
                            api=req_api,
                            spec_module=self._get_header("X-Spec-Module", "") or None,
                            spec_class=self._get_header("X-Spec-Class", "") or None,
                            mode=mode, input_schema=input_schema, attrs=attrs,
                            tmp_in_path=tmp_in, input_count=input_count,
                            device_id=opts["device_id"],
                            use_device=self.use_device, output_dir=req_dir,
                            runtime=runtime, param_order=param_order)
                        if "env" in opts:                  # sandbox=none 分支才设 env
                            kwargs["env"] = opts["env"]
                        if self.sandbox == "docker":       # cpu+docker 保现状走容器（仅 --cap-drop）
                            image = self.docker_images.get(provider)
                            if image is None:
                                result = {"ok": False, "http_status": 500,
                                          "error": f"no docker image for provider {provider}",
                                          "api": req_api}
                            else:
                                result = _run_in_container(
                                    kwargs, deadline=deadline, image=image,
                                    memory=self.docker_memory, network=self.docker_network,
                                    use_device=self.use_device,
                                    docker_args=opts.get("docker_args") or [])
                        else:
                            result = _run_in_subprocess(kwargs, deadline=deadline)
            finally:
                if n is not None and n != "cpu":           # None 守卫（dry_run）+ cpu 守卫
                    _device_locks[n].release()
            # status 三分（双层职责：500 服务端错 / <500 客户端错）：ok→info / ≥500→error（堆栈由 executor :846
            # exception 记，handler 只补上下文）/ <500（400/424）→warning（客户端错）。
            if result.get("ok"):
                logging.info("_handle_run ok: api=%s provider=%s", req_api, provider)
                self._send_run_ok(result)
            else:
                status = result.get("http_status", 500)
                if status >= 500:
                    logging.error("_handle_run fail: status=%s api=%s provider=%s device=%s err=%s",
                                  status, req_api, provider, n, result.get("error"))
                else:
                    logging.warning("_handle_run client-err: status=%s api=%s err=%s",
                                    status, req_api, result.get("error"))
                self._send_json(status,
                                {"error": result.get("error"), "missing": result.get("missing")},
                                env=result)
        finally:
            shutil.rmtree(req_dir, ignore_errors=True)
            if gate_held:
                try:
                    gate.release()
                except ValueError:        # never over-release
                    pass

    def _send_run_ok(self, env):
        output_path = env.get("output_path")
        has_body = bool(output_path and os.path.exists(output_path))
        file_size = os.path.getsize(output_path) if has_body else 0
        self.send_response(200)
        self.send_header("X-Output-Count", str(env.get("output_count", 0)))
        if has_body:
            self.send_header("X-Output-Shapes", json.dumps(env.get("shapes", [])))
            self.send_header("X-Output-Schema", json.dumps(env.get("schema", [])))   # 替 X-Output-Dtypes
        if env.get("perf") is not None:
            self.send_header("X-Perf", json.dumps(env["perf"]))
        if env.get("api"):
            self.send_header("X-API", env["api"])
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        if has_body:
            with open(output_path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    self.wfile.write(chunk)


def _dry_run_env(req_dir):
    """Random-output envelope for dry-run, mirroring _run_in_subprocess's return."""
    outs = [np.random.randn(4, 8).astype(np.float32)]
    path = os.path.join(req_dir, "out.npz")
    np.savez_compressed(path, **{f"a{i}": o for i, o in enumerate(outs)})
    return {"ok": True, "http_status": 200, "output_path": path,
            "output_count": len(outs),
            "shapes": [list(o.shape) for o in outs],
            "schema": [{"index": i, "dtype": str(o.dtype)} for i, o in enumerate(outs)],   # 替 dtypes
            "perf": None, "api": None}


def _heartbeat_watcher(tenant_manager: TenantManager):
    while True:
        time.sleep(60)
        try:
            tenant_manager.cleanup_expired()
        except Exception as e:
            logging.error(f"Heartbeat watcher error: {e}")


def _detect_frameworks() -> dict:
    """Detect installed frameworks by module lookup — no import (fast, no GIL flood).

    Returns {provider_name: True}; keys are PROVIDER names (the convention used
    by --provider, third_party dicts, X-Provider header) — NOT import names.
    tensorflow advertises as 'tf' so the client's provider routing matches.
    Only the keys are used (advertised via /v1/heartbeat). The actual framework
    import is deferred to the executor (subprocess/container) when a request
    needs it — importing here would block startup ~5.5s on tensorflow's C-lib
    init + flood ~700 log lines and starve the HTTP handler.
    """
    import importlib.util
    providers = {}
    if importlib.util.find_spec("torch") is not None:
        providers["torch"] = True
    if importlib.util.find_spec("tensorflow") is not None:
        providers["tf"] = True       # provider name; the import is 'tensorflow'
    return providers


def run_server(port: int, dry_run: bool = False, devices: str = None,
               bind: str = "127.0.0.1", config_path: str = None):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    import sys
    try:
        cfg = load_server_config(config_path)   # raises ValueError on bad yaml -> sys.exit(1)

        # CLI args > yaml > defaults
        bind = bind or cfg["bind"]
        port = port or cfg["port"]

        # device 决策（2 分支）：
        #   devices 显式（含 "cpu"/""）→ 用户主张，探测仅取 role（ids 丢弃）
        #   devices None → /dev auto-detect
        if devices is not None:
            # bool 防护 --devices ""（int("") 崩）；"cpu"/"" → 显式 cpu，不信任探测
            # （防 heartbeat 报假硬件）
            use_device = bool(devices) and devices != "cpu"
            if use_device:
                role, _ = detect_hardware(cfg["hardware_config"])  # 取 role，ids 丢弃
            else:
                role = "cpu"
            device_ids = [int(x) for x in devices.split(",")] if use_device else ["cpu"]
        else:
            use_device = True
            role, device_ids = detect_hardware(cfg["hardware_config"])

        XpuRequestHandler.hardware = role
        XpuRequestHandler.profile = cfg["hardware_config"].get(role, {})
        XpuRequestHandler.device_ids = device_ids
        XpuRequestHandler.hardware_config = cfg["hardware_config"]
        XpuRequestHandler.provider_framework = cfg["provider_framework"]
        _init_device_locks(device_ids)
        # max_concurrent < device count is allowed (data_gate caps total concurrency;
        # extra devices idle). Warn so user notices under-utilization, not assert.
        device_count = len([d for d in device_ids if d != "cpu"])
        if device_count > 0 and cfg["max_concurrent"] < device_count:
            logging.warning(
                f"max_concurrent ({cfg['max_concurrent']}) < device count ({device_count}); "
                f"data_gate caps total concurrency, {device_count - cfg['max_concurrent']} "
                f"device(s) may idle")
    except ValueError as e:
        logging.critical(f"startup failed: {e}")
        sys.exit(1)

    global SYNC_BASE_DIR, TMP_ROOT
    SYNC_BASE_DIR = cfg["sync_dir"]
    TMP_ROOT = cfg["tmp_dir"]
    os.makedirs(SYNC_BASE_DIR, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)

    tenant_manager = TenantManager(SYNC_BASE_DIR)

    frameworks = _detect_frameworks()

    XpuRequestHandler.tenant_manager = tenant_manager
    XpuRequestHandler.dry_run = dry_run
    XpuRequestHandler.frameworks = frameworks
    XpuRequestHandler.device_count = len([d for d in device_ids if d != "cpu"])
    XpuRequestHandler.use_device = use_device
    XpuRequestHandler.sandbox = cfg["sandbox"]
    XpuRequestHandler.docker_images = cfg["docker_images"]
    XpuRequestHandler.docker_memory = cfg["docker_memory"]
    XpuRequestHandler.docker_network = cfg["docker_network"]
    XpuRequestHandler.data_gate = threading.BoundedSemaphore(cfg["max_concurrent"])
    XpuRequestHandler.sync_base_dir = SYNC_BASE_DIR
    XpuRequestHandler.tmp_root = TMP_ROOT
    XpuRequestHandler.gate_wait_s = cfg["gate_wait_s"]
    XpuRequestHandler.run_deadline_s = cfg["run_deadline_s"]
    try:
        os.makedirs(TMP_ROOT, exist_ok=True)
    except OSError as e:
        logging.warning(f"cannot create TMP_ROOT {TMP_ROOT}: {e}")

    watcher = threading.Thread(target=_heartbeat_watcher,
                               args=(tenant_manager,), daemon=True)
    watcher.start()

    class _Server(ThreadingHTTPServer):
        # Default backlog (5) is tight for concurrent fan-out (16+ in-flight
        # requests); raise it so bursts aren't refused under load, and reuse
        # the address so rapid restarts don't hit TIME_WAIT bind failures.
        request_queue_size = 128
        allow_reuse_address = True

    server = _Server((bind, port), XpuRequestHandler)

    # mTLS: wrap socket if enabled
    if cfg["tls_enabled"]:
        tls_ca = cfg["tls_ca_cert"]
        tls_cert = cfg["tls_server_cert"]
        tls_key = cfg["tls_server_key"]
        if tls_ca and tls_cert and tls_key:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(tls_ca)
            ctx.load_cert_chain(tls_cert, tls_key)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            logging.info("mTLS enabled (client cert required)")
        else:
            logging.error("tls.enabled=true but cert paths empty; refusing to start without TLS")
            sys.exit(1)

    logging.info(f"xpu_server listening on {bind}:{port} (dry_run={dry_run})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="TTK XPU Remote Execution Server")
    parser.add_argument("--port", type=int, default=None,
                        help="Listen port (default: from config yaml or 9090)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: return random outputs")
    parser.add_argument("--bind", default="",
                        help="Bind address (default: from config or 127.0.0.1)")
    parser.add_argument("--config", default=None,
                        help="Path to xpu_server.yaml")
    parser.add_argument("--devices", default=None,
                        help="Device IDs (e.g. 0,1) or 'cpu'")
    args = parser.parse_args()
    run_server(args.port, dry_run=args.dry_run, devices=args.devices,
               bind=args.bind, config_path=args.config)


if __name__ == "__main__":
    main()
