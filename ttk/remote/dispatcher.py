#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Dispatcher - Worker-side remote execution dispatch.

Sends numpy inputs to xpu_server, handles 424 dependency retry with automatic
/v1/sync upload, returns numpy outputs.
"""

import base64
import hashlib
import http.client
import io
import json
import logging
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ttk.remote import has_data
from ttk.utilities.container_utils import deep_flatten

CHUNK_SIZE = 64 * 1024  # 64KB chunks for file streaming


class RemoteExecutionError(Exception):
    """Remote execution failed."""


class RemoteConnectionError(Exception):
    """Failed to connect to xpu_server."""


class RemoteBusyError(RemoteExecutionError):
    """Server returned 503 (busy) and the retry budget was exhausted.

    Subclasses RemoteExecutionError so callers catching the base class still
    work, while TTK can tell 'server gave up under load' (this) apart from
    'operator genuinely failed' (plain RemoteExecutionError from a 500).
    """


@dataclass
class RemoteResult:
    """Outputs + perf from a successful /v1/run.

    dispatch_to_remote returns a plain list by default; pass
    return_result=True to get this wrapper carrying both outputs and perf.
    ``api`` carries the server-resolved API (X-API header) so the collector
    can show the real API rather than a client-side guess.
    """

    outputs: list
    perf: Optional[dict] = None
    api: Optional[str] = None


def _parse_client_mode(raw) -> int:
    """Convert mode string ('data'/'data_perf') to bitmask integer.

    Accepts 'data', 'data_perf', or None. Defaults to DATA mode.
    """
    from ttk.remote import DATA, PERF

    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        return {"data": DATA, "perf": PERF}.get(raw.strip().lower(), DATA)
    return DATA


def _parse_perf_header(raw):
    """Parse the X-Perf header (JSON); None if absent/invalid."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_api_header(raw):
    """Parse the X-API response header (raw string, NOT json).

    The server echoes back the resolved API as a plain dotted string
    (e.g. 'torch.add'); None if absent/empty/whitespace.
    """
    if not raw:
        return None
    return raw.strip() or None


def _backoff_delay(n: int, base: float, max_delay: float, jitter: float, rng=random.uniform) -> float:
    """Exponential backoff with jitter: min(base*2**n, max_delay) * (1 ± jitter)."""
    capped = min(base * (2**n), max_delay)
    return capped * (1 + rng(-jitter, jitter))


def _cfg(field: str, default, cast=float):
    """Generic config getter from RemoteConfig with fallback.

    Args:
        field: Config field name (e.g., 'backoff_base_s')
        default: Default value if config is None or field is None
        cast: Type cast function (default: float)

    Returns:
        Config value cast to specified type, or default
    """
    from ttk.remote.config import get_remote_config

    config = get_remote_config()
    val = getattr(config, field, default) if config else default
    v = val if val is not None else default
    if field == "backoff_jitter":
        return cast(max(0.0, min(float(v), 1.0)))
    return cast(v)


# Response bodies <= this stay in memory; larger stream to a per-request file.
RESP_MEM_THRESHOLD = 64 << 20


def _serialize_to_file(inputs: list, dir=None) -> str:
    """Serialize numpy inputs to a temporary .npz file.

    deep_flatten 展开嵌套（每叶子一个 a{i}），过滤 None。dir: 临时文件目录
    （默认系统 tempdir）。返回路径，caller 清理。
    """
    leaves = [a for a in deep_flatten(inputs) if a is not None]
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=dir) as tmp:
        np.savez(tmp, **{f"a{i}": a for i, a in enumerate(leaves)})
    return tmp.name


def _reinterpret_dtype(arr, dtype_name):
    from ..utilities.dtypes import numpy_bfloat16, numpy_float8_e4m3fn, numpy_float8_e5m2, numpy_float8_e8m0

    try:
        if dtype_name == "bfloat16":
            return arr.view(numpy_bfloat16())
        if dtype_name == "float8_e5m2":
            return arr.view(numpy_float8_e5m2())
        if dtype_name == "float8_e4m3fn":
            return arr.view(numpy_float8_e4m3fn())
        if dtype_name == "float8_e8m0":
            return arr.view(numpy_float8_e8m0())
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
        return arr
    return arr


def _resp_to_npz_source(resp, req_dir, threshold=RESP_MEM_THRESHOLD):
    """Return an np.load()-able source for the response body.
    Large Content-Length -> req_dir/resp.npz; small/missing -> BytesIO."""
    try:
        cl = int(resp.getheader("Content-Length") or 0)
    except (TypeError, ValueError):
        cl = 0
    if cl > threshold:
        path = os.path.join(req_dir, "resp.npz")
        with open(path, "wb") as f:
            shutil.copyfileobj(resp, f, CHUNK_SIZE)
        return path
    return io.BytesIO(resp.read())


def _load_npz_outputs(source, schema):
    """按 schema 恢复嵌套顶层 slots（不 flatten）。
    schema: [{index|indices|null, dtype}, ...]"""
    npz = np.load(source, allow_pickle=False)
    out = []
    for entry in schema:
        idx = entry.get("index")
        if idx is None:
            indices = entry.get("indices")
            if indices is None:
                out.append(None)
            else:
                # tensor-list slot：恢复 list of arrays
                dtype = entry.get("dtype")
                out.append([_reinterpret_dtype(npz[f"a{i}"], dtype) for i in indices])
        else:
            arr = npz[f"a{idx}"]
            out.append(_reinterpret_dtype(arr, entry.get("dtype")))
    return out


def _dtype_name(arr):
    """Best-effort numpy dtype name for the wire (e.g. 'float32', 'bfloat16').

    Carried in X-Input-Schema so the server can convert precisely even when its
    own numpy can't represent the dtype (bfloat16 arrives as raw bytes / void).
    """
    if arr is None:
        return None
    try:
        return arr.dtype.name
    except AttributeError:
        return None


def _build_input_schema(inputs: list, input_names: list, input_formats: Optional[list] = None) -> list:
    """Build X-Input-Schema from inputs and their names.

    嵌套为真相源：顶层 zip(input_names, inputs) 位置对齐，按 slot 类型分派。
    None slot → index:null；ndarray slot → 单 index；list/tuple slot → indices
    （deep_flatten 取叶子，过滤 None）。deep_flatten 不作用于 ndarray slot
    （其作容器元素时原子保留；裸 ndarray 当 sequence 入参会按行切，故 ndarray
    分支直接用 slot）。

    ``input_formats`` 为逐输入 format（与 input_names 位置对齐，可为空）——
    嵌入每个 schema 条目（name/index/dtype/**format**），server 侧可据此把
    format 传给三方 compose（广播/归约轴判定用），不另设独立 header。

    注意：tensor-list slot 的 dtype 取自首个叶子（leaves[0]），假设同一 list
    内 dtype 同质；混合 dtype 的 list slot 未完全支持——server 每个 name 只应用
    一种 dtype。
    """
    if not input_names:
        return []

    assert len(inputs) <= len(input_names), (
        f"inputs({len(inputs)}) > names({len(input_names)}): slots/names 长度不匹配（caller 构造 bug）"
    )

    fmts = list(input_formats) if input_formats else []
    fi = 0
    schema: list = []
    for i, (name, slot) in enumerate(zip(input_names, inputs)):
        fmt = fmts[i] if i < len(fmts) else None
        if slot is None:
            schema.append({"name": name, "index": None, "dtype": None, "format": fmt})
        elif isinstance(slot, (list, tuple)):
            leaves = [x for x in deep_flatten(slot) if x is not None]
            schema.append(
                {
                    "name": name,
                    "indices": [fi + i for i in range(len(leaves))],
                    "dtype": _dtype_name(leaves[0]) if leaves else None,
                    "format": fmt,
                }
            )
            fi += len(leaves)
        else:  # ndarray（含 0-d）/ numpy 标量 —— 单叶子,直接用 slot
            schema.append({"name": name, "index": fi, "dtype": _dtype_name(slot), "format": fmt})
            fi += 1
    # names 多于 inputs:尾部补 index:null
    for name in input_names[len(inputs) :]:
        schema.append({"name": name, "index": None, "dtype": None, "format": None})
    return schema


def _schema_leaf_count(schema: list) -> int:
    """schema 引用的非 None 叶子数 = npz a{i} 数 = X-Input-Count。

    `dispatch_to_remote` 内 `effective_count` 用此派生 header，使 X-Input-Count
    按构造等于 schema 叶子数（非第三个独立 deep_flatten 表达式）。
    """
    return sum(len(e["indices"]) if "indices" in e else (1 if e.get("index") is not None else 0) for e in schema)


def _find_spec_file(module_name: str, search_roots: list) -> Optional[str]:
    """Search for a .py file matching module_name in the spec directory trees.

    Converts dotted module names to paths, tries both as .py file and __init__.py.
    """
    rel_path = module_name.replace(".", os.sep)
    candidates = [f"{rel_path}.py", os.path.join(rel_path, "__init__.py")]

    for root in search_roots:
        for cand in candidates:
            full = os.path.join(root, cand)
            if os.path.isfile(full):
                return full

    # Try a shallow name-only match (e.g. "util" → any "util.py" under root)
    base = module_name.rsplit(".", 1)[-1]
    for root in search_roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if f"{base}.py" in filenames:
                return os.path.join(dirpath, f"{base}.py")

    return None


def _read_file_with_hash(file_path: str) -> tuple:
    """Read file, return (base64_content, sha256_hex).

    Computes the hash over the raw bytes at no extra read cost.
    """
    with open(file_path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def _do_http_sync(
    missing: str, search_roots: list, endpoint_host: str, endpoint_port: int, tenant_id: str, timeout: int
) -> bool:
    """Find and sync a missing dependency file to xpu_server.

    Returns True if the file was found and synced successfully.
    """
    file_path = _find_spec_file(missing, search_roots)
    if not file_path:
        logging.warning(f"424: could not find file for missing module '{missing}'")
        return False

    rel_path = os.path.basename(file_path)
    content_b64, file_hash = _read_file_with_hash(file_path)

    sync_body = json.dumps({"files": {rel_path: {"content": content_b64, "hash": file_hash}}})

    try:
        conn = _create_connection(endpoint_host, endpoint_port, timeout=timeout)
        conn.request(
            "POST", "/v1/sync", body=sync_body, headers={"Content-Type": "application/json", "X-Tenant-ID": tenant_id}
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status == 200:
            logging.info(f"424: synced '{rel_path}' for missing module '{missing}'")
            return True
        logging.warning(f"424: sync failed for '{rel_path}' (status {resp.status})")
        return False
    except Exception as e:
        logging.warning(f"424: sync error for '{missing}': {e}")
        return False


def _sync_missing_dependency(
    missing: str, search_roots: list, endpoint_host: str, endpoint_port: int, tenant_id: str, timeout: int
) -> bool:
    """Sync a missing dependency, serialized across workers by semaphore.

    The first worker to acquire the per-(endpoint, module) lock performs the
    upload; concurrent workers poll ``get_semaphore`` and reuse its result,
    avoiding duplicate uploads and concurrent-write races on xpu_server.
    """
    from ttk.core_modules.tbe_multiprocessing.pool import get_process_context

    ctx = get_process_context()
    if ctx is None:
        # Single-process mode (no pool): sync directly — no concurrent dedup needed.
        return _do_http_sync(missing, search_roots, endpoint_host, endpoint_port, tenant_id, timeout)
    sync_id = f"xpu_sync_{endpoint_host}:{endpoint_port}:{missing}"

    if ctx.acquire_semaphore(sync_id):
        try:
            ok = _do_http_sync(missing, search_roots, endpoint_host, endpoint_port, tenant_id, timeout)
            ctx.set_semaphore(sync_id, "ok" if ok else "fail")
            return ok
        except Exception as e:
            ctx.set_semaphore(sync_id, f"err:{e}")
            raise

    # Another worker holds the lock — wait for its result. Normally the wait
    # ends quickly: holder set_semaphore on success/failure, or pool.py's
    # dead-worker handler fills the semaphore via _semaphore_dead_sequence if
    # the holder process crashed. The timeout below is a last-resort safety
    # net for edge cases where neither path fires (e.g. holder stuck in
    # zombie state, pool detection delayed).
    logging.debug(f"424: waiting for another worker to sync '{missing}' ({sync_id})")
    wait_deadline = time.monotonic() + timeout
    while (result := ctx.get_semaphore(sync_id)) is None:
        if time.monotonic() >= wait_deadline:
            raise RemoteExecutionError(
                f"timed out waiting for sync of '{missing}' (holder may have crashed); sync_id={sync_id}"
            )
        time.sleep(0.5)
    return result == "ok"


def _create_connection(host, port, timeout):
    """HTTP or HTTPS connection based on RemoteConfig TLS fields (shared tls module)."""
    from ttk.remote.config import get_remote_config
    from ttk.remote.tls import build_tls_connection, tls_from_config

    return build_tls_connection(host, port, timeout, tls_from_config(get_remote_config()))


def dispatch_to_remote(
    op_name: str,
    inputs: list,
    *,
    input_names: Optional[list] = None,
    op_type: Optional[str] = None,
    provider: str = "torch",
    attrs: Optional[dict] = None,
    input_formats: Optional[list] = None,
    endpoint_host: str = "127.0.0.1",
    endpoint_port: int = 9090,
    tenant_id: str = "unknown",
    timeout: int = 300,
    max_retries: int = 5,
    mode: str = "data",
    spec_search_roots: Optional[list] = None,
    api: Optional[str] = None,
    execution_type: str = "api",
    spec_module: Optional[str] = None,
    spec_class: Optional[str] = None,
    spec_file: Optional[str] = None,
    return_result: bool = False,
    tmp_root: Optional[str] = None,
    runtime: int = 3,
    param_order: Optional[list] = None,
):
    """Send inputs to remote xpu_server, return numpy outputs.

    On 424 (missing dependency), automatically searches spec_search_roots
    for the missing .py file and uploads it via /v1/sync before retrying.
    """
    attrs = attrs or {}
    input_names = input_names or []
    spec_search_roots = list(spec_search_roots or [])

    # In spec mode, ensure the spec source dir is searchable for 424 dep sync.
    if execution_type == "spec" and spec_file:
        _fdir = os.path.dirname(os.path.abspath(spec_file))
        if _fdir not in spec_search_roots:
            spec_search_roots.insert(0, _fdir)

    # KERNEL no-spec (api=None) relies on the server's _resolve_3party_api to
    # derive the API from op_name/op_type — do NOT infer client-side.
    schema = _build_input_schema(inputs, input_names, input_formats)
    effective_count = _schema_leaf_count(schema)
    mode_int = _parse_client_mode(mode)

    headers = {
        "X-Execution-Type": execution_type,
        "X-Provider": provider,
        "X-Attrs": json.dumps(attrs),
        "X-Input-Schema": json.dumps(schema),
        "X-Input-Count": str(effective_count),
        "X-Mode": str(mode_int),
        "X-Runtime": str(runtime),
        "X-Tenant-ID": tenant_id,
        "Content-Type": "application/octet-stream",
    }
    if param_order:
        headers["X-Param-Order"] = json.dumps(param_order)
    if execution_type == "api":
        if api:
            headers["X-API"] = api
    else:  # spec
        if spec_module:
            headers["X-Spec-Module"] = spec_module
        if spec_class:
            headers["X-Spec-Class"] = spec_class
    if op_name:
        headers["X-Op-Name"] = op_name
    if op_type:
        headers["X-Op-Type"] = op_type

    def _done(outputs, perf, api=None):
        return RemoteResult(outputs=outputs, perf=perf, api=api) if return_result else outputs

    deadline_at = time.monotonic() + _cfg("dispatch_deadline_s", 300.0, int)
    budget_503 = _cfg("max_503_retries", 10.0, int)
    budget_conn = _cfg("max_conn_retries", 5.0, int)
    budget_424 = max_retries
    used_503 = 0
    used_conn = 0

    req_dir = None
    try:
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        req_dir = tempfile.mkdtemp(prefix=f"req_{tenant_id}_", dir=tmp_root)
        tmp_path = _serialize_to_file(inputs, dir=req_dir)
        while True:
            if time.monotonic() >= deadline_at:
                raise RemoteExecutionError(f"dispatch deadline ({_cfg('dispatch_deadline_s', 300.0, int)}s) exceeded")
            conn = None
            try:
                conn = _create_connection(endpoint_host, endpoint_port, timeout=timeout)
                file_size = os.path.getsize(tmp_path)
                headers["Content-Length"] = str(file_size)
                conn.putrequest("POST", "/v1/run")
                for key, value in headers.items():
                    conn.putheader(key, value)
                conn.endheaders()
                with open(tmp_path, "rb") as f:
                    while True:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        conn.send(chunk)

                resp = conn.getresponse()

                if resp.status == 200:
                    perf = _parse_perf_header(resp.getheader("X-Perf"))
                    api = _parse_api_header(resp.getheader("X-API"))
                    if not has_data(mode_int):
                        resp.read()
                        return _done([], perf, api)
                    output_count = int(resp.getheader("X-Output-Count", "1"))
                    if output_count == 0:
                        resp.read()
                        return _done([], perf, api)
                    _schema_raw = resp.getheader("X-Output-Schema")
                    _schema = json.loads(_schema_raw) if _schema_raw else []
                    _source = _resp_to_npz_source(resp, req_dir)
                    return _done(_load_npz_outputs(_source, _schema), perf, api)

                if resp.status == 424:
                    resp_body = resp.read()
                    budget_424 -= 1
                    if budget_424 < 0:
                        raise RemoteExecutionError(
                            f"424 dependency retries ({max_retries}) exceeded: {resp_body.decode()}"
                        )
                    logging.warning(f"424 retry ({max_retries - budget_424}/{max_retries}): {resp_body.decode()}")
                    try:
                        missing_info = json.loads(resp_body)
                        missing_module = missing_info.get("missing", "")
                    except json.JSONDecodeError:
                        missing_module = ""
                    if missing_module and spec_search_roots:
                        # Pre-check: can we find this file locally?
                        # If not, this is an environment dep (e.g. scipy not installed
                        # on the server), not a spec file — retrying won't help.
                        if _find_spec_file(missing_module, spec_search_roots) is None:
                            raise RemoteExecutionError(
                                f"server requires module '{missing_module}' which is not "
                                f"in spec search paths; likely a server-side "
                                f"environment dependency (pip install) is missing"
                            )
                        synced = _sync_missing_dependency(
                            missing_module, spec_search_roots, endpoint_host, endpoint_port, tenant_id, timeout
                        )
                        if not synced:
                            logging.warning(
                                f"Could not sync '{missing_module}', "
                                f"retrying ({max_retries - budget_424}/{max_retries})"
                            )
                    continue  # does NOT touch 503 budget

                if resp.status == 503:
                    resp.read()
                    if used_503 >= budget_503:
                        raise RemoteBusyError(f"server busy (503), retry budget ({budget_503}) exhausted")
                    used_503 += 1
                    time.sleep(
                        _backoff_delay(
                            used_503 - 1,
                            _cfg("backoff_base_s", 0.5),
                            _cfg("backoff_max_s", 10.0),
                            _cfg("backoff_jitter", 0.25),
                        )
                    )
                    continue  # does NOT touch 424 budget

                # 400 / 500 / other -> genuine failure, do not retry.
                resp_body = resp.read()
                raise RemoteExecutionError(f"Server returned {resp.status}: {resp_body.decode()}")

            except RemoteExecutionError:
                raise
            except (http.client.HTTPException, ConnectionRefusedError, OSError) as e:
                # _mark_endpoint_dead removed: round-robin + HB 11s eviction replace it
                if used_conn >= budget_conn:
                    raise RemoteConnectionError(f"connection retries ({budget_conn}) exhausted: {e}") from e
                used_conn += 1
                logging.warning(f"conn error, retry ({used_conn}/{budget_conn}): {e}")
                time.sleep(
                    _backoff_delay(
                        used_conn - 1,
                        _cfg("backoff_base_s", 0.5),
                        _cfg("backoff_max_s", 10.0),
                        _cfg("backoff_jitter", 0.25),
                    )
                )
                continue
            finally:
                if conn:
                    conn.close()

    finally:
        if req_dir:
            shutil.rmtree(req_dir, ignore_errors=True)
