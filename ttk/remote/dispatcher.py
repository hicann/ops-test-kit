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

import atexit
import base64
import contextlib
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


_FEATURE_CACHE: dict = {}
FEATURE_CACHE_TTL_S = 300.0


def _forget_server_features(host, port):
    """连接层出错时丢弃缓存: 对端可能已被换成另一个实例。"""
    _FEATURE_CACHE.pop((host, port), None)


def _features_from_health(host, port):
    """从心跳写的 health 文件取该 endpoint 的能力; 取不到返回 None(表示"不知道")。

    返回 None 与返回 [] 含义不同: [] 是"探过了, 服务端没有任何能力", None 是"没查到,
    请调用方自己探" —— 混淆两者会让旧服务端被误判成新服务端。
    """
    try:
        from ttk.remote.health_file import read_health_file

        health = read_health_file()
        if not health:
            return None
        entry = (health.get("endpoints") or {}).get(f"{host}:{port}")
        if not isinstance(entry, dict) or "features" not in entry:
            return None
        return list(entry.get("features") or [])
    except Exception:  # noqa: BLE001  取不到就退回自己探, 绝不因此让派发失败
        return None


def _server_features(host, port, timeout):
    """探一次服务端能力并按 (host, port) 缓存(带 TTL)。探不到一律按"不支持"处理。

    能力来自服务端自己在 /v1/heartbeat 里的声明, **不按端口号硬编码判断新旧**。
    但 (host, port) 并不唯一标识一个服务实例 —— 服务端升级重启、或隧道被重新
    指向另一个实例, 同一个地址背后就换了人。所以缓存必须有 TTL, 且连接出错时作废。
    """
    key = (host, port)
    hit = _FEATURE_CACHE.get(key)
    if hit is not None and time.monotonic() - hit[0] < FEATURE_CACHE_TTL_S:
        return hit[1]
    # 先读心跳写的 health 文件: 本模块的 _FEATURE_CACHE 是**模块级**字典, 而跑批的每个
    # 用例跑在独立 worker 进程里 —— 进程间不共享, 每个 worker 都会自己探一次(慢链路上
    # 就是每 worker 一个多余 RTT)。心跳本来就在周期性调 /v1/heartbeat, 能力已写进 health
    # 文件, 直接取即可; 取不到再退回自己探(单机无心跳、health 文件还没生成等场景)。
    shared = _features_from_health(host, port)
    if shared is not None:
        _FEATURE_CACHE[key] = (time.monotonic(), shared)
        return shared
    feats = []
    try:
        conn = _create_connection(host, port, timeout=min(timeout, 15))
        try:
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 200:
                feats = json.loads(body).get("features") or []
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001  探测失败只影响是否启用优化
        logging.debug("feature probe failed for %s:%s (%s)", host, port, e)
    _FEATURE_CACHE[key] = (time.monotonic(), feats)
    logging.info("server %s:%s features=%s", host, port, feats)
    return feats


def _plan_zero_upload(inputs: list, input_recipes: Optional[dict]):
    """能免传就返回 {recipes, digests}，否则 None（调用方回落整包上传）。

    判据是**逐片可证**的：把要发的每片叶子算出内容指纹，到配方表里查；
    全查得到才成立。任何一片查不到（非连续视图、input 插件原地改过数据、
    该片没播种……）就整体放弃——宁可多传，不可两端数据不同却当成相同。
    """
    if not input_recipes:
        return None
    from .input_recipe import digest_of

    leaves = [a for a in deep_flatten(inputs) if a is not None]
    if not leaves:
        return None
    digests = []
    recipes = []
    for leaf in leaves:
        key = digest_of(leaf)
        recipe = input_recipes.get(key)
        if recipe is None:
            return None
        digests.append(key)
        recipes.append(recipe)
    return {"recipes": recipes, "digests": digests}


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


def _slot_logical_dtype(entry):
    """Schema 每 slot 单值；TensorList 声明的嵌套 dtype 取首叶子。

    与物理 dtype 的 leaves[0] 取法一致（同质假设：list 内 dtype 一致）。
    """
    if isinstance(entry, (list, tuple)):
        return entry[0] if entry else None
    return entry


def _build_input_schema(
    inputs: list, input_names: list, input_formats: Optional[list] = None, input_dtypes: Optional[list] = None
) -> list:
    """Build X-Input-Schema from inputs and their names.

    嵌套为真相源：顶层 zip(input_names, inputs) 位置对齐，按 slot 类型分派。
    None slot → index:null；ndarray slot → 单 index；list/tuple slot → indices
    （deep_flatten 取叶子，过滤 None）。deep_flatten 不作用于 ndarray slot
    （其作容器元素时原子保留；裸 ndarray 当 sequence 入参会按行切，故 ndarray
    分支直接用 slot）。

    ``input_formats`` 为逐输入 format（与 input_names 位置对齐，可为空）——
    嵌入每个 schema 条目（name/index/dtype/format），server 侧可据此把
    format 传给三方 compose（广播/归约轴判定用），不另设独立 header。

    ``input_dtypes`` 为逐输入**逻辑** dtype（与 input_names 位置对齐，可为空），
    即 CSV 声明值（如 complex32），区别于条目 ``dtype`` 字段的物理 numpy dtype
    ——complex32 在 npz 里物理上是 float16+尾维[2]，服务端靠 logical_dtype
    才能还原成 torch.complex32 逻辑张量。

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
    ldts = list(input_dtypes) if input_dtypes else []
    fi = 0
    schema: list = []
    for i, (name, slot) in enumerate(zip(input_names, inputs)):
        fmt = fmts[i] if i < len(fmts) else None
        ldt = _slot_logical_dtype(ldts[i] if i < len(ldts) else None)
        if slot is None:
            schema.append({"name": name, "index": None, "dtype": None, "format": fmt, "logical_dtype": ldt})
        elif isinstance(slot, (list, tuple)):
            leaves = [x for x in deep_flatten(slot) if x is not None]
            schema.append(
                {
                    "name": name,
                    "indices": [fi + i for i in range(len(leaves))],
                    "dtype": _dtype_name(leaves[0]) if leaves else None,
                    "format": fmt,
                    "logical_dtype": ldt,
                }
            )
            fi += len(leaves)
        else:  # ndarray（含 0-d）/ numpy 标量 —— 单叶子,直接用 slot
            schema.append({"name": name, "index": fi, "dtype": _dtype_name(slot), "format": fmt, "logical_dtype": ldt})
            fi += 1
    # names 多于 inputs:尾部补 index:null
    for name in input_names[len(inputs) :]:
        schema.append({"name": name, "index": None, "dtype": None, "format": None, "logical_dtype": None})
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


# ---------------------------------------------------------------------------
# 连接复用(keep-alive)
#
# 每请求新建一条 TCP, 在 SSH 隧道上就是新开一条 channel。实测经公网跳板时
# 新建成本随使用逐步劣化(同隧道连续探测 18ms→57ms→290ms→752ms), 而复用同一
# 连接恒为即时。故按 (host, port) 在**进程内**缓存连接。
#
# 三条上限防止"一条连接挂一整晚": 服务端是 ThreadingHTTPServer, 一条持久连接
# 占住一个线程不放; 长时间空闲的连接还会被 NAT/跳板静默丢弃, 老 channel 本身
# 也会劣化。任一上限触发就主动换新。
#
# 关键前提: **必须假设连接随时会被对端关掉**。所有上限只是"在可控时机主动关",
# 真正兜底的是复用失败时立即换新连接重试一次(见 dispatch_to_remote 的 reused 分支)。
_CONN_POOL: dict = {}
KEEPALIVE_IDLE_S = 60.0  # 空闲这么久就不再复用(批次之间的长间隔不占资源)
KEEPALIVE_MAX_USES = 100  # 定期换新 channel, 避免老化劣化
KEEPALIVE_MAX_AGE_S = 300.0  # 兜底: 任何路径下都不会无限持有


def _acquire_connection(host, port, timeout):
    """取一条可用连接。返回 (conn, reused)；reused=True 表示来自缓存。"""
    key = (host, port)
    entry = _CONN_POOL.pop(key, None)
    if entry is not None:
        now = time.monotonic()
        fresh = (
            now - entry["last"] < KEEPALIVE_IDLE_S
            and now - entry["born"] < KEEPALIVE_MAX_AGE_S
            and entry["uses"] < KEEPALIVE_MAX_USES
        )
        if fresh:
            entry["uses"] += 1
            _CONN_POOL[key] = entry
            return entry["conn"], True
        with contextlib.suppress(OSError):
            entry["conn"].close()
    conn = _create_connection(host, port, timeout=timeout)
    _CONN_POOL[key] = {"conn": conn, "uses": 1, "born": time.monotonic(), "last": time.monotonic()}
    return conn, False


def _response_allows_reuse(resp) -> bool:
    """只在服务端明确允许时才复用: HTTP/1.1 且未声明 Connection: close。

    这样遇到未升级的服务端(BaseHTTPRequestHandler 默认 HTTP/1.0)会自动退化成
    每次新建, 不需要开关, 也不需要版本协商。
    """
    if getattr(resp, "version", 10) < 11:
        return False
    return "close" not in (resp.getheader("Connection") or "").lower()


def _settle_connection(host, port, conn, reusable: bool):
    """请求收尾: 可复用则留在池里(并刷新 last), 否则关掉并从池中移除。"""
    key = (host, port)
    if reusable:
        entry = _CONN_POOL.get(key)
        if entry is not None and entry["conn"] is conn:
            entry["last"] = time.monotonic()
            return
    _CONN_POOL.pop(key, None)
    with contextlib.suppress(OSError):
        conn.close()


def close_pooled_connections():
    """进程退出/worker 收尾时关掉缓存连接。SIGKILL 覆盖不到 —— 那只能靠服务端
    自己的空闲超时回收(见 xpu_server 的 handler timeout)。"""
    for key in list(_CONN_POOL):
        entry = _CONN_POOL.pop(key, None)
        if entry:
            with contextlib.suppress(OSError):
                entry["conn"].close()


def _forget_pool_after_fork():
    """fork 后子进程必须丢弃继承来的连接引用(不关闭: 那是父进程的 socket)。

    TTK 的 worker 走 fork/forkserver。若连接在 fork 前入池, 父子会持有**同一个**
    socket fd, 两边往同一条连接上收发 → 流内容交错, 症状是诡异的间歇性失败。
    并行度本来就来自进程数(--pc N)而非连接数, 每进程各自建连才是正确语义。
    """
    _CONN_POOL.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_pool_after_fork)

atexit.register(close_pooled_connections)


def dispatch_to_remote(
    op_name: str,
    inputs: list,
    *,
    input_names: Optional[list] = None,
    op_type: Optional[str] = None,
    provider: str = "torch",
    attrs: Optional[dict] = None,
    input_formats: Optional[list] = None,
    input_dtypes: Optional[list] = None,
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
    input_recipes: Optional[dict] = None,
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
    schema = _build_input_schema(inputs, input_names, input_formats, input_dtypes)
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

    # 免上传：仅当每片叶子的内容指纹都能在配方表里查到，才敢声明"服务端可自行重算"。
    # 查不到（非连续视图、input 插件改过数据、未播种……）就老老实实整包上传。
    # 必须先协商: 旧服务端不认这两个头, 会把"没有 body"当成参数绑定失败而回 400。
    if input_recipes and "zero_upload" not in _server_features(endpoint_host, endpoint_port, timeout):
        logging.info("[%s] zero-upload disabled: server does not advertise support", op_name)
        input_recipes = None
    zero_upload = _plan_zero_upload(inputs, input_recipes)
    if zero_upload:
        headers["X-Input-Recipes"] = json.dumps(zero_upload["recipes"])
        headers["X-Input-Digests"] = json.dumps(zero_upload["digests"])
        logging.info("[%s] zero-upload: %d leaf recipes sent", op_name, len(zero_upload["recipes"]))
    elif input_recipes is not None:
        # 配方表在但没能全部对上指纹 —— 静默回落最危险: "生效了"和"没生效"
        # 在日志上长得一样, 事后根本分不清。这里必须留痕。
        logging.info("[%s] zero-upload not applicable (some leaf has no matching recipe)", op_name)
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
        # 免传时不序列化（省的就是这份 npz 与它的上行带宽）；一旦服务端回 409
        # 说重算对不上，再在原地序列化并按老路重发。
        tmp_path = None if zero_upload else _serialize_to_file(inputs, dir=req_dir)
        while True:
            if time.monotonic() >= deadline_at:
                raise RemoteExecutionError(f"dispatch deadline ({_cfg('dispatch_deadline_s', 300.0, int)}s) exceeded")
            conn = None
            conn_reused = False
            keep_conn = False
            try:
                conn, conn_reused = _acquire_connection(endpoint_host, endpoint_port, timeout)
                file_size = 0 if tmp_path is None else os.path.getsize(tmp_path)
                headers["Content-Length"] = str(file_size)
                conn.putrequest("POST", "/v1/run")
                for key, value in headers.items():
                    conn.putheader(key, value)
                conn.endheaders()
                _t_send = time.monotonic()  # TEMP-XFER-PROBE
                if tmp_path is not None:
                    _sent = 0  # TEMP-XFER-PROBE
                    with open(tmp_path, "rb") as f:
                        while True:
                            chunk = f.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            conn.send(chunk)
                            _sent += len(chunk)  # TEMP-XFER-PROBE
                    _d = time.monotonic() - _t_send  # TEMP-XFER-PROBE
                    logging.info(
                        "XFER_PROBE mode=UPLOAD bytes=%d sec=%.3f rate=%.2fMB/s",
                        _sent,
                        _d,
                        _sent / max(_d, 1e-9) / 1048576,
                    )  # TEMP-XFER-PROBE
                else:
                    logging.info(
                        "XFER_PROBE mode=ZERO_UPLOAD bytes=0 sec=%.3f", time.monotonic() - _t_send
                    )  # TEMP-XFER-PROBE

                _t_rt = time.monotonic()  # TEMP-XFER-PROBE
                resp = conn.getresponse()
                logging.info("RESP_PROBE blocked=%.3f", time.monotonic() - _t_send)  # TEMP-XFER-PROBE

                if resp.status == 200:
                    keep_conn = _response_allows_reuse(resp)
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

                if resp.status == 409 and zero_upload:
                    # 服务端按配方重算后摘要对不上 —— 两端生成环境有差异
                    # （numpy/scipy 版本、生成代码不同步等）。不是错误结论，
                    # 而是免传不成立：回落成整包上传，本次请求原样重发。
                    detail = resp.read().decode(errors="replace")[:200]
                    logging.warning(
                        "[%s] zero-upload rejected by server, falling back to full upload: %s", op_name, detail
                    )
                    zero_upload = None
                    headers.pop("X-Input-Recipes", None)
                    headers.pop("X-Input-Digests", None)
                    tmp_path = _serialize_to_file(inputs, dir=req_dir)
                    continue

                if resp.status == 424:
                    resp_body = resp.read()
                    keep_conn = _response_allows_reuse(resp)
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
                    keep_conn = _response_allows_reuse(resp)
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
                # 复用连接失败是 keep-alive 的固有竞态: 服务端可能正好在我们发请求的
                # 瞬间关掉了这条空闲连接。这不算"连接不上", 立即换新连接重试一次,
                # 不消耗重试预算也不退避 —— 否则每次池过期都要白等一轮 backoff。
                if conn_reused:
                    logging.debug(f"pooled connection stale, reconnecting: {e}")
                    continue
                _forget_server_features(endpoint_host, endpoint_port)
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
                    _settle_connection(endpoint_host, endpoint_port, conn, keep_conn)

    finally:
        if req_dir:
            shutil.rmtree(req_dir, ignore_errors=True)
