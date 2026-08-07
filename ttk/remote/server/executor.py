"""executor — runs ONE /v1/run inside a fresh child process (isolation).

Module-level so it is picklable for multiprocessing. execute_request returns an
envelope dict and never raises for 400/424/500; only a hard crash (segfault/OOM)
kills the child, which the parent sees as a nonzero exitcode -> 500.

Deployment constraint: MUST NOT import outside ttk.remote.server (no
ttk.core_modules, no ttk.remote) — the server deploys standalone on the XPU box.
Only stdlib + numpy + sibling server modules + lazy torch.
"""
import importlib
import inspect
import logging
import os
import sys
import time
from collections.abc import Callable as _Callable
from functools import partial

import numpy as np

try:
    from . import execution_container as _ec
except ImportError:
    # Container entrypoint imports executor as a top-level module (no parent
    # package); fall back to absolute (execution_container is standalone).
    import execution_container as _ec

(UnknownParamError, bind_params, format_device,
 has_data, has_perf, match_params_v1, resolve_callable) = (
    _ec.UnknownParamError, _ec.bind_params, _ec.format_device,
    _ec.has_data, _ec.has_perf, _ec.match_params_v1, _ec.resolve_callable)


class _MissingSpecDependency(ImportError):
    """Spec-module import failed -> 424 (syncable), not 500."""


# Module-level cache for the torch backend extension module (the object
# returned by getattr(torch, profile["torch_lib"])). executor runs in a
# per-request throwaway child process, so a module-global is safe (one request
# == one process lifetime). Set ONCE in execute_request after the backend
# import (Step 4a); read by _device_available and _run_perf pass-2 instead of
# repeating `getattr(torch, profile["torch_lib"])`.
_TORCH_DEV_MODULE = None

# third_party key aliases — mirrors client-side aliases (profiling.py) so spec
# writers can use "tensorflow"/"np"; server is standalone (no ttk import) so
# duplicated here. Keep in sync with the client copy.
_TP_ALIASES = {"tensorflow": "tf", "np": "numpy"}


def _api_from_kwargs(kwargs):
    """从请求 kwargs 取 client 携带的 api 标识（X-API 或 spec_class）。

    用于 server-side 错误路径回传 X-API：成功路径用
    executor 推导的 resolved api（api_label），错误路径只能回传 client 发来的原始值。
    """
    return kwargs.get("api") or kwargs.get("spec_class")


def _ok(output_path, count, shapes, dtypes, perf, api=None, schema=None):
    return {"ok": True, "http_status": 200, "output_path": output_path,
            "output_count": count, "shapes": shapes, "dtypes": dtypes,
            "perf": perf, "missing": None, "error": None, "api": api, "schema": schema}


def _err(status, error, missing=None, api=None):
    return {"ok": False, "http_status": status, "output_path": None,
            "output_count": 0, "shapes": [], "dtypes": [], "perf": None,
            "missing": missing, "error": error, "api": api}


def _client_error(e) -> str:
    """Client-facing error string for a 500: exception TYPE + MESSAGE only.

    Security (OWASP Improper Error Handling / OTG-ERR-002): the full traceback
    carries server FS paths, line numbers and source lines, so it is logged
    server-side only (logging.exception) and NEVER put in the wire envelope.
    The exception message can still embed a path (residual risk, accepted). To
    tighten the contract later (generic message, add error_id, ...), change ONLY
    this function — both 500 exit points (execute_request, child_main) route
    through it, so it is the single control point.
    """
    return f"{type(e).__name__}: {e}"


def _resolve_attr_path(module, dotted):
    obj = module
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


# --- KERNEL api resolve: faithful port of backup get_torch_func / get_tf_func ---
# Dict literals copied VERBATIM (KERNEL-proven); they reference torch/tf symbols,
# so they are built inside each resolver where the lazy import has run. Do not
# paraphrase or trim entries.
# ACLNN branch (aclnn-prefixed names); this section covers only
# the torch/tf dual-form KERNEL port.


def _torch_resolve(name):
    """Port of backup get_torch_func. Single snake-form arg.

    Only invoked when provider == 'torch' (gated by _resolve_3party_api's
    dispatch); a TF request never reaches this import.
    """
    import torch
    # === VERBATIM torch_funcs dict (backup __init__.py:2808-2824) ===
    torch_funcs = {
        "kl_div": torch.nn.functional.kl_div,
        "ctc_loss_v2": torch.nn.functional.ctc_loss,
        "equal": torch.eq,
        "gelu": torch.nn.GELU(),
        "floor_mod": torch.fmod,
        "floor_div": partial(torch.div, rounding_mode='floor'),
        "pows": torch.pow,
        "real_div": torch.div,
        "top_k_v2": torch.topk,
        "select": torch.where,
        "select_v2": torch.where,
        "is_inf": torch.isinf,
        "is_nan": torch.isnan,
        "is_close": torch.isclose,
        "is_finite": torch.isfinite,
    }
    if name in torch_funcs:
        return torch_funcs[name]
    sources = [torch, torch.ops.aten]
    result = None
    for source in sources:
        try:
            result = getattr(source, name, None)
        except RuntimeError:
            pass
        if result is not None:
            return result
    # suffix stripping — faithful to backup (serial-if, NOT elif). The bare `if`
    # chain does last-match-wins: each matching suffix UNCONDITIONALLY reassigns
    # `result`, so a later matching suffix overwrites an earlier one. In practice
    # a name ends in exactly ONE of _d/_grad/_v2 (they are mutually exclusive
    # endings), so last-match-wins == short-circuit return — behaviorally
    # equivalent. EXCEPTION: a name like `xxx_d_v2` matches BOTH `_d` (step 1,
    # result=resolve(xxx_v2)) AND `_v2` (step 3, result=resolve(xxx_d)); the `_v2`
    # step runs last and wins, so `xxx_d_v2` resolves via the `_v2` path. We
    # mirror backup's verbatim structure (don't second-guess KERNEL-proven code).
    if name.endswith("_d"):
        result = _torch_resolve(name[:-2])
    if name.endswith("_grad"):
        result = _torch_resolve(name[:-5] + "_backward")
    if name.endswith("_v2"):
        result = _torch_resolve(name[:-3])
    return result


def _tf_resolve(name):
    """Port of backup get_tf_func. Single Camel-form arg."""
    import tensorflow as tf
    from tensorflow.python.ops import gen_nn_ops, gen_state_ops, gen_array_ops
    from tensorflow.python.ops import gen_math_ops, nn_impl, gen_bitwise_ops, gen_image_ops
    from tensorflow import raw_ops
    # === VERBATIM tf_func_map dict (backup platform.py:87-91) ===
    tf_func_map = {
        "batch_norm_v3": nn_impl.fused_batch_norm,
        "spence": tf.math.special.spence,
        "space_to_batch": gen_array_ops.SpaceToBatch
    }
    if name in tf_func_map:
        return tf_func_map[name]
    sources = [tf, tf.math, gen_image_ops, gen_nn_ops, tf.nn, gen_state_ops,
               gen_array_ops, gen_math_ops, nn_impl, gen_bitwise_ops, raw_ops]

    try:
        from tensorflow.python.training import training_ops
        sources.append(training_ops)
    except (ModuleNotFoundError, ImportError):
        pass

    try:
        from tensorflow.python.ops import gen_stateless_random_ops_v2
        sources.append(gen_stateless_random_ops_v2)
    except (ModuleNotFoundError, ImportError):
        pass

    for source in sources:
        result = getattr(source, name, None)
        if result is not None:
            return result
    # suffix stripping (faithful: backup's replace-based logic, kept verbatim)
    if "_d" in name:
        r = _tf_resolve(name.replace("_d", ""))
        if r is not None:
            return r
    if "_v2" in name:
        r = _tf_resolve(name.replace("_v2", ""))
        if r is not None:
            return r
    if "_v1" in name:
        r = _tf_resolve(name.replace("_v1", ""))
        if r is not None:
            return r
    if "_v3" in name:
        r = _tf_resolve(name.replace("_v3", ""))
        if r is not None:
            return r
    if "_1d" in name:
        r = _tf_resolve(name.replace("_1d", "1d"))
        if r is not None:
            return r
    if "_2d" in name:
        r = _tf_resolve(name.replace("_2d", "2d"))
        if r is not None:
            return r
    if "_3d" in name:
        r = _tf_resolve(name.replace("_3d", "3d"))
        if r is not None:
            return r
    if "_" in name:
        r = _tf_resolve(name.replace("_", ""))
        if r is not None:
            return r
    return None


# --- ACLNN api resolve: verbatim port of golden_generation's torch-search ----
# segment. Source: ttk/utilities/string_utils.py:70 (camel_to_snake)
#         (_find_torch_api / _auto_import_from_torch as methods on GoldenGeneration).
# Deployment constraint: MUST NOT import outside ttk.remote.server — these helpers
# are COPIED here (de-methodized: drop self; module-level functions). golden_generation
# also does a plugin lookup before the torch fallback; that lookup is not portable
# (needs plugin_path / OpApiInfoKeeper), so only the torch-search segment is ported.
# Lazy torch import inside function bodies is permitted by the constraint.


def camel_to_snake(camel_name: str) -> str:
    """
    Operator Registered Camel name convert to snake name.
    Verbatim copy from ttk/utilities/string_utils.py:70 (do NOT import ttk.*).
    """
    snake_name = ""
    sub_head = False
    name_list = list(camel_name)
    for _idx, _char in enumerate(name_list):
        if _char.islower():
            sub_head = False
        if _char.isdigit():
            sub_head = True
        if _char.isupper() and _idx != 0:
            if not sub_head:
                snake_name += "_"
                sub_head = True
            else:
                _idx_next = _idx + 1
                if _idx_next < len(name_list):
                    if name_list[_idx_next].islower():
                        snake_name += "_"
        snake_name += _char

    return snake_name.lower()


def _find_torch_api(torch_module, snake_name: str):
    # find torch.snake_name. Verbatim port of golden_generation._find_torch_api
    # (de-methodized: self dropped, explicit torch_module/snake_name args).
    # The logging.debug call from the original is dropped (no self._ctx here).
    try:
        torch_api = getattr(torch_module, snake_name, None)
    except RuntimeError:
        # older torch version, 1.11.0 for exp,
        # will throw RuntimeError when absent.
        torch_api = None
    if torch_api and isinstance(torch_api, _Callable):
        return torch_api
    else:
        if snake_name.startswith("inplace_"):
            return _find_torch_api(torch_module, snake_name[len("inplace_"):])
        elif snake_name.endswith("_scalar"):
            return _find_torch_api(torch_module, snake_name[:-len("_scalar")])
        elif snake_name.endswith("_tensor"):
            return _find_torch_api(torch_module, snake_name[:-len("_tensor")])
        elif snake_name.endswith("_v2"):
            return _find_torch_api(torch_module, snake_name[:-len("_v2")])
        elif snake_name.endswith("s"):
            return _find_torch_api(torch_module, snake_name[:-len("s")])
        elif not snake_name.startswith("_"):  # _add_relu
            return _find_torch_api(torch_module, "_" + snake_name)
        else:
            return None


def _auto_import_from_torch(snake_name: str):
    """Search torch / torch.nn.functional / torch.ops.aten.

    ACLNN-only path (provider=='torch'): sole caller is _aclnn_resolve; a TF
    request never reaches here. Verbatim port of golden_generation.
    """
    import torch
    # torch api first
    torch_api = _find_torch_api(torch, snake_name)
    if torch_api is not None:
        return torch_api
    torch_nn_api = _find_torch_api(torch.nn.functional, snake_name)
    if torch_nn_api is not None:
        return torch_nn_api
    return _find_torch_api(torch.ops.aten, snake_name)


def _aclnn_resolve(api_name):
    """ACLNN: strip 'aclnn' prefix + camel_to_snake + torch search.

    Returns (callable, name) | (None, None). name = strip+snake 后的实际搜索 key
    （如 aclnnAdd → add），供 _resolve_callable 拼 provider.name（如 torch.add）——
    X-API 回传的是 server 实际执行的 API，不是请求的原始 op_name。

    Port of golden_generation._import_golden_funcs's torch-search segment (NOT
    the plugin lookup — that's not portable). The original returns (func, src);
    here we return func plus the resolved snake key.
    """
    if not api_name or not api_name.startswith("aclnn"):
        return None, None
    snake = camel_to_snake(api_name[5:])  # strip "aclnn" (5 chars) + Camel→snake
    f = _auto_import_from_torch(snake)
    return (f, snake) if f is not None else (None, None)


def _resolve_3party_api(op_name, op_type, provider):
    """KERNEL/ACLNN: resolve op_name(snake)/op_type(Camel) -> (callable, name).

    aclnn-prefixed op_name routes through _aclnn_resolve (strip + torch search),
    returning (callable, op_name). Otherwise dual-form KERNEL port: try op_name
    then op_type, first hit wins (tf benefits from Camel; torch resolves via
    snake). name = 命中的搜索入参（op_name 或 op_type），非 callable 的
    __name__（suffix stripping 脱钩）。Returns (None, None) if unresolvable.
    """
    # ACLNN branch: aclnn-prefixed op_name -> torch search. ACLNN is torch-only;
    # provider is assumed torch (not asserted — _aclnn_resolve does a torch
    # search regardless of provider, which is correct only for torch).
    if op_name and op_name.startswith("aclnn"):
        return _aclnn_resolve(op_name)
    resolve = _torch_resolve if provider == "torch" else _tf_resolve
    for name in (op_name, op_type):
        if not name:
            continue
        f = resolve(name)
        if f is not None:
            return f, name           # name = 命中的搜索入参
    return None, None


def _resolve_callable(exec_type, provider, api, op_name, op_type, spec_module,
                      spec_class):
    """Resolve the callable to execute. Returns (callable, api_label).

    api_label = 实际执行 api 的标识，供 X-API response header 回传：
      - spec 模式: spec_class（dotted 路径，e.g. "MySpec.AddSpec"）
      - api 模式 (explicit api): api 字符串本身（e.g. "torch.nn.functional.softmax"）
      - api=None 推导模式: provider.命中name（e.g. "torch.add"、"tf.Relu"、"torch.aclnnAdd"）
    """
    if exec_type == "spec":
        if not spec_class:
            raise ValueError("spec mode requires spec_class")
        try:
            mod = importlib.import_module(spec_module)
        except ImportError as e:
            name = getattr(e, "name", None) or spec_module
            err = _MissingSpecDependency(name)
            err.name = name              # ImportError.name isn't set by manual construction
            raise err from e
        cls = _resolve_attr_path(mod, spec_class) if spec_class else mod
        tp = getattr(cls, "third_party", None)
        if tp is None:
            return cls, spec_class
        if isinstance(tp, str):
            return resolve_callable(tp), spec_class
        if isinstance(tp, dict):
            tp = {_TP_ALIASES.get(k, k): v for k, v in tp.items()}
            if provider not in tp:
                raise ValueError(f"provider '{provider}' not in third_party")
            v = tp[provider]
            return (resolve_callable(v) if isinstance(v, str) else v), spec_class
        raise ValueError("unsupported third_party format")
    # api mode
    if api:
        return resolve_callable(api), api
    # api=None -> KERNEL/ACLNN: server derives via _resolve_3party_api
    f, name = _resolve_3party_api(op_name, op_type, provider)
    if f is None:
        raise ValueError(
            f"cannot resolve api for op_name={op_name!r} op_type={op_type!r} "
            f"provider={provider!r}")
    return f, f"{provider}.{name}"       # 推导 api_label = provider.命中name


def _bind(func, named, attrs, device, warn_leftover=True):
    merged = dict(attrs or {})
    merged.update(named)
    return bind_params(func, merged, device=device, warn_leftover=warn_leftover)


def _invoke(callable_fn, named, attrs, provider, device_id, use_device,
            profile=None):
    """Run the callable once, return RAW outputs (no numpy cast)."""
    if inspect.isclass(callable_fn):
        cls = callable_fn
        dev = format_device(provider, profile,
                            "cpu" if not use_device else device_id)
        # 契约: __init__/__call__ 参数(除kwargs)并集 ⊆ inputs∪attrs; input/attr
        # 喂给声明它的方法(都声明则都喂)。device 保留注入, 有默认值用默认。
        # warn_leftover=False: 一个方法没消耗的可能喂给另一个, 不报 leftover 误报。
        if cls.__init__ is object.__init__:          # 无自定义 __init__: 直接实例化
            inst = cls()
        else:
            ia, ik = _bind(cls.__init__, named, attrs, dev, warn_leftover=False)
            inst = cls(*ia, **ik)
        ca, ck = _bind(inst.__call__, named, attrs, dev, warn_leftover=False)
        return inst(*ca, **ck)
    # Three-phase invoke:
    # 1. kwargs as-is (when input names match API param names)
    # 2. positional (torch uses input/other which don't match schema names)
    # 3. inspect signature, zip positional values to real param names (TF raw_ops)
    # TODO: a TypeError RAISED BY THE OP ITSELF (not a binding mismatch) is
    # indistinguishable from a binding error here and gets retried through all 3
    # phases — phase 3 may then fail with the count-mismatch branch and mask the
    # op's real TypeError. Future: inspect the traceback (frame origin) to tell a
    # binding TypeError (raised inside the call dispatch) from an op TypeError
    # (raised inside the op body) and avoid the wasted retries / masking.
    try:
        return callable_fn(**dict(named, **(attrs or {})))
    except TypeError:
        try:
            return callable_fn(*list(named.values()), **(attrs or {}))
        except TypeError:
            sig = inspect.signature(callable_fn)
            param_names = [
                p for p, v in sig.parameters.items()
                if v.kind not in (inspect.Parameter.VAR_KEYWORD,
                                  inspect.Parameter.VAR_POSITIONAL)
                and p != 'self'
            ]
            vals = list(named.values())
            if len(vals) > len(param_names):
                raise TypeError(
                    f"{getattr(callable_fn, '__name__', callable_fn)} expects "
                    f"{len(param_names)} params {param_names}, "
                    f"got {len(vals)} inputs {list(named.keys())}")
            return callable_fn(**dict(zip(param_names, vals), **(attrs or {})))


def _to_numpy_pair(v, provider):
    """Convert one output to (numpy_array, semantic_dtype_name), provider-aware.

    bfloat16 can't round-trip through numpy savez, so it's stored as raw int16
    bits with 'bfloat16' declared (client reinterprets). float8 likewise has no
    numpy storage class — stored as raw uint8 bits with the float8 dtype name
    declared. torch is imported ONLY for the torch path; tf/other paths never
    touch it (tf bfloat16 ships a numpy bf16 dtype via
    tensorflow.bfloat16.as_numpy_dtype).
    """
    if provider == "torch":
        try:
            import torch
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bfloat16:
                    return v.contiguous().view(torch.int16).cpu().numpy(), "bfloat16"
                if "float8" in str(v.dtype):
                    return v.contiguous().view(torch.uint8).cpu().numpy(), str(v.dtype).replace("torch.", "")
                return v.detach().cpu().numpy(), str(v.dtype).replace("torch.", "")
        except ImportError:
            pass
    # tf tensor / numpy / array-like — np.asarray handles it; never imports torch.
    a = np.asarray(v)
    dt = a.dtype.name
    if "bfloat16" in dt or "bf16" in dt:
        return np.ascontiguousarray(a).view(np.int16), "bfloat16"
    return a, dt


def _outputs_to_numpy(outputs, provider):
    """保留嵌套（不 flatten）。遍历顶层 slots，构造 schema。
    返回 (schema, arrays_叶子)。schema: [{index|indices|null, dtype}, ...]"""
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    schema = []
    arrays = []
    npz_idx = 0
    for slot in outputs:
        if slot is None:
            schema.append({"index": None, "dtype": None})   # None 只在顶层 slot（op 输出占位）
            continue
        if isinstance(slot, (list, tuple)):
            # tensor-list slot：叶子均为实际 tensor（op 输出约束；None 只在顶层 slot 处理）
            leaves = []
            dt = None
            for leaf in slot:
                arr, dt = _to_numpy_pair(leaf, provider)
                arrays.append(arr)
                leaves.append(npz_idx)
                npz_idx += 1
            schema.append({"indices": leaves, "dtype": dt})   # dt 取末叶子；list 内假设 dtype 同质（op 输出约束；异质 list 未支持）
        else:
            arr, dt = _to_numpy_pair(slot, provider)
            arrays.append(arr)
            schema.append({"index": npz_idx, "dtype": dt})
            npz_idx += 1
    return schema, arrays


def _to_vendor_tensor(value, provider, device_str, dtype_name=None):
    """Framework H2D: numpy input -> provider tensor on device.

    Inputs arrive as numpy (restored from the tmp_in savez). torch/tf callables
    need tensors, so convert before binding. ``dtype_name`` is the dtype declared
    in X-Input-Schema by the client (whose numpy knows the real dtype) — used to
    convert dtypes the server's numpy can't represent (bfloat16) without guessing.
    None / lists recurse; non-numpy / unknown provider pass through.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return type(value)(
            _to_vendor_tensor(v, provider, device_str, dtype_name) for v in value)
    if provider == "torch":
        try:
            import torch
        except ImportError:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(device_str)
        if isinstance(value, np.ndarray):
            if dtype_name == "bfloat16":
                # numpy has no native bfloat16 (wire form is raw int16 bits);
                # reinterpret then view as bfloat16.
                return torch.from_numpy(
                    value.view(np.int16)).view(torch.bfloat16).to(device_str)
            return torch.from_numpy(value).to(device_str)
        return value
    if provider == "tf":
        try:
            import tensorflow as tf
            if dtype_name == "bfloat16" and isinstance(value, np.ndarray):
                # wire form is raw int16 bits; tf ships a numpy bf16 dtype
                # (tensorflow.bfloat16.as_numpy_dtype) we can view into.
                value = value.view(tf.bfloat16.as_numpy_dtype)
            # Pin to the assigned device — otherwise tf.convert_to_tensor lands on
            # TF's default (device 0), wrong when multiple devices present. TF2 soft
            # device placement makes tf.device a no-op when the device is absent.
            if device_str and device_str != "cpu":
                with tf.device(device_str):
                    return tf.convert_to_tensor(value)
            return tf.convert_to_tensor(value)
        except ImportError:
            return value
    return value


def _device_available(provider, profile) -> bool:
    """Provider-aware device availability — never imports torch for a tf request.

    profile-driven: torch -> getattr(torch, torch_lib).is_available();
    tf -> tf_type = profile.get("tf_device_type"); False when tf_type is missing
    (graceful degrade, NOT a config error — distinguishes from format_device's
    ValueError on the same missing field). The torch_{lib} extension is imported
    upstream in execute_request before this is reached.
    """
    if provider == "torch":
        # Defensive on torch_lib: tf branch degrades via .get(); mirror that so a
        # missing key degrades to False rather than KeyError. Prefer the cached
        # backend module (set in execute_request) to avoid repeating getattr.
        # (import torch sits right under the provider=='torch' gate so the static
        # guard recognizes it as vendor-gated.)
        try:
            import torch
        except ImportError:
            return False
        lib_name = profile.get("torch_lib")
        if lib_name is None:
            return False
        dev = _TORCH_DEV_MODULE if _TORCH_DEV_MODULE is not None else getattr(torch, lib_name, None)
        if dev is None:
            return False
        return dev.is_available()
    if provider == "tf":
        tf_type = profile.get("tf_device_type")
        if tf_type is None:
            return False
        try:
            import tensorflow as tf
            return bool(tf.config.list_physical_devices(tf_type))
        except ImportError:
            return False
    return False


def _device_time(evt, device) -> float:
    """server inline 2-candidate self_ time. is not None guard
    (not truthiness) so a genuine 0.0 on candidate1 is NOT skipped."""
    v = getattr(evt, "self_device_time_total", None)      # candidate1: torch 2.7+
    if v is not None:
        return v
    return getattr(evt, f"self_{device}_time_total", 0.0)  # candidate2: <2.7 (self_<lib>)


def _is_device_kernel(evt) -> bool:
    """只保留真实 设备 内核条目，供 device_us 求和使用。

    key_averages() 是把 profiler 的**事件树拍平**后的列表。树上同一段 设备 时间会以
    三种不同身份各出现一次，若不加区分地全部相加，等于把它重复计入：

        key_averages()
        │
        ├─ ProfilerStep*        容器 span（schedule() 每轮自动插入，覆盖整轮）
        │                       含义：这一轮的【时间跨度】——从首个内核开始到末个内核结束，
        │                             其间 设备 等待 CPU 下发下一个 kernel 的空闲也计入
        │                       剔除：① 量纲不同，它是"经过了多久"而非"设备忙了多久"；
        │                             ② 它在层级上已经包住下面两类，留下即整轮再加一遍。
        │                             对 launch-bound 负载（算子多、单个数据量小，设备 大
        │                             部分时间在等下发），跨度远大于忙碌时间，会使结果
        │                             整体虚高一个数量级
        │
        ├─ aten::xxx            CPU 侧算子（算子调用的主机端记录）
        │                       含义：其 self device time 挂的是【该算子所启动内核的耗时】
        │                       剔除：与下面的内核条目是同一段时间的两种视角（主机端视角
        │                             vs 设备端视角），两者数值成对相同，同留即翻倍
        │
        └─ void ...kernel<>     真实 设备 内核（设备上实际执行的那段时间）
                                含义：设备真正忙碌的时间
                                保留：彼此互不重叠，相加即为本轮 设备 忙碌总时间

    过滤条件两条缺一不可：ProfilerStep* 自身的 device_type 也被标为设备类型，只看
    device_type 滤不掉它；而 aten:: 条目是 CPU 类型，只按 key 前缀又滤不干净。
    """
    if str(getattr(evt, "key", "")).startswith("ProfilerStep"):
        return False
    dev_t = getattr(evt, "device_type", None)
    if dev_t is None:            # 无该字段的 torch 版本：不过滤，退回旧行为
        return True
    try:
        from torch.autograd import DeviceType
    except ImportError:
        return True
    return dev_t != DeviceType.CPU


def _run_perf(callable_fn, named, attrs, provider, device_id, use_device,
              profile=None, runtime=3):
    """PERF timing via profiler (torch: Self device time; TF: xplane.pb device plane).

    Two passes: (1) profiler pass for device_us (no empty_cache); (2) peak pass
    for peak_memory_mb (separate, reset_peak then re-invoke). device_us is the
    per-iteration average over ``runtime`` active iterations (μs, 3 sig figs).

    CPU / no-device -> device_us=NA. Device EXCLUSIVITY is enforced by the PARENT
    (it holds a lock around this child); this function does NOT lock. reset_peak
    is per-device global, so the parent's lock keeps concurrent PERF from
    corrupting it.
    """
    profile = profile or {}
    perf = {"device_us": "NA", "peak_memory_mb": "NA"}
    # use_device is first in the `and` chain so _device_available is NOT called
    # when profile={} (cpu short-circuit upstream + and-use_device double-guard).
    torch_dev = use_device and provider == "torch" and _device_available(provider, profile)
    tf_dev = use_device and provider == "tf" and _device_available(provider, profile)

    # --- Pass 1: profiler (timing) ---
    outputs = None
    device_us = 0.0          # init before branch: CPU branch never assigns it
    if torch_dev:
        outputs, device_us = _torch_profiler_pass(
            callable_fn, named, attrs, provider, device_id, use_device,
            profile, runtime)
    elif tf_dev:
        outputs, device_us = _tf_profiler_pass(
            callable_fn, named, attrs, provider, device_id, use_device,
            profile, runtime)
    else:
        # CPU or no device: single invoke for outputs only
        outputs = _invoke(callable_fn, named, attrs, provider, device_id,
                          use_device, profile=profile)

    if device_us > 0:
        perf["device_us"] = float(f"{device_us:.3g}")

    # --- Pass 2: peak (memory) ---
    # Pass 2 is best-effort: the op already executed in pass 1 (an op error would
    # have propagated -> FAIL), so any failure here is memory-API breakage ->
    # peak_memory_mb=NA (logged).
    if torch_dev:
        try:
            lib = _TORCH_DEV_MODULE
            lib.reset_peak_memory_stats(device_id)
            _invoke(callable_fn, named, attrs, provider, device_id, use_device,
                    profile=profile)
            lib.synchronize()
            perf["peak_memory_mb"] = (
                lib.max_memory_allocated(device_id) / 1e6)
        except Exception:
            logging.exception("torch peak_memory measure failed; peak_memory_mb=NA")
    elif tf_dev:
        try:
            import tensorflow as tf
            tf_device = f"{profile['tf_device_type']}:{device_id}"
            tf.config.experimental.reset_memory_stats(tf_device)
            _invoke(callable_fn, named, attrs, provider, device_id, use_device,
                    profile=profile)
            info = tf.config.experimental.get_memory_info(tf_device)
            perf["peak_memory_mb"] = info.get("peak", 0) / 1e6
        except Exception:
            logging.exception("tf peak_memory measure failed; peak_memory_mb=NA")

    return outputs, perf


def _torch_profiler_pass(callable_fn, named, attrs, provider, device_id,
                         use_device, profile, runtime):
    """torch.profiler Self device time. Returns (outputs, device_us).

    device_us = per-iteration average (sum of self_device_time_total over the
    active window / runtime), 仅累加真实 设备 内核条目——ProfilerStep* 容器 span 与
    CPU 侧 aten:: 条目会让同一段时间被重复计入，详见 _is_device_kernel。
    torch 2.7+ renamed self_cuda_time_total ->
    self_device_time_total; the 2-candidate _device_time helper covers both.

    Error policy: an OP execution failure PROPAGATES (-> execute_request FAIL) —
    it must never be masked as device_us=NA + PASS. Only PROFILER machinery
    (start / stop / key_averages readout) degrades to NA, and each such failure
    is logged so it is debuggable server-side. A misconfigured profile (missing
    torch_profiler/activities or an unknown activity enum) raises RuntimeError
    OUTSIDE the try block (server config error -> 500, not an NA degrade).
    """
    # gated: only called when provider == "torch" and torch_dev (see _run_perf)
    import torch
    # device + activities resolved OUTSIDE the try block: a missing/unknown
    # value is a server-side config error -> RuntimeError -> 500 (not an NA
    # degrade). cpu path never reaches here (and-use_device short-circuit).
    device = profile["torch_lib"]                       # backend lib name, NOT the role
    try:
        activities_cfg = profile["torch_profiler"]["activities"]
    except KeyError:
        raise RuntimeError("profile missing torch_profiler/activities")
    activities = []
    for name in activities_cfg:
        try:
            activities.append(getattr(torch.profiler.ProfilerActivity, name))
        except AttributeError:
            raise RuntimeError(f"unknown ProfilerActivity: {name}")
    device_us = 0.0
    outputs = None
    # Manual ctx (not `with`): start needs its own try-scope (NA+log on failure)
    # separate from the op loop below (whose errors must propagate, not NA).
    try:
        _ctx = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=runtime, repeat=1))
        prof = _ctx.__enter__()
    except Exception:
        logging.exception("torch.profiler start failed; device_us=NA")
        return outputs, device_us
    # OP execution — errors propagate (-> FAIL).
    try:
        for _ in range(runtime + 2):
            outputs = _invoke(callable_fn, named, attrs, provider,
                              device_id, use_device, profile=profile)
            prof.step()
    finally:
        try:
            _ctx.__exit__(None, None, None)
        except Exception:
            logging.exception("torch.profiler stop failed")
    # Profiler machinery — readout. Failure -> NA + log; outputs preserved.
    try:
        total = sum(_device_time(e, device) for e in prof.key_averages()
                    if _is_device_kernel(e))
        if runtime > 0:
            total /= runtime
        device_us = total
    except Exception:
        logging.exception("torch.profiler key_averages failed; device_us=NA")
    return outputs, device_us


def _tf_profiler_pass(callable_fn, named, attrs, provider, device_id,
                      use_device, profile, runtime):
    """tf.profiler.experimental + xplane.pb. Returns (outputs, device_us).

    device_us = per-iteration average (sum of ev.duration_ps/1e6 over the device
    plane events / runtime). num_occurrences is NOT multiplied (proto3 unset = 0).
    Two warmup invokes precede the profiled runtime invokes. logdir is removed in
    a finally guard so a stop() crash still cleans up.

    Error policy: an OP execution failure PROPAGATES (-> execute_request FAIL).
    Only TF profiler machinery (start / stop / xplane parse) degrades to NA, and
    each such failure is logged so it is debuggable server-side.
    """
    import tempfile, shutil
    from pathlib import Path
    try:
        from tensorflow.core.profiler.protobuf import xplane_pb2
    except ImportError:
        xplane_pb2 = None

    device_us = 0.0
    outputs = None

    logdir = tempfile.mkdtemp(prefix="tfprof_")
    try:
        import tensorflow as tf
        # OP warmup — errors propagate (-> FAIL).
        for _ in range(2):    # warmup (dropped)
            _invoke(callable_fn, named, attrs, provider, device_id, use_device,
                    profile=profile)
        # Profiler machinery — start. Failure -> NA + log.
        try:
            tf.profiler.experimental.start(logdir)
        except Exception:
            logging.exception("tf.profiler start failed; device_us=NA")
            return outputs, device_us
        # OP active window — errors propagate; stop() always runs.
        try:
            for _ in range(runtime):
                outputs = _invoke(callable_fn, named, attrs, provider, device_id,
                                  use_device, profile=profile)
        finally:
            try:
                tf.profiler.experimental.stop()
            except Exception:
                logging.exception("tf.profiler stop failed")
        # Profiler machinery — parse. Failure -> NA + log; outputs preserved.
        pb_path = next(Path(logdir).rglob("*.xplane.pb"), None)
        if pb_path and xplane_pb2:
            try:
                xs = xplane_pb2.XSpace()
                xs.ParseFromString(pb_path.read_bytes())
                total = 0.0
                # Safety cap: a pathological/malformed proto (or a huge trace) can
                # blow up iteration time; cap device-plane event scanning and warn.
                # 100000 comfortably exceeds real op-trace sizes.
                _TF_XPLANE_EVENT_CAP = 100000
                _seen = 0
                _truncated = False
                for plane in xs.planes:
                    if plane.name.startswith("/device"):
                        for line in plane.lines:
                            for ev in line.events:
                                if _seen >= _TF_XPLANE_EVENT_CAP:
                                    _truncated = True
                                    break
                                total += ev.duration_ps / 1e6
                                _seen += 1
                            if _truncated:
                                break
                    if _truncated:
                        break
                if _truncated:
                    logging.warning(
                        "TF xplane event count exceeded %d, truncating",
                        _TF_XPLANE_EVENT_CAP)
                if runtime > 0:
                    total /= runtime
                device_us = total
            except Exception:
                logging.exception("tf xplane parse failed; device_us=NA")
    finally:
        shutil.rmtree(logdir, ignore_errors=True)
    return outputs, device_us


def execute_request(*, tenant_sync_dir, exec_type, provider, api, spec_module,
                    spec_class, mode, input_schema, attrs, tmp_in_path,
                    input_count, device_id, use_device, output_dir,
                    profile=None, op_name=None, op_type=None, runtime=3,
                    **_extra):
    """Run one request. Returns an envelope dict (never raises for 4xx/5xx)."""
    try:
        if tenant_sync_dir and tenant_sync_dir not in sys.path:
            sys.path.insert(0, tenant_sync_dir)
        importlib.invalidate_caches()

        # On-demand import of the torch backend extension. Non-
        # default backends are extension packages; importing torch_{lib}
        # registers the torch.<lib> namespace used by _device_available /
        # memory / ProfilerActivity below. The default backend is built in
        # (already registered -> hasattr True -> skip); a missing extension
        # raises RuntimeError -> 500. torch-only (TF never touches the torch
        # namespace); cpu short-circuits (use_device=False).
        if use_device and provider == "torch" and profile:
            lib = profile.get("torch_lib")
            if lib is None:
                raise RuntimeError("profile missing torch_lib")
            import torch
            if not hasattr(torch, lib):
                try:
                    importlib.import_module(f"torch_{lib}")
                except ModuleNotFoundError as e:
                    raise RuntimeError(f"torch backend '{lib}' unavailable: {e}")
            # Cache the resolved backend module (the getattr(torch, lib) object)
            # for the rest of this request (see module-level _TORCH_DEV_MODULE).
            # global statement: child process writes once per request.
            global _TORCH_DEV_MODULE
            _TORCH_DEV_MODULE = getattr(torch, lib)

        named = {}
        if input_count and tmp_in_path:
            npz = np.load(tmp_in_path)
            flat = [npz[f"a{i}"] for i in range(input_count)]
            named = match_params_v1(input_schema, flat)
        # Framework H2D: numpy inputs -> provider tensors on the target device.
        # dtype is declared per-input in X-Input-Schema (the client's numpy knows
        # the real dtype; the server's may not, e.g. bfloat16).
        device_str = format_device(provider, profile,
                                   "cpu" if not use_device else device_id)
        _dtypes = {e.get("name"): e.get("dtype") for e in (input_schema or [])}
        named = {k: _to_vendor_tensor(v, provider, device_str, _dtypes.get(k))
                 for k, v in named.items()}

        callable_fn, api_label = _resolve_callable(
            exec_type, provider, api, op_name, op_type, spec_module, spec_class)
        if has_perf(mode):
            raw_outputs, perf = _run_perf(callable_fn, named, attrs or {},
                                          provider, device_id, use_device,
                                          profile=profile, runtime=runtime)
        else:
            raw_outputs = _invoke(callable_fn, named, attrs or {},
                                  provider, device_id, use_device,
                                  profile=profile)
            perf = None
        schema, outs = _outputs_to_numpy(raw_outputs, provider)

        if has_data(mode):
            path = os.path.join(output_dir, "out.npz")
            np.savez_compressed(path, **{f"a{i}": o for i, o in enumerate(outs)})
            return _ok(path, len(schema), [list(o.shape) for o in outs],
                       dtypes=None, perf=perf, api=api_label, schema=schema)
        return _ok(None, 0, [], [], perf=perf, api=api_label, schema=[])

    except _MissingSpecDependency as e:
        # 424 is syncable: client will _sync_missing_dependency + retry. Not a
        # failure — info, not warning (avoids noise on the expected first miss).
        logging.info("spec dependency missing: %s (awaiting client sync)", e.name)
        return _err(424, f"missing spec dependency: {e}", missing=e.name,
                    api=api or spec_class)
    except ImportError as e:
        logging.exception("request failed: import error (api=%s)", api or spec_class)
        return _err(500, f"import failed: {e}", api=api or spec_class)
    except (UnknownParamError, ValueError) as e:
        logging.warning("request failed: bad params (api=%s): %s",
                        api or spec_class, e)
        return _err(400, str(e), api=api or spec_class)
    except Exception as e:
        logging.exception("request failed (api=%s)", api or spec_class)
        return _err(500, _client_error(e), api=api or spec_class)


def child_main(conn, kwargs):
    """Child-process entry point: run execute_request, send the envelope back.

    A hard crash (segfault/OOM) kills the child before conn.send -> the parent
    sees a nonzero exitcode and no message -> 500.
    """
    # pop env BEFORE execute_request: sandbox=none execution isolation passes the
    # assigned device via env; if it leaked through, **_extra would silently
    # swallow it and isolation would silently fail. os imported
    # at module top.
    os.environ.update(kwargs.pop("env", {}))
    try:
        conn.send(execute_request(**kwargs))
    except Exception as e:
        logging.exception("child_main: execute_request raised (api=%s)",
                          _api_from_kwargs(kwargs))
        conn.send(_err(500, _client_error(e), api=_api_from_kwargs(kwargs)))
    finally:
        conn.close()
