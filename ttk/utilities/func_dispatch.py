#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Golden/callable dispatch helpers: framework classification + name-based binding.

Note: bind_by_name mirrors ttk/remote/server/execution_container.py:bind_params in
semantics, but server is a ttk-free hermetic module — core cannot depend on it, so
both copies are maintained independently (isolation boundary)."""

import inspect
from typing import Optional, Tuple

import numpy


class UnknownParamError(TypeError):
    """A parameter name in the golden spec is neither a known input/attribute
    nor self/**kwargs. Turns a silent mis-computation from a typo into a loud failure."""


def framework_of(func) -> Optional[str]:
    """Classify a callable by its framework: 'numpy'/'torch'/'tf'/None (custom).

    Reads the object's real ``__module__`` — does NOT guess from signature style.

    Caveats (verified empirically — do NOT simplify):
    - numpy ufunc has no ``__module__`` (access raises AttributeError) → must use getattr fallback.
    - torch top-level functions: ``type(func).__module__ == 'builtins'``, only
      ``func.__module__ == 'torch'`` → check func BEFORE type(func).
    - Do NOT use ``inspect.getmodule``: returns None for ufuncs and torch.ops.aten (OpOverloadPacket).
    """
    for obj in (func, type(func)):
        mod = getattr(obj, "__module__", "") or ""
        if mod.startswith("numpy"):
            return "numpy"
        if mod.startswith("torch"):
            return "torch"
        if mod.startswith(("tensorflow", "tf")):
            return "tf"
    if isinstance(func, numpy.ufunc):
        return "numpy"
    return None


def bind_by_name(func, pool: dict) -> Tuple[list, dict]:
    """Bind parameters of ``func`` from ``pool`` by name.

    - Parameters before ``*`` (POSITIONAL_OR_KEYWORD) → positional args.
    - Parameters after ``*`` (KEYWORD_ONLY) → keyword kwargs.
    - ``*args`` collects unconsumed pool entries (in insertion order) as
      positional args — use this to receive inputs whose names clash with
      Python reserved slots (e.g. an ACLNN param named ``self``).
    - ``**kwargs`` absorbs leftover pool entries (excluding ``self``).
    - A parameter name present in the signature but missing from pool
      (and not self/*args/**kwargs) → ``UnknownParamError``.

    ``self`` is skipped by name; callers should pass a bound method
    (``inst.__call__``) so ``self`` is auto-stripped by Python.
    A pool entry named ``self`` is never injected into ``**kwargs`` to avoid
    ``multiple values for argument 'self'``; it is reachable via ``*args``.
    """
    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_kw = False
    has_var_pos = False
    consumed = set()
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            seen_star = True
            has_var_pos = True
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
            continue
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            seen_star = True
        if name in pool:
            consumed.add(name)
            if seen_star:
                kwargs[name] = pool[name]
            else:
                args.append(pool[name])
        elif p.default is inspect.Parameter.empty:
            # Required param missing from pool: raise (not greedy-bind an
            # unrelated unconsumed entry) so a golden/input name typo fails
            # loudly instead of silently computing against the wrong value.
            raise UnknownParamError(
                f"parameter '{name}' of {getattr(func, '__qualname__', func)} is not a known input or attribute name"
            )
        # has a default: leave it to Python (skip, use the default)
    leftover = [(k, v) for k, v in pool.items() if k not in consumed]
    if has_var_pos:
        args.extend(v for _, v in leftover)
    if has_var_kw:
        kwargs.update({k: v for k, v in leftover if k != "self"})
    return args, kwargs


def resolve_callable_str(s: str):
    """Resolve a dotted-path string like 'numpy.abs' / 'torch.mm' / 'tf.raw_ops.Add' into a callable.

    Lazy import：只 import 字符串引用的框架（避免解析 'numpy.add' 时 eager-import
    torch/tensorflow —— 某些环境的 tensorflow C 扩展 import 即 segfault）。
    """
    ns = {"numpy": numpy, "np": numpy}
    if s == "torch" or s.startswith("torch."):
        import torch

        ns["torch"] = torch
    elif s == "tf" or s == "tensorflow" or s.startswith("tf.") or s.startswith("tensorflow."):
        import tensorflow as tf

        ns["tf"] = tf
        ns["tensorflow"] = tf
    try:
        return eval(s, ns)
    except Exception as e:
        raise ValueError(f"Cannot resolve golden callable {s!r}: {e}") from e
