"""
ExecutionContainer - Dual-mode dispatch and parameter matching.

Deployment constraint: This file MUST NOT import ttk.*.
"""

import importlib
import inspect
import logging
from typing import Optional

_FRAMEWORK_RESERVED = {"device"}
_PROVIDER_MODULE_ALIASES = {"tf": "tensorflow"}

# X-Mode bitmask. Defined HERE (not imported from ttk.remote) so the
# server package deploys standalone on the XPU box with no TTK framework. Keep
# the bit values in sync with ttk/remote/__init__.py (the client-side copy).
DATA = 0b01  # return device outputs
PERF = 0b10  # collect performance


def has_data(mode: int) -> bool:
    return bool(mode & DATA)


def has_perf(mode: int) -> bool:
    return bool(mode & PERF)


class UnknownParamError(ValueError):
    """A parameter name is neither a known input/attribute nor
    self / a framework-reserved name / **kwargs.

    Turns a misspelled parameter name from a silent mis-computation into a
    loud failure.
    """


def bind_params(func, name_to_value: dict, device: Optional[str] = None, warn_leftover: bool = True) -> tuple:
    """Bind a name->value map onto ``func``'s signature BY PARAMETER NAME.

    Returns ``(args, kwargs)``. Each declared param (except self / VAR) is resolved by:
      * name in ``name_to_value`` -> take value (positional before ``*``, keyword after);
      * framework-reserved name (``device``) -> inject ``device`` (when not None);
      * has a default -> use the default (skip, no error);
      * otherwise -> :class:`UnknownParamError` (neither input/attr nor defaulted).
    ``*args`` collects unconsumed entries (in insertion order) as positional args.
    ``**kwargs`` absorbs leftover entries (excluding ``self``); otherwise
    ``warn_leftover`` controls whether unconsumed entries log a warning (default True).
    A pool entry named ``self`` is never injected into ``**kwargs``; use ``*args``
    to receive it positionally.
    """
    sig = inspect.signature(func)
    args: list = []
    kwargs: dict = {}
    seen_star = False
    has_var_keyword = False
    has_var_positional = False
    consumed = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        kind = param.kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if kind is inspect.Parameter.VAR_POSITIONAL:
            seen_star = True
            has_var_positional = True
            continue
        if kind is inspect.Parameter.KEYWORD_ONLY:
            seen_star = True
        if name in _FRAMEWORK_RESERVED:
            consumed.add(name)
            if device is not None:
                kwargs[name] = device
            continue
        if name in name_to_value:
            consumed.add(name)
            value = name_to_value[name]
            if seen_star:
                kwargs[name] = value
            else:
                args.append(value)
        elif param.default is inspect.Parameter.empty:
            # Required param missing from the pool: raise (not greedy-bind an
            # unrelated unconsumed entry) so an input/attr name typo fails
            # loudly instead of silently binding the wrong value.
            qual = getattr(func, "__qualname__", getattr(func, "__name__", func))
            raise UnknownParamError(f"parameter '{name}' of {qual} is not a known input or attribute name")
        # has a default: leave it to Python (skip, use the default)
    leftover = {k: v for k, v in name_to_value.items() if k not in consumed}
    if has_var_positional:
        args.extend(leftover.values())
    if has_var_keyword:
        kwargs.update({k: v for k, v in leftover.items() if k != "self"})
    elif not has_var_positional and leftover and warn_leftover:
        logging.warning("dispatch: inputs/attrs not consumed by signature: %s", sorted(leftover))
    return args, kwargs


def format_device(provider, profile, device_id):
    """Build the framework device spec. cpu short-circuits
    before reading profile; otherwise the lib/type comes from profile."""
    if str(device_id) == "cpu":
        return "cpu"
    if provider == "torch":
        return f"{profile['torch_lib']}:{device_id}"
    if provider == "tf":
        tf_type = profile.get("tf_device_type")
        if tf_type is None:
            raise ValueError("hardware has no tf_device_type, tf provider unavailable")
        return f"/device:{tf_type}:{device_id}"
    raise ValueError(f"unknown provider: {provider}")


def to_device(value, device_str: str, provider: str):
    """Move a value onto the device (Mode B framework-side H2D).

    torch is imported ONLY for the torch provider; tf/other providers never touch
    it. No-op when device is None/'cpu', or for non-tensor values.
    """
    if value is None or device_str in (None, "cpu"):
        return value
    if provider == "torch":
        try:
            import torch
        except ImportError:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(device_str)
        if isinstance(value, (list, tuple)):
            moved = [to_device(v, device_str, provider) for v in value]
            return type(value)(moved)
    return value


def resolve_callable(api_str: str):
    """Resolve a dotted ``module.attr.sub`` string to a callable (api mode).

    Rejects classes: api mode is defined for stateless functions;
    a class (e.g. ``torch.nn.Softmax``) is a misuse -> ValueError.

    Raises:
        ValueError: if the string has no dot, or resolves to a class.
        ImportError/AttributeError: if the module/attr does not exist
            (propagated; the executor maps these to the appropriate status).
    """
    parts = api_str.split(".")
    if len(parts) < 2:
        raise ValueError(f"api '{api_str}' must be a dotted 'module.attr' path")
    # Map short names (e.g. "tf") to real package names (e.g. "tensorflow")
    if parts[0] in _PROVIDER_MODULE_ALIASES:
        parts[0] = _PROVIDER_MODULE_ALIASES[parts[0]]
    obj = importlib.import_module(parts[0])
    for attr in parts[1:]:
        obj = getattr(obj, attr)
    if inspect.isclass(obj):
        raise ValueError(f"api mode requires a function, got a class: {api_str}")
    return obj


def match_params_v1(schema: list, flat_arrays: list) -> dict:
    """Restore named inputs from X-Input-Schema and flat arrays.

    Args:
        schema: List of {"name": str, "index"|"indices"|None}
        flat_arrays: Flat list of numpy arrays (no None gaps)

    Returns:
        Dict mapping name to value (single ndarray, list of ndarrays, or None)
    """
    result = {}
    for entry in schema:
        name = entry["name"]
        if "indices" in entry:
            result[name] = [flat_arrays[i] for i in entry["indices"]]
        elif "index" in entry:
            idx = entry["index"]
            result[name] = flat_arrays[idx] if idx is not None else None
    return result
