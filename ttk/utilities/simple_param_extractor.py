#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""
Simple parameter extractor for torch/torch_npu APIs.

Multi-strategy fallback:
  1. TypeError fallback — call with wrong args, parse error message
  2. Manual override config — for APIs that can't be auto-parsed or have incorrect parsing

Returns APIParamInfo with typed parameter list including Tensor/List[Tensor]/Scalar/etc.
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParamInfo:
    name: str
    type: str = ""
    default: Any = None
    is_optional: bool = False
    is_keyword_only: bool = False
    is_var_positional: bool = False

    @property
    def is_tensor(self):
        """True if any member type is a single Tensor."""
        return any(t in ("Tensor", "tensor") for t in self.type.split("|"))

    @property
    def is_tensor_list(self):
        """True if any member type is a sequence of Tensors."""
        return any(
            t in ("tuple of Tensors", "list of Tensors", "Tensor[]", "List[Tensor]") for t in self.type.split("|")
        )

    @property
    def is_scalar(self):
        """True if any member type is a scalar (Number, int, float, bool, str)."""
        return any(t in ("Number", "Scalar", "int", "float", "bool", "str") for t in self.type.split("|"))

    @property
    def is_scalar_list(self):
        """True if any member type is a list of scalars."""
        return any(t in ("Scalar[]", "List[Scalar]", "list of Numbers") for t in self.type.split("|"))

    @property
    def is_tensor_like(self):
        """True if the param accepts Tensor data (single tensor or tensor list)."""
        return self.is_tensor or self.is_tensor_list


@dataclass
class OverloadTensorLayout:
    """Tensor parameter layout for a single overload, split by input/out role."""

    input_params: List[ParamInfo]
    required_input_count: int
    input_count: int
    has_var_input: bool
    out_param: Optional[ParamInfo]
    is_out_required: bool
    is_out_tensor_list: bool
    out_expected_count: int

    @staticmethod
    def build(overload_params: List[ParamInfo], return_count: int = 0) -> "OverloadTensorLayout":
        input_params = [p for p in overload_params if p.is_tensor_like and p.name != "out"]
        has_var_input = any(getattr(p, "is_var_positional", False) for p in input_params)
        required_input_count = sum(
            1 for p in input_params if not p.is_optional and not getattr(p, "is_var_positional", False)
        )

        out_param = next((p for p in overload_params if p.name == "out" and p.is_tensor_like), None)
        is_out_required = out_param is not None and not out_param.is_optional
        is_out_tensor_list = out_param.is_tensor_list if out_param else False

        if out_param is None:
            out_expected_count = 0
        elif is_out_tensor_list:
            out_expected_count = return_count if return_count > 0 else 0
        else:
            out_expected_count = 1

        return OverloadTensorLayout(
            input_params=input_params,
            required_input_count=required_input_count,
            input_count=len(input_params),
            has_var_input=has_var_input,
            out_param=out_param,
            is_out_required=is_out_required,
            is_out_tensor_list=is_out_tensor_list,
            out_expected_count=out_expected_count,
        )


@dataclass
class OverloadInfo:
    """Single overload: its parameter list and pre-computed tensor layout."""

    params: List[ParamInfo]
    return_count: int = 0
    layout: OverloadTensorLayout = field(init=False, repr=False)

    def __post_init__(self):
        self.layout = OverloadTensorLayout.build(self.params, self.return_count)


def _normalize_to_overload_infos(overloads, return_counts=None):
    """Convert List[List[ParamInfo]] or List[OverloadInfo] → List[OverloadInfo]."""
    if return_counts is None:
        return_counts = []
    result = []
    for i, ov in enumerate(overloads):
        if isinstance(ov, OverloadInfo):
            result.append(ov)
        elif isinstance(ov, list):
            rc = return_counts[i] if i < len(return_counts) else 0
            result.append(OverloadInfo(params=ov, return_count=rc))
    return result


@dataclass
class APIParamInfo:
    api_name: str
    params: List[ParamInfo] = field(default_factory=list)
    source: str = ""
    overloads: list = field(default_factory=list)
    is_tensor_method: bool = False
    is_inplace: bool = False
    inplace_param_indexes: List[int] = field(default_factory=list)
    _return_counts: List[int] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.overloads:
            self.overloads = _normalize_to_overload_infos(self.overloads, self._return_counts)
            best = max(self.overloads, key=lambda oi: sum(1 for p in oi.params if p.is_optional))
            self.params = best.params
        elif self.params and not self.overloads:
            self.overloads = _normalize_to_overload_infos([self.params], self._return_counts)

    def _resolve_api_flags(self):
        parts = self.api_name.split(".") if self.api_name else []
        self.is_tensor_method = len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor"
        func_name = parts[-1] if parts else ""
        is_inplace_name = func_name.endswith("_") and not func_name.startswith("_")
        if is_inplace_name:
            try:
                self._detect_inplace_from_schema()
            except Exception:
                pass
            if not self.is_inplace:
                self.is_inplace = True
                tensor_params = [i for i, p in enumerate(self.params) if p.is_tensor_like]
                if tensor_params:
                    self.inplace_param_indexes = [tensor_params[0]]

    def _detect_inplace_from_schema(self):
        obj = _resolve_function(self.api_name)
        if obj is None:
            return
        schema = None
        if hasattr(obj, "default") and hasattr(obj.default, "_schema"):
            schema = obj.default._schema
        if schema is None:
            return
        return_aliases = set()
        for ret in schema.returns:
            ai = ret.alias_info
            if ai and ai.is_write:
                return_aliases |= ai.before_set
        if not return_aliases:
            return
        self.is_inplace = True
        tensor_idx = 0
        for arg in schema.arguments:
            ai = arg.alias_info
            if ai and ai.is_write and (ai.before_set & return_aliases):
                self.inplace_param_indexes.append(tensor_idx)
            if ai is not None or (arg.type and str(arg.type) == "Tensor"):
                tensor_idx += 1

    @property
    def tensors(self):
        return [p for p in self.params if p.is_tensor_like]

    @property
    def scalars(self):
        return [p for p in self.params if p.is_scalar or p.is_scalar_list]

    @property
    def tensor_count(self):
        return len(self.tensors)

    @property
    def scalar_count(self):
        return len(self.scalars)

    @property
    def tensor_distribution(self):
        """Return distribution tuple: -1 for TensorList, 0 for single Tensor."""
        dist = []
        for p in self.tensors:
            if p.is_tensor_list:
                dist.append(-1)
            else:
                dist.append(0)
        return tuple(dist)

    def match_overload(self, tensor_count, nested_flags=None, skip_flags=None):
        """Check if testcase matches ANY overload signature.

        Args:
            tensor_count: Number of top-level input tensors (excluding out).
            nested_flags: Optional list of bools — True if position is nested (TensorList).
            skip_flags: Optional list of bools — True to skip type check at that position
                        (used for None elements which match any type).
        Returns:
            (matched, input_params_list, overload_index) or (False, None, -1).
        """
        for oidx, ov in enumerate(self.overloads):
            layout = ov.layout
            if layout.has_var_input:
                if tensor_count < layout.required_input_count:
                    continue
            else:
                if tensor_count < layout.required_input_count or tensor_count > layout.input_count:
                    continue
            if nested_flags is not None:
                type_ok = True
                non_var = [p for p in layout.input_params if not getattr(p, "is_var_positional", False)]
                for idx in range(tensor_count):
                    if skip_flags and idx < len(skip_flags) and skip_flags[idx]:
                        continue
                    if idx < len(non_var):
                        param = non_var[idx]
                    else:
                        continue
                    is_nested = nested_flags[idx]
                    if param.is_tensor_list and not is_nested:
                        type_ok = False
                        break
                    if param.is_tensor and is_nested:
                        type_ok = False
                        break
                if not type_ok:
                    continue
            return True, layout.input_params, oidx
        return False, None, -1


def _split_by_comma(s: str) -> List[str]:
    parts = []
    current = []
    depth = 0
    for char in s:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_params_from_signature(sig: str) -> Optional[List[ParamInfo]]:
    if not sig or sig.isspace():
        return None
    sig = sig.strip()
    if sig.startswith("(") and sig.endswith(")"):
        sig = sig[1:-1]
    params = []
    parts = _split_by_comma(sig)
    keyword_only = False
    for part in parts:
        part = part.strip()
        if not part or part == "/":
            continue
        if part in ("*", "\\*"):
            keyword_only = True
            continue
        default = None
        is_optional = keyword_only
        is_var_positional = False
        if "=" in part:
            main_part, default = part.split("=", 1)
            main_part = main_part.strip()
            default = default.strip()
            is_optional = True
        else:
            main_part = part
        # Detect *args (VAR_POSITIONAL) prefix, e.g. "*operands" or "*tensors"
        if main_part.startswith("*") and not main_part.startswith("**"):
            is_var_positional = True
            main_part = main_part[1:].strip()
        tokens = main_part.split()
        if len(tokens) >= 2 and not tokens[0].startswith('"'):
            first = tokens[0]
            if ":" in first:
                name = first.split(":")[0]
                type_hint = first.split(":", 1)[1].strip() or tokens[1]
                type_hint = _normalize_doc_type(type_hint)
            else:
                type_hint = tokens[0]
                name = tokens[-1].rstrip("?)")
                if type_hint in ("tuple", "list") and len(tokens) >= 3:
                    type_hint = f"{tokens[0]} of {tokens[2]}"
        elif len(tokens) == 1:
            tok = tokens[0].strip("\"'")
            if ":" in tok:
                name, type_hint = tok.split(":", 1)
                name = name.strip()
                type_hint = type_hint.strip()
                type_hint = _normalize_doc_type(type_hint)
                if not type_hint:
                    # Priority: default value inference > name inference
                    # _infer_type_from_value returns empty string for unrecognizable
                    # values (e.g., 'torch.xxx') and 'Optional[Tensor]' for None —
                    # in both cases, fall back to name-based inference.
                    type_hint = ""
                    if default is not None:
                        type_hint = _infer_type_from_value(default)
                    if not type_hint or type_hint == "Optional[Tensor]":
                        type_hint = _infer_type_from_name(name)
            else:
                name = tok
                # Priority: default value inference > name inference
                type_hint = ""
                if default is not None:
                    type_hint = _infer_type_from_value(default)
                if not type_hint or type_hint == "Optional[Tensor]":
                    type_hint = _infer_type_from_name(name)
        else:
            continue
        if default is not None and isinstance(default, str):
            value_type = _infer_type_from_value(default)
            if type_hint == "Tensor" and value_type not in ("Tensor", "Optional[Tensor]", "tuple of Tensors"):
                type_hint = value_type
            default = _coerce_default_value(default, type_hint)
        # Strip trailing '?' from type — it indicates optional in some
        # docstring formats (e.g., 'str?', 'int[]?'). Mark as optional
        # and keep the default value if present, otherwise default to None.
        if type_hint.endswith("?"):
            type_hint = type_hint[:-1]
            is_optional = True
            if default is None:
                default = None
        params.append(
            ParamInfo(
                name=name,
                type=type_hint,
                default=default,
                is_optional=is_optional,
                is_keyword_only=keyword_only,
                is_var_positional=is_var_positional,
            )
        )
    return params if params else None


def _normalize_doc_type(type_str: str) -> str:
    """Normalize type annotations from docstrings.

    Examples: Tensor, Optional[Tensor], List[Tensor], int, float, bool, str
    """
    type_str = type_str.strip()
    if not type_str:
        return ""
    if type_str.startswith("Optional["):
        inner = type_str[len("Optional[") : -1]
        return inner
    if type_str.startswith("List["):
        inner = type_str[len("List[") : -1]
        if inner == "Tensor":
            return "List[Tensor]"
    return type_str


def _parse_docstring_args_section(doc: str) -> Dict[str, Tuple[str, bool]]:
    """Parse param types from docstring Args/Keyword arguments sections.

    Matches lines like:
        input (Tensor): the input tensor.
        other (Tensor or Number): ...
        out (Tensor, optional): ...

    Returns dict of {param_name: (raw_type_string, is_optional)}.
    """
    if not doc:
        return {}
    type_map = {}
    in_args = False
    args_indent = 0
    args_terminators = {
        "Returns:",
        "Example::",
        "Examples::",
        "Raises:",
        "Note:",
        "Notes:",
        "Warning:",
        "Warnings:",
        "See Also:",
        "References::",
        "References:",
        "Todo:",
    }
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Keyword arguments:", "Keyword Args:", "Keyword args:", "Kwargs:"):
            in_args = True
            args_indent = len(line) - len(line.lstrip())
            continue
        if not in_args:
            continue
        if (
            stripped in args_terminators
            or stripped.startswith(tuple(args_terminators))
            or (not stripped.startswith((" ", "\t", "*", ".")) and stripped.endswith(":") and stripped == line.rstrip())
        ):
            break
        # Sphinx directives like ".. versionchanged::" appear inside Args
        # sections but should not terminate parsing — just skip them.
        if stripped.startswith(".."):
            continue
        if not stripped:
            continue
        line_indent = len(line) - len(line.lstrip())
        # Skip continuation lines: indent strictly deeper than the first
        # parameter definition level (args_indent + 4).  Using relative
        # indentation instead of a hardcoded threshold (>= 12) ensures
        # correctness for deeply-indented docstrings (e.g. torch.meshgrid).
        if line_indent > args_indent + 4:
            continue
        m = re.match(r"\*{0,2}(\w+)\s*\(([^)]+)\).*:", stripped)
        if m:
            name = m.group(1)
            type_str = m.group(2).strip()
            is_opt = "optional" in type_str.lower()
            type_str = re.sub(r",\s*optional\s*$", "", type_str, flags=re.IGNORECASE)
            type_str = re.sub(r",\s*required\s*$", "", type_str, flags=re.IGNORECASE)
            if type_str:
                type_map[name] = (type_str, is_opt)
            continue
        m = re.match(r"\*{0,2}(\w+)\s*:\s*(.+)", stripped)
        if m:
            name = m.group(1)
            desc = m.group(2).strip()
            if name.lower().startswith(("default", "attr", "note", "example", "deprecated", "version", "see", "if")):
                continue
            if desc:
                type_map.setdefault(name, (desc, False))
    return type_map


def _parse_return_count_from_first_line(first_line: str) -> int:
    """Extract the number of return tensors from a docstring first-line ``->`` annotation.

    Examples::

        "func(...) -> Tensor"                          → 1
        "func(...) -> (Tensor, LongTensor)"            → 2
        "func(...) -> (Tensor, Tensor, Tensor)"        → 3
        "func(...) -> None"                             → 0
        "func(...) -> torch.return_types.sort"          → 0  (unresolvable custom type)
        "func(...)"                                     → 0  (no annotation)
    """
    m = re.search(r"\)\s*->\s*(.+)$", first_line)
    if not m:
        return 0
    ret = m.group(1).strip()
    if ret.startswith("(") and ret.endswith(")"):
        parts = [p.strip() for p in ret[1:-1].split(",") if p.strip()]
        return len(parts)
    if ret.lower() in ("none", ""):
        return 0
    # Single named type like "Tensor" → 1; unresolvable types like
    # "torch.return_types.sort" → 0 (we cannot count fields).
    if re.match(r"^[A-Za-z_]\w*$", ret):
        return 1
    return 0


_ARGS_TYPE_NORMALIZE = {
    "Tensor": "Tensor",
    "tensor": "Tensor",
    "``Tensor``": "Tensor",
    "Number": "Number",
    "Scalar": "Number",
    "``scalar``": "Number",
    "bool": "bool",
    ":class:`bool`": "bool",
    "SymBool or bool": "bool",
    "SymInt or int": "int",
    "int": "int",
    "``int``": "int",
    "integer": "int",
    "int32": "int",
    "int64": "int",
    "float": "float",
    "str": "str",
    "dtype": "Dtype",
    ":class:`torch.dtype`": "Dtype",
    "torch.dtype": "Dtype",
    "device": "Device",
    ":class:`torch.device`": "Device",
    "torch.device": "Device",
    "layout": "Layout",
    ":class:`torch.layout`": "Layout",
    "torch.layout": "Layout",
    "torch.Size": "tuple of ints",
    ":class:`torch.Size`": "tuple of ints",
    ":class:`torch.Generator`": "Generator",
    "torch.Generator": "Generator",
    "Generator": "Generator",
    ":class:`torch.memory_format`": "MemoryFormat",
    "torch.memory_format": "MemoryFormat",
    "memory_format": "MemoryFormat",
    "BoolTensor": "Tensor",
    "IntTensor or LongTensor": "Tensor",
    "LongTensor": "Tensor",
    "torch.ByteTensor": "Tensor",
    "Tensors...": "tuple of Tensors",
    "Tensor[]": "tuple of Tensors",
    "List[Tensor]": "tuple of Tensors",
    "list of Tensor": "tuple of Tensors",
    "sequence of Tensors": "tuple of Tensors",
    "sequence of float": "list of Numbers",
    "torch.Tensor": "Tensor",
    "torch.BoolTensor": "Tensor",
    "torch.IntTensor": "Tensor",
    "torch.LongTensor": "Tensor",
    "torch.FloatTensor": "Tensor",
    "torch.DoubleTensor": "Tensor",
}


def _normalize_args_type(type_str: str) -> str:
    """Normalize Args section type to our type system.

    Handles: "Tensor", "int or tuple of ints", "``Tensor``", ":class:`bool`", etc.

    Union types ("X or Y") are handled as follows:
    - Tensor-containing members (Tensor, tuple of Tensors) are excluded because
      those overloads are already captured by TypeError multi-overload resolution.
    - Remaining scalar-level members are normalized and joined with '|'.
    - If only one non-Tensor member remains, it is returned directly (no union).
    - If no non-Tensor members survive, returns empty string.

    Examples:
        'int or Tuple[int]'    → 'int|tuple of ints'
        'float or Tuple[float]'→ 'float|tuple of floats'
        'Tensor or Number'     → 'Number'
        'Tensor or float'      → 'float'
        'int or list or tuple of ints' → 'int|tuple of ints'
    """
    type_str = type_str.strip()
    if not type_str:
        return ""
    # Handle Optional[X] — strip the wrapper and normalize the inner type
    opt_match = re.match(r"Optional\[(.+)\]$", type_str)
    if opt_match:
        return _normalize_args_type(opt_match.group(1))
    if type_str in _ARGS_TYPE_NORMALIZE:
        return _ARGS_TYPE_NORMALIZE[type_str]
    # Handle backtick-quoted types: ``int``, ``list of int``
    if type_str.startswith("``") and type_str.endswith("``"):
        inner = type_str[2:-2]
        if inner in _ARGS_TYPE_NORMALIZE:
            return _ARGS_TYPE_NORMALIZE[inner]
        return ""
    # Handle :class:`...` pattern
    cm = re.match(r":class:`([^`]+)`", type_str)
    if cm:
        return _normalize_args_type(cm.group(1))
    # Union types: "int or Tuple[int]", "Tensor or Number", etc.
    # Tensor-containing members are excluded (handled by overload resolution).
    # Scalar-level members are joined with '|' for _coerce_value to try in order.
    if " or " in type_str:
        parts = [p.strip() for p in type_str.split(" or ")]
        _TENSOR_TYPES = frozenset({"Tensor", "tuple of Tensors", "list of Tensors", "List[Tensor]", "Tensor[]"})
        normalized_parts = []
        for part in parts:
            n = _normalize_args_type(part)
            if n and n not in _TENSOR_TYPES:
                normalized_parts.append(n)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for n in normalized_parts:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        if len(unique) >= 2:
            return "|".join(unique)
        elif len(unique) == 1:
            return unique[0]
        return ""
    # Handle comma-separated: "int, float, inf, -inf, 'fro', 'nuc'"
    # Treated as union types (same as "X or Y"), each member normalized
    # and joined with '|'. Quoted values like 'fro' are recognized as 'str'.
    # Skipped for types that genuinely contain commas (e.g., "tuple of ints").
    if "," in type_str and not type_str.startswith(("tuple", "list", "Tuple", "List")):
        parts = [p.strip() for p in type_str.split(",")]
        _TENSOR_TYPES = frozenset({"Tensor", "tuple of Tensors", "list of Tensors", "List[Tensor]", "Tensor[]"})
        normalized_parts = []
        for part in parts:
            # Quoted string values like 'fro', 'nuc' → 'str'
            if (part.startswith("'") and part.endswith("'")) or (part.startswith('"') and part.endswith('"')):
                normalized_parts.append("str")
                continue
            n = _normalize_args_type(part)
            if n and n not in _TENSOR_TYPES:
                normalized_parts.append(n)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for n in normalized_parts:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        if len(unique) >= 2:
            return "|".join(unique)
        elif len(unique) == 1:
            return unique[0]
        return ""
    # Handle "tuple of ints", "list of Tensors"
    if type_str.startswith("tuple of ") or type_str.startswith("list of "):
        inner = type_str.split(" of ")[1].strip()
        if inner in ("ints", "int"):
            return "tuple of ints"
        if inner in ("floats", "float"):
            return "tuple of floats"
        if inner in ("Tensors", "Tensor"):
            return "tuple of Tensors"
        if inner in ("Scalars", "Scalar", "numbers", "Number"):
            return "list of Numbers"
    # Handle "Tuple[int]", "Tuple[int, int]", "List[float]", etc.
    # Extract element type from generic bracket syntax and map to our type system.
    bracket_match = re.match(r"(Tuple|List)\[([^\]]+)\]", type_str)
    if bracket_match:
        container = bracket_match.group(1)
        inner = bracket_match.group(2).strip()
        parts = [p.strip() for p in inner.split(",")]
        unique = set(parts)
        if unique <= {"int", "int64", "int32"}:
            return "tuple of ints"
        if unique <= {"float", "double"}:
            return "tuple of floats"
        if unique == {"bool"}:
            return "tuple of bools"
        if unique == {"Tensor"}:
            return "tuple of Tensors"
        if len(unique) == 1 and parts[0] in ("float", "int", "bool", "str"):
            return f"tuple of {parts[0]}s"
        return "tuple of ints" if container == "Tuple" else "list of Numbers"
    # If it's a known simple type, return it
    known_simple = {
        "Tensor",
        "tensor",
        "Number",
        "Scalar",
        "bool",
        "int",
        "float",
        "str",
        "int64",
        "int32",
        "object",
        "Object",
        "Generator",
        "function",
        "callable",
    }
    if type_str in known_simple:
        return _ARGS_TYPE_NORMALIZE.get(type_str, type_str)
    # Descriptive text — return empty, let caller fall back to name inference
    return ""


def _infer_type_from_name(name: str) -> str:
    tensor_names = {
        "input",
        "output",
        "other",
        "self",
        "tensor",
        "weight",
        "bias",
        "mat1",
        "mat2",
        "x",
        "y",
        "src",
        "dst",
        "query",
        "key",
        "mask",
        "grad",
        "indices",
        "values",
        "h_0",
        "c_0",
        "hidden",
        "cell",
        "source",
        "target",
    }
    tensor_list_names = {
        "tensors",
        "inputs",
        "targets",
        "features",
        "labels",
        "boxes",
    }
    scalar_names = {
        "dim",
        "dims",
        "axis",
        "size",
        "sizes",
        "shape",
        "shapes",
        "stride",
        "strides",
        "padding",
        "dilation",
        "alpha",
        "beta",
        "gamma",
        "epsilon",
        "momentum",
        "p",
        "dropout",
        "training",
        "inplace",
    }
    special_types = {
        "dtype": "Dtype",
        "layout": "Layout",
        "device": "Device",
        "memory_format": "MemoryFormat",
        "requires_grad": "bool",
        "pin_memory": "bool",
        "inplace": "bool",
        "training": "bool",
        "keepdim": "bool",
        "keepdims": "bool",
        "approximate": "str",
        "ceil_mode": "bool",
        "return_indices": "bool",
        "kernel_size": "int",
        "output_size": "int",
        "eps": "float",
        "epsilon": "float",
        "exponential_average_factor": "float",
        "align_corners": "bool",
        "transposed": "bool",
        "groups": "int",
        "numel": "int",
        "n_bins": "int",
        "ratio": "float",
        "bit_width": "int",
        "count": "int",
        "counts": "int",
        "mode": "int",
        "hidden_size": "int",
        "num_layers": "int",
        "batch_first": "bool",
        "bidirectional": "bool",
        "has_biases": "bool",
        "reverse": "bool",
        "bias_defined": "bool",
        "n": "int",
        "c": "int",
        "h": "int",
        "w": "int",
        "hxw": "int",
        "group": "int",
        "full": "bool",
        "reduction": "str",
        "log_input": "bool",
        "observer_on": "bool",
        "fake_quant_on": "bool",
        "averaging_const": "float",
        "quant_min": "int",
        "quant_max": "int",
        "ch_axis": "int",
        "interpolation_mode": "int",
        "padding_mode": "int",
        "benchmark": "bool",
        "deterministic": "bool",
        "allow_tf32": "bool",
        "weight_stride0": "int",
        "dropout_state": "bool",
        "use_input_stats": "bool",
        "cudnn_enabled": "bool",
        "train": "bool",
        "reduce": "str",
        "include_self": "bool",
    }
    name_lower = name.lower().strip("\"'")
    if name_lower in special_types:
        return special_types[name_lower]
    if name_lower in tensor_list_names:
        return "tuple of Tensors"
    if name_lower in tensor_names:
        return "Tensor"
    if name_lower in scalar_names:
        if name_lower in ("p", "alpha", "beta", "gamma", "epsilon", "dropout"):
            return "float"
        return "int"
    # Pattern-based inference for parameter names not in exact lists
    # dim0/dim1/axis0/axis1 etc.
    if name_lower.startswith(("dim", "axis")):
        return "int"
    # chunks/split_size/chunk_size etc.
    if "chunk" in name_lower or "split" in name_lower:
        return "int"
    # threshold/lambd/tol/tolerance
    if any(k in name_lower for k in ("threshold", "lambd", "tol")):
        return "float"
    # pad/padding_mode (but 'padding' already in scalar_names)
    if name_lower.startswith("pad") and name_lower not in tensor_names:
        return "int"
    # scale/scale_value
    if name_lower.startswith("scale"):
        return "float"
    # negative_slope/lower/upper/min_val/max_val
    if name_lower in (
        "negative_slope",
        "lower",
        "upper",
        "min_val",
        "max_val",
        "correction",
        "fill_value",
        "margin",
        "value",
    ):
        return "float"
    return "Tensor"


def _infer_type_from_value(value: str) -> str:
    """Infer parameter type from its default value string.

    Priority: None→Optional[Tensor], True/False→bool, quoted→str,
    numeric (e.g., 1.0, 3)→float/int.
    Module attribute references like 'torch.xxx' are NOT treated as float
    even though they contain '.'.
    """
    value = value.strip()
    if value == "None":
        return "Optional[Tensor]"
    if value in ("True", "False"):
        return "bool"
    if value.startswith(("'", '"')):
        return "str"
    # Try list/tuple literal: [1, 2], (0, 0) → tuple of ints/floats
    if value.startswith(("[", "(")) and value.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                if all(isinstance(x, bool) for x in parsed):
                    return "tuple of bools"
                if all(isinstance(x, int) for x in parsed):
                    return "tuple of ints"
                return "tuple of floats"
        except (ValueError, SyntaxError):
            pass
    # Try parsing as numeric first — only values that are actually numbers
    # should be classified as int/float. Strings like 'torch.preserve_format'
    # or 'fro' will fail numeric parsing and fall through to return "".
    try:
        float(value)
        if "." in value or "e" in value.lower():
            return "float"
        return "int"
    except (ValueError, OverflowError):
        return ""


def _extract_params_from_type_error(obj) -> Optional[Tuple[List[ParamInfo], str, List[List[ParamInfo]]]]:
    """Parse TypeError from calling obj() with wrong args.

    Returns (primary_params, source, all_overloads).
    For multi-overload APIs, all_overloads contains all parsed signatures.
    For single-signature APIs, all_overloads is empty (use primary_params).
    """
    try:
        obj()
    except TypeError as e:
        error_msg = str(e)
        if "expected one of:" in error_msg:
            all_overloads = []
            for line in error_msg.split("\n"):
                line = line.strip()
                if line.startswith("*"):
                    sig = line[1:].strip()
                    if sig.startswith("(") and sig.endswith(")"):
                        sig = sig[1:-1]
                    parsed = _parse_params_from_signature(sig)
                    if parsed:
                        all_overloads.append(parsed)
            if all_overloads:
                return all_overloads[0], f"TypeError(multi-overload, {len(all_overloads)} signatures)", all_overloads
        m = re.search(r"expected\s*\(([^)]+(?:\([^)]*\))*[^)]*)\)", error_msg)
        if m:
            sig = m.group(1)
            params = _parse_params_from_signature(sig)
            if params:
                return params, "TypeError(single-signature)", []
    except Exception:
        pass
    return None


def _extract_params_from_tensor_call(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Call with a Tensor argument to trigger more detailed TypeError.

    For C builtins that only return 'missing N required positional argument: "name"'
    when called with no args, passing a Tensor triggers parameter-type-aware errors
    that include full parameter names and sometimes type hints.
    """
    try:
        import torch

        x = torch.randn(1)
    except ImportError:
        return None
    try:
        obj(x)
    except TypeError as e:
        return _parse_simple_type_error(str(e))
    except Exception:
        pass
    return None


def _parse_simple_type_error(
    error_msg: str, source: str = "TypeError(simple)"
) -> Optional[Tuple[List[ParamInfo], str]]:
    """Parse 'missing N required positional argument(s): "name1", "name2"' format."""
    m = re.search(r"missing \d+ required positional arguments?[=:] (.+)", error_msg)
    if not m:
        return None
    param_str = m.group(1)
    names = [p.strip().strip("\"'") for p in param_str.split(",")]
    params = [ParamInfo(name=n, type=_infer_type_from_name(n)) for n in names]
    if not params:
        return None
    return params, source


def _resolve_alias_from_docstring(api_name: str) -> Optional[str]:
    doc = getattr(_resolve_function(api_name), "__doc__", "") or ""
    m = re.search(r"Alias for :func:`([^`]+)`", doc)
    if m:
        return m.group(1)
    return None


def _extract_params_from_docstring(obj, api_name: str) -> Optional[Tuple[List[ParamInfo], str, int]]:
    """Parse params from __doc__, supporting three docstring formats:

    Format 1: first line is a signature like "func(input, other, *, out=None) -> Tensor"
              → parse first line for names/defaults, enrich types from Args section
    Format 2: first line is description, but Args section has typed params like "input (Tensor): ..."
              → extract names/types directly from Args section
    Format 3: first line has inline type annotations like "func(input: Tensor, *, out: Optional[Tensor])"
              → parse types directly from first line, enrich from Args section

    Returns (params, source, return_count) where return_count is the number of return
    tensors parsed from the ``->`` annotation (0 when absent or unresolvable).
    """
    doc = getattr(obj, "__doc__", None)
    if not doc:
        return None

    args_types = _parse_docstring_args_section(doc)
    first_line = doc.strip().split("\n")[0].strip()
    m = re.match(r"[\w.]+\((.+?)\)\s*(?:->.*)?$", first_line)

    if m:
        # Format 1 or 3: first line contains a signature
        sig = m.group(1)
        params = _parse_params_from_signature(sig)
        if params:
            # Priority 1: Args/Keyword Args section types
            if args_types:
                for p in params:
                    if p.name in args_types:
                        raw_type, is_opt = args_types[p.name]
                        normalized = _normalize_args_type(raw_type)
                        if normalized:
                            p.type = normalized
                        elif any(kw in raw_type.lower() for kw in ("list", "tuple")):
                            elem_type = _infer_type_from_name(p.name)
                            p.type = f"tuple of {elem_type}s"
                        if is_opt:
                            p.is_optional = True
            # Priority 3: alias API types (for params with inferred types only)
            alias_name = _resolve_alias_from_docstring(api_name)
            if alias_name:
                alias_result = extract_api_params(alias_name)
                if alias_result:
                    alias_map = {p.name: p for p in alias_result.params if p.type}
                    for p in params:
                        if p.name in alias_map and _is_inferred_type(p.type):
                            p.type = alias_map[p.name].type
            return (
                params,
                "docstring(first-line+Args)" if args_types else "docstring(first-line)",
                _parse_return_count_from_first_line(first_line),
            )

    # Format 2: no signature in first line, but Args section has typed params
    # Only use Args-only if at least some params had (Type) bracket format.
    # If all params were parsed from "name: description" fallback (no brackets),
    # the types are unreliable — let inspect.signature take priority.
    if args_types:
        has_bracket_types = any(
            _normalize_args_type(raw_type) not in (None, "") for _, (raw_type, _) in args_types.items()
        )
        if not has_bracket_types:
            return None
        params = []
        for name, (raw_type, is_opt) in args_types.items():
            normalized = _normalize_args_type(raw_type)
            params.append(
                ParamInfo(
                    name=name,
                    type=normalized or _infer_type_from_name(name),
                    default=None,
                    is_optional=is_opt or normalized in ("Layout", "Device", "Dtype"),
                )
            )
        if params:
            # Format 2 has no signature line, so is_optional and default come
            # only from the Args section's ", optional" marker.  Many params
            # have default values in the actual Python signature (e.g.
            # ``size=None``) that the docstring omits the optional tag for.
            # Fall back to inspect.signature to enrich these fields.
            inspect_result = _extract_params_from_inspect(obj)
            if inspect_result:
                _enrich_types_from_annotations(params, inspect_result[0])
            return params, "docstring(Args-only)", 0

    return None


def _coerce_default_value(raw: str, type_hint: str):
    if raw == "None":
        return None
    if type_hint == "str":
        if len(raw) >= 2 and ((raw[0] == '"' and raw[-1] == '"') or (raw[0] == "'" and raw[-1] == "'")):
            return raw[1:-1]
        return raw
    if type_hint == "bool":
        if isinstance(raw, bool):
            return raw
        return raw.lower() in ("true", "1")
    if type_hint == "int":
        if isinstance(raw, str):
            _REDUCTION_CPP = {
                "at::Reduction::None": 0,
                "at::Reduction::Mean": 1,
                "at::Reduction::Sum": 2,
                "Reduction::None": 0,
                "Reduction::Mean": 1,
                "Reduction::Sum": 2,
            }
            if raw in _REDUCTION_CPP:
                return _REDUCTION_CPP[raw]
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    if type_hint == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    return raw


def _extract_params_from_op_declaration(api_name: str) -> Optional[Tuple[List[ParamInfo], str, int]]:
    """Extract params from torch_npu C++ Declaration by calling obj._op().

    Handles both torch_npu.xxx and torch.npu_xxx API naming conventions.
    Uses obj._op() (not obj()) to trigger RuntimeError containing the
    C++ Declaration string with typed parameter info.

    Returns:
        (params, source, return_count) or None
    """
    # Only handle NPU APIs: torch_npu.xxx or torch.npu_xxx
    if not (api_name.startswith("torch_npu.") or api_name.startswith("torch.npu_")):
        return None
    obj = _resolve_function(api_name)
    if obj is None:
        return None
    # Use _op() to trigger Declaration — works for both torch_npu.xxx
    # and torch.npu_xxx (deprecated alias). obj() may hit deprecation
    # warnings instead of producing a Declaration for torch.npu_xxx.
    error_msg = None
    try:
        obj._op()
    except Exception as e:
        error_msg = str(e)
    if not error_msg:
        return None
    decl_match = re.search(r"Declaration:\s*[\w:]+::([\w.]+)\(", error_msg)
    if not decl_match:
        return None
    rest = error_msg[decl_match.end() :]
    depth = 1
    end = len(rest) - 1
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    sig = rest[:end]
    params = _parse_npu_declaration(sig)
    if params:
        return_count = _parse_return_count_from_error(rest, end, error_msg)
        return params, "torch_npu._op()", return_count
    return None


def _parse_return_count_from_error(rest, paren_end, error_msg):
    """Parse the number of return tensors from a C++ Declaration error message.

    Handles:
      -> (Tensor, Tensor, Tensor, Tensor) → 4
      -> Tensor(a!) → 1
      -> Tensor → 1
    """
    after_paren = rest[paren_end + 1 :].strip()
    arrow_match = re.match(r"\s*->\s*(.+)", after_paren)
    if not arrow_match:
        return 0
    ret_str = arrow_match.group(1).strip()
    if ret_str.startswith("("):
        inner = ret_str[1:]
        close = inner.find(")")
        if close >= 0:
            inner = inner[:close]
        parts = [p.strip() for p in _split_by_comma(inner)]
        return sum(1 for p in parts if "Tensor" in p)
    if "Tensor" in ret_str:
        return 1
    return 0


def _parse_npu_declaration(sig: str) -> Optional[List[ParamInfo]]:
    if not sig:
        return None
    params = []
    saw_star = False
    parts = _split_by_comma(sig)
    for part in parts:
        part = part.strip()
        if not part or part.startswith("->"):
            continue
        if part == "*":
            saw_star = True
            continue
        m = re.match(r"(\w+(?:\([^)]*\))?(?:\[\d*\])?)\??\s+(\w+)(?:=(.+))?", part)
        if m:
            type_hint = m.group(1)
            type_hint = re.sub(r"\([^)]*\)", "", type_hint)
            name = m.group(2)
            default = m.group(3) or None
            type_hint = _normalize_npu_type(type_hint)
            is_optional = default is not None or "?" in part.split()[0]
            if default is not None:
                default = _coerce_default_value(default, type_hint)
            params.append(
                ParamInfo(name=name, type=type_hint, default=default, is_optional=is_optional, is_keyword_only=saw_star)
            )
    return params if params else None


def _normalize_npu_type(t: str) -> str:
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_npu_type(inner)
    if t.startswith("List[") and t.endswith("]"):
        inner = t[len("List[") : -1]
        inner_mapped = _normalize_npu_type(inner)
        if inner_mapped == "Tensor":
            return "tuple of Tensors"
        if inner_mapped == "Number":
            return "tuple of Numbers"
        return f"tuple of {inner_mapped}s"
    mapping = {
        "Tensor": "Tensor",
        "TensorList": "tuple of Tensors",
        "Tensor[]": "tuple of Tensors",
        "Scalar": "Number",
        "ScalarList": "list of Numbers",
        "Scalar[]": "list of Numbers",
        "number": "Number",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "str": "str",
        "complex": "Number",
        "int?": "int",
        "float?": "float",
        "bool?": "bool",
        "str?": "str",
        "int[]": "tuple of ints",
        "float[]": "tuple of floats",
        "bool[]": "tuple of bools",
        "str[]": "tuple of strs",
        "int64[]": "tuple of ints",
        "int32[]": "tuple of ints",
        "SymInt": "int",
        "SymInt?": "int",
        "SymInt[]": "tuple of ints",
        "SymBool": "bool",
        "ScalarType": "torch.dtype",
        "ScalarType?": "torch.dtype",
        "Device": "str",
        "Device?": "str",
        "Layout": "int",
        "Layout?": "int",
        "MemoryFormat": "int",
        "MemoryFormat?": "int",
        "Generator": "None",
        "Generator?": "None",
        "AnyEnumType": "int",
        "Storage": "None",
        "Storage?": "None",
    }
    return mapping.get(t, t)


def _extract_params_from_aten_schemas(
    api_name: str,
) -> Optional[Tuple[List[ParamInfo], str, List[List[ParamInfo]], List[int]]]:
    obj = _resolve_function(api_name)
    if obj is None or not hasattr(obj, "_schemas"):
        return None
    schemas = obj._schemas
    if not schemas:
        return None
    all_overloads = []
    return_counts = []
    for schema in schemas.values():
        params = _parse_aten_schema_arguments(schema)
        if params:
            all_overloads.append(params)
            return_counts.append(len(schema.returns) if hasattr(schema, "returns") else 0)
    if not all_overloads:
        return None
    primary = max(all_overloads, key=lambda ov: sum(1 for p in ov if p.is_optional))
    _first_schema = next(iter(schemas.values()))
    namespace = getattr(_first_schema, "name", api_name.rsplit(".", 1)[0]).split("::")[0]
    source = (
        f"{namespace}._schemas({len(all_overloads)} overloads)" if len(all_overloads) > 1 else f"{namespace}._schemas"
    )
    return primary, source, all_overloads, return_counts


def _parse_aten_schema_arguments(schema) -> Optional[List[ParamInfo]]:
    params = []
    for arg in schema.arguments:
        name = arg.name
        if not name:
            continue
        type_str = _normalize_aten_schema_type(str(arg.type))
        default = None
        is_optional = False
        if arg.has_default_value():
            default = arg.default_value
            is_optional = True
            if isinstance(default, (list, tuple)):
                default = list(default)
        elif str(arg.type).startswith("Optional["):
            is_optional = True
        is_keyword_only = arg.kwarg_only if hasattr(arg, "kwarg_only") else False
        params.append(
            ParamInfo(
                name=name, type=type_str, default=default, is_optional=is_optional, is_keyword_only=is_keyword_only
            )
        )
    return params if params else None


def _normalize_aten_schema_type(t: str) -> str:
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_aten_schema_type(inner)
    if t.startswith("List[") and t.endswith("]"):
        inner = t[len("List[") : -1]
        inner_mapped = _normalize_aten_schema_type(inner)
        if inner_mapped == "Tensor":
            return "tuple of Tensors"
        return f"tuple of {inner_mapped}s"
    return _normalize_npu_type(t)


_SCALAR_ANNOTATION_TYPES = {bool: "bool", int: "int", float: "float", str: "str"}

_SEQUENCE_INNER_TYPES = {"Tensor": "tuple of Tensors", "int": "tuple of ints", "float": "tuple of floats"}


def _torch_tensor_type():
    """torch.Tensor;torch 不可用时返回一个不会与任何注解相等的哨兵。"""
    try:
        import torch

        return torch.Tensor
    except ImportError:
        return object()


def _torch_dtype_type():
    """torch.dtype;torch 不可用时返回一个不会与任何注解相等的哨兵。"""
    try:
        import torch

        return torch.dtype
    except ImportError:
        return object()


def _sequence_annotation_to_type(annotation) -> str:
    """list/tuple/Sequence[...] 按其元素类型归一成 tuple of xxx。"""
    args = getattr(annotation, "__args__", ())
    if not args:
        return "int"
    return _SEQUENCE_INNER_TYPES.get(_annotation_to_type(args[0]), "int")


def _annotation_to_type(annotation) -> str:
    """Convert a type annotation object to our type string.

    Returns empty string for None or unrecognized annotations.
    """
    if annotation is None:
        return ""
    import types as _types
    import typing

    # PEP 604 的 `str | None` 是 types.UnionType,没有 __origin__(torch>=2.6 的
    # .pyi/源码普遍改用这种写法),故 __origin__ 取不到时用 get_origin 兜底。
    origin = getattr(annotation, "__origin__", None) or typing.get_origin(annotation)
    # 不能写成 origin in (..., getattr(_types, 'UnionType', None)):Python < 3.10 无 UnionType,
    # getattr 返回 None,而无 origin 的注解(int/float/str/bool)其 origin 恰为 None,
    # 会整片落进 Union 分支、args 为空后误返回 "Tensor"。
    if origin in (typing.Optional, typing.Union) or (hasattr(_types, "UnionType") and origin is _types.UnionType):
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        return _annotation_to_type(non_none[0]) if non_none else "Tensor"
    if origin in (list, tuple, typing.List, typing.Tuple, typing.Sequence):
        return _sequence_annotation_to_type(annotation)
    scalar = _SCALAR_ANNOTATION_TYPES.get(annotation)
    if scalar:
        return scalar
    try:
        if isinstance(annotation, type) and issubclass(annotation, int):
            return "int"
    except TypeError:
        pass
    if annotation is _torch_dtype_type():
        return "Dtype"
    if annotation is _torch_tensor_type():
        return "Tensor"
    return ""


def _extract_params_from_annotations(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Extract params from obj.__annotations__.

    Works for any Python object that has __annotations__ (functions, classes, etc).
    Does NOT require inspect.signature — safe to call on any object.
    Default values are obtained from inspect.signature if available, otherwise None.
    """
    ann = getattr(obj, "__annotations__", None)
    if not ann:
        return None

    param_names = [k for k in ann if k != "return"]
    if not param_names:
        return None

    import inspect as _inspect

    defaults_map = {}
    kw_only_set = set()
    try:
        sig = _inspect.signature(obj)
        for pname, param in sig.parameters.items():
            if param.default != _inspect.Parameter.empty:
                defaults_map[pname] = param.default
            if param.kind == _inspect.Parameter.KEYWORD_ONLY:
                kw_only_set.add(pname)
    except (ValueError, TypeError):
        pass

    params = []
    for pname in param_names:
        type_hint = _annotation_to_type(ann[pname])
        if not type_hint:
            type_hint = _infer_type_from_name(pname)
        default_raw = defaults_map.get(pname)
        is_optional = pname in defaults_map
        default = _coerce_default_value(str(default_raw), type_hint) if is_optional else None
        params.append(
            ParamInfo(
                name=pname,
                type=type_hint,
                default=default,
                is_optional=is_optional,
                is_keyword_only=pname in kw_only_set,
            )
        )

    return (params, "__annotations__") if params else None


def _extract_params_from_inspect(obj) -> Optional[Tuple[List[ParamInfo], str]]:
    """Try inspect.signature for Python-callable functions."""
    import inspect as _inspect

    try:
        sig = _inspect.signature(obj)
    except (ValueError, TypeError):
        return None

    ann = getattr(obj, "__annotations__", None) or {}
    params = []
    for pname, param in sig.parameters.items():
        if param.kind == _inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == _inspect.Parameter.VAR_POSITIONAL:
            type_hint = _infer_type_from_name(pname)
            if type_hint in ("Tensor", "list of Tensors", "tuple of Tensors", "Tensor[]", "List[Tensor]"):
                params.append(
                    ParamInfo(name=pname, type="Tensor", default=None, is_optional=False, is_var_positional=True)
                )
            continue

        if pname in ann:
            type_hint = _annotation_to_type(ann[pname])
        elif param.annotation is not None and param.annotation != _inspect.Parameter.empty:
            type_hint = _annotation_to_type(param.annotation)
        else:
            type_hint = _infer_type_from_name(pname)
        if _is_inferred_type(type_hint) and param.default != _inspect.Parameter.empty:
            if isinstance(param.default, bool):
                type_hint = "bool"
            elif isinstance(param.default, int):
                type_hint = "int"
            elif isinstance(param.default, float):
                type_hint = "float"
            elif isinstance(param.default, str):
                type_hint = "str"
        is_optional = param.default != _inspect.Parameter.empty
        default = None if not is_optional else _coerce_default_value(str(param.default), type_hint)
        # In Python, parameters after *args are implicitly keyword-only
        # (e.g. meshgrid(*tensors, indexing=None)).  Mapping this correctly
        # ensures build_args() routes such params to kwargs instead of args.
        is_kw_only = param.kind == _inspect.Parameter.KEYWORD_ONLY
        params.append(
            ParamInfo(name=pname, type=type_hint, default=default, is_optional=is_optional, is_keyword_only=is_kw_only)
        )
    return (params, "inspect.signature") if params else None


def _is_inferred_type(type_str: str) -> bool:
    """Check if a type was inferred from name rather than from an authoritative source.

    For union types (e.g., 'int|tuple of ints'), returns True only if ALL members
    are inferred types. If any member comes from an authoritative source (e.g.,
    Args section), the whole type is considered non-inferred.
    """
    _INFERRED = frozenset(
        {
            "Tensor",
            "float",
            "int",
            "bool",
            "str",
            "tuple of Tensors",
            "tuple of ints",
            "tuple of floats",
            "list of Numbers",
            "Number",
            "Dtype",
            "Device",
            "Layout",
        }
    )
    return all(t in _INFERRED for t in type_str.split("|"))


def _enrich_types_from_annotations(params: List[ParamInfo], ann_params: List[ParamInfo]) -> List[ParamInfo]:
    """Replace inferred types with real types from __annotations__.

    Only overwrites types that came from _infer_type_from_name.
    Authoritative types from docstring Args section or TypeError are kept.
    Union types (containing '|') from Args section are considered authoritative
    and are not overwritten by annotations.
    Also fills in defaults from annotations/signature when missing.
    """
    if not ann_params:
        return params
    ann_map = {p.name: p for p in ann_params if p.type}
    for p in params:
        ann_p = ann_map.get(p.name)
        if ann_p is None:
            continue
        # Union types from Args section are authoritative — don't overwrite
        if _is_inferred_type(p.type) and "|" not in p.type and ann_p.type:
            p.type = ann_p.type
        if p.default is None and ann_p.default is not None:
            p.default = ann_p.default
        if not p.is_optional and ann_p.is_optional:
            p.is_optional = ann_p.is_optional
    return params


def _strip_inplace_if_separate_exists(api_name: str, params: List[ParamInfo]) -> List[ParamInfo]:
    if not api_name.startswith("torch.") or api_name.startswith("torch.nn."):
        return params
    if api_name.endswith("_"):
        params[:] = [p for p in params if p.name != "inplace"]
        return params
    inplace_name = api_name + "_"
    inplace_obj = _resolve_function(inplace_name)
    if inplace_obj is None:
        return params
    params[:] = [p for p in params if p.name != "inplace"]
    return params


def _detect_keyword_only_by_probing(obj, params: List[ParamInfo]):
    """Probe a C++ builtin to detect keyword-only boundary and validate param names.

    Strategy:
    1. Extract actual C++ param names from "missing N required positional" TypeError
    2. Detect max positional boundary by progressive calling
    3. Remove params not in C++ signature, recover via convention (size_average→reduction)

    Only called for C++ builtins (no inspect.signature available).
    """
    import torch as _torch

    if not params or any(p.is_keyword_only for p in params):
        return

    _PROBE_MAP = {
        "int": 0,
        "float": 1.0,
        "Number": 1.0,
        "bool": False,
        "str": "mean",
        "torch.dtype": _torch.float32,
        "Dtype": _torch.float32,
        "torch.memory_format": _torch.contiguous_format,
    }

    dummy = _torch.zeros(2, 3)
    probes = []
    for p in params:
        if p.is_tensor_like:
            probes.append(dummy)
        else:
            val = _PROBE_MAP.get(p.type)
            if val is None:
                return
            probes.append(val)

    # Step 1: Extract actual C++ param names from TypeError
    cpp_param_set = None
    try:
        obj()
    except TypeError as e:
        msg = str(e)
        m = re.search(r"missing \d+ required positional arguments?: (.+)", msg)
        if m:
            cpp_param_set = set(n.strip().strip("\"'") for n in m.group(1).split(","))
        else:
            cpp_param_set = None
    except Exception:
        cpp_param_set = None

    # Step 2: Detect max positional boundary
    def _classify(n):
        try:
            obj(*probes[:n])
            return "ok"
        except TypeError as e:
            msg = str(e).lower()
            if "too many" in msg or ("takes" in msg and "positional" in msg):
                return "too_many"
            if "invalid combination" in msg:
                return "too_many"
            if "missing" in msg:
                return "missing"
            return "error"
        except Exception:
            return "error"

    max_pos = 0
    found_valid = False
    for n in range(len(params) + 1):
        result = _classify(n)
        if result in ("ok", "error"):
            max_pos = n
            found_valid = True
        elif result == "missing":
            continue
        elif result == "too_many" and found_valid:
            break

    for i in range(max_pos, len(params)):
        params[i].is_keyword_only = True

    # Step 3: Remove deprecated params not in C++ signature (if we have ground truth)
    _DEPRECATED_LOSS_PARAMS = {"size_average": "bool", "reduce": "bool"}
    if cpp_param_set is not None:
        to_remove = []
        removed_names = set()
        for i, p in enumerate(params):
            if (
                p.name in _DEPRECATED_LOSS_PARAMS
                and _DEPRECATED_LOSS_PARAMS[p.name] == p.type
                and p.name not in cpp_param_set
            ):
                to_remove.append(i)
                removed_names.add(p.name)
        for idx in reversed(to_remove):
            params.pop(idx)

        if removed_names:
            first_req = 0
            for i, p in enumerate(params):
                if p.is_optional:
                    first_req = i
                    break
            else:
                first_req = len(params)
            base = probes[:first_req]
            _try_recover_removal(obj, base, params, removed_names)

    # Step 4: Verify keyword-only params are actually accepted by C++.
    kw_only_params = [(i, params[i], probes[i]) for i in range(len(params)) if params[i].is_keyword_only]
    if not kw_only_params:
        return

    first_req = 0
    for i, p in enumerate(params):
        if p.is_optional:
            first_req = i
            break
    else:
        first_req = len(params)
    base = probes[:first_req]

    to_remove = []
    for idx, p, val in kw_only_params:
        try:
            obj(*base, **{p.name: val})
        except TypeError as _te:
            _te_msg = str(_te).lower()
            if "unexpected keyword" not in _te_msg:
                continue
            _ALT_PROBES = {
                "str": [("int", 1)],
                "int": [("str", "mean")],
                "Dtype": [("int", 0)],
                "torch.dtype": [("int", 0)],
                "bool": [("str", "mean")],
            }
            fixed = False
            for alt_type, alt_val in _ALT_PROBES.get(p.type, []):
                try:
                    obj(*base, **{p.name: alt_val})
                    p.type = alt_type
                    p.default = alt_val
                    fixed = True
                    break
                except TypeError:
                    continue
                except Exception:
                    p.type = alt_type
                    p.default = alt_val
                    fixed = True
                    break
            if not fixed:
                to_remove.append(idx)
        except Exception:
            pass
    for idx in reversed(to_remove):
        params.pop(idx)

    # Step 3 fallback: verify optional params by kwarg probing
    first_optional = len(params)
    for i in range(len(params)):
        if params[i].is_optional:
            first_optional = i
            break

    if first_optional < len(params):
        kw_params = [(i, params[i], probes[i]) for i in range(first_optional, len(params))]
        to_remove = []
        for idx, p, val in kw_params:
            test_pos = [probes[j] for j in range(max_pos) if j != idx]
            try:
                obj(*test_pos, **{p.name: val})
            except TypeError as _te:
                _te_msg = str(_te).lower()
                if "missing" in _te_msg and "required positional" in _te_msg:
                    continue
                if "unexpected keyword" not in _te_msg and "got an unexpected keyword" not in _te_msg:
                    continue
                _ALT_PROBES = {
                    "str": [("int", 1)],
                    "int": [("str", "mean")],
                    "Dtype": [("int", 0)],
                    "torch.dtype": [("int", 0)],
                    "bool": [("str", "mean")],
                    "torch.Tensor": "Tensor",
                    "torch.BoolTensor": "Tensor",
                    "torch.IntTensor": "Tensor",
                    "torch.LongTensor": "Tensor",
                    "torch.FloatTensor": "Tensor",
                    "torch.DoubleTensor": "Tensor",
                }
                fixed = False
                for alt_type, alt_val in _ALT_PROBES.get(p.type, []):
                    try:
                        obj(*test_pos, **{p.name: alt_val})
                        p.type = alt_type
                        p.default = alt_val
                        fixed = True
                        break
                    except TypeError:
                        continue
                    except Exception:
                        p.type = alt_type
                        p.default = alt_val
                        fixed = True
                        break
                if not fixed:
                    to_remove.append(idx)
            except Exception:
                pass
        removed_names = set()
        for idx in reversed(to_remove):
            removed_names.add(params[idx].name)
            params.pop(idx)

        if removed_names:
            _try_recover_removal(obj, probes[:first_optional], params, removed_names)


def _try_recover_removal(obj, pos_probes, params, removed_names):
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return

    _DEPRECATED_TO_MODERN = {
        "size_average": ("reduction", "int", 1),
        "reduce": ("reduction", "int", 1),
    }

    for old_name in removed_names:
        if old_name not in _DEPRECATED_TO_MODERN:
            continue
        new_name, new_type, new_default = _DEPRECATED_TO_MODERN[old_name]
        existing = next((p for p in params if p.name == new_name), None)
        if existing is not None:
            existing.type = new_type
            existing.default = new_default
            existing.is_keyword_only = True
            continue
        try:
            obj(*pos_probes, **{new_name: new_default})
        except TypeError:
            continue
        except Exception:
            pass
        existing_kw = [p for p in params if p.is_keyword_only]
        insert_idx = len(params)
        if existing_kw:
            insert_idx = params.index(existing_kw[0])
        params.insert(
            insert_idx,
            ParamInfo(name=new_name, type=new_type, default=new_default, is_optional=True, is_keyword_only=True),
        )


def _strip_alias_extra_params(api_name: str, params: List[ParamInfo]) -> List[ParamInfo]:
    """Post-process parsed params for C++ builtins.

    1. Remove 'inplace' if a separate inplace API exists (design: test inplace separately).
    2. Remove params whose type resolved to 'name' (named-dimension artefact).
    3. For C++ builtins: probe to detect keyword-only boundary and remove
       params the C++ function does not accept at all.
    """
    _strip_inplace_if_separate_exists(api_name, params)

    if not api_name.startswith("torch."):
        return params

    params[:] = [p for p in params if p.type != "name"]

    obj = _resolve_function(api_name)
    is_cpp = obj is not None and type(obj).__name__ == "builtin_function_or_method"

    if is_cpp:
        _detect_keyword_only_by_probing(obj, params)

    return params


def _extract_params_from_functional_alias(api_name: str) -> Optional[Tuple[List[ParamInfo], str]]:
    """Fallback: try torch.nn.functional.XXX when torch.XXX fails.

    Also handles inplace APIs (torch.abs_ → torch.abs) by stripping the trailing '_'.
    """
    if not api_name.startswith("torch."):
        return None
    func_name = api_name.split(".")[-1]

    candidates = [f"torch.nn.functional.{func_name}"]

    if func_name.startswith("npu_"):
        candidates.append(f"torch_npu.{func_name}")

    if func_name.endswith("_") and not func_name.startswith("_"):
        base_name = func_name[:-1]
        candidates.append(f"torch.{base_name}")
        candidates.append(f"torch.nn.functional.{base_name}")

    for candidate in candidates:
        obj = _resolve_function(candidate)
        if obj is None:
            continue
        doc_result = _extract_params_from_docstring(obj, candidate)
        te_result = _extract_params_from_type_error(obj)
        ann_result = _extract_params_from_annotations(obj)

        if te_result:
            params, source, overloads = te_result
            if overloads and len(overloads) > 1:
                _enrich_types_from_annotations(overloads[0], ann_result[0] if ann_result else [])
                _strip_alias_extra_params(api_name, overloads[0])
                return overloads[0], f"alias({candidate}, {source})"

        if doc_result and te_result:
            merged = _merge_docstring_and_type_error(doc_result[0], te_result[0])
            if merged:
                if ann_result:
                    _enrich_types_from_annotations(merged, ann_result[0])
                _strip_alias_extra_params(api_name, merged)
                return merged, f"alias({candidate}, merge({doc_result[1]}, {te_result[1]}))"

        if doc_result:
            params, doc_source, _ = doc_result
            if ann_result:
                _enrich_types_from_annotations(params, ann_result[0])
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, {doc_source})"

        if te_result:
            params, te_source, _ = te_result
            if ann_result:
                _enrich_types_from_annotations(params, ann_result[0])
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, {te_source})"

        if ann_result:
            params = ann_result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, __annotations__)"

        result = _extract_params_from_inspect(obj)
        if result:
            params = result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, inspect.signature)"
        result = _extract_params_from_op_declaration(candidate)
        if result:
            params = result[0]
            _strip_alias_extra_params(api_name, params)
            return params, f"alias({candidate}, torch_npu._op())"
    return None


def _merge_docstring_and_type_error(
    doc_params: List[ParamInfo], te_params: List[ParamInfo]
) -> Optional[List[ParamInfo]]:
    """Merge docstring and TypeError results as equal-priority sources.

    Rules:
    - Use the longer param list as base (more info is better).
    - For each param present in both, take the more authoritative type:
      - If one is clearly inferred (generic Tensor/int/float) and the other
        is specific (Number, torch.dtype, tuple of ints, etc.), prefer the specific one.
      - Otherwise keep the base param's type.
    - Preserve defaults from whichever source has them.
    """
    if not doc_params or not te_params:
        return doc_params or te_params

    # Build name→ParamInfo lookup for the secondary source
    base = doc_params if len(doc_params) >= len(te_params) else te_params
    other = te_params if base is doc_params else doc_params
    other_map = {p.name: p for p in other}

    merged = []
    for bp in base:
        op = other_map.get(bp.name)
        if op is None:
            merged.append(bp)
            continue

        name = bp.name
        # Pick type: prefer authoritative over inferred
        if bp.type == op.type:
            ptype = bp.type
        elif _is_inferred_type(bp.type) and not _is_inferred_type(op.type):
            ptype = op.type
        elif not _is_inferred_type(bp.type) and _is_inferred_type(op.type):
            ptype = bp.type
        else:
            # Both are specific but different — prefer TypeError (C++ ground truth)
            ptype = op.type if other is te_params else bp.type

        # Pick default: prefer non-empty
        default = bp.default if bp.default is not None else op.default

        # Pick optional: either source says optional
        is_optional = bp.is_optional or op.is_optional

        # Pick keyword_only: TypeError is authoritative for positional/keyword boundary
        is_kw = op.is_keyword_only if other is te_params else bp.is_keyword_only

        merged.append(ParamInfo(name=name, type=ptype, default=default, is_optional=is_optional, is_keyword_only=is_kw))

    # Add params only in 'other' that aren't in 'base'
    base_names = {p.name for p in base}
    for op in other:
        if op.name not in base_names:
            merged.append(op)

    return merged if merged else None


def _try_upgrade_to_inspect_if_var_pos(obj, params, api_name, ann_result=None):
    """Try replacing *params* with inspect.signature result when VAR_POSITIONAL is needed.

    Docstring and annotation sources cannot capture ``*args`` tensor parameters:
    - ``__annotations__`` omits ``*args`` entries entirely
    - Docstring Args sections list ``*tensors`` as a normal param without
      the ``is_var_positional`` flag

    When the current extraction source has no VAR_POSITIONAL tensor param
    but inspect.signature finds one, the inspect result is strictly more
    accurate and should be used instead.

    Args:
        obj: Resolved callable.
        params: Current extraction result (list of ParamInfo).
        api_name: Full API name (e.g. "torch.meshgrid").
        ann_result: Optional annotation result for type enrichment.

    Returns:
        APIParamInfo built from inspect params if an upgrade was performed,
        otherwise None.
    """
    has_var_pos = any(getattr(p, "is_var_positional", False) for p in params)
    if has_var_pos:
        return None
    inspect_result = _extract_params_from_inspect(obj)
    if not inspect_result:
        return None
    inspect_has_var_pos = any(getattr(p, "is_var_positional", False) for p in inspect_result[0])
    if not inspect_has_var_pos:
        return None
    inspect_params = inspect_result[0]
    inspect_names = {p.name for p in inspect_params}
    if inspect_names <= {"args", "kwargs"}:
        return None
    if ann_result:
        _enrich_types_from_annotations(inspect_result[0], ann_result[0])
    _strip_alias_extra_params(api_name, inspect_result[0])
    return APIParamInfo(api_name=api_name, params=inspect_result[0], source="inspect.signature")


def _try_upgrade_to_inspect_if_kw_mismatch(obj, params, api_name):
    """Upgrade to inspect.signature when keyword-only params are missing.

    Docstring-based extraction cannot detect keyword-only boundaries (the ``*``
    separator is not preserved).  If inspect.signature reports KEYWORD_ONLY
    params but the current params have none, inspect is strictly more accurate.
    """
    has_kw = any(getattr(p, "is_keyword_only", False) for p in params)
    if has_kw:
        return None
    inspect_result = _extract_params_from_inspect(obj)
    if not inspect_result:
        return None
    inspect_has_kw = any(getattr(p, "is_keyword_only", False) for p in inspect_result[0])
    if not inspect_has_kw:
        return None
    inspect_names = {p.name for p in inspect_result[0]}
    if inspect_names <= {"args", "kwargs"}:
        return None
    _strip_alias_extra_params(api_name, inspect_result[0])
    return APIParamInfo(api_name=api_name, params=inspect_result[0], source="inspect.signature")


def extract_api_params(api_name: str) -> Optional[APIParamInfo]:
    try:
        return _extract_api_params_impl(api_name)
    except Exception as e:
        logging.warning(f"extract_api_params({api_name}) raised {type(e).__name__}: {e}")
        return None


def _extract_api_params_impl(api_name: str) -> Optional[APIParamInfo]:
    # The extension package is imported by FrameworkApiInfoKeeper.get before
    # this is reached; no import here.

    # Any API whose resolved object carries _schemas (an OpOverloadPacket)
    # has an authoritative FunctionSchema; parse it regardless of api_name form.
    result = _extract_params_from_aten_schemas(api_name)
    if result:
        params, source, overloads, return_counts = result
        return APIParamInfo(
            api_name=api_name, params=params, source=source, overloads=overloads, _return_counts=return_counts
        )

    # NPU APIs (torch_npu.xxx, torch.npu_xxx, torch.ops.*):
    # Declaration is authoritative
    if (
        api_name.startswith("torch.ops.")
        or api_name.startswith("torch_npu.")
        or (api_name.startswith("torch.npu_") and "torch_npu" not in api_name)
    ):
        result = _extract_params_from_op_declaration(api_name)
        if result:
            params, source, return_count = result
            return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=[return_count])

    obj = _resolve_function(api_name)
    if obj is None:
        return None

    # torch_npu APIs: fallback to docstring/inspect
    if api_name.startswith("torch_npu."):
        doc_result = _extract_params_from_docstring(obj, api_name)
        if doc_result:
            params, source, doc_rc = doc_result
            return APIParamInfo(
                api_name=api_name, params=params, source=source, _return_counts=[doc_rc] if doc_rc else []
            )
        result = _extract_params_from_inspect(obj)
        if result:
            params, source = result
            return APIParamInfo(api_name=api_name, params=params, source=source)
        return None

    doc_result = _extract_params_from_docstring(obj, api_name)
    te_result = _extract_params_from_type_error(obj)
    ann_result = _extract_params_from_annotations(obj)

    # TypeError multi-overload is unique — no other source provides multiple signatures
    if te_result:
        params, source, overloads = te_result
        if overloads and len(overloads) > 1:
            if ann_result:
                _enrich_types_from_annotations(overloads[0], ann_result[0])
            _strip_alias_extra_params(api_name, overloads[0])
            return APIParamInfo(api_name=api_name, params=overloads[0], source=source, overloads=overloads)

    # docstring and TypeError single-signature are equal-priority sources.
    # Merge when both succeed: use longer param list as base, enrich types from the other.
    if doc_result and te_result:
        doc_params, doc_source, doc_rc = doc_result
        te_params, te_source, _ = te_result
        merged = _merge_docstring_and_type_error(doc_params, te_params)
        if merged:
            if ann_result:
                _enrich_types_from_annotations(merged, ann_result[0])
            _strip_alias_extra_params(api_name, merged)
            return APIParamInfo(
                api_name=api_name,
                params=merged,
                source=f"merge({doc_source}, {te_source})",
                _return_counts=[doc_rc] if doc_rc else [],
            )

    if doc_result:
        params, source, doc_rc = doc_result
        upgraded = _try_upgrade_to_inspect_if_kw_mismatch(obj, params, api_name)
        if upgraded:
            return upgraded
        upgraded = _try_upgrade_to_inspect_if_var_pos(obj, params, api_name, ann_result)
        if upgraded:
            return upgraded
        if ann_result:
            _enrich_types_from_annotations(params, ann_result[0])
        _strip_alias_extra_params(api_name, params)
        return APIParamInfo(api_name=api_name, params=params, source=source, _return_counts=[doc_rc] if doc_rc else [])

    if te_result:
        params, source, overloads = te_result
        if ann_result:
            _enrich_types_from_annotations(params, ann_result[0])
        _strip_alias_extra_params(api_name, params)
        return APIParamInfo(api_name=api_name, params=params, source=source, overloads=[params])

    if ann_result:
        params, source = ann_result
        # __annotations__ omits *args (VAR_POSITIONAL) params entirely.
        # When annotations produced zero tensor params but inspect finds
        # some, upgrade to inspect so *args tensor params are captured.
        upgraded = _try_upgrade_to_inspect_if_var_pos(obj, params, api_name)
        if upgraded:
            return upgraded
        _strip_alias_extra_params(api_name, params)
        return APIParamInfo(api_name=api_name, params=params, source=source)

    # inspect.signature
    result = _extract_params_from_inspect(obj)
    if result:
        params, source = result
        _strip_alias_extra_params(api_name, params)
        return APIParamInfo(api_name=api_name, params=params, source=source)

    # alias (functional/inplace/npu)
    result = _extract_params_from_functional_alias(api_name)
    if result:
        params, source = result
        return APIParamInfo(api_name=api_name, params=params, source=source)

    # pyi stub file
    result = _extract_params_from_pyi(api_name)
    if result:
        params, source, overloads = result
        if overloads and len(overloads) > 1:
            _strip_alias_extra_params(api_name, overloads[0])
            return APIParamInfo(api_name=api_name, params=overloads[0], source=source, overloads=overloads)
        _strip_alias_extra_params(api_name, params)
        return APIParamInfo(api_name=api_name, params=params, source=source, overloads=[params])

    return None


def _normalize_pyi_type(t: str) -> str:
    t = t.strip()
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        return _normalize_pyi_type(inner)
    if t.startswith("Union[") and t.endswith("]"):
        inner = t[len("Union[") : -1]
        parts = [p.strip() for p in inner.split(",")]
        for p in parts:
            if p == "None":
                continue
            return _normalize_pyi_type(p)
        return "Tensor"
    if t == "Tensor":
        return "Tensor"
    if t in ("_int", "SymInt", "int"):
        return "int"
    if t in ("_float", "float"):
        return "float"
    if t in ("_bool", "bool"):
        return "bool"
    if t in ("str",):
        return "str"
    if t in ("Number", "_complex"):
        return "Number"
    if t in ("_dtype", "torch.dtype"):
        return "Dtype"
    if t in ("_layout", "torch.layout"):
        return "Layout"
    if t in ("_device", "DeviceLikeType", "torch.device"):
        return "Device"
    if t in ("MemoryFormat", "torch.memory_format"):
        return "MemoryFormat"
    if t.startswith("Sequence["):
        inner = t[len("Sequence[") : -1]
        inner_type = _normalize_pyi_type(inner)
        if inner_type == "int":
            return "tuple of ints"
        if inner_type == "float":
            return "tuple of floats"
        if inner_type == "Tensor":
            return "tuple of Tensors"
        return f"tuple of {inner_type}s"
    if t.startswith("tuple[") or t.startswith("Tuple["):
        inner = t.split("[", 1)[1].rstrip("]")
        inner_type = _normalize_pyi_type(inner.split(",")[0].strip())
        if inner_type == "Tensor":
            return "tuple of Tensors"
        if inner_type == "int":
            return "tuple of ints"
        return f"tuple of {inner_type}s"
    if t.startswith("list[") or t.startswith("List["):
        inner = t.split("[", 1)[1].rstrip("]")
        inner_type = _normalize_pyi_type(inner)
        if inner_type == "Tensor":
            return "tuple of Tensors"
        return f"list of {inner_type}s"
    return t


def _parse_pyi_param_str(param_str: str) -> Optional[List[ParamInfo]]:
    parts = _split_by_comma(param_str)
    params = []
    kw_only = False
    for part in parts:
        part = part.strip()
        if not part or part == "/":
            continue
        if part == "*":
            kw_only = True
            continue
        default = None
        is_optional = kw_only
        if "=" in part:
            main_part, default_str = part.split("=", 1)
            main_part = main_part.strip()
            default_str = default_str.strip()
            is_optional = True
        else:
            main_part = part
            default_str = None
        if ":" in main_part:
            name, type_hint = main_part.split(":", 1)
            name = name.strip()
            type_hint = _normalize_pyi_type(type_hint.strip())
        else:
            name = main_part.strip()
            type_hint = _infer_type_from_name(name)
        if not type_hint:
            type_hint = _infer_type_from_name(name)
        params.append(
            ParamInfo(name=name, type=type_hint, default=default, is_optional=is_optional, is_keyword_only=kw_only)
        )
    return params if params else None


_PYI_CACHE = None


def _load_pyi_signatures():
    global _PYI_CACHE
    if _PYI_CACHE is not None:
        return _PYI_CACHE
    _PYI_CACHE = {}
    try:
        import os

        import torch

        pyi_path = os.path.join(os.path.dirname(torch.__file__), "_C", "_VariableFunctions.pyi")
        if not os.path.exists(pyi_path):
            return _PYI_CACHE
        with open(pyi_path) as f:
            lines = f.readlines()
        for func_name, param_str in _iter_pyi_defs(lines):
            full_name = f"torch.{func_name}"
            parsed = _parse_pyi_param_str(param_str)
            if parsed:
                if full_name not in _PYI_CACHE:
                    _PYI_CACHE[full_name] = []
                _PYI_CACHE[full_name].append(parsed)
    except Exception:
        pass

    _load_nn_functional_pyi(_PYI_CACHE)
    return _PYI_CACHE


def _iter_pyi_defs(lines):
    """从 .pyi 行序列里产出 (函数名, 参数串)。

    torch>=2.6 起长签名会被折成多行:
        def conv1d(
            input: Tensor,
            ...
        ) -> Tensor: ...
    逐行正则只能匹到单行 def(如 `def abs(...)->Tensor:`),多行的整条漏掉,
    torch.conv1d 这类就进不了缓存。这里按括号配平把整条 def 拼回一行再匹。
    """
    buf = None
    depth = 0
    for line in lines:
        stripped = line.strip()
        if buf is None:
            if not stripped.startswith("def "):
                continue
            buf = stripped
            depth = stripped.count("(") - stripped.count(")")
        else:
            buf += " " + stripped
            depth += stripped.count("(") - stripped.count(")")
        if depth > 0:
            continue
        m = re.match(r"def\s+(\w+)\s*\((.*)\)\s*(?:->.*?)?:", buf)
        buf = None
        if m:
            yield m.group(1), m.group(2)


def _iter_pyi_reexports(lines):
    """产出 .pyi 里 re-export 的 (被导出名, 本地名)。

    两种写法都要认:torch<2.6 是单行 from-import-as;torch>=2.6 起改成括号包起来的
    多行块,每行一个 "X as Y"。只认单行形式会把整块漏掉,torch.nn.functional.*
    全部取不到签名。
    """
    single = re.compile(r"from\s+torch\s+import\s+(\w+)\s+as\s+(\w+)")
    block_open = re.compile(r"from\s+torch\s+import\s*\(\s*$")
    block_item = re.compile(r"(\w+)\s+as\s+(\w+)\s*,?\s*$")
    in_block = False
    for line in lines:
        stripped = line.strip()
        if in_block:
            in_block = not stripped.startswith(")")
            m = block_item.match(stripped)
        elif block_open.match(stripped):
            in_block = True
            m = None
        else:
            m = single.match(stripped)
        if m:
            yield m.group(1), m.group(2)


def _load_nn_functional_pyi(cache):
    try:
        import os

        import torch

        pyi_path = os.path.join(os.path.dirname(torch.__file__), "nn", "functional.pyi")
        if not os.path.exists(pyi_path):
            return
        with open(pyi_path) as f:
            lines = f.readlines()

        for torch_name, local_name in _iter_pyi_reexports(lines):
            full_name = f"torch.nn.functional.{local_name}"
            torch_full = f"torch.{torch_name}"
            if torch_full in cache and full_name not in cache:
                cache[full_name] = cache[torch_full]

        for func_name, param_str in _iter_pyi_defs(lines):
            full_name = f"torch.nn.functional.{func_name}"
            parsed = _parse_pyi_param_str(param_str)
            if parsed:
                if full_name not in cache:
                    cache[full_name] = []
                cache[full_name].append(parsed)
    except Exception:
        pass


def _extract_params_from_pyi(api_name: str):
    if not api_name.startswith("torch."):
        return None
    sigs = _load_pyi_signatures().get(api_name)
    if not sigs:
        return None
    all_overloads = []
    for sig in sigs:
        all_overloads.append(sig)
    primary = all_overloads[0]
    source = f"pyi-stub({len(all_overloads)} overloads)" if len(all_overloads) > 1 else "pyi-stub"
    return primary, source, all_overloads


def _resolve_function(api_name: str):
    try:
        parts = api_name.split(".")
        if len(parts) < 2:
            return None
        if api_name.startswith("torch.ops."):
            import torch

            if len(parts) >= 4:
                namespace = parts[2]
                op_name = parts[3]
                return getattr(torch.ops, namespace).__getattr__(op_name)
        if len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor":
            import torch

            method_name = ".".join(parts[2:])
            return getattr(torch.Tensor, method_name, None)
        mod_name = ".".join(parts[:-1])
        func_name = parts[-1]
        import importlib

        mod = importlib.import_module(mod_name)
        return getattr(mod, func_name, None)
    except (ImportError, AttributeError):
        return None


_MANUAL_OVERRIDES: Dict[str, APIParamInfo] = {}


def register_api_params(api_name: str, params: List[ParamInfo], source: str = "manual"):
    _MANUAL_OVERRIDES[api_name] = APIParamInfo(api_name=api_name, params=params, source=source)


def _is_tensor_method(api_name: str) -> bool:
    parts = api_name.split(".")
    return len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor"


def get_api_params(api_name: str) -> Optional[APIParamInfo]:
    if api_name in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[api_name]
    result = extract_api_params(api_name)
    if result is not None and _is_tensor_method(api_name):
        # Functional alias params already include the self-equivalent tensor
        # (e.g. 'input' from torch.nn.functional.relu_), so skip insertion.
        is_alias = "alias(" in (result.source or "")
        if not is_alias:
            self_param = ParamInfo(name="self", type="Tensor")
            if result.overloads:
                for ov in result.overloads:
                    ov.params.insert(0, self_param)
                    ov.layout = OverloadTensorLayout.build(ov.params, ov.return_count)
                best = max(result.overloads, key=lambda oi: sum(1 for p in oi.params if p.is_optional))
                result.params = best.params
            else:
                result.params.insert(0, self_param)
    if result is not None:
        result._resolve_api_flags()
    return result
