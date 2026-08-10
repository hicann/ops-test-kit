#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""
C++ source generator for GEIR tests.

Op-level template: one CPP per operator (cached), per-case data (shapes,
dtypes, formats, attrs) written as JSON config and read at runtime.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from .proto_loader import ProtoLoader

_DTYPE_TO_GE_ENUM = {
    "float32": "DT_FLOAT",
    "float16": "DT_FLOAT16",
    "bfloat16": "DT_BF16",
    "int32": "DT_INT32",
    "int16": "DT_INT16",
    "int8": "DT_INT8",
    "uint8": "DT_UINT8",
    "uint16": "DT_UINT16",
    "uint32": "DT_UINT32",
    "int64": "DT_INT64",
    "uint64": "DT_UINT64",
    "float64": "DT_DOUBLE",
    "bool": "DT_BOOL",
    "float8_e4m3fn": "DT_FLOAT8_E4M3FN",
    "float8_e5m2": "DT_FLOAT8_E5M2",
    "float8_e8m0": "DT_FLOAT8_E8M0",
    "hifloat8": "DT_HIFLOAT8",
}

_FORMAT_TO_GE_ENUM = {
    "ND": "FORMAT_ND",
    "NC1HWC0": "FORMAT_NC1HWC0",
    "FRACTAL_Z": "FORMAT_FRACTAL_Z",
    "FRACTAL_NZ": "FORMAT_FRACTAL_NZ",
    "NCHW": "FORMAT_NCHW",
    "NHWC": "FORMAT_NHWC",
    "HWCN": "FORMAT_HWCN",
    "C1HWNCoC0": "FORMAT_C1HWNCoC0",
    "NDHWC": "FORMAT_NDHWC",
    "NCDHW": "FORMAT_NCDHW",
    "NDC1HWC0": "FORMAT_NDC1HWC0",
}

_PROTO_ATTR_TYPE_TO_CPP = {
    "Bool": "bool",
    "Int": "int64_t",
    "Int64": "int64_t",
    "UInt64": "int64_t",
    "Float": "float",
    "Double": "double",
    "String": "string",
    "ListInt": "vector_int64",
    "ListInt64": "vector_int64",
    "ListFloat": "vector_float",
    "ListDouble": "vector_double",
    "ListBool": "vector_int64",
    "ListString": "vector_string",
    "ListType": "vector_int64",
    "Type": "int64_t",
    "DataType": "int64_t",
    "Format": "int64_t",
}

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _render_template(name: str, **ctx) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template(name)
    return tpl.render(**ctx)


def _resolve_dtype(dtype_str: str) -> str:
    return _DTYPE_TO_GE_ENUM.get(dtype_str, "DT_FLOAT")


def _resolve_format(fmt_str: str) -> str:
    return _FORMAT_TO_GE_ENUM.get(fmt_str, "FORMAT_ND")


def _attr_value_to_json(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_attr_value_to_json(x) for x in v]
    return str(v)


class GeirGraphBuilder:
    """Generate op-level C++ source and per-case JSON config for GEIR tests."""

    def __init__(self, switches):
        self._switches = switches
        self._proto_loader = ProtoLoader()
        self._work_dir: Optional[str] = None
        self._op_dir: Optional[str] = None
        self._last_proto_file: Optional[str] = None

    @property
    def work_dir(self) -> Optional[str]:
        return self._work_dir

    @property
    def op_dir(self) -> Optional[str]:
        return self._op_dir

    @property
    def last_proto_file(self) -> Optional[str]:
        return self._last_proto_file

    def _compute_dirs(self, mode: str, work_dir: str = None):
        is_dynamic = mode.startswith("dynamic")
        is_binary = "binary" in mode
        base_dir = "dynamic" if is_dynamic else "const"
        sub_dir = os.path.join(base_dir, "binary") if is_binary else base_dir
        root = getattr(self._switches, "root_path", os.getcwd())
        self._work_dir = work_dir or os.path.join(root, "geir", sub_dir)
        self._op_dir = os.path.join(root, "geir", "ops")
        os.makedirs(self._work_dir, exist_ok=True)
        os.makedirs(self._op_dir, exist_ok=True)

    def _build_attr_entries(self, proto_info) -> List[Dict[str, str]]:
        input_name_set = set(proto_info.inputs)
        attr_entries = []
        for attr_name, proto_type in proto_info.attrs:
            if attr_name in input_name_set:
                continue
            if attr_name[0] in ("!", "#", "@"):
                continue
            cpp_type = _PROTO_ATTR_TYPE_TO_CPP.get(proto_type, "")
            if not cpp_type:
                raise RuntimeError(
                    f"Unsupported proto attr type '{proto_type}' for attr '{attr_name}' "
                    f"on operator '{proto_info.op_class}'"
                )
            attr_entries.append({
                "name": attr_name,
                "proto_type": proto_type,
                "cpp_type": cpp_type,
            })
        return attr_entries

    def generate_op_source(self, op_name: str, mode="const", work_dir: str = None) -> Optional[str]:
        """Generate op-level C++ source (one per operator, cached). Returns file path."""
        self._compute_dirs(mode, work_dir)

        proto_info = self._proto_loader.get_op_info(op_name)
        if proto_info is None:
            logging.error("No proto info for operator '%s'", op_name)
            return None
        self._last_proto_file = proto_info.proto_file

        attr_entries = self._build_attr_entries(proto_info)

        source = _render_template(
            "geir_op_template.cpp.j2",
            proto_file=proto_info.proto_file,
            op_class=proto_info.op_class,
            input_names=proto_info.inputs[:],
            output_names=proto_info.outputs[:],
            dynamic_input_names=(proto_info.dynamic_inputs or []),
            attr_entries=attr_entries,
            dtype_map=[(v, v) for v in _DTYPE_TO_GE_ENUM.values()],
            format_map=[(v, v) for v in _FORMAT_TO_GE_ENUM.values()],
        )

        source_path = os.path.join(self._op_dir, "%s.cpp" % op_name)
        if os.path.isfile(source_path):
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    if f.read() == source:
                        logging.info("GEIR op source unchanged: %s", source_path)
                        return source_path
            except OSError:
                pass
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source)
        logging.info("Generated GEIR op source: %s", source_path)
        return source_path

    def write_case_config(self, testcase, mode="const", work_dir: str = None) -> Optional[str]:
        """Write per-case JSON config. Returns file path."""
        if work_dir:
            self._work_dir = work_dir
        if not self._work_dir:
            self._compute_dirs(mode)

        op_name = testcase.op_name
        input_shapes = testcase.input_shapes
        dyn_input_shapes = getattr(testcase, "dyn_input_shapes", None)
        input_dtypes = testcase.input_dtypes
        input_formats = getattr(testcase, "input_formats", None) or ()
        input_ori_formats = getattr(testcase, "input_ori_formats", None) or ()
        input_ori_shapes = getattr(testcase, "input_ori_shapes", None) or ()
        output_ori_shapes = getattr(testcase, "output_ori_shapes", None) or ()
        output_shapes = testcase.output_shapes
        output_dtypes = testcase.output_dtypes
        output_formats = getattr(testcase, "output_formats", None) or ()
        output_ori_formats = getattr(testcase, "output_ori_formats", None) or ()
        attrs = testcase.attributes or {}

        is_dynamic = mode.startswith("dynamic")
        is_binary = "binary" in mode

        proto_info = self._proto_loader.get_op_info(op_name)
        if proto_info is None:
            logging.error("No proto info for operator '%s'", op_name)
            return None
        self._last_proto_file = proto_info.proto_file

        input_names = proto_info.inputs[:]
        out_names = proto_info.outputs[:]
        attr_keys = set(attrs.keys())

        # ---- inputs ----
        dynamic_input_names = set(getattr(proto_info, "dynamic_inputs", None) or [])
        inputs_json: List[Optional[Dict[str, Any]]] = []
        data_idx = 0
        for i, name in enumerate(input_names):
            if i >= len(input_shapes):
                inputs_json.append(None)
                continue
            if input_shapes[i] is None:
                inputs_json.append(None)
                continue
            dtype_str = (
                input_dtypes[i]
                if isinstance(input_dtypes, (list, tuple)) and i < len(input_dtypes)
                else str(input_dtypes[0])
            )
            fmt_str = (
                (input_formats[i] if isinstance(input_formats, (list, tuple)) and i < len(input_formats) else "ND")
                if input_formats
                else "ND"
            )
            ori_fmt_str = (
                (
                    input_ori_formats[i]
                    if isinstance(input_ori_formats, (list, tuple)) and i < len(input_ori_formats)
                    else "ND"
                )
                if input_ori_formats
                else "ND"
            )
            # DYNAMIC_INPUT(TensorList)端口:input_shapes[i] 是嵌套的 shape 列表,
            # 逐元素展开为 elements,模板按变参口(create_dynamic_input_X)接线。
            if name in dynamic_input_names and isinstance(input_shapes[i], (list, tuple)) and input_shapes[i] and isinstance(input_shapes[i][0], (list, tuple)):
                elems = []
                elt_dtypes = input_dtypes[i] if (isinstance(input_dtypes, (list, tuple)) and i < len(input_dtypes) and isinstance(input_dtypes[i], (list, tuple))) else [dtype_str] * len(input_shapes[i])
                for j, eshape in enumerate(input_shapes[i]):
                    edt = elt_dtypes[j] if j < len(elt_dtypes) else elt_dtypes[-1]
                    elems.append({
                        "data_idx": data_idx + j,
                        "data_shape": list(eshape),
                        "desc_shape": list(eshape),
                        "dtype": _resolve_dtype(edt),
                        "format": _resolve_format(fmt_str),
                        "ori_format": _resolve_format(ori_fmt_str),
                    })
                inputs_json.append({
                    "name": name,
                    "is_const": False,
                    "dynamic": True,
                    "count": len(elems),
                    "elements": elems,
                })
                data_idx += len(elems)
                continue
            data_shape = list(input_shapes[i])
            if is_dynamic and dyn_input_shapes and i < len(dyn_input_shapes) and dyn_input_shapes[i] is not None:
                desc_shape = list(dyn_input_shapes[i])
            elif is_dynamic:
                desc_shape = [-1 for _ in data_shape]
            else:
                desc_shape = data_shape
            inputs_json.append({
                "name": name,
                "is_const": (name in attr_keys) or bool(os.environ.get("GEIR_CONST_FEED")),
                "data_idx": data_idx,
                "data_shape": data_shape,
                "desc_shape": desc_shape,
                "dtype": _resolve_dtype(dtype_str),
                "format": _resolve_format(fmt_str),
                "ori_format": _resolve_format(ori_fmt_str),
                "ori_shape": (list(input_ori_shapes[i])
                              if isinstance(input_ori_shapes, (list, tuple)) and i < len(input_ori_shapes)
                                 and input_ori_shapes[i] is not None
                              else list(input_shapes[i])),
            })
            data_idx += 1

        # ---- outputs ----
        outputs_json: List[Optional[Dict[str, Any]]] = []
        for i, name in enumerate(out_names):
            if i >= len(output_shapes):
                outputs_json.append(None)
                continue
            dtype_str = (
                output_dtypes[i]
                if isinstance(output_dtypes, (list, tuple)) and i < len(output_dtypes)
                else str(output_dtypes[0])
            )
            fmt_str = (
                (output_formats[i] if isinstance(output_formats, (list, tuple)) and i < len(output_formats) else "ND")
                if output_formats
                else "ND"
            )
            ori_fmt_str = (
                (
                    output_ori_formats[i]
                    if isinstance(output_ori_formats, (list, tuple)) and i < len(output_ori_formats)
                    else "ND"
                )
                if output_ori_formats
                else "ND"
            )
            out_data_shape = list(output_shapes[i])
            if is_dynamic:
                out_desc_shape = [-1 for _ in out_data_shape]
            else:
                out_desc_shape = out_data_shape
            outputs_json.append({
                "name": name,
                "data_shape": out_data_shape,
                "desc_shape": out_desc_shape,
                "dtype": _resolve_dtype(dtype_str),
                "format": _resolve_format(fmt_str),
                "ori_format": _resolve_format(ori_fmt_str),
                "ori_shape": (list(output_ori_shapes[i])
                              if isinstance(output_ori_shapes, (list, tuple)) and i < len(output_ori_shapes)
                                 and output_ori_shapes[i] is not None
                              else list(output_shapes[i])),
            })

        # ---- attrs (exclude input names and special prefixes) ----
        input_name_set = set(input_names)
        attrs_json: Dict[str, Any] = {}
        for k, v in attrs.items():
            if k in input_name_set or str(k)[0] in ("!", "#", "@"):
                continue
            attrs_json[k] = _attr_value_to_json(v)

        # ---- build options ----
        jit_compile = "0" if is_binary else "1"
        compile_dynamic_mode = "1" if (is_dynamic and not is_binary) else ""
        ge_deterministic_level = ""
        if getattr(self._switches, "deterministic_level", 0) == 1:
            ge_deterministic_level = "1"
        build_options = {
            "ge.jit_compile": jit_compile,
            "ge.compile_dynamic_mode": compile_dynamic_mode,
            "ge.deterministicLevel": ge_deterministic_level,
        }

        config = {
            "inputs": inputs_json,
            "outputs": outputs_json,
            "attrs": attrs_json,
            "build_options": build_options,
        }

        config_path = os.path.join(self._work_dir, "%s.json" % testcase.testcase_name)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logging.info("Generated GEIR case config: %s", config_path)
        return config_path
