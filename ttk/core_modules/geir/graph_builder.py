#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""
C++ source generator for GEIR tests.  Uses a Jinja2 template to produce a
complete g++-compilable program from CSV-level testcase parameters.
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
}

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _render_template(name: str, **ctx) -> str:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)
    tpl = env.get_template(name)
    return tpl.render(**ctx)


def _cpp_shape(shape) -> str:
    """Convert a Python list/tuple to a C++ brace-init list string."""
    return "{" + ", ".join(str(x) for x in shape) + "}"


def _attr_value_str(v) -> str:
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(str(x) for x in v) + "}"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    return str(v)


class GeirGraphBuilder:
    """Generate and optionally compile a self-contained C++ GEIR test binary."""

    def __init__(self, switches):
        self._switches = switches
        self._proto_loader = ProtoLoader()
        self._work_dir: Optional[str] = None

    @property
    def work_dir(self) -> Optional[str]:
        return self._work_dir

    def generate(self, testcase, mode="const", work_dir: str = None) -> Optional[str]:
        """Generate C++ source for *testcase*, return file path."""
        op_name = testcase.op_name
        input_shapes = testcase.input_shapes
        dyn_input_shapes = getattr(testcase, "dyn_input_shapes", None)
        input_dtypes = testcase.input_dtypes
        input_formats = getattr(testcase, "input_formats", None) or ()
        input_ori_formats = getattr(testcase, "input_ori_formats", None) or ()
        output_shapes = testcase.output_shapes
        output_dtypes = testcase.output_dtypes
        output_formats = getattr(testcase, "output_formats", None) or ()
        output_ori_formats = getattr(testcase, "output_ori_formats", None) or ()
        attrs = testcase.attributes or {}

        # mode: "const" / "dynamic" / "const_binary" / "dynamic_binary"
        is_dynamic = mode.startswith("dynamic")
        is_binary = "binary" in mode

        proto_info = self._proto_loader.get_op_info(op_name)
        if proto_info is None:
            logging.error("No proto info for operator '%s'", op_name)
            return None

        # ---- inputs ----
        input_names = proto_info.inputs[:]
        attr_keys = set(attrs.keys())
        input_entries: List[Dict[str, Any]] = []
        data_idx = 0
        for i, name in enumerate(input_names):
            if i >= len(input_shapes):
                break
            if input_shapes[i] is None:
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
            data_shape = input_shapes[i]
            if is_dynamic and dyn_input_shapes and i < len(dyn_input_shapes) and dyn_input_shapes[i] is not None:
                desc_shape = dyn_input_shapes[i]
            elif is_dynamic:
                # Default: -1 for each dim (same as kernel _dynamicize)
                desc_shape = tuple(-1 for _ in data_shape)
            else:
                desc_shape = data_shape
            entry = {
                "name": name,
                "data_shape": _cpp_shape(data_shape),
                "desc_shape": _cpp_shape(desc_shape),
                "dtype_enum": _DTYPE_TO_GE_ENUM.get(dtype_str, "DT_FLOAT"),
                "format_enum": _FORMAT_TO_GE_ENUM.get(fmt_str, "FORMAT_ND"),
                "ori_format_enum": _FORMAT_TO_GE_ENUM.get(ori_fmt_str, "FORMAT_ND"),
                "is_const": name in attr_keys,
                "is_dynamic": is_dynamic,
                "data_idx": data_idx,
            }
            input_entries.append(entry)
            data_idx += 1

        # ---- outputs ----
        out_names = proto_info.outputs[:]
        output_entries: List[Dict[str, Any]] = []
        for i, name in enumerate(out_names):
            if i >= len(output_shapes):
                break
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
            out_data_shape = output_shapes[i]
            if is_dynamic:
                out_desc_shape = tuple(-1 for _ in out_data_shape)
            else:
                out_desc_shape = out_data_shape
            output_entries.append(
                {
                    "name": name,
                    "shape": _cpp_shape(out_data_shape),
                    "desc_shape": _cpp_shape(out_desc_shape),
                    "dtype_enum": _DTYPE_TO_GE_ENUM.get(dtype_str, "DT_FLOAT"),
                    "format_enum": _FORMAT_TO_GE_ENUM.get(fmt_str, "FORMAT_ND"),
                    "ori_format_enum": _FORMAT_TO_GE_ENUM.get(ori_fmt_str, "FORMAT_ND"),
                }
            )

        # ---- attributes (exclude keys that are proto inputs, e.g. tile.multiples) ----
        input_name_set = set(input_names)
        attr_entries = [
            {"name": k, "value": _attr_value_str(v)}
            for k, v in attrs.items()
            if k not in input_name_set and str(k)[0] not in ("!", "#", "@")
        ]

        # ---- render ----
        jit_compile = "0" if is_binary else "1"
        compile_dynamic_mode = "1" if (is_dynamic and not is_binary) else ""
        source = _render_template(
            "geir_test_template.cpp.j2",
            proto_file=proto_info.proto_file,
            op_class=proto_info.op_class,
            inputs=input_entries,
            outputs=output_entries,
            attributes=attr_entries,
            jit_compile=jit_compile,
            compile_dynamic_mode=compile_dynamic_mode,
        )

        base_dir = "dynamic" if is_dynamic else "const"
        sub_dir = os.path.join(base_dir, "binary") if is_binary else base_dir
        self._work_dir = work_dir or os.path.join(getattr(self._switches, "root_path", os.getcwd()), "geir", sub_dir)
        os.makedirs(self._work_dir, exist_ok=True)
        source_path = os.path.join(self._work_dir, "%s.cpp" % testcase.testcase_name)
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source)

        logging.info("Generated GEIR source: %s", source_path)
        return source_path
