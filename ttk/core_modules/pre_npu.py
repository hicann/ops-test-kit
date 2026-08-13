#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build runtime context and execute the optional TestSpec pre-NPU stage."""

import csv
import inspect
import logging
import pathlib
import re
import types
from types import MappingProxyType
from typing import Any, Callable, ForwardRef, Mapping, Union, get_args, get_origin

from ttk.core_modules.manual_data import (
    case_directory_name,
    manual_data_prepare_roles,
)
from ttk.test_spec import get_spec_attr
from ttk.test_spec.pre_npu import PreNpuResult
from ttk.test_spec.runtime_context import (
    RuntimeKernelProfile,
    RuntimeProfile,
    TtkContext,
)
from ttk.utilities.string_utils import stable_path_component

_PRE_NPU_WARMUP_COUNT = 5


def _read_only_mapping(value) -> Mapping[str, Any]:
    return MappingProxyType({
        key: _freeze_option(item)
        for key, item in dict(value if value is not None else {}).items()
    })


def _freeze_option(value):
    """Keep hook-visible runtime options stable without exposing mutable switches."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_option(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_option(item) for item in value)
    if hasattr(value, "__slots__"):
        slots = value.__slots__
        names = (slots,) if isinstance(slots, str) else slots
        return MappingProxyType({
            name: _freeze_option(getattr(value, name))
            for name in names if not name.startswith("_") and hasattr(value, name)
        })
    if hasattr(value, "__dict__"):
        return MappingProxyType({
            name: _freeze_option(item)
            for name, item in value.__dict__.items() if not name.startswith("_")
        })
    return value


def _runtime_options(switches) -> Mapping[str, Any]:
    names = getattr(switches, "__slots__", ())
    if isinstance(names, str):
        names = (names,)
    options = {
        name: _freeze_option(getattr(switches, name))
        for name in names
        if not name.startswith("_") and hasattr(switches, name)
    }
    options["task_prof"] = bool(getattr(switches, "TASK_PROFILING", False))
    options["run_time"] = int(getattr(switches, "run_time", 1))
    options["compile_only"] = bool(getattr(switches, "compile_only", False))
    options["output_file_name"] = getattr(switches, "output_file_name", None)
    options["manual_data_dirs"] = tuple(
        pathlib.Path(path).expanduser().resolve()
        for path in (getattr(switches, "manual_data_dirs", ()) or ())
    )
    return MappingProxyType(options)


def resolve_manual_case_dir(testcase, switches, manual_case=None):
    """Return the selected or uniquely addressable testcase data directory."""
    if manual_case is not None:
        return pathlib.Path(manual_case.case_dir).resolve()
    roots = tuple(getattr(switches, "manual_data_dirs", ()) or ())
    if len(roots) != 1:
        return None
    root = pathlib.Path(roots[0]).expanduser().resolve()
    return root / case_directory_name(str(testcase.testcase_name))


def build_ttk_context(testcase, switches, case_type, manual_case=None):
    """Create optional hook context once for one testcase invocation."""
    api_name = getattr(testcase, "api_name", None)
    if api_name is None:
        api_name = getattr(testcase, "op_name", "")
    manual_case_dir = resolve_manual_case_dir(testcase, switches, manual_case)
    _write_inputs, write_goldens = manual_data_prepare_roles(switches)
    return TtkContext(
        api_name=str(api_name or ""),
        testcase_name=str(getattr(testcase, "testcase_name", "")),
        case_type=case_type,
        input_tensors=getattr(testcase, "tensors", None),
        input_scalars=getattr(testcase, "scalars", ()),
        attributes=_read_only_mapping(getattr(testcase, "attributes", None)),
        csv_fields=_read_only_mapping(getattr(testcase, "original_dict", None)),
        options=_runtime_options(switches),
        manual_data_mode=getattr(switches, "manual_data_mode", None),
        manual_data_writes_goldens=write_goldens,
        manual_data_dirs=tuple(
            pathlib.Path(path).expanduser().resolve()
            for path in (getattr(switches, "manual_data_dirs", ()) or ())
        ),
        manual_case_dir=manual_case_dir,
        manual_data_format=(
            str(manual_case.file_format) if manual_case is not None else None
        ),
    )


def refresh_ttk_context(context, testcase, input_tensors=None, input_scalars=None):
    """Refresh generated/restored values without replacing process-local state."""
    context.input_tensors = (
        getattr(testcase, "tensors", None) if input_tensors is None else input_tensors
    )
    context.input_scalars = (
        getattr(testcase, "scalars", ()) if input_scalars is None else input_scalars
    )


def _is_ttk_context_annotation(annotation):
    if annotation is TtkContext:
        return True
    if isinstance(annotation, ForwardRef):
        return _is_ttk_context_annotation(annotation.__forward_arg__)
    if isinstance(annotation, str):
        names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", annotation))
        context_names = {"TtkContext", "ttk.test_spec.TtkContext"}
        wrappers = {
            "None",
            "NoneType",
            "Optional",
            "Union",
            "typing.Optional",
            "typing.Union",
        }
        return bool(names & context_names) and names <= context_names | wrappers
    union_types = (Union,)
    if hasattr(types, "UnionType"):
        union_types += (types.UnionType,)
    if get_origin(annotation) not in union_types:
        return False
    members = get_args(annotation)
    return any(_is_ttk_context_annotation(item) for item in members) and all(
        item is type(None) or _is_ttk_context_annotation(item) for item in members
    )


def add_context_if_declared(func, kwargs, context):
    """Inject context only into a parameter explicitly typed as ``TtkContext``."""
    parameter = inspect.signature(func).parameters.get("context")
    if parameter is None or not _is_ttk_context_annotation(parameter.annotation):
        return
    if parameter.kind == inspect.Parameter.VAR_KEYWORD:
        return
    if parameter.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError("context must be a keyword or keyword-only parameter")
    if "context" in kwargs:
        raise ValueError(
            "CSV attribute context conflicts with the reserved TestSpec hook parameter"
        )
    kwargs["context"] = context


def _read_aclnn_profile(result_path, repeat_count):
    """Read the native ACLNN profiler summary into the public hook result."""
    statistic_files = tuple(result_path.glob("**/op_statistic_*.csv"))
    if not statistic_files:
        return RuntimeProfile(True, repeat_count, 0.0, result_path)
    statistic_file = max(
        statistic_files,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )

    kernels = []
    with statistic_file.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            core_type = str(row.get("Core Type", ""))
            suffix = "AiCpu" if "cpu" in core_type.lower() else "AiCore"
            kernels.append(
                RuntimeKernelProfile(
                    name=f"{row['OP Type']}_{suffix}",
                    device_us=float(row["Total Time(us)"]),
                    calls=int(row["Count"]),
                    avg_us=float(row["Avg Time(us)"]),
                    max_us=float(row["Max Time(us)"]),
                    min_us=float(row["Min Time(us)"]),
                )
            )
    elapsed_us = sum(kernel.device_us for kernel in kernels) / repeat_count
    return RuntimeProfile(True, repeat_count, elapsed_us, result_path, tuple(kernels))


def build_pre_npu_profile_runner(
    testcase,
    switches,
    case_type,
    synchronize: Callable[[], None] = None,
    device_id=None,
):
    """Build a generic profiler used only when a hook calls ``run_profiled``."""
    testcase_name = str(getattr(testcase, "testcase_name", ""))
    api_name = str(
        getattr(testcase, "api_name", "") or getattr(testcase, "op_name", "")
    )
    case_component = stable_path_component(
        f"{case_type}-{api_name}-{testcase_name}", "testcase"
    )
    root_path = pathlib.Path(getattr(switches, "root_path", pathlib.Path.cwd()))

    def run_profiled(stage_name, operation):
        if not isinstance(stage_name, str) or not stage_name.strip():
            raise ValueError("profiled hook stage_name must be a non-empty string")
        if not callable(operation):
            raise TypeError("profiled hook operation must be callable")

        profiling_enabled = bool(getattr(switches, "TASK_PROFILING", True))
        repeat_count = (
            max(int(getattr(switches, "run_time", 1) or 1), 1)
            if profiling_enabled else 1
        )
        if profiling_enabled and bool(getattr(switches, "warmup", False)):
            for _ in range(_PRE_NPU_WARMUP_COUNT):
                operation()
            if synchronize is not None:
                synchronize()

        if not profiling_enabled:
            operation()
            if synchronize is not None:
                synchronize()
            return RuntimeProfile(False, repeat_count, 0.0, None)

        stage_component = stable_path_component(stage_name, "stage")
        result_path = (
            root_path
            / "msprof"
            / "pre_npu"
            / case_component
            / stage_component
        )
        if case_type == "aclnn":
            if device_id is None:
                raise RuntimeError("ACLNN hook profiling requires a device id")
            from ttk.core_modules.msprof import MsProfiler, TtkMsProfType

            with MsProfiler(
                device_id,
                str(result_path),
                TtkMsProfType.API,
                start_step=0,
            ) as profiler:
                for _ in range(repeat_count):
                    profiler.step()
                    operation()
            profile = _read_aclnn_profile(result_path, repeat_count)
        else:
            from ttk.core_modules.framework_api.profiler import NpuProfiler

            profiler = NpuProfiler(None, result_path=str(result_path))
            with profiler:
                for _ in range(repeat_count):
                    operation()
                if synchronize is not None:
                    synchronize()
            result = profiler.result(None, repeat_count)
            details = result.kernel_details
            kernels = tuple(
                RuntimeKernelProfile(
                    name=kernel.name,
                    device_us=kernel.device_us,
                    calls=kernel.calls,
                    avg_us=kernel.avg_us,
                    max_us=kernel.max_us,
                    min_us=kernel.min_us,
                )
                for kernel in (details.kernels if details is not None else ())
            )
            profile = RuntimeProfile(
                True, repeat_count, result.elapsed_us, result_path, kernels
            )
        logging.info(
            "[%s] profiled pre-NPU stage %s: %.3f us, kernels=%d, result=%s",
            testcase_name,
            stage_name,
            profile.elapsed_us,
            len(profile.kernels),
            result_path,
        )
        return profile

    return run_profiled


def resolve_pre_npu(testcase, switches):
    """Resolve the optional stage once so no-plugin paths avoid device setup."""
    return get_spec_attr(testcase.api_name, "pre_npu", switches.plugin_path)


def _build_pre_npu_call(testcase, switches, context, func):
    """Build the regular API callback arguments before optionally adding context."""
    plan = testcase.get_param_plan() if hasattr(testcase, "get_param_plan") else None
    args = []
    kwargs = {}
    extra_attrs = dict(getattr(testcase, "attributes", None) or {})
    if plan is not None and hasattr(plan, "build_args"):
        build_args = plan.build_args
        tensors = testcase.tensors
        scalars = getattr(testcase, "scalars", ())
        attributes = getattr(testcase, "attributes", None)
        try:
            inspect.signature(build_args).bind(tensors, scalars, attributes)
        except TypeError:
            args, plan_kwargs, extra_attrs = build_args(tensors)
            kwargs.update(plan_kwargs)
        else:
            args, extra_attrs = build_args(tensors, scalars, attributes)

    extra = {
        "testcase_name": getattr(testcase, "testcase_name", None),
        "short_soc_version": getattr(switches, "short_soc_version", None),
        "tensor_formats": getattr(testcase, "tensor_formats", None),
        "tensor_dtypes": getattr(testcase, "tensor_dtypes", None),
        "scalar_dtypes": getattr(testcase, "scalar_dtypes", None),
        "input_ranges": getattr(testcase, "input_data_ranges", None),
        "use_torch": getattr(testcase, "is_torch_dtype_support", lambda: False)(),
    }
    for name in ("batch_axis", "batch_slice_info", "batch_seed"):
        value = getattr(testcase, name, None)
        if value is not None:
            extra[name] = value
    extra.update(extra_attrs)

    signature = inspect.signature(func)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        kwargs.update(extra)
    else:
        kwargs.update({
            name: value for name, value in extra.items()
            if name in signature.parameters
        })
    add_context_if_declared(func, kwargs, context)

    accepts_args = any(
        parameter.name != "context" and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        for parameter in signature.parameters.values()
    )
    return (args if accepts_args else ()), kwargs


def execute_pre_npu(
    testcase, switches, context, aclnn_runner=None, profile_runner=None,
    pre_npu_func=None,
):
    """Execute an operator stage once and normalize its stop/continue result."""
    func = pre_npu_func
    if func is None:
        func = resolve_pre_npu(testcase, switches)
    if func is None:
        return PreNpuResult()

    context._aclnn_runner = aclnn_runner
    context._profile_runner = profile_runner
    try:
        args, kwargs = _build_pre_npu_call(testcase, switches, context, func)
        result = func(*args, **kwargs)
    finally:
        context._aclnn_runner = None
        context._profile_runner = None

    if result is None:
        return PreNpuResult()
    if not isinstance(result, PreNpuResult):
        raise TypeError(
            "TestSpec.pre_npu must return None or ttk.test_spec.PreNpuResult"
        )
    return result
