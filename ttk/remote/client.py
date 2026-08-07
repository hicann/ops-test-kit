#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the CANN Open Software License Agreement Version 2.0
# (the "License"). You may not use this file except in compliance with
# the License. See LICENSE in the root of the software repository for the
# full text of the License.
"""
XPU 第三方输出采集客户端 —— 面向 core_modules 的门面。

把 TestSpec.third_party 翻译成 ExecutionSpec，经 EndpointView 解析可用 provider，
派发到远端 xpu_server 执行，提取返回的第三方输出数组。

本模块是 Kernel / GEIR / ACLNN / E2E 各模式共用的通用编排层，不耦合任何模式
专属上下文（OpInfoKeeper / OpApiInfoKeeper / TestcaseOp / TestcaseAclnn 等）。
各模式在调用前自行准备：op_name、inputs（逻辑 ori-shape 数组）、input_names、
op_type、attributes、testcase_name、switches。
"""

__all__ = [
    "extract_spec_providers",
    "build_spec",
    "xpu_mode_of",
    "extract_third_party",
    "dispatch_xpu",
    "collect_third_party",
]

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from . import DATA, PERF, ExecutionSpec, get_tenant_id

# third_party key aliases — mirrors server executor.py:_TP_ALIASES; keep in sync.
_TP_ALIASES = {"tensorflow": "tf", "np": "numpy"}


def extract_spec_providers(tp) -> List[str]:
    """Spec-layer provider keys (priority order = insertion order).

    dict -> keys; str -> single provider derived from API prefix; None/empty -> [].
    (Empty -> caller lets EndpointView.resolve_providers use detect∩yaml∩alive.)
    """
    if isinstance(tp, dict):
        return [_TP_ALIASES.get(k, k) for k in tp.keys()]
    if isinstance(tp, str):
        from ttk.remote import _derive_provider_from_api

        return [_derive_provider_from_api(tp, "torch")]
    return []


def build_spec(provider: str, tp, spec_file: Optional[str], spec_class: Optional[str]) -> ExecutionSpec:
    """Build one ExecutionSpec. api source: third_party dict | str | None.

    - dict[str, str]      -> type='api', api=value
    - dict[str, type]     -> type='spec' (impl class; server resolves
                             cls.third_party[provider]); spec_file/class
                             carried so the server can sync the module
    - str                 -> type='api', api=tp
    - None / no spec      -> type='api', api=None (server _resolve_3party_api
                             derives api from op_name + op_type)
    """
    if isinstance(tp, dict) and provider in tp:
        v = tp[provider]
        if isinstance(v, str):
            return ExecutionSpec(provider=provider, type="api", api=v)
        return ExecutionSpec(
            provider=provider,
            type="spec",
            spec_file=spec_file,
            spec_module=Path(spec_file).stem if spec_file else None,
            spec_class=spec_class,
        )
    if isinstance(tp, str):
        return ExecutionSpec(provider=provider, type="api", api=tp)
    return ExecutionSpec(provider=provider, type="api", api=None)


def xpu_mode_of(switches, need_data: bool) -> int:
    """按位或：xpu_perf→PERF，need_data→DATA。返回 0/PERF/DATA/DATA|PERF。"""
    mode = 0
    if getattr(switches, "xpu_perf", False):
        mode |= PERF
    if need_data:
        mode |= DATA
    return mode


def extract_third_party(xpu_results, priority: Optional[str]):
    """从 priority provider 取 outputs（纯函数，直接索引，不靠 dict 序）。

    fail-closed：无 results / 无 priority / 非 PASS / 无 outputs → None
    （cross_check → GOLDEN_FAILURE）。
    """
    if not xpu_results or priority is None:
        return None
    entry = xpu_results.get(priority, {})
    if entry.get("status") != "PASS" or "outputs" not in entry:
        return None
    return entry["outputs"]


def dispatch_xpu(
    *,
    op_name: str,
    inputs,
    input_names: List[str],
    op_type: Optional[str],
    attributes: dict,
    testcase_name: str,
    switches,
    need_data: bool,
):
    """Run XPU dispatch，返回 (xpu_results, priority_provider)。

    resolve_providers 失败 → xpu_results={} + priority=None
    （→ extract_third_party None → GOLDEN_FAILURE）。

    参数化：不读 OpInfoKeeper / context.op_name / get_global_storage，
    全部由调用方注入。
    """
    from ttk.remote.endpoint_view import EndpointView, _parse_provider_filter
    from ttk.test_spec import get_spec_attr, get_spec_class_meta

    paths = getattr(switches, "plugin_path", None) or ()
    tp = get_spec_attr(op_name, "third_party", paths)
    if isinstance(tp, dict):
        tp = {_TP_ALIASES.get(k, k): v for k, v in tp.items()}
    meta = get_spec_class_meta(op_name, paths)
    spec_file = meta["spec_file"] if meta else None
    spec_class = meta["class_name"] if meta else None

    ev = EndpointView()
    spec_providers = extract_spec_providers(tp)
    cli_providers = _parse_provider_filter(getattr(switches, "provider_filter", None))

    try:
        providers = ev.resolve_providers(spec_providers, cli_providers)
    except RuntimeError as e:
        logging.error("[%s] XPU resolve failed: %s", testcase_name or op_name, e)
        return {}, None

    specs = [build_spec(p, tp, spec_file, spec_class) for p in providers]

    # E2E: op_name is a dotted API path (e.g. "torch.add"); use it as api
    # when no explicit third_party is configured, so the server resolves it
    # via resolve_callable (dotted) instead of _resolve_3party_api (snake_case).
    if tp is None and op_name and "." in op_name:
        for s in specs:
            if s.api is None:
                s.api = op_name

    from ttk.remote.xpu_collector import collect_xpu_results

    _tmp_root = os.path.join(getattr(switches, "root_path", os.getcwd()), ".ttk", "xpu_tmp")
    xpu_results = collect_xpu_results(
        specs,
        inputs=inputs,
        input_names=input_names,
        mode=xpu_mode_of(switches, need_data),
        tenant_id=get_tenant_id(),
        op_name=op_name,
        op_type=op_type,
        attrs=attributes or {},
        tmp_root=_tmp_root,
        runtime=getattr(switches, "run_time", 3),
    )
    return xpu_results, (specs[0].provider if specs else None)


def collect_third_party(
    *,
    op_name: str,
    inputs,
    input_names: List[str],
    op_type: Optional[str],
    attributes: dict,
    testcase_name: str,
    switches,
    need_data: bool = True,
) -> Tuple[Optional[str], Optional[list], Optional[dict]]:
    """门面：采集第三方输出，返回 (priority_provider, flat_third_parties, xpu_results)。

    各模式在 cross_check / xpu_perf 场景调用：传入逻辑 ori-shape 的 inputs、
    算子参数名 input_names，本函数完成 spec 解析 → endpoint 解析 → 派发 → 提取 → 展平。

    - 远端不可用 / 无 provider / 执行失败 → (None, None, None)
      （调用方传 None 给 compare → cross_check 返回 GOLDEN_FAILURE）
    - cross_check 成功 → (priority, [np.ndarray, ...], xpu_results)
    - PERF-only 成功 → (priority, None, xpu_results)
    """
    from ttk.utilities import deep_flatten

    xpu_mode = xpu_mode_of(switches, need_data)
    if not xpu_mode:
        return None, None, None

    xpu_results, priority = dispatch_xpu(
        op_name=op_name,
        inputs=inputs,
        input_names=input_names,
        op_type=op_type,
        attributes=attributes,
        testcase_name=testcase_name,
        switches=switches,
        need_data=need_data,
    )

    if need_data:
        nested = extract_third_party(xpu_results, priority)
        if nested is None:
            if priority is None and not xpu_results:
                logging.warning(
                    "[%s] cross_check configured but no third_party output "
                    "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                    testcase_name or op_name,
                )
            return priority, None, xpu_results
        return priority, list(deep_flatten(nested)), xpu_results

    return priority, None, xpu_results
