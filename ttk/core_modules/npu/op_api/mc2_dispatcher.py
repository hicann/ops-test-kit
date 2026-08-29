#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). See LICENSE in the root of the software repository for the
# full text of the License.

"""MC2 multi-device dispatcher — framework entry point.

All operator-specific golden logic has been migrated to per-operator
tests/assets/ directories in ops-transformer. This module retains only:
  - __golden_multi_device_compare: migration dispatcher: SKIP if no plugin, delegate if plugin present
  - GMM attribute patching helpers (used by profiling layer for sendCounts/recvCounts)
  - Public framework_* stubs for backward compatibility

TTK = multi-device test framework only. Golden computation lives in
ops-transformer/mc2/<op>/tests/assets/impl/golden.py, loaded via --plugin.
"""

import logging
from typing import Dict, List

import numpy

from ...testcase_manager import TestcaseAclnn

# ---------------------------------------------------------------------------
# Migration dispatcher
# ---------------------------------------------------------------------------

_MIGRATED_APIS = (
    "AlltoAllAllGatherBatchMatMul",
    "BatchMatMulReduceScatterAlltoAll",
    "GroupedMatMulAlltoAllv",
    "AlltoAllvGroupedMatMul",
    "MoeDistributeDispatch",
    "MoeDistributeCombine",
    "AllGatherMatmul",
    "MatmulReduceScatter",
    "MatmulAlltoAll",
    "AlltoAllMatmul",
    # A5 operators (quant / V2 / add_rms_norm variants)
    "MatmulAllReduce",
    "WeightQuantMatmulAllReduce",
    "QuantMatmulAllReduce",
    "AllGatherMatmulV2",
    "MatmulReduceScatterV2",
    "InplaceMatmulAllReduceAddRmsNorm",
    "MatmulAllReduceAddRmsNorm",
    "GroupedMatMulAllReduce",
    "QuantAllReduce",
    "QuantReduceScatter",
    "QuantGroupedMatMulAlltoAllv",
    "AlltoAllvQuantGroupedMatMul",
)


def __golden_multi_device_compare(
    thread_contexts: Dict[int, TestcaseAclnn], device_ids: List[int], all_precision: list
):
    """Multi-device golden dispatcher.

    All MC2 operators have been migrated to ops-transformer. When --plugin
    is provided, the plugin's multi_device_golden is called by the framework
    before reaching here. If we reach here without a plugin, the operator
    is marked SKIP(MIGRATED).

    DistributeBarrier is a synchronization-only API with no value output.
    """
    api_name = next(iter(thread_contexts.values())).api_name

    if api_name.startswith("aclnnDistributeBarrier"):
        for did in device_ids:
            thread_contexts[did].golden_tensors = []
            all_precision.append(f"rank{did}:PASS(EXECUTED)")
        return

    if any(m in api_name for m in _MIGRATED_APIS):
        logging.warning(
            f"{api_name}: golden has been migrated to ops-transformer; "
            f"use --plugin <op>/tests/assets to run precision comparison"
        )
        for did in device_ids:
            thread_contexts[did].golden_tensors = []
            all_precision.append(f"rank{did}:SKIP(MIGRATED)")
        return

    # Non-migrated APIs: mark as no golden available
    logging.warning(f"{api_name}: no multi-device golden implementation")
    for did in device_ids:
        thread_contexts[did].golden_tensors = []
        all_precision.append(f"rank{did}:SKIP(NO_GOLDEN)")


# ---------------------------------------------------------------------------
# GMM attribute patching (framework setup, used by profiling layer)
# ---------------------------------------------------------------------------


def _generate_gmm_alltoallv_matrix(a_array_val, exp_per_card, seed):
    n = len(a_array_val)
    rng = numpy.random.default_rng(seed)
    total = sum(a_array_val)
    if total % n != 0:
        return [[total // n] * (exp_per_card * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in a_array_val:
        if a % n != 0:
            return [[col_sum // (exp_per_card)] * (exp_per_card * n) for _ in range(n)]
        k = a // n
        k_values.append(max(k, exp_per_card))
    blocks = []
    for k in k_values:
        block = numpy.zeros((exp_per_card, n), dtype=int)
        for col in range(n):
            counts = rng.multinomial(k - exp_per_card, [1.0 / exp_per_card] * exp_per_card)
            block[:, col] = counts + 1
        blocks.append(block)
    tmp = numpy.vstack(blocks)
    return [list(col) for col in zip(*tmp)]


def _get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws):
    exp_per_card = first_ctx.tensor_view_shapes[1][0] if len(first_ctx.tensor_view_shapes) > 1 else 1
    seed_val = 0
    remark = first_ctx.remark or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "seed":
            try:
                seed_val = int(kv[1].strip())
            except ValueError:
                pass
    bsk = first_ctx.tensor_view_shapes[0][0] if first_ctx.tensor_view_shapes else 0
    a_array = [bsk] * ep_ws
    return _generate_gmm_alltoallv_matrix(a_array, exp_per_card, seed_val)


def get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws):
    return _get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws)


def generate_gmm_alltoallv_matrix(a_array_val, exp_per_card, seed):
    return _generate_gmm_alltoallv_matrix(a_array_val, exp_per_card, seed)


def patch_gmm_rank_attributes(ctx, rank_idx, world_size):
    api_name = ctx.api_name
    is_alltoallv_gmm = "AlltoAllvGroupedMatMul" in api_name
    is_gmm_alltoallv = "GroupedMatMulAlltoAllv" in api_name
    if not is_alltoallv_gmm and not is_gmm_alltoallv:
        return
    attrs = ctx.attributes
    ep_ws = attrs.get("epWorldSize", world_size)
    exp_per_card = ctx.tensor_view_shapes[1][0] if len(ctx.tensor_view_shapes) > 1 else 1
    seed_val = 0
    remark = ctx.remark or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "seed":
            try:
                seed_val = int(kv[1].strip())
            except ValueError:
                pass
    if is_alltoallv_gmm:
        bsk = ctx.tensor_view_shapes[0][0] if ctx.tensor_view_shapes else 0
        a_array = [bsk] * ep_ws
        exp_token_nums = _generate_gmm_alltoallv_matrix(a_array, exp_per_card, seed_val)
        send_counts = exp_token_nums[rank_idx]
        recv_counts = []
        for i in range(ep_ws):
            recv_counts.extend(exp_token_nums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card])
        attrs["sendCounts"] = send_counts
        attrs["recvCounts"] = recv_counts
    elif is_gmm_alltoallv:
        m_per_rank = ctx.tensor_view_shapes[0][0] if ctx.tensor_view_shapes else 0
        a_array = [m_per_rank] * ep_ws
        exp_token_nums = _generate_gmm_alltoallv_matrix(a_array, exp_per_card, seed_val)
        recv_counts = exp_token_nums[rank_idx]
        send_counts = []
        for i in range(ep_ws):
            send_counts.extend(exp_token_nums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card])
        attrs["sendCounts"] = send_counts
        attrs["recvCounts"] = recv_counts
        ctx._pure_attrs = None
        logging.info(
            f"[GMM patch] api={api_name} rank={rank_idx} ep_ws={ep_ws} "
            f"seed={seed_val} send_counts={send_counts[:4]}... recv_counts={recv_counts[:4]}..."
        )


def patch_gmm_weight_transpose(ctx):
    pass


# ---------------------------------------------------------------------------
# Public framework API stubs (backward compatibility)
# ---------------------------------------------------------------------------
# These are kept for any external code that may still import them.
# The actual implementations now live in ops-transformer's golden_utils.py.


def framework_to_torch_f32(t):
    import torch

    if t is None:
        return None
    if isinstance(t, numpy.ndarray):
        dtype_str = str(t.dtype)
        if "e8m0" in dtype_str or "float8_e8m0" in dtype_str:
            raw = t.view(numpy.uint8).astype(numpy.float64)
            arr = numpy.power(2.0, raw - 127).astype(numpy.float32)
        else:
            try:
                arr = t.astype(numpy.float32, copy=False)
            except (TypeError, ValueError):
                arr = t.view(numpy.uint8).astype(numpy.float32, copy=False)
        return torch.from_numpy(arr)
    if hasattr(t, "dtype") and str(t.dtype).replace("torch.", "") in ("float8_e4m3fn", "float8_e5m2", "hifloat8"):
        return t.float()
    return t.float()
