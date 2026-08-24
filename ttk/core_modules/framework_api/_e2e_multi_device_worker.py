#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib
import logging
import os
import subprocess
import sys
import json
import socket
import datetime
import time
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

_moe_dispatch_module = None
_moe_combine_module = None
_moe_context_manager_cls = None
_mega_moe_module = None


_TORCH_NPU_MOE_TENSOR_PARAMS = {
    "torch_npu.npu_moe_distribute_dispatch": ("x", "expert_ids"),
    "torch_npu.npu_moe_distribute_dispatch_v2": ("x", "expert_ids"),
    "torch_npu.npu_moe_distribute_dispatch_setup": ("x", "expert_ids"),
    "torch_npu.npu_moe_distribute_dispatch_teardown": (
        "x", "y", "expert_ids", "comm_cmd_info"),
    "torch_npu.npu_moe_distribute_combine": (
        "expand_x", "expert_ids", "expand_idx", "ep_send_counts", "expert_scales"),
    "torch_npu.npu_moe_distribute_combine_v2": (
        "expand_x", "expert_ids", "assist_info_for_combine", "ep_send_counts", "expert_scales"),
    "torch_npu.npu_moe_distribute_combine_add_rms_norm": (
        "expand_x", "expert_ids", "expand_idx", "ep_send_counts", "expert_scales",
        "residual_x", "gamma"),
    "torch_npu.npu_moe_distribute_combine_setup": (
        "expand_x", "expert_ids", "assist_info_for_combine"),
    "torch_npu.npu_moe_distribute_combine_teardown": (
        "expand_x", "quant_expand_x", "expert_ids", "expand_idx",
        "expert_scales", "comm_cmd_info"),
}


def _uses_private_moe_backend(api_name):
    """Return whether the API needs TTK's private CANN MoE extension path."""
    return (api_name.startswith("cann_ops_transformer.") or
            api_name.startswith("torch.ops.cann_ops_transformer."))


def _call_torch_npu_moe(api_name, dev_tensors, hcomm, world_size, attrs):
    """Call public torch_npu MoE APIs without the private context tensor."""
    resolved = _resolve_api(api_name)
    kw = dict(attrs)
    kw.pop("_tensor_dtypes", None)
    tensor_count = len(_TORCH_NPU_MOE_TENSOR_PARAMS[api_name])
    tensors = list(dev_tensors[:tensor_count])
    ep_world_size = int(kw.pop("ep_world_size", world_size))
    ep_rank_id = int(kw.pop("ep_rank_id", kw.pop("_rank", 0)))
    moe_expert_num = int(kw.pop("moe_expert_num"))
    kw.pop("group_ep", None)
    kw["group_tp"] = kw.get("group_tp", "")
    return resolved(*tensors, hcomm, ep_world_size, ep_rank_id,
                    moe_expert_num, **kw)


def _call_torch_npu_token_exchange(api_name, dev_tensors, hcomm, world_size, attrs):
    """Call public Attention/FFN token exchange APIs."""
    resolved = _resolve_api(api_name)
    kw = dict(attrs)
    kw.pop("group", None)
    kw.pop("hcom", None)
    kw.pop("world_size", None)
    if api_name == "torch_npu.npu_attention_to_ffn":
        return resolved(
            *dev_tensors, hcomm, world_size,
            tuple(kw.pop("ffn_token_info_table_shape")),
            tuple(kw.pop("ffn_token_data_shape")),
            tuple(kw.pop("attn_token_info_table_shape")),
            int(kw.pop("moe_expert_num")), **kw)
    return resolved(
        *dev_tensors, hcomm, world_size,
        tuple(kw.pop("token_info_table_shape")),
        tuple(kw.pop("token_data_shape")), **kw)


def _get_moe_ccl_buffer_size(world_size, num_tokens, hidden, num_experts, topk,
                             shared_experts=0, shared_expert_ranks=0, comm_alg=""):
    """Mirror MoeDistributeBuffer.get_low_latency_ccl_buffer_size without JIT imports."""
    def align(value, base):
        return (value + base - 1) // base * base

    expert_world_size = world_size - shared_expert_ranks
    if expert_world_size <= 0:
        raise ValueError(
            "shared_expert_ranks must be less than world_size "
            f"(got {shared_expert_ranks} and {world_size})")
    local_experts = num_experts // expert_world_size
    token_bytes = align(hidden * 2, 32) + 44
    if comm_alg == "fullmesh_v2":
        dispatch_token_bytes = align(token_bytes, 480) // 480 * 512
    else:
        dispatch_token_bytes = align(token_bytes, 512)
    combine_token_bytes = align(hidden * 2, 512)
    minimum_bytes = 2 * (
        num_tokens * dispatch_token_bytes * world_size * local_experts
        + num_tokens * combine_token_bytes * (topk + shared_experts)
    ) + 1024 * 1024
    return align(align(minimum_bytes, 1024 * 1024) // (1024 * 1024), 2) // 2



def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# ========== Golden functions (CPU simulation of collective ops) ==========

def _golden_mm_all_reduce(cpu_inputs_per_rank):
    """MatmulAllReduce: each rank local matmul(+bias), then all_reduce(sum).

    Mirrors mc2_test get_cpu (op_class/aclnnMatmulAllReduce.py:105-113):
    fp32 matmul + bias, then all_reduce SUM. All ranks share weight, so
    golden = sum of all ranks' local matmul.
    """
    import torch
    world_size = len(cpu_inputs_per_rank)
    local_results = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0].float()
        x2 = cpu_inputs_per_rank[r][1].float()
        mm_out = torch.matmul(x1, x2)
        bias = cpu_inputs_per_rank[r][2] if len(cpu_inputs_per_rank[r]) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() > 0:
            mm_out = mm_out + bias.float()
        local_results.append(mm_out)
    total = torch.zeros_like(local_results[0])
    for lr in local_results:
        total = total + lr
    del local_results
    return [total.contiguous() for _ in range(world_size)]


def _golden_all_gather_mm(cpu_inputs_per_rank):
    world_size = len(cpu_inputs_per_rank)
    all_x1 = [inp[0].float() for inp in cpu_inputs_per_rank]
    gathered = torch.cat(all_x1, dim=0)
    del all_x1
    goldens = []
    for r in range(world_size):
        x2 = cpu_inputs_per_rank[r][1].float()
        g = torch.matmul(gathered, x2)
        goldens.append(g.contiguous())
    del gathered
    return goldens


def _golden_reduce_scatter_mm(cpu_inputs_per_rank, attrs=None):
    import torch as _torch
    world_size = len(cpu_inputs_per_rank)
    # Match ACLNN golden (mc2_golden.py:208-240): CPU fp32 matmul -> bf16 truncation -> CPU bf16 accumulate
    orig_dtype = _torch.float32
    if attrs and '_tensor_dtypes' in attrs:
        dt_str = str(attrs['_tensor_dtypes'][0]) if attrs['_tensor_dtypes'] else 'float32'
        if dt_str == 'bfloat16':
            orig_dtype = _torch.bfloat16
        elif dt_str == 'float16':
            orig_dtype = _torch.float16
    local_results = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0]
        x2 = cpu_inputs_per_rank[r][1]
        x1_f = x1.float()
        x2_f = x2.float()
        mm_out = _torch.matmul(x1_f, x2_f)
        mm_out = mm_out.to(orig_dtype).float()
        local_results.append(mm_out.to(orig_dtype))
    total = _torch.zeros_like(local_results[0])
    for lr in local_results:
        total = total + lr
    total = total.float()
    del local_results
    M = total.shape[0]
    chunk_m = M // world_size
    return [total[r * chunk_m:(r + 1) * chunk_m, :].contiguous() for r in range(world_size)]


def _golden_matmul_all_to_all(cpu_inputs_per_rank, attrs):
    """Mirror ACLNN __golden_matmul_allto_all exactly.

    For each target rank, iterate over all source ranks, compute local matmul,
    chunk along N (permute(1,0,2) + chunk dim=0), and pick the target's chunk.
    Finally cat + reshape(-1, chunk_n).
    """
    world_size = len(cpu_inputs_per_rank)
    t_x1 = bool(attrs.get('transposeX1', False))
    t_x2 = bool(attrs.get('transposeX2', False))

    # Precompute each rank's mm_out (float32) with bias if present
    all_mm_out = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0]
        x2 = cpu_inputs_per_rank[r][1]
        bias = cpu_inputs_per_rank[r][2] if len(cpu_inputs_per_rank[r]) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        input_mat = x1.t().contiguous() if t_x1 else x1
        weight_mat = x2.t().contiguous() if t_x2 else x2
        mm_out = torch.matmul(input_mat.float(), weight_mat.float())
        if bias is not None:
            mm_out = mm_out + bias.float()
        all_mm_out.append(mm_out)

    M = all_mm_out[0].shape[0]
    N = all_mm_out[0].shape[1]
    chunk_n = N // world_size

    goldens = []
    for target_r in range(world_size):
        all_to_all_results = []
        for src_r in range(world_size):
            s_mm = all_mm_out[src_r]
            s_chunks = s_mm.view(M, world_size, chunk_n).permute(1, 0, 2).contiguous()
            s_chunks = s_chunks.view(world_size, M * chunk_n)
            send_chunks = s_chunks.chunk(world_size, dim=0)
            all_to_all_results.append(send_chunks[target_r].clone())
            del s_chunks, send_chunks
        received = torch.cat(all_to_all_results, dim=0)
        received = received.reshape(-1, chunk_n).contiguous()
        goldens.append(received)
        del all_to_all_results
    del all_mm_out
    return goldens


def _golden_all_to_all_matmul(cpu_inputs_per_rank, attrs):
    """Mirror ACLNN __golden_allto_all_matmul exactly.

    For each target rank: collect each source rank's x1 chunk[target_idx],
    stack + permute(1,0,2) + reshape(M_chunk, ws*K), then matmul with own weight.
    """
    world_size = len(cpu_inputs_per_rank)
    t_x1 = bool(attrs.get('transposeX1', False))
    t_x2 = bool(attrs.get('transposeX2', False))

    # Precompute each rank's input_mat and weight_mat (float32)
    input_mats = []
    weight_mats = []
    biases = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0]
        x2 = cpu_inputs_per_rank[r][1]
        bias = cpu_inputs_per_rank[r][2] if len(cpu_inputs_per_rank[r]) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        input_mats.append((x1.t().contiguous() if t_x1 else x1).float())
        weight_mats.append((x2.t().contiguous() if t_x2 else x2).float())
        biases.append(bias.float() if bias is not None else None)

    M_total = input_mats[0].shape[0]
    K = input_mats[0].shape[1]
    M_chunk = M_total // world_size

    goldens = []
    for target_r in range(world_size):
        recv_chunks = []
        for src_r in range(world_size):
            s_input = input_mats[src_r]
            s_reshaped = s_input.view(world_size, M_chunk, K)
            recv_chunks.append(s_reshaped[target_r])

        recv_tensor = torch.stack(recv_chunks, dim=0)
        a2a_out = recv_tensor.permute(1, 0, 2).reshape(M_chunk, world_size * K).contiguous()

        weight_mat = weight_mats[target_r]
        mm_out = torch.matmul(a2a_out, weight_mat)
        bias = biases[target_r]
        if bias is not None:
            mm_out = mm_out + bias
        goldens.append(mm_out)

    del input_mats, weight_mats, biases
    return goldens


def _generate_gmm_matrix(A_array, m, seed=1):
    """Replicate ACLNN __generate_gmm_alltoallv_matrix exactly."""
    n = len(A_array)
    rng = np.random.default_rng(seed)
    total = sum(A_array)
    if total % n != 0:
        return [[total // n] * (m * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in A_array:
        if a % n != 0:
            return [[col_sum // m] * (m * n) for _ in range(n)]
        k = a // n
        k_values.append(max(k, m))
    blocks = []
    for k in k_values:
        block = np.zeros((m, n), dtype=int)
        for col in range(n):
            counts = rng.multinomial(k - m, [1.0 / m] * m)
            block[:, col] = counts + 1
        blocks.append(block)
    tmp = np.vstack(blocks)
    return [list(col) for col in zip(*tmp)]


def _golden_a2a_ag_bmm(cpu_inputs_per_rank, attrs, rank, ws, dist_avail=True):
    x_cpu = cpu_inputs_per_rank[rank][0].float()
    w_cpu = cpu_inputs_per_rank[rank][1].float()
    shard_type = int(attrs.get('shard_type', attrs.get('xShardType', 0)))
    ep_ws = int(attrs.get('epWorldSize', 0))
    tp_ws = int(attrs.get('tpWorldSize', 0))
    act_type_val = attrs.get('actType', 0)
    if isinstance(act_type_val, str):
        act_map = {'none': 0, 'gelu': 1, 'silu': 2, 'relu': 3, 'fastgelu': 4}
        act_type = act_map.get(act_type_val.lower(), 0)
    else:
        act_type = int(act_type_val)
    is_bias = bool(attrs.get('isBias', False))
    need_ag_out = bool(attrs.get('needAllgatherOut', attrs.get('need_allgather_out', False)))
    need_act_feat = bool(attrs.get('needActivationFeature', attrs.get('need_activation_feature', False)))
    if not ep_ws or not tp_ws:
        return None

    def _apply_act(x, act):
        if act == 0:
            return x
        elif act == 1:
            return torch.nn.functional.gelu(x, approximate="tanh")
        elif act == 2:
            return x * torch.sigmoid(x)
        elif act == 3:
            return torch.nn.functional.relu(x)
        elif act == 4:
            return x / (1.0 + torch.exp(-1.702 * x))
        return x

    E_local = x_cpu.shape[0]
    E_div_ep = E_local // ep_ws

    if shard_type == 0:
        C = x_cpu.shape[1]
        H_div_tp = x_cpu.shape[2]
    else:
        C_div_tp = x_cpu.shape[1]
        H = x_cpu.shape[2]

    n_ep_groups = ws // ep_ws
    n_tp_groups = ws // tp_ws
    ep_groups = [list(range(g * ep_ws, (g + 1) * ep_ws)) for g in range(n_ep_groups)]
    tp_groups = [[g + e * ep_ws for e in range(tp_ws)] for g in range(n_tp_groups)]

    all_x_cache = {}
    for r in range(ws):
        all_x_cache[r] = cpu_inputs_per_rank[r][0].float()

    a2a_per_rank = {}
    for group_ranks in ep_groups:
        chunks_per_rank = {}
        for local_idx, r in enumerate(group_ranks):
            chunks_per_rank[local_idx] = all_x_cache[r].chunk(ep_ws, dim=0)
        for target_local, target_rank in enumerate(group_ranks):
            result_chunks = [chunks_per_rank[src_local][target_local] for src_local in range(len(group_ranks))]
            a2a_out = torch.cat(result_chunks, dim=0)
            if shard_type == 0:
                a2a_out = a2a_out.reshape(ep_ws, E_div_ep, C, H_div_tp).permute(1, 0, 2, 3).contiguous()
            else:
                a2a_out = a2a_out.reshape(ep_ws, E_div_ep, C_div_tp, H).permute(1, 0, 2, 3).contiguous()
            a2a_per_rank[target_rank] = a2a_out

    for tp_group_ranks in tp_groups:
        all_parts = [a2a_per_rank[r] for r in tp_group_ranks]
        gathered = torch.cat(all_parts, dim=0)
        del all_parts
        if shard_type == 0:
            gathered = gathered.reshape(tp_ws, E_div_ep, ep_ws, C, H_div_tp)
            gathered = gathered.permute(1, 2, 3, 0, 4).contiguous()
            gathered = gathered.reshape(E_div_ep, ep_ws * C, H_div_tp * tp_ws)
        else:
            gathered = gathered.reshape(tp_ws, E_div_ep, ep_ws, C_div_tp, H)
            gathered = gathered.permute(1, 2, 0, 3, 4).contiguous()
            gathered = gathered.reshape(E_div_ep, ep_ws * tp_ws * C_div_tp, H)

        for r in tp_group_ranks:
            if r == rank:
                w_r = cpu_inputs_per_rank[r][1].float()
                bmm_out = torch.bmm(gathered, w_r)
                del w_r
                if is_bias and len(cpu_inputs_per_rank[r]) > 2:
                    bias_r = cpu_inputs_per_rank[r][2].float()
                    if bias_r.numel() > 0:
                        if bias_r.dim() == 2:
                            bias_r = bias_r.reshape(bias_r.shape[0], 1, bias_r.shape[1])
                        bmm_out = bmm_out + bias_r
                    del bias_r
                activated = _apply_act(bmm_out, act_type)
                in_dtype = cpu_inputs_per_rank[r][0].dtype
                goldens = {'main': activated.to(in_dtype).float()}
                if need_ag_out:
                    goldens['allgather'] = gathered.to(in_dtype).float()
                if need_act_feat:
                    goldens['bmm'] = bmm_out.to(in_dtype).float()
                return goldens
        del gathered
    return None


def _golden_bmm_rs_a2a(cpu_inputs_per_rank, attrs, rank, ws, dist_avail=True):
    """Pure CPU golden for npu_bmm_reducescatter_alltoall (matching ACLNN
    __golden_bmm_reduce_scatter_allto_all exactly)."""
    shard_type = int(attrs.get('shard_type', attrs.get('yShardType', 0)))
    ep_ws = int(attrs.get('epWorldSize', 0))
    tp_ws = int(attrs.get('tpWorldSize', 0))
    if not ep_ws or not tp_ws:
        return None

    # in_dtype must be the NPU compute dtype (bf16/fp16), not the CPU float32 copy.
    # cpu_inputs_per_rank tensors are already .float() in worker; recover original dtype.
    in_dtype = torch.float32
    if attrs and '_tensor_dtypes' in attrs:
        dt_str = str(attrs['_tensor_dtypes'][0]) if attrs['_tensor_dtypes'] else 'float32'
        if 'bfloat16' in dt_str or 'bf16' in dt_str:
            in_dtype = torch.bfloat16
        elif 'float16' in dt_str or 'fp16' in dt_str:
            in_dtype = torch.float16
    E_div_ep = cpu_inputs_per_rank[0][0].shape[0]
    x_dim1 = cpu_inputs_per_rank[0][0].shape[1]
    H = cpu_inputs_per_rank[0][1].shape[2]
    is_bias = bool(attrs.get('isBias', False))

    if shard_type == 0:
        C = x_dim1 // ep_ws
    else:
        C_div_tp = x_dim1 // ep_ws // tp_ws

    # Step 1: reduce_scatter across TP groups (compute rs for all ranks, matching ACLNN)
    rs_per_rank = {}
    n_tp_groups = ws // tp_ws
    for g in range(n_tp_groups):
        group_ranks = [g + e * ep_ws for e in range(tp_ws)]
        all_parts = []
        for r in group_ranks:
            x = cpu_inputs_per_rank[r][0].float()
            weight = cpu_inputs_per_rank[r][1].float()
            bmm_out = torch.bmm(x, weight)
            bmm_out = bmm_out.to(in_dtype).float()
            del weight
            if shard_type == 0:
                r1 = bmm_out.reshape(E_div_ep, ep_ws * C, tp_ws, H // tp_ws)
                r1 = r1.permute(2, 0, 1, 3).contiguous()
                r1 = r1.reshape(tp_ws * E_div_ep, ep_ws * C, H // tp_ws)
            else:
                r1 = bmm_out.reshape(E_div_ep, ep_ws, tp_ws, C_div_tp, H)
                r1 = r1.permute(2, 0, 1, 3, 4).contiguous()
                r1 = r1.reshape(tp_ws * E_div_ep, ep_ws * C_div_tp, H)
            all_parts.append(r1.to(in_dtype))
            del bmm_out
        n_tp = len(group_ranks)
        for local_idx, r in enumerate(group_ranks):
            start = (local_idx + 1) % n_tp
            acc = all_parts[start][local_idx * E_div_ep:(local_idx + 1) * E_div_ep].clone().float()
            for step in range(1, n_tp):
                src_idx = (start + step) % n_tp
                src_chunk = all_parts[src_idx][local_idx * E_div_ep:(local_idx + 1) * E_div_ep]
                acc = acc + src_chunk.float()
            chunk = acc.to(in_dtype)
            if is_bias and len(cpu_inputs_per_rank[r]) > 2:
                bias = cpu_inputs_per_rank[r][2].float()
                if bias.numel() > 0:
                    if bias.dim() == 2:
                        bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
                    chunk = chunk.to(in_dtype).float() + bias
                    chunk = chunk.to(in_dtype)
                del bias
            rs_per_rank[r] = chunk
        del all_parts

    # Step 2: all_to_all across EP groups (only compute for own rank)
    n_ep_groups = ws // ep_ws
    for g in range(n_ep_groups):
        group_ranks = list(range(g * ep_ws, (g + 1) * ep_ws))
        for target_local, target_rank in enumerate(group_ranks):
            if target_rank != rank:
                continue
            if shard_type == 0:
                all_chunks = []
                for src_local, src_rank in enumerate(group_ranks):
                    rs = rs_per_rank[src_rank]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C, H // tp_ws)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C, H // tp_ws)
            else:
                all_chunks = []
                for src_local, src_rank in enumerate(group_ranks):
                    rs = rs_per_rank[src_rank]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C_div_tp, H)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C_div_tp, H)
            del gathered, all_chunks
            return out
    return None


def _golden_gmm_alltoallv(cpu_inputs_per_rank, attrs, rank, ws, dist_avail=True):
    """Pure CPU golden for npu_gmm_alltoallv (matching ACLNN __golden_gmm_alltoallv).

    Flow: gmm(x, weight) per rank -> unpermute -> all_to_all (CPU simulated).
    """
    import numpy as _np
    epc = int(attrs.get('expPerCard', 1))
    seed = int(attrs.get('seed', 1))
    ep_ws = int(attrs.get('ep_ws', ws))
    expTokenNums = attrs.get('expTokenNums')
    if expTokenNums is None:
        gmm_x_cpu = cpu_inputs_per_rank[0][0].float()
        A = gmm_x_cpu.shape[0]
        A_array = [A] * ep_ws
        expTokenNums = _generate_gmm_matrix(A_array, epc, seed=seed)
    trans_gmm = bool(attrs.get('transGmmWeight', attrs.get('trans_gmm_weight', False)))

    # Step 1: per-rank grouped matmul (CPU fp32)
    all_gmm_out = {}
    all_unpermuted = {}
    for r in range(ws):
        gmm_x = cpu_inputs_per_rank[r][0].float()
        gmm_weight = cpu_inputs_per_rank[r][1].float()
        if trans_gmm:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
        # recv group list for rank r
        recv_gl = []
        for j in range(epc):
            total = sum(expTokenNums[i][r * epc + j] for i in range(ep_ws))
            recv_gl.append(total)
        B_list = list(torch.unbind(gmm_weight, dim=0))
        A_groups = torch.split(gmm_x, recv_gl, dim=0)
        gmm_results = []
        for i in range(len(recv_gl)):
            a = A_groups[i].numpy()
            b = B_list[i].numpy()
            gmm_results.append(torch.from_numpy(np.matmul(a, b)))
        gmm_out = torch.cat(gmm_results, dim=0).float()
        all_gmm_out[r] = gmm_out
        # unpermute (CPU, matching __unpermute_mc2)
        empty_arr = _np.zeros((ep_ws, epc), dtype=_np.int64)
        for i in range(ep_ws):
            for j in range(epc):
                empty_arr[i][j] = int(expTokenNums[i][r * epc + j])
        tmp1 = empty_arr.T
        sum_list1 = _np.sum(tmp1, axis=1)
        sum_list2 = _np.cumsum(sum_list1, axis=0)
        offsets = [0] + [int(s) for s in sum_list2[:-1]]
        sum_list = _np.cumsum(tmp1, axis=1)
        indices_list = []
        for i in range(epc):
            tmp = []
            for j in range(ep_ws):
                if j == 0:
                    tmp.append(list(range(offsets[i], offsets[i] + int(sum_list[i][j]))))
                else:
                    tmp.append(list(range(offsets[i] + int(sum_list[i][j-1]),
                                          offsets[i] + int(sum_list[i][j]))))
            indices_list.append(tmp)
        selected = []
        for i in range(ep_ws):
            for j in range(epc):
                indices = torch.tensor(indices_list[j][i], dtype=torch.long)
                selected.append(gmm_out[indices])
        all_unpermuted[r] = torch.cat(selected, dim=0).float()

    # Step 2: CPU-simulated all_to_all (matching __simulate_alltoallv)
    N = all_unpermuted[0].shape[1] if all_unpermuted[0].dim() > 1 else 1
    output_chunks = []
    for src_r in range(ws):
        src_data = all_unpermuted[src_r]
        # input_splits for src_r
        input_splits = [sum(expTokenNums[src_r][t * epc:(t + 1) * epc]) for t in range(ep_ws)]
        offset = 0
        for t in range(ep_ws):
            if t == rank:
                chunk = src_data[offset:offset + input_splits[t]].clone()
                output_chunks.append(chunk)
            offset += input_splits[t]
    main_golden = torch.cat(output_chunks, dim=0) if output_chunks else torch.zeros(0, N)

    # mm_out (optional) - use this rank's own mm_x/mm_weight (matching ACLNN)
    mm_golden = None
    if len(cpu_inputs_per_rank[rank]) > 2 and cpu_inputs_per_rank[rank][2] is not None:
        mm_x = cpu_inputs_per_rank[rank][2].float()
        mm_weight = cpu_inputs_per_rank[rank][3].float()
        trans_mm = bool(attrs.get('transMmWeight', attrs.get('trans_mm_weight', False)))
        if trans_mm:
            mm_weight = mm_weight.t().contiguous()
        mm_golden = torch.mm(mm_x, mm_weight)

    del all_gmm_out, all_unpermuted
    if mm_golden is not None:
        return (main_golden, mm_golden)
    return main_golden


def _golden_alltoallv_gmm(cpu_inputs_per_rank, attrs, rank, ws, dist_avail=True):
    """Pure CPU golden for npu_alltoallv_gmm (matching ACLNN __golden_alltoallv_gmm).

    Flow: all_to_all(x) -> permute -> grouped_matmul.
    """
    import numpy as _np
    epc = int(attrs.get('expPerCard', 1))
    seed = int(attrs.get('seed', 1))
    ep_ws = int(attrs.get('ep_ws', ws))
    expTokenNums = attrs.get('expTokenNums')
    if expTokenNums is None:
        gmm_x_cpu = cpu_inputs_per_rank[0][0].float()
        A = gmm_x_cpu.shape[0]
        A_array = [A] * ep_ws
        expTokenNums = _generate_gmm_matrix(A_array, epc, seed=seed)
    trans_gmm = bool(attrs.get('transGmmWeight', attrs.get('trans_gmm_weight', False)))
    trans_mm = bool(attrs.get('transMmWeight', attrs.get('trans_mm_weight', False)))
    permute_out_flag = bool(attrs.get('permuteOutFlag', attrs.get('permute_out_flag', False)))

    # Step 1: all_to_all (CPU simulated, matching ACLNN __golden_alltoallv_gmm)
    all_a2a_inputs = {}
    all_send_segments = {}
    for r in range(ws):
        src_x = cpu_inputs_per_rank[r][0].float()
        all_a2a_inputs[r] = src_x
        my_row = expTokenNums[r]
        segments = []
        offset = 0
        for t in range(ep_ws):
            cs = sum(my_row[t * epc:(t + 1) * epc])
            segments.append(src_x[offset:offset + cs])
            offset += cs
        all_send_segments[r] = segments

    # a2a output for this rank
    output_splits = [sum(expTokenNums[i][rank * epc:(rank + 1) * epc]) for i in range(ep_ws)]
    recv_offsets = [0] + list(_np.cumsum(output_splits)[:-1])
    K = all_a2a_inputs[0].shape[1] if all_a2a_inputs[0].dim() > 1 else 1
    gathered = torch.zeros(sum(output_splits), K, dtype=torch.float32)
    for src_r in range(ws):
        chunk = all_send_segments[src_r][rank]
        gathered[recv_offsets[src_r]:recv_offsets[src_r] + chunk.shape[0]] = chunk

    # Step 2: permute (CPU, matching __permute_a2a_gmm)
    indices = torch.zeros(epc, ep_ws, dtype=torch.long)
    for j in range(epc):
        for i in range(ep_ws):
            indices[j][i] = expTokenNums[i][j + epc * rank]
    trans = indices.permute(1, 0).reshape(-1)
    cum = torch.cumsum(trans, dim=0)
    tmp = []
    for i in range(len(cum)):
        if i == 0:
            tmp.append(range(0, int(cum[i].item())))
        else:
            tmp.append(range(int(cum[i-1].item()), int(cum[i].item())))
    parts = []
    expert_sizes = []
    for e in range(epc):
        exp_token = []
        for r in range(ep_ws):
            exp_token += list(tmp[e + r * epc])
        combined = torch.tensor(exp_token, dtype=torch.long)
        parts.append(gathered.index_select(0, combined))
        expert_sizes.append(len(exp_token))
    permuted = torch.zeros(sum(expert_sizes), K, dtype=torch.float32)
    offset = 0
    for e in range(epc):
        permuted[offset:offset + expert_sizes[e]] = parts[e]
        offset += expert_sizes[e]

    # Step 3: grouped matmul (CPU, matching __grouped_matmul_cpu)
    gmm_weight = cpu_inputs_per_rank[rank][1].float()
    if trans_gmm:
        gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
    B_list = list(torch.unbind(gmm_weight, dim=0))
    A_groups = torch.split(permuted, expert_sizes, dim=0)
    gmm_results = []
    for i in range(len(expert_sizes)):
        a = A_groups[i].numpy()
        b = B_list[i].numpy()
        gmm_results.append(torch.from_numpy(np.matmul(a, b)))
    main_golden = torch.cat(gmm_results, dim=0).float()

    permute_ret = permuted.contiguous() if permute_out_flag else None

    # mm_out (optional) - use this rank's own mm_x/mm_weight (matching ACLNN)
    mm_golden = None
    if len(cpu_inputs_per_rank[rank]) > 2 and cpu_inputs_per_rank[rank][2] is not None:
        mm_x = cpu_inputs_per_rank[rank][2].float()
        mm_weight = cpu_inputs_per_rank[rank][3].float()
        if trans_mm:
            mm_weight = mm_weight.t().contiguous()
        mm_golden = torch.mm(mm_x, mm_weight)

    del all_a2a_inputs, all_send_segments
    return main_golden, mm_golden, permute_ret



def _compute_golden(api_name, cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
    if 'npu_all_gather_base_mm' in api_name:
        return _golden_all_gather_mm(cpu_inputs_per_rank)
    elif 'npu_mm_reduce_scatter_base' in api_name:
        return _golden_reduce_scatter_mm(cpu_inputs_per_rank, attrs)
    elif 'npu_mm_all_reduce_base' in api_name:
        return _golden_mm_all_reduce(cpu_inputs_per_rank)
    elif 'npu_matmul_all_to_all' in api_name:
        return _golden_matmul_all_to_all(cpu_inputs_per_rank, attrs)
    elif 'npu_all_to_all_matmul' in api_name:
        return _golden_all_to_all_matmul(cpu_inputs_per_rank, attrs)
    elif 'npu_gmm_alltoallv' in api_name:
        golden = _golden_gmm_alltoallv(cpu_inputs_per_rank, attrs, rank, ws, dist_avail)
        return [golden]
    elif 'npu_alltoallv_gmm' in api_name:
        gmm_out, mm_out, permute_out = _golden_alltoallv_gmm(
            cpu_inputs_per_rank, attrs, rank, ws, dist_avail)
        return [gmm_out, mm_out, permute_out]
    elif 'bmm_reducescatter_alltoall' in api_name:
        golden = _golden_bmm_rs_a2a(cpu_inputs_per_rank, attrs, rank, ws, dist_avail)
        return [golden]
    elif 'alltoall_allgather_bmm' in api_name:
        golden = _golden_a2a_ag_bmm(cpu_inputs_per_rank, attrs, rank, ws, dist_avail)
        return [golden]
    return [None] * len(cpu_inputs_per_rank)


def _compare(npu_out_np, golden_np, rtol=0.01, atol=1.0):
    diff = np.abs(npu_out_np - golden_np)
    max_diff = float(np.max(diff))
    close = np.sum(diff < (atol + rtol * np.abs(golden_np)))
    pct = float(close / diff.size * 100)
    return max_diff, pct


def _call_api(api_name, dev_tensors, hcomm, world_size, attrs):
    kw = dict(attrs)

    if _uses_private_moe_backend(api_name) and 'elastic_buffer_moe_ep_chain' in api_name:
        from cann_ops_transformer.ops.elastic_buffer import ElasticBuffer
        num_experts = int(kw.get('num_experts', 4))
        num_max_tokens = int(kw.get('num_max_tokens_per_rank', dev_tensors[0].shape[0]))
        topk = int(dev_tensors[1].shape[1])
        hidden = int(dev_tensors[0].shape[1])
        topk_ids = torch.arange(
            dev_tensors[1].numel(), device=dev_tensors[1].device, dtype=torch.int32
        ).reshape(dev_tensors[1].shape) % num_experts
        elastic = ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=num_max_tokens,
            hidden=hidden,
            num_topk=topk,
        )
        try:
            recv_x, _, _, handle = elastic.dispatch(
                dev_tensors[0], topk_idx=topk_ids, num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens, expert_alignment=1,
                do_cpu_sync=False,
            )
            combined_x, _ = elastic.combine(recv_x, handle)
            return combined_x
        finally:
            elastic.destroy()

    if _uses_private_moe_backend(api_name) and 'mega_moe' in api_name:
        context = kw.pop('_moe_context')
        ccl_buffer_size = int(kw.pop('_moe_ccl_buffer_size'))
        topo_type = int(kw.pop('_moe_topo_type'))
        rank_num_per_server = int(kw.pop('_moe_rank_num_per_server'))
        topk_ids = torch.arange(
            dev_tensors[1].numel(), device=dev_tensors[1].device, dtype=torch.int32
        ).reshape(dev_tensors[1].shape) % int(kw.get('moe_expert_num'))
        args = (
            context, dev_tensors[0], topk_ids, dev_tensors[2],
            [dev_tensors[3]], [dev_tensors[4]],
            int(kw.get('moe_expert_num')), int(kw.get('ep_world_size', world_size)),
            ccl_buffer_size, [dev_tensors[6]], [dev_tensors[7]], None, None, None,
            None, None, None, None, None, None,
            int(kw.get('max_recv_token_num',
                       kw.get('num_max_tokens_per_rank', dev_tensors[0].shape[0])
                       * int(kw.get('ep_world_size', world_size)))),
            int(kw.get('dispatch_quant_mode', 4)),
            int(kw.get('combine_quant_mode', 0)), str(kw.get('comm_alg', '')),
            int(kw.get('num_max_tokens_per_rank', dev_tensors[0].shape[0])),
            'swiglu', [], int(kw.get('dispatch_quant_out_dtype', 23)),
            None, None, topo_type, rank_num_per_server, 0)
        try:
            return _mega_moe_module.npu_mega_moe(*args)
        except TypeError:
            # The older extension omits shared-expert tensors and the final
            # topkWeightsType argument from its Python binding.
            return _mega_moe_module.npu_mega_moe(
                context, dev_tensors[0], topk_ids, dev_tensors[2],
                [dev_tensors[3]], [dev_tensors[4]],
                int(kw.get('moe_expert_num')), int(kw.get('ep_world_size', world_size)),
                ccl_buffer_size, [dev_tensors[6]], [dev_tensors[7]], None, None, None,
                int(kw.get('max_recv_token_num',
                           kw.get('num_max_tokens_per_rank', dev_tensors[0].shape[0])
                           * int(kw.get('ep_world_size', world_size)))),
                int(kw.get('dispatch_quant_mode', 4)),
                int(kw.get('combine_quant_mode', 0)), str(kw.get('comm_alg', '')),
                int(kw.get('num_max_tokens_per_rank', dev_tensors[0].shape[0])),
                'swiglu', None, int(kw.get('dispatch_quant_out_dtype', 23)),
                topo_type, rank_num_per_server)

    if _uses_private_moe_backend(api_name) and 'npu_moe_distribute_dispatch' in api_name:
        context = kw.pop('_moe_context')
        ccl_buffer_size = int(kw.pop('_moe_ccl_buffer_size'))
        ep_rank_id = int(kw.pop('_moe_rank'))
        ep_world_size = int(kw.pop('ep_world_size', world_size))
        moe_expert_num = int(kw.pop('moe_expert_num'))
        return _moe_dispatch_module.npu_moe_distribute_dispatch(
            context, dev_tensors[1], dev_tensors[2], ep_world_size,
            ep_rank_id, moe_expert_num, ccl_buffer_size,
            dev_tensors[3], dev_tensors[4], dev_tensors[5], dev_tensors[6], dev_tensors[7],
            int(kw.get('tp_world_size', 0)), int(kw.get('tp_rank_id', 0)),
            int(kw.get('expert_shard_type', 0)), int(kw.get('shared_expert_num', 1)),
            int(kw.get('shared_expert_rank_num', 0)), int(kw.get('quant_mode', 0)),
            int(kw.get('global_bs', 0)), int(kw.get('expert_token_nums_type', 1)),
            str(kw.get('comm_alg', '')), int(kw.get('zero_expert_num', 0)),
            int(kw.get('copy_expert_num', 0)), int(kw.get('const_expert_num', 0)),
            kw.get('y_dtype'), kw.get('x_dtype'), kw.get('scales_dtype'))
    if _uses_private_moe_backend(api_name) and 'npu_moe_distribute_combine' in api_name:
        context = kw.pop('_moe_context')
        ccl_buffer_size = int(kw.pop('_moe_ccl_buffer_size'))
        ep_rank_id = int(kw.pop('_moe_rank'))
        ep_world_size = int(kw.pop('ep_world_size', world_size))
        moe_expert_num = int(kw.pop('moe_expert_num'))
        expert_ids = dev_tensors[2]
        expert_scales = dev_tensors[5]
        dispatch_x = dev_tensors[1][:expert_ids.shape[0]].contiguous()
        dispatch = _moe_dispatch_module.npu_moe_distribute_dispatch(
            context, dispatch_x, expert_ids, ep_world_size, ep_rank_id,
            moe_expert_num, ccl_buffer_size, None, None, expert_scales, None, None,
            0, 0, int(kw.get('expert_shard_type', 0)), 0, 0, 0,
            int(kw.get('global_bs', 0)), int(kw.pop('expert_token_nums_type', 0)),
            str(kw.get('comm_alg', '')), 0, 0, 0, None, None, None)
        expand_x, _, assist_info, _, ep_send_counts, tp_send_counts, expand_scales = dispatch
        return _moe_combine_module.npu_moe_distribute_combine(
            context, expand_x, expert_ids, assist_info, ep_send_counts,
            expert_scales, ep_world_size, ep_rank_id, moe_expert_num,
            ccl_buffer_size, tp_send_counts, dev_tensors[7], expand_scales,
            None, None, None, None, None, None, None,
            int(kw.get('tp_world_size', 0)), int(kw.get('tp_rank_id', 0)),
            int(kw.get('expert_shard_type', 0)), int(kw.get('shared_expert_num', 1)),
            int(kw.get('shared_expert_rank_num', 0)), int(kw.get('global_bs', 0)),
            int(kw.get('comm_quant_mode', 0)), str(kw.get('comm_alg', '')),
            int(kw.get('zero_expert_num', 0)), int(kw.get('copy_expert_num', 0)),
            int(kw.get('const_expert_num', 0)))

    if api_name in _TORCH_NPU_MOE_TENSOR_PARAMS:
        return _call_torch_npu_moe(api_name, dev_tensors, hcomm, world_size, attrs)
    if api_name in ("torch_npu.npu_attention_to_ffn", "torch_npu.npu_ffn_to_attention"):
        return _call_torch_npu_token_exchange(
            api_name, dev_tensors, hcomm, world_size, attrs)
    if api_name == "torch_npu._npu_distribute_barrier":
        resolved = _resolve_api(api_name)
        return resolved(dev_tensors[0], hcomm, world_size)
    if api_name == "torch_npu.npu_moe_update_expert":
        resolved = _resolve_api(api_name)
        kw = dict(attrs)
        kw.pop("_tensor_dtypes", None)
        if kw.get("local_rank_id", -1) < 0:
            kw["local_rank_id"] = int(kw.pop("_rank", 0))
        return resolved(*dev_tensors, **kw)

    resolved = _resolve_api(api_name)

    if 'npu_quant_gmm_alltoallv' in api_name or 'npu_alltoallv_quant_gmm' in api_name:
        send_counts = kw.pop('sendCounts', kw.pop('send_counts', []))
        recv_counts = kw.pop('recvCounts', kw.pop('recv_counts', []))
        ep_ws = int(kw.pop('ep_world_size', kw.pop('epWorldSize', world_size)))
        gmm_y_dtype = int(kw.pop('gmm_y_dtype', kw.pop('gmmYType', 1)))
        gmm_x_quant_mode = kw.pop('gmm_x_quant_mode', None)
        gmm_weight_quant_mode = kw.pop('gmm_weight_quant_mode', None)
        # ACLNN exposes comm_mode, but the torch_npu quantized E2E schema does
        # not.  Keep the attribute in ACLNN cases without forwarding it here.
        kw.pop('comm_mode', None)
        kw.pop('epWorldSize', None)
        args = (dev_tensors[0], dev_tensors[1], dev_tensors[2], dev_tensors[3],
                hcomm, ep_ws, send_counts, recv_counts, gmm_y_dtype)
        if 'npu_alltoallv_quant_gmm' in api_name:
            return resolved(*args,
                            gmm_x_quant_mode=gmm_x_quant_mode,
                            gmm_weight_quant_mode=gmm_weight_quant_mode)
        return resolved(*args,
                        gmm_x_quant_mode=gmm_x_quant_mode,
                        gmm_weight_quant_mode=gmm_weight_quant_mode)

    # Remove ACLNN-only params that don't exist in torch_npu API
    for k in ['transposeX1', 'transposeX2', 'group', 'is_trans_b', 'network_name',
              'graph_type', 'seed', 'isBias', 'weight_same',
              'hcom', 'world_size', 'expPerCard', 'commTurn',
              'streamMode', 'gather_index', 'mm_out_flag',
              'trans_gmm_weight', 'trans_mm_weight', 'reduceOp',
              'ep_world_size', 'is_trans', 'is_bias',
              'groupEp', 'groupTp', '_tensor_dtypes',
              'alltoAllAxesOptional', 'all2all_axes_optional',
              'isTrans', 'is_transX', 'is_transX2', 'is_transY',
              'expTokenNums', 'ep_ws']:
        kw.pop(k, None)

    # Convert string bool values to actual bool
    for k in ['gather_output']:
        if k in kw and isinstance(kw[k], str):
            kw[k] = kw[k].lower() in ('true', '1', 'yes')

    if 'npu_all_gather_base_mm' in api_name:
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_all_gather_quant_mm' in api_name:
        kw.pop('comm_mode', None)
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_mm_reduce_scatter_base' in api_name:
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_quant_mm_reduce_scatter' in api_name:
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_mm_all_reduce_add_rms_norm' in api_name:
        kw.pop('world_size', None)
        reduce_op = kw.pop('reduce_op', 'sum')
        epsilon = kw.pop('epsilon', 1e-6)
        return resolved(dev_tensors[0], dev_tensors[1], dev_tensors[2], dev_tensors[3],
                        hcomm, reduce_op=reduce_op, epsilon=epsilon, **kw)
    elif 'npu_mm_all_reduce_base' in api_name:
        bias = dev_tensors[2] if len(dev_tensors) > 2 and dev_tensors[2] is not None else None
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, bias=bias)
    elif 'npu_matmul_all_to_all' in api_name:
        # Match mc2_test: explicitly pass bias=None and all2all_axes=None
        kw.setdefault('bias', None)
        kw.setdefault('all2all_axes', None)
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_all_to_all_matmul' in api_name:
        kw.setdefault('bias', None)
        kw.setdefault('all2all_axes', None)
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_all_to_all_quant_matmul' in api_name:
        return resolved(dev_tensors[0], dev_tensors[1], hcomm, world_size, **kw)
    elif 'npu_gmm_alltoallv' in api_name:
        send_counts = kw.pop('sendCounts', kw.pop('send_counts', []))
        recv_counts = kw.pop('recvCounts', kw.pop('recv_counts', []))
        ep_ws = int(kw.pop('ep_world_size', kw.pop('epWorldSize', world_size)))
        kw.pop('epWorldSize', None)
        if 'transGmmWeight' in kw:
            kw['trans_gmm_weight'] = kw.pop('transGmmWeight')
        if 'transMmWeight' in kw:
            kw['trans_mm_weight'] = kw.pop('transMmWeight')
        # dev_tensors: [gmm_x, gmm_weight, (mm_x, mm_weight) if mm_out]
        gmm_x = dev_tensors[0]
        gmm_weight = dev_tensors[1]
        mm_x = dev_tensors[2] if len(dev_tensors) > 2 else None
        mm_weight = dev_tensors[3] if len(dev_tensors) > 3 else None
        return resolved(gmm_x=gmm_x, gmm_weight=gmm_weight,
                        ep_world_size=ep_ws, hcom=hcomm,
                        send_counts=send_counts, recv_counts=recv_counts,
                        send_counts_tensor=None, recv_counts_tensor=None,
                        mm_x=mm_x, mm_weight=mm_weight, **kw)
    elif 'npu_alltoallv_gmm' in api_name:
        send_counts = kw.pop('sendCounts', kw.pop('send_counts', []))
        recv_counts = kw.pop('recvCounts', kw.pop('recv_counts', []))
        ep_ws = int(kw.pop('ep_world_size', kw.pop('epWorldSize', world_size)))
        kw.pop('epWorldSize', None)
        if 'transGmmWeight' in kw:
            kw['trans_gmm_weight'] = kw.pop('transGmmWeight')
        if 'transMmWeight' in kw:
            kw['trans_mm_weight'] = kw.pop('transMmWeight')
        if 'permuteOutFlag' in kw:
            kw['permute_out_flag'] = kw.pop('permuteOutFlag')
        gmm_x = dev_tensors[0]
        gmm_weight = dev_tensors[1]
        mm_x = dev_tensors[2] if len(dev_tensors) > 2 else None
        mm_weight = dev_tensors[3] if len(dev_tensors) > 3 else None
        return resolved(gmm_x=gmm_x, gmm_weight=gmm_weight,
                        ep_world_size=ep_ws, hcom=hcomm,
                        send_counts=send_counts, recv_counts=recv_counts,
                        send_counts_tensor=None, recv_counts_tensor=None,
                        mm_x=mm_x, mm_weight=mm_weight, **kw)
    elif 'bmm_reducescatter_alltoall' in api_name:
        ep_ws = int(kw.pop('group_ep_worldsize', kw.pop('epWorldSize', 0)))
        tp_ws = int(kw.pop('group_tp_worldsize', kw.pop('tpWorldSize', 0)))
        ep_hc = kw.pop('ep_hcomm', hcomm)
        tp_hc = kw.pop('tp_hcomm', hcomm)
        shard = int(kw.pop('shard_type', kw.pop('yShardType', 0)))
        is_bias = bool(kw.pop('isBias', False))
        bias = dev_tensors[2] if (is_bias and len(dev_tensors) > 2) else None
        # Drop all remaining attrs not in API schema
        kw.clear()
        return resolved(dev_tensors[0], dev_tensors[1],
                        ep_hc, ep_ws, tp_hc, tp_ws,
                        bias=bias, shard_type=shard)
    elif 'alltoall_allgather_bmm' in api_name:
        ep_ws = int(kw.pop('group_ep_worldsize', kw.pop('epWorldSize', 0)))
        tp_ws = int(kw.pop('group_tp_worldsize', kw.pop('tpWorldSize', 0)))
        ep_hc = kw.pop('ep_hcomm', hcomm)
        tp_hc = kw.pop('tp_hcomm', hcomm)
        act = kw.pop('act_type', kw.pop('actType', 'none'))
        if isinstance(act, int):
            act_map = {0: 'none', 1: 'gelu', 2: 'silu', 3: 'relu', 4: 'fastgelu'}
            act = act_map.get(act, 'none')
        shard = int(kw.pop('shard_type', kw.pop('xShardType', 0)))
        need_ag = bool(kw.pop('need_allgather_out', kw.pop('needAllgatherOut', False)))
        need_act = bool(kw.pop('need_activation_feature', kw.pop('needActivationFeature', False)))
        # Drop attrs not in API schema
        for k in ['isBias', 'isTrans', 'epWorldSize', 'tpWorldSize',
                  'xShardType', 'actType', 'needAllgatherOut', 'needActivationFeature']:
            kw.pop(k, None)
        bias = dev_tensors[2] if len(dev_tensors) > 2 else None
        return resolved(dev_tensors[0], dev_tensors[1], ep_hc, ep_ws, tp_hc, tp_ws,
                        bias=bias, shard_type=shard, act_type=act,
                        need_allgather_out=need_ag, need_activation_feature=need_act)
    else:
        kw['hcom'] = hcomm
        kw['world_size'] = world_size
        return resolved(*dev_tensors, **kw)


class _MC2GraphModel(torch.nn.Module):
    """nn.Module wrapper for mc2 collective ops, mirroring mc2_test's
    MatmulAlltoAllGraphModel pattern. forward delegates to _call_api so all
    mc2 ops (matmul_all_to_all, all_gather_mm, gmm_alltoallv, ...) are supported."""

    def __init__(self, api_name, hcomm, world_size, attrs):
        super().__init__()
        self._api_name = api_name
        self._hcomm = hcomm
        self._world_size = world_size
        self._attrs = attrs

    def forward(self, *dev_tensors):
        return _call_api(self._api_name, list(dev_tensors),
                         self._hcomm, self._world_size, self._attrs)


class _AlltoAllvGmmGraphModel(torch.nn.Module):
    """Dynamo-friendly GraphModel for npu_alltoallv_gmm / npu_gmm_alltoallv.

    dynamo cannot trace _MC2GraphModel because:
    (1) forward(*dev_tensors) uses *args (variable positional)
    (2) _call_api resolves the op via a big if-elif and calls with kwargs
        like resolved(gmm_x=..., gmm_weight=...) which dynamo mis-traces.
    (3) dynamo converts self._send_counts (list[int]) to immutable_list,
        but the C++ op requires List[int]. Passing as forward arg makes
        dynamo treat it as a constant (not traced).
    This model mirrors mc2_test's AlltoAllvGroupedMatmulGraphModel: explicit
    positional forward signature, direct torch_npu call inside.
    """

    def __init__(self, api_name):
        super().__init__()
        self._api_name = api_name

    def forward(self, gmm_x, gmm_weight, hcom, ep_ws, send_counts, recv_counts,
                trans_gmm, trans_mm, permute, mm_x=None, mm_weight=None):
        is_v2 = 'gmm_alltoallv' in self._api_name
        fn = torch_npu.npu_gmm_alltoallv if is_v2 else torch_npu.npu_alltoallv_gmm
        send_counts = [int(x) for x in send_counts]
        recv_counts = [int(x) for x in recv_counts]
        if is_v2:
            return fn(gmm_x, gmm_weight, hcom, ep_ws,
                      send_counts, recv_counts,
                      send_counts_tensor=None, recv_counts_tensor=None,
                      mm_x=mm_x, mm_weight=mm_weight,
                      trans_gmm_weight=trans_gmm, trans_mm_weight=trans_mm)
        return fn(gmm_x, gmm_weight, hcom, ep_ws,
                  send_counts, recv_counts,
                  send_counts_tensor=None, recv_counts_tensor=None,
                  mm_x=mm_x, mm_weight=mm_weight,
                  trans_gmm_weight=trans_gmm, trans_mm_weight=trans_mm,
                  permute_out_flag=permute)


def _run_mc2_graph(api_name, dev_tensors, hcomm, world_size, attrs, graph_mode):
    """Run mc2 op in GE graph mode via torch.compile + torchair backend.

    Mirrors mc2_test's define_model(graph_type=2): torch.compile(model,
    backend=npu_backend, dynamic=True/False). Returns NPU output tensor
    (same shape/dtype as eager) or raises.

    For npu_alltoallv_gmm / npu_gmm_alltoallv, uses _AlltoAllvGmmGraphModel
    (dynamo-friendly explicit forward signature) instead of the generic
    _MC2GraphModel, because dynamo cannot trace *args + kwarg call inside
    _call_api.
    """
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    is_a2av_gmm = ('npu_alltoallv_gmm' in api_name or
                   'npu_gmm_alltoallv' in api_name)
    if is_a2av_gmm:
        model = _AlltoAllvGmmGraphModel(api_name)
        invoke = _build_a2av_gmm_invoke(dev_tensors, hcomm, world_size, attrs)
    else:
        model = _MC2GraphModel(api_name, hcomm, world_size, attrs)
        invoke = dev_tensors
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    dynamic = (graph_mode == "dynamic")
    compiled = torch.compile(model, backend=npu_backend, dynamic=dynamic)
    result = compiled(*invoke)
    torch.npu.synchronize()
    return result


def _build_a2av_gmm_invoke(dev_tensors, hcomm, world_size, attrs):
    """Build positional invoke args for _AlltoAllvGmmGraphModel.forward.
    forward(gmm_x, gmm_weight, hcom, ep_ws, send_counts, recv_counts,
            trans_gmm, trans_mm, permute, mm_x=None, mm_weight=None).
    Non-tensor args are passed as forward params so dynamo treats them as
    constants (avoids immutable_list conversion of self attributes)."""
    gmm_x = dev_tensors[0]
    gmm_weight = dev_tensors[1]
    mm_x = dev_tensors[2] if len(dev_tensors) > 2 else None
    mm_weight = dev_tensors[3] if len(dev_tensors) > 3 else None
    ep_ws = int(attrs.get('ep_world_size', attrs.get('epWorldSize', world_size)))
    send_counts = list(attrs.get('send_counts', attrs.get('sendCounts', [])))
    recv_counts = list(attrs.get('recv_counts', attrs.get('recvCounts', [])))
    trans_gmm = bool(attrs.get('trans_gmm_weight', attrs.get('transGmmWeight', False)))
    trans_mm = bool(attrs.get('trans_mm_weight', attrs.get('transMmWeight', False)))
    permute = bool(attrs.get('permute_out_flag', attrs.get('permuteOutFlag', False)))
    return [gmm_x, gmm_weight, hcomm, ep_ws, send_counts, recv_counts,
            trans_gmm, trans_mm, permute, mm_x, mm_weight]


def worker(rank, world_size, port, input_path, plan_path, result_path, error_path, graph_mode=''):
    try:
        # Set longer HCCL timeouts for large BMM golden computation (CPU-side ~100s)
        os.environ['HCCL_EXEC_TIMEOUT'] = '3600'
        os.environ['HCCL_LINK_TIMEOUT'] = '3600'
        os.environ['HCCL_CONNECT_TIMEOUT'] = '3600'
        torch.npu.set_device(rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                                init_method=f"tcp://127.0.0.1:{port}",
                                timeout=datetime.timedelta(seconds=3600))
        hcomm = str(dist.group.WORLD._get_backend(
            torch.device("npu")).get_hccl_comm_name(rank))

        with open(plan_path) as f:
            plan_info = json.load(f)
        api_name = plan_info['api_name']
        attrs = plan_info.get('attributes', {})
        golden_disabled = bool(plan_info.get('golden_disabled', False))
        remark = plan_info.get('remark', '')
        tensor_view_shapes = plan_info.get('tensor_view_shapes', [])
        # Pass tensor_dtypes to attrs so golden functions can simulate NPU dtype precision
        tensor_dtypes = plan_info.get('tensor_dtypes', [])
        if tensor_dtypes:
            attrs = dict(attrs)
            attrs['_tensor_dtypes'] = tensor_dtypes

        moe_group = None
        moe_hcomm = None
        if (_uses_private_moe_backend(api_name) and
                ('npu_moe_distribute_' in api_name or 'mega_moe' in api_name)):
            # Dispatch/combine require an EP-only communication domain. WORLD is
            # retained for TTK barriers/result gathering and must not be reused.
            moe_group = dist.new_group(ranks=list(range(world_size)), backend="hccl")
            moe_hcomm = str(moe_group._get_backend(
                torch.device("npu")).get_hccl_comm_name(rank, init_comm=True))

        is_dual = ('bmm_reducescatter_alltoall' in api_name or
                   'alltoall_allgather_bmm' in api_name)
        if is_dual:
            dist.barrier()
            ep_ws = int(attrs.get('epWorldSize', 0))
            tp_ws = int(attrs.get('tpWorldSize', 0))
            if ep_ws and tp_ws:
                # EP/TP group layout matching mc2_test reference:
                # EP groups: consecutive ranks [i*ep_ws:(i+1)*ep_ws] for i in range(tp_ws)
                # TP groups: strided ranks [i, i+ep_ws, i+2*ep_ws, ...] for i in range(ep_ws)
                ep_groups = []
                for i in range(tp_ws):
                    ep_ranks = [x + ep_ws * i for x in range(ep_ws)]
                    ep_groups.append(dist.new_group(backend="hccl", ranks=ep_ranks))
                tp_groups = []
                for i in range(ep_ws):
                    tp_ranks = [x * ep_ws + i for x in range(tp_ws)]
                    tp_groups.append(dist.new_group(backend="hccl", ranks=tp_ranks))
                ep_rank = rank % ep_ws
                tp_rank = rank // ep_ws
                attrs = dict(attrs)
                attrs['ep_hcomm'] = str(
                    ep_groups[tp_rank]._get_backend(
                        torch.device("npu")).get_hccl_comm_name(rank))
                attrs['tp_hcomm'] = str(
                    tp_groups[ep_rank]._get_backend(
                        torch.device("npu")).get_hccl_comm_name(rank))

        # Each rank loads its own inputs from npz (keyed by inp_{rank}_{idx})
        data = np.load(input_path, allow_pickle=False)
        my_prefix = f'inp_{rank}_'
        my_input_keys = sorted([k for k in data.files if k.startswith(my_prefix)],
                               key=lambda x: int(x.split('_')[2]))
        if not my_input_keys:
            # Fallback: single-set inputs (all ranks share same inputs)
            my_input_keys = sorted([k for k in data.files if k.startswith('inp_') and k.count('_') == 1],
                                   key=lambda x: int(x.split('_')[1]))
        # Determine which indices we need. For shared weight ops, missing idx>=1
        # tensors are filled from rank 0 (shared weight only saved once).
        my_indices = [int(k.split('_')[2]) for k in my_input_keys]
        if my_indices:
            rank0_keys = sorted([k for k in data.files if k.startswith('inp_0_')],
                                key=lambda x: int(x.split('_')[2]))
            rank0_by_idx = {int(k.split('_')[2]): k for k in rank0_keys}
            max_idx = max(max(my_indices), max(rank0_by_idx.keys()) if rank0_by_idx else 0)
            for idx in range(max_idx + 1):
                if idx not in my_indices and idx in rank0_by_idx:
                    my_input_keys.append(rank0_by_idx[idx])
            my_input_keys.sort(key=lambda x: int(x.split('_')[2]))
        my_inputs_np = [data[k] for k in my_input_keys]

        def _flat_declared_dtype(index):
            flat = []
            for item in tensor_dtypes:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            return flat[index] if index < len(flat) else None

        def _np_to_torch(np_arr, index):
            from ml_dtypes import bfloat16 as np_bf16
            if np_arr.dtype == np_bf16 or (
                hasattr(np_arr.dtype, 'itemsize') and np_arr.dtype.itemsize == 2
                and np_arr.dtype.kind == 'V'):
                t = torch.from_numpy(np_arr.view(np.uint16)).view(torch.bfloat16)
                return t.reshape(np_arr.shape)
            declared_dtype = _flat_declared_dtype(index)
            if declared_dtype in ("float8_e5m2", "float8_e4m3fn"):
                return torch.from_numpy(np_arr.view(np.uint8)).view(
                    getattr(torch, declared_dtype))
            if declared_dtype in ("float8_e8m0", "float8_e8m0fnu"):
                return torch.from_numpy(np_arr.view(np.uint8)).view(torch.float8_e8m0fnu)
            return torch.from_numpy(np_arr)

        # CPU copy (float32 for golden) and NPU copy (original dtype for API call)
        is_none_input = lambda value: value.dtype == object and value.shape == () and value.item() is None
        cpu_inputs_for_rank = [None if is_none_input(x) else _np_to_torch(x.copy(), i).float()
                               for i, x in enumerate(my_inputs_np)]
        dev_tensors = [None if is_none_input(x) else _np_to_torch(x.copy(), i).npu(rank)
                       for i, x in enumerate(my_inputs_np)]

        moe_context_manager = None
        if (_uses_private_moe_backend(api_name) and
                ('npu_moe_distribute_' in api_name or 'mega_moe' in api_name)):
            custom_ccl_buffer_size = int(os.environ.get("HCCL_BUFFSIZE", "0")) * 2 * 1024 * 1024
            try:
                moe_context_manager = _moe_context_manager_cls(
                    moe_hcomm,
                    world_size,
                    backend={
                        "Ascend910B": "kfc",
                        "Ascend910_93": "kfc",
                        "Ascend950": "channel",
                    },
                    customCclBufferSize=custom_ccl_buffer_size,
                )
            except TypeError:
                # Older installed comm_context extensions only accept a string
                # backend and do not expose customCclBufferSize.
                moe_context_manager = _moe_context_manager_cls(
                    moe_hcomm, world_size, backend="kfc")
            attrs = dict(attrs)
            attrs['_moe_context'] = moe_context_manager.create_context()
            attrs['_moe_ccl_buffer_size'] = moe_context_manager.ccl_buffer_size
            attrs['_moe_topo_type'] = getattr(moe_context_manager, 'topo_type', 0)
            attrs['_moe_rank_num_per_server'] = getattr(
                moe_context_manager, 'rank_num_per_server', 0)
            attrs['_moe_rank'] = rank

        # Transpose handling
        if attrs.get('transposeX2', False) or attrs.get('is_trans_b', False):
            dev_tensors[1] = dev_tensors[1].t().contiguous()
        if attrs.get('transposeX1', False):
            dev_tensors[0] = dev_tensors[0].t().contiguous()

        is_gmm = ('npu_gmm_alltoallv' in api_name or 'npu_alltoallv_gmm' in api_name)
        if is_gmm:
            # Generate sendCounts/recvCounts matching ACLNN __patch_gmm_rank_attributes
            # seed comes from remark (e.g. "seed=1"), ep_ws from attrs
            seed_val = 1
            for part in (remark or '').split(','):
                kv = part.split('=', 1)
                if len(kv) == 2 and kv[0].strip() == 'seed':
                    try:
                        seed_val = int(kv[1].strip())
                    except ValueError:
                        pass
            ep_ws = int(attrs.get('epWorldSize', world_size))
            epc = int(dev_tensors[1].shape[0]) if len(dev_tensors) > 1 else 1
            M_per_rank = int(dev_tensors[0].shape[0]) if len(dev_tensors) > 0 else 0
            A_array = [M_per_rank] * ep_ws
            expTokenNums = _generate_gmm_matrix(A_array, epc, seed=seed_val)
            attrs = dict(attrs)
            if 'npu_gmm_alltoallv' in api_name:
                # gmm_alltoallv: recv = expTokenNums[rank], send = collected from all ranks
                recv_counts = list(expTokenNums[rank])
                send_counts = []
                for i in range(ep_ws):
                    send_counts.extend(expTokenNums[i][rank * epc:(rank + 1) * epc])
            else:
                # alltoallv_gmm: send = expTokenNums[rank], recv = collected from all ranks
                send_counts = list(expTokenNums[rank])
                recv_counts = []
                for i in range(ep_ws):
                    recv_counts.extend(expTokenNums[i][rank * epc:(rank + 1) * epc])
            attrs['sendCounts'] = send_counts
            attrs['recvCounts'] = recv_counts
            attrs['expTokenNums'] = expTokenNums
            attrs['expPerCard'] = epc
            attrs['ep_ws'] = ep_ws
            attrs['seed'] = seed_val

        # Call NPU API
        if (api_name.startswith("torch_npu.npu_moe_distribute_") or
                api_name == "torch_npu.npu_moe_update_expert"):
            attrs = dict(attrs)
            attrs["_rank"] = rank
        result = _call_api(api_name, dev_tensors, hcomm, world_size, attrs)
        torch.npu.synchronize(rank)

        if result is None:
            all_prec_strings = [None] * world_size
            dist.all_gather_object(all_prec_strings, f'rank{rank}:PASS(EXECUTED)')
            if rank == 0:
                np.savez(result_path,
                         precision_0=np.array(','.join(all_prec_strings)),
                         pass_0=np.array('PASS'))
            dist.barrier()
            dist.destroy_process_group()
            return

        npu_out = result[0] if isinstance(result, (tuple, list)) else result

        if golden_disabled:
            all_prec_strings = [None] * world_size
            dist.all_gather_object(all_prec_strings, f'rank{rank}:PASS(EXECUTED)')
            if rank == 0:
                np.savez(result_path,
                         precision_0=np.array(','.join(all_prec_strings)),
                         pass_0=np.array('PASS'))
            dist.barrier()
            dist.destroy_process_group()
            return

        # Gather all ranks' CPU inputs to every rank via dist (data transport only, NOT golden)
        # Use torch_npu dist all_gather to move CPU inputs across cards
        # OPTIMIZATION: Only gather tensors that golden actually needs from other ranks.
        # For MC2 matmul ops, weight (idx 1+) is per-rank/shared, only x (idx 0) needs cross-rank.
        # For BMM_RS_A2A / A2A_AG_BMM: golden needs all ranks' x but only own weight.
        # This avoids OOM when weight is large (e.g. 16x8192x2560 ~0.5GB * 8 ranks = 4GB).
        cpu_inputs_per_rank = [None] * world_size
        cpu_inputs_per_rank[rank] = cpu_inputs_for_rank

        # Determine which tensors need cross-rank gather.
        # Default: only x (idx 0), weight is shared (own copy suffices).
        needs_cross_rank = [True] + [False] * (len(cpu_inputs_for_rank) - 1)
        if 'npu_gmm_alltoallv' in api_name or 'npu_alltoallv_gmm' in api_name:
            # GMM golden needs all ranks' gmm_x AND gmm_weight
            needs_cross_rank = [True] * len(cpu_inputs_for_rank)
        # MatmulAlltoAll / AlltoAllMatmul: x2 is NOT shared (per-rank different)
        if 'npu_matmul_all_to_all' in api_name or 'npu_all_to_all_matmul' in api_name:
            needs_cross_rank = [True] * min(len(cpu_inputs_for_rank), 2)
        # BMM_RS_A2A: weight is per-rank different (NOT shared).
        # golden needs all ranks' x AND weight. weight is too large for dist.all_gather
        # (8 ranks * 2.68GB = 21GB > NPU memory). Use file sharing instead.
        is_bmm_rs_a2a = 'bmm_reducescatter_alltoall' in api_name
        if is_bmm_rs_a2a:
            needs_cross_rank = [True] * len(cpu_inputs_for_rank)
            t0 = time.time()
            rank_file = os.path.join(
                os.path.dirname(result_path), f'ttk_e2e_md_rank_{rank}.pt')
            torch.save(cpu_inputs_for_rank, rank_file)
            logging.info('[BMM_IO rank=%s] saved t=%.1fs', rank, time.time() - t0)
            dist.barrier()
            logging.info('[BMM_IO rank=%s] barrier1 t=%.1fs', rank, time.time() - t0)
            for r in range(world_size):
                if r == rank:
                    continue
                peer_file = os.path.join(
                    os.path.dirname(result_path), f'ttk_e2e_md_rank_{r}.pt')
                cpu_inputs_per_rank[r] = torch.load(peer_file, weights_only=True)
                logging.info('[BMM_IO rank=%s] loaded r=%s t=%.1fs', rank, r, time.time() - t0)
            dist.barrier()
            logging.info('[BMM_IO rank=%s] barrier2 t=%.1fs', rank, time.time() - t0)
            try:
                os.remove(rank_file)
            except Exception:
                pass
        else:
            for tensor_idx in range(len(cpu_inputs_for_rank)):
                if not needs_cross_rank[tensor_idx]:
                    continue
                t_cpu = cpu_inputs_for_rank[tensor_idx]
                # Move to NPU for all_gather transport, then back to CPU for golden
                t_npu = t_cpu.npu(rank)
                gather_list = [torch.empty_like(t_npu) for _ in range(world_size)]
                dist.all_gather(gather_list, t_npu)
                for r in range(world_size):
                    if r == rank:
                        continue
                    if cpu_inputs_per_rank[r] is None:
                        cpu_inputs_per_rank[r] = [None] * len(cpu_inputs_for_rank)
                    cpu_inputs_per_rank[r][tensor_idx] = gather_list[r].cpu().float()
                del t_npu, gather_list

            # Fill in other ranks' non-gathered tensors with own copy (shared weight assumption)
            for r in range(world_size):
                if r == rank:
                    continue
                if cpu_inputs_per_rank[r] is None:
                    cpu_inputs_per_rank[r] = list(cpu_inputs_for_rank)
                else:
                    for idx in range(len(cpu_inputs_for_rank)):
                        if cpu_inputs_per_rank[r][idx] is None:
                            cpu_inputs_per_rank[r][idx] = cpu_inputs_for_rank[idx]

        # Pure CPU golden (each golden function references ACLNN mc2_golden.py logic)
        goldens = _compute_golden(api_name, cpu_inputs_per_rank, attrs,
                                  rank=rank, ws=world_size, dist_avail=(is_gmm or is_dual))
        golden = goldens[0] if (is_gmm or is_dual) else goldens[rank]

        # Handle multi-output golden (e.g. GMM returns (main, mm) tuple, A2A_AG_BMM returns dict)
        npu_mm_out = None
        if isinstance(result, (tuple, list)) and len(result) > 1:
            npu_mm_out = result[1]
        if isinstance(golden, dict):
            # A2A_AG_BMM: golden is {'main':..., 'allgather':..., 'bmm':...}
            # NPU result is (main, allgather, bmm) tuple
            golden = golden.get('main')
            if isinstance(result, (tuple, list)) and len(result) > 1:
                npu_mm_out = result[1]  # allgather
                # bmm is result[2] if present
        if isinstance(golden, (tuple, list)):
            golden_mm = golden[1] if len(golden) > 1 else None
            golden = golden[0]
        else:
            golden_mm = None

        # Cast golden to NPU output dtype (matching ACLNN: golden.to(output_dtype))
        golden_t = None
        if golden is not None and isinstance(golden, torch.Tensor):
            npu_dtype = npu_out.dtype
            golden_t = golden.to(npu_dtype)

        golden_mm_t = None
        if golden_mm is not None and isinstance(golden_mm, torch.Tensor) and npu_mm_out is not None:
            golden_mm_t = golden_mm.to(npu_mm_out.dtype)

        # Each rank compares its own NPU output vs its own CPU golden using torch.isclose
        npu_dtype_str = str(npu_out.dtype).split('.')[-1]
        if golden_t is not None:
            dtype_str = npu_dtype_str
            rtol = 0.001 if dtype_str in ('float16', 'bfloat16') else 0.0001
            atol = 1e-8
            ptol = 0.001 if dtype_str in ('float16', 'bfloat16') else 0.0001
            npu_flat = npu_out.cpu().contiguous().view(-1)
            gold_flat = golden_t.cpu().contiguous().view(-1)
            close = torch.isclose(npu_flat, gold_flat, rtol=rtol, atol=atol, equal_nan=True)
            precision = float(close.sum().item()) / float(close.numel())
            is_pass = (1.0 - precision) <= ptol

            # Also compare mm_out if present
            if golden_mm_t is not None and npu_mm_out is not None:
                npu_mm_flat = npu_mm_out.cpu().contiguous().view(-1)
                gold_mm_flat = golden_mm_t.cpu().contiguous().view(-1)
                close_mm = torch.isclose(npu_mm_flat, gold_mm_flat, rtol=rtol, atol=atol, equal_nan=True)
                precision_mm = float(close_mm.sum().item()) / float(close_mm.numel())
                is_pass_mm = (1.0 - precision_mm) <= ptol
                # Combined precision: average of both outputs
                precision = (precision + precision_mm) / 2
                is_pass = is_pass and is_pass_mm

            save_arrays = {
                'precision_0': np.array(f'{precision*100}%'),
                'pass_0': np.array('PASS' if is_pass else 'FAIL'),
            }
        else:
            save_arrays = {
                'out_0': npu_out.cpu().float().numpy(),
                'out_dtype_0': np.array(npu_dtype_str),
            }

        # Gather precision from all ranks (rank 0 collects and writes combined result)
        if golden_t is not None:
            rank_status = 'PASS' if is_pass else 'FAIL'
            rank_precision_str = f'rank{rank}:{rank_status}({precision*100}%)'
        elif golden_disabled:
            rank_precision_str = f'rank{rank}:PASS(EXECUTED)'
        else:
            rank_precision_str = f'rank{rank}:FAIL'
        # Use dist to gather precision strings from all ranks
        all_prec_strings = [None] * world_size
        dist.all_gather_object(all_prec_strings, rank_precision_str)

        if rank == 0:
            # Combine per-rank precision (matching ACLNN format: rank0:PASS(99.97%),rank1:PASS(...),...)
            combined_prec = ",".join(all_prec_strings)
            has_fail = any('FAIL' in s for s in all_prec_strings)
            save_arrays = {
                'precision_0': np.array(combined_prec),
                'pass_0': np.array('PASS' if not has_fail else 'FAIL'),
            }
            np.savez(result_path, **save_arrays)

        # ===== Multi-device mc2 graph mode (mirrors mc2_test define_model) =====
        # Runs the same mc2 op through torch.compile + torchair NPU backend
        # (dynamic=True when graph_mode=="dynamic"), then compares against the
        # same CPU golden used for eager. Result is surfaced as GRAPH DYN/CST
        # in the framework_api return structure.
        if graph_mode in ("dynamic", "static") and golden_t is not None:
            dist.barrier()
            graph_precision = None
            graph_is_pass = False
            graph_err = None
            try:
                graph_result = _run_mc2_graph(api_name, dev_tensors, hcomm,
                                              world_size, attrs, graph_mode)
                graph_out = graph_result[0] if isinstance(graph_result, (tuple, list)) else graph_result
                g_flat = graph_out.cpu().contiguous().view(-1)
                close_g = torch.isclose(g_flat, golden_t.cpu().contiguous().view(-1),
                                        rtol=rtol, atol=atol, equal_nan=True)
                graph_precision = float(close_g.sum().item()) / float(close_g.numel())
                graph_is_pass = (1.0 - graph_precision) <= ptol
            except Exception as ge:
                graph_err = str(ge)[:500]

            g_rank_status = 'PASS' if graph_is_pass else 'FAIL'
            if graph_err:
                g_rank_str = f'rank{rank}:ERROR({graph_err[:80]})'
            else:
                g_rank_str = f'rank{rank}:{g_rank_status}({graph_precision*100}%)'
            all_g_strings = [None] * world_size
            dist.all_gather_object(all_g_strings, g_rank_str)

            if rank == 0:
                combined_g = ",".join(all_g_strings)
                g_has_fail = any('FAIL' in s or 'ERROR' in s for s in all_g_strings)
                # Reload save_arrays (rank 0 wrote it above) and append graph fields
                loaded_so_far = np.load(result_path, allow_pickle=False)
                save_arrays = {k: loaded_so_far[k] for k in loaded_so_far.files}
                save_arrays['graph_precision_0'] = np.array(combined_g)
                save_arrays['graph_pass_0'] = np.array('PASS' if not g_has_fail else 'FAIL')
                save_arrays['graph_mode_0'] = np.array(graph_mode)
                np.savez(result_path, **save_arrays)

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        import traceback
        if rank == 0:
            with open(error_path, 'w') as f:
                f.write(traceback.format_exc())
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _resolve_api(api_name):
    if 'bmm_reducescatter_alltoall' in api_name:
        from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
        return npu_bmm_reducescatter_alltoall
    if 'alltoall_allgather_bmm' in api_name:
        from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
        return npu_alltoall_allgather_bmm
    if 'npu_mm_all_reduce_add_rms_norm' in api_name:
        from mindspeed.ops.npu_mm_all_reduce_add_rms_norm import npu_mm_all_reduce_add_rms_norm
        return npu_mm_all_reduce_add_rms_norm
    parts = api_name.split(".")
    obj = importlib.import_module(parts[0])
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj


if __name__ == '__main__':
    input_path = os.environ['TTK_E2E_INPUT']
    plan_path = os.environ['TTK_E2E_PLAN']
    result_path = os.environ['TTK_E2E_RESULT']
    error_path = os.environ['TTK_E2E_ERROR']
    device_ids = [int(d) for d in os.environ['TTK_E2E_DEVICES'].split(',')]
    ndev = int(os.environ['TTK_E2E_NDEV'])
    graph_mode = os.environ.get('TTK_E2E_GRAPH_MODE', '')

    rank_value = os.environ.get('TTK_E2E_RANK')
    if rank_value is not None:
        with open(plan_path) as plan_file:
            worker_api_name = json.load(plan_file).get('api_name', '')
        if (_uses_private_moe_backend(worker_api_name) and
                ('npu_moe_distribute_' in worker_api_name or 'mega_moe' in worker_api_name)):
            import importlib.util
            extension_root = os.path.expanduser(
                f'~/.cache/torch_extensions/py{sys.version_info.major}{sys.version_info.minor}_cpu')
            def load_extension(name):
                path = os.path.join(extension_root, name, f'{name}.so')
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            comm_context_module = load_extension('comm_context')
            _moe_context_manager_cls = comm_context_module.CommContextManager
            _moe_dispatch_module = load_extension('npu_moe_distribute_dispatch')
            if 'combine' in worker_api_name:
                _moe_combine_module = load_extension('npu_moe_distribute_combine')
            if 'mega_moe' in worker_api_name:
                _mega_moe_module = load_extension('npu_mega_moe')
        ready_path = f"{result_path}.rank{rank_value}.ready"
        with open(ready_path, 'w') as ready_file:
            ready_file.write('ready')
        start_path = f"{result_path}.start"
        import time
        while not os.path.exists(start_path):
            time.sleep(0.05)
        worker(int(rank_value), ndev, int(os.environ['TTK_E2E_PORT']), input_path,
               plan_path, result_path, error_path, graph_mode)
        sys.exit(1 if os.path.exists(error_path) else 0)

    with open(plan_path) as plan_file:
        parent_plan = json.load(plan_file)
    parent_api_name = parent_plan.get('api_name', '')
    if (_uses_private_moe_backend(parent_api_name) and
            ('npu_moe_distribute_' in parent_api_name or 'mega_moe' in parent_api_name)):
        parent_attrs = parent_plan.get('attributes', {})
        input_shapes = parent_plan.get('tensor_view_shapes', [])
        x_shape = input_shapes[1]
        expert_ids_shape = input_shapes[2]
        num_tokens = int(x_shape[0])
        # Combine CSV carries the maximum dispatched buffer as x. Its paired
        # dispatch input size is encoded by expert_ids.
        if 'combine' in parent_api_name:
            num_tokens = int(expert_ids_shape[0])
        hidden = int(x_shape[1])
        topk = int(expert_ids_shape[1])
        if 'mega_moe' in parent_api_name:
            # Mirror get_mega_moe_ccl_buffer_size for the E2E worker without
            # importing the JIT-loading Python wrapper in the parent process.
            align = lambda value, base: (value + base - 1) // base * base
            local_experts = int(parent_attrs['moe_expert_num']) // ndev
            compare_count = align(num_tokens * topk * 4, 256) // 4
            mask_slot_size = align(compare_count // 8, 32) + 32
            mask_recv_size = align(local_experts * ndev * mask_slot_size, 512)
            token_bytes = align(align(hidden, 256) + (hidden + 31) // 32, 32)
            total = (60 * 1024 + mask_recv_size
                     + align(num_tokens * token_bytes, 512)
                     + align(num_tokens * hidden * topk * 2, 512))
            buffer_size = align(align(total, 1024 * 1024) // (1024 * 1024), 2) // 2
        else:
            buffer_size = _get_moe_ccl_buffer_size(
                ndev, num_tokens, hidden, int(parent_attrs['moe_expert_num']), topk,
                int(parent_attrs.get('shared_expert_num', 0)),
                int(parent_attrs.get('shared_expert_rank_num', 0)),
                str(parent_attrs.get('comm_alg', '')))
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        os.environ['HCCL_BUFFSIZE'] = str(buffer_size)

    port = find_free_port()
    processes = []
    for rank in range(ndev):
        child_env = os.environ.copy()
        child_env['TTK_E2E_RANK'] = str(rank)
        child_env['TTK_E2E_PORT'] = str(port)
        processes.append(subprocess.Popen([sys.executable, __file__], env=child_env))
        ready_path = f"{result_path}.rank{rank}.ready"
        deadline = time.monotonic() + 120
        while not os.path.exists(ready_path):
            if processes[-1].poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"Rank {rank} failed during E2E worker initialization")
            time.sleep(0.05)

    with open(f"{result_path}.start", 'w') as start_file:
        start_file.write('start')

    for p in processes:
        return_code = p.wait()
        if return_code != 0:
            if not os.path.exists(error_path):
                with open(error_path, 'w') as f:
                    f.write(f"Worker process exited with code {return_code}")
            break

    if not os.path.exists(result_path) and not os.path.exists(error_path):
        with open(error_path, 'w') as f:
            f.write("No result file and no error file produced")

    if os.path.exists(error_path):
        sys.exit(1)
