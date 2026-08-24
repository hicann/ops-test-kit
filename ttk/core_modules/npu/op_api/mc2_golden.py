#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# mc2 multi-device golden implementations
# Extracted from profiling.py for maintainability

import logging
import numpy
from typing import Dict, List

from ...testcase_manager import TestcaseAclnn
from .comparison import Comparator


def _to_torch_f32(t):
    """Convert torch.Tensor or numpy.ndarray to float32 torch.Tensor."""
    import torch
    if t is None:
        return None
    if isinstance(t, numpy.ndarray):
        dtype_str = str(t.dtype)
        if 'e8m0' in dtype_str or 'float8_e8m0' in dtype_str:
            raw = t.view(numpy.uint8).astype(numpy.float64)
            arr = numpy.power(2.0, raw - 127).astype(numpy.float32)
        else:
            try:
                arr = t.astype(numpy.float32, copy=False)
            except (TypeError, ValueError):
                arr = t.view(numpy.uint8).astype(numpy.float32, copy=False)
        return torch.from_numpy(arr)
    if hasattr(t, 'dtype') and str(t.dtype).replace('torch.', '') in (
        'float8_e4m3fn', 'float8_e5m2', 'hifloat8'):
        return t.float()
    return t.float()


def _e8m0_to_f32(scale_tensor):
    """Convert e8m0 scales to float32 using 2^(e-127) formula."""
    import torch
    if scale_tensor is None:
        return None
    if isinstance(scale_tensor, numpy.ndarray):
        raw = scale_tensor.view(numpy.uint8).astype(numpy.float64)
        return numpy.power(2.0, raw - 127).astype(numpy.float32)
    dtype_str = str(scale_tensor.dtype).replace('torch.', '')
    if 'e8m0' in dtype_str:
        raw = scale_tensor.view(torch.uint8).to(torch.float64)
        return torch.pow(2.0, raw - 127).to(torch.float32)
    return scale_tensor.float()


def _quant_grouped_matmul_cpu(gmm_x, gmm_weight, group_list,
                                gmm_x_scale=None, gmm_weight_scale=None,
                                is_mxfp=False, is_tt=False):
    """Quant-aware grouped matmul, applying scales per RDV logic.

    MX: per-expert mxfp_cpu_compute (block-scaled matmul).
    TT: plain matmul then multiply by (x_scale * weight_scale).
    """
    import torch
    import numpy as np
    if not is_mxfp and not is_tt:
        return __grouped_matmul_cpu(gmm_x, gmm_weight, group_list)

    if is_mxfp:
        xs_np = gmm_x_scale if isinstance(gmm_x_scale, np.ndarray) else gmm_x_scale.numpy()
        ws_np = gmm_weight_scale if isinstance(gmm_weight_scale, np.ndarray) else gmm_weight_scale.numpy()
        xs_np = xs_np.reshape(xs_np.shape[0], -1)
        ep, k_groups, N, pair = ws_np.shape
        ws_np = ws_np.transpose(0, 1, 3, 2).reshape(ep, k_groups * pair, N)
        x_np = gmm_x.numpy()
        w_np = gmm_weight.numpy()
        results = []
        offset = 0
        for i, gl in enumerate(group_list):
            if gl <= 0:
                continue
            x_chunk = x_np[offset:offset + gl]
            xs_chunk = xs_np[offset:offset + gl]
            w_expert = w_np[i]
            ws_expert = ws_np[i]
            rep_x1s = np.repeat(xs_chunk, 32, axis=-1)
            rep_x2s = np.repeat(ws_expert, 32, axis=-2)
            k_pad = rep_x1s.shape[1] - x_chunk.shape[1]
            if k_pad > 0:
                x_chunk = np.pad(x_chunk, ((0, 0), (0, k_pad)))
                w_expert = np.pad(w_expert, ((0, k_pad), (0, 0)))
            out = np.matmul(x_chunk * rep_x1s[:, :x_chunk.shape[1]],
                            w_expert * rep_x2s[:w_expert.shape[0], :])
            results.append(torch.from_numpy(out))
            offset += gl
        return torch.cat(results, dim=0).to(torch.float32)
    else:
        gmm_out = __grouped_matmul_cpu(gmm_x, gmm_weight, group_list)
        combined = gmm_x_scale * gmm_weight_scale
        if combined.dim() == 0:
            combined = combined.unsqueeze(0).unsqueeze(0)
        elif combined.dim() == 1:
            combined = combined.unsqueeze(0)
        gmm_out = gmm_out * combined
        return gmm_out


def _to_torch_keep(t):
    """Convert numpy.ndarray to torch.Tensor, keep dtype if torch.Tensor."""
    import torch
    if t is None:
        return None
    if isinstance(t, numpy.ndarray):
        return torch.from_numpy(t)
    return t


def __fmt_compare_result(cr):
    """Format compare result with mere/mare for stat_rel_err or precision for close."""
    _m = cr.metrics.get(0, {}) if cr.metrics else {}
    if _m and 'mere' in _m:
        return f" mere={_m.get('mere'):.4e} mare={_m.get('mare'):.4e} th={_m.get('threshold'):.4e}"
    return f" prec={cr.precision}"


def __golden_multi_device_compare(thread_contexts: Dict[int, TestcaseAclnn],
                                      device_ids: List[int],
                                      all_precision: list):
    """Generate golden and compare for multi-device testcases.

    Supports:
      - aclnnMatmulAlltoAll: matmul(x1, x2) -> all_to_all -> output
      - aclnnAlltoAllMatmul: all_to_all(x1) -> matmul(a2a_out, x2) -> output
      - aclnnMatmulReduceScatter: matmul(x1, x2) -> reduce_scatter(split M by rank)
    """
    api_name = next(iter(thread_contexts.values())).api_name
    world_size = len(device_ids)

    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes

    # Barrier is a synchronization-only API.  It has no value output and
    # therefore cannot use the matmul/MoE golden routines below.
    if api_name.startswith("aclnnDistributeBarrier"):
        for did in device_ids:
            thread_contexts[did].golden_tensors = []
            all_precision.append(f"rank{did}:PASS(EXECUTED)")
        return

    is_allto_all_matmul = "AlltoAllMatmul" in api_name or "AlltoAllQuantMatmul" in api_name
    is_matmul_alltoall = "MatmulAlltoAll" in api_name
    is_quant_alltoall = "QuantMatmulAlltoAll" in api_name
    is_all_reduce = "AllReduce" in api_name
    is_reduce_scatter = "ReduceScatter" in api_name
    is_all_gather = "AllGather" in api_name
    is_grouped_matmul = "GroupedMatMul" in api_name
    is_bmm_rs_a2a = "BatchMatMulReduceScatter" in api_name
    is_a2a_ag_bmm = "AlltoAllAllGather" in api_name
    is_moe_dispatch = "MoeDistributeDispatch" in api_name
    is_moe_combine = "MoeDistributeCombine" in api_name

    if is_a2a_ag_bmm:
        __golden_a2a_ag_bmm(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_bmm_rs_a2a:
        __golden_bmm_reduce_scatter_allto_all(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_grouped_matmul:
        __golden_grouped_matmul_compare(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_moe_dispatch:
        __golden_moe_distribute_dispatch(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_moe_combine:
        __golden_moe_distribute_combine(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_all_reduce:
        __golden_all_reduce_compare(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_reduce_scatter:
        __golden_reduce_scatter_compare(thread_contexts, device_ids, all_precision, world_size)
        return

    if is_all_gather:
        __golden_all_gather_compare(thread_contexts, device_ids, all_precision, world_size)
        return

    alltoAllAxesOptional = attrs.get('alltoAllAxesOptional', None)
    transposeX1 = attrs.get('transposeX1', False)
    transposeX2 = attrs.get('transposeX2', False)

    import torch
    t_x1 = bool(transposeX1) if transposeX1 is not None else False
    t_x2 = bool(transposeX2) if transposeX2 is not None else False

    rank_goldens = {}
    is_quant_alltoall = "QuantMatmulAlltoAll" in api_name
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None

        if is_quant_alltoall:
            x1scale = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            x2scale = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
            golden = __golden_matmul_allto_all(
                thread_contexts, device_ids, did, x1, x2, None,
                t_x1, t_x2, world_size,
                x1scale=x1scale, x2scale=x2scale)
        elif is_allto_all_matmul:
            golden = __golden_allto_all_matmul(
                thread_contexts, device_ids, did, x1, x2, bias,
                t_x1, t_x2, world_size)
        else:
            golden = __golden_matmul_allto_all(
                thread_contexts, device_ids, did, x1, x2, bias,
                t_x1, t_x2, world_size)
        rank_goldens[did] = golden

    # 真·小算子级联 third_party（参考 mc2_test get_hccl_mm）
    # aclnnMatmulAlltoAll: matmul -> all_to_all
    # aclnnAlltoAllMatmul: all_to_all -> matmul
    rank_third_parties = None
    if is_allto_all_matmul:
        try:
            from .hccl_cascade import run_alltoall_matmul_cascade
            # 推断 is_alltoall_output：看 output_tensor_indexes 是否包含 alltoall 输出
            # ttk CSV 用 output_tensor_indexes 标注输出位置；
            # 若有 >=2 个输出，第二个是 alltoall_output
            first_tc = next(iter(thread_contexts.values()))
            is_a2a_out = (len(first_tc.output_tensor_indexes) >= 2)
            cascade_outs = run_alltoall_matmul_cascade(
                thread_contexts, device_ids,
                transpose_x1=t_x1, transpose_x2=t_x2,
                is_alltoall_output=is_a2a_out)
            # cascade_outs[did] = {'main': tensor, 'alltoall': tensor|None}
            # rank_third_parties[did] 期望 list[0]=main, list[1]=alltoall
            rank_third_parties = {
                did: [cascade_outs[did]['main'], cascade_outs[did].get('alltoall')]
                for did in device_ids
            }
            logging.info("AlltoAllMatmul: real HCCL cascade succeeded")
        except Exception:
            logging.exception("AlltoAllMatmul: real HCCL cascade failed, no third_party")
            rank_third_parties = None
    else:
        try:
            from .hccl_cascade import run_matmul_alltoall_cascade
            cascade_outs = run_matmul_alltoall_cascade(
                thread_contexts, device_ids,
                transpose_x1=t_x1, transpose_x2=t_x2)
            rank_third_parties = {did: [cascade_outs[did]] for did in device_ids}
            logging.info("MatmulAlltoAll: real HCCL cascade succeeded")
        except Exception:
            logging.exception("MatmulAlltoAll: real HCCL cascade failed, no third_party")
            rank_third_parties = None

    __apply_a2a_goldens_and_compare(thread_contexts, device_ids, rank_goldens, all_precision,
                                       rank_third_parties=rank_third_parties)


def __apply_a2a_goldens_and_compare(thread_contexts, device_ids, rank_goldens, all_precision,
                                       rank_third_parties=None):
    import torch as _torch
    dtype_map = {
        'float16': _torch.float16, 'fp16': _torch.float16,
        'float32': _torch.float32, 'fp32': _torch.float32,
        'bfloat16': _torch.bfloat16, 'bf16': _torch.bfloat16,
    }
    for did in device_ids:
        tc = thread_contexts[did]
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        goldens = rank_goldens[did]
        third_party = rank_third_parties[did] if rank_third_parties else None
        golden_list = []
        third_parties_list = None
        for out_idx in tc.output_tensor_indexes:
            if isinstance(goldens, dict):
                if out_idx == tc.output_tensor_indexes[0]:
                    g = goldens['main']
                elif len(tc.output_tensor_indexes) > 1 and out_idx == tc.output_tensor_indexes[1]:
                    g = goldens.get('alltoall')
                else:
                    g = _torch.zeros(tc.tensor_view_shapes[out_idx])
                if g is None:
                    g = _torch.zeros(tc.tensor_view_shapes[out_idx])
            else:
                g = goldens
            dt_idx = list(tc.output_tensor_indexes).index(out_idx)
            if dt_idx < len(out_dtypes):
                target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                if target_dtype is not None:
                    g = g.to(target_dtype)
            golden_list.append(g.contiguous())
        # 准备 third_parties：main 输出（output_tensor_indexes[0]）放真级联 main
        # AlltoAllMatmul 的第二输出 alltoall_output 放真级联 alltoall（若存在）
        if third_party is not None:
            tp_items = third_party if isinstance(third_party, (list, tuple)) else [third_party]
            # tp_items 顺序：[main, alltoall|None, ...]
            third_parties_list = []
            for oi_idx, out_idx in enumerate(tc.output_tensor_indexes):
                if oi_idx < len(tp_items):
                    tp = tp_items[oi_idx]
                    third_parties_list.append(tp.contiguous() if isinstance(tp, _torch.Tensor) else tp)
                else:
                    third_parties_list.append(None)
        tc.golden_tensors = golden_list
        del rank_goldens[did]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({__fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(f"Multi-device: rank dev={did} comparison FAILED: {cr.precision} metrics={cr.metrics}")
            else:
                logging.info(f"Multi-device: rank dev={did} comparison PASSED")
        except Exception:
            logging.exception(f"Multi-device: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def __apply_goldens_and_compare(thread_contexts, device_ids, rank_goldens, all_precision, rank_third_parties=None):
    import torch as _torch
    import numpy as _np
    dtype_map = {
        'float16': _torch.float16, 'fp16': _torch.float16,
        'float32': _torch.float32, 'fp32': _torch.float32,
        'bfloat16': _torch.bfloat16, 'bf16': _torch.bfloat16,
    }
    # dtypes that torch supports natively (for torch.isclose path)
    torch_native_dtypes = {
        'float16', 'fp16', 'float32', 'fp32', 'bfloat16', 'bf16',
        'int8', 'int32', 'int64', 'uint8', 'bool',
    }
    for did in device_ids:
        tc = thread_contexts[did]
        golden = rank_goldens[did]
        multi_output_goldens = None
        if isinstance(golden, dict):
            multi_output_goldens = [golden.get('main'), golden.get('gather')]
            golden = multi_output_goldens[0]
        third_party = rank_third_parties[did] if rank_third_parties else None
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        # Decide golden format based on whether the case uses torch-native dtypes.
        # When the case contains non-torch dtypes (hif8/fp8/e8m0), output bytes are
        # decoded as numpy arrays (is_torch_output=False). Golden must then also be
        # numpy (otherwise Comparator tries golden.detach().cpu().numpy() which
        # fails for bfloat16). Use float32 numpy for non-torch cases.
        use_torch = tc.is_torch_dtype_support()
        if len(out_dtypes) > 0:
            out_dtype_str = out_dtypes[0]
            target_dtype = dtype_map.get(out_dtype_str, None)
            if target_dtype is not None and use_torch:
                golden = golden.to(target_dtype)
            else:
                # Non-torch path (hif8/fp8 inputs): golden stays as numpy.
                # But if output dtype is bf16/fp16, NPU output is quantized to
                # that dtype. Golden must be quantized too, otherwise 1-ULP
                # shifts in bf16/fp16 output fail isclose(rtol=0.001) for
                # large-magnitude elements (e.g. |v|>6000 -> 1 ULP=8 > rtol*|v|=6).
                golden_t = (golden.float() if hasattr(golden, 'float')
                            else _torch.from_numpy(_np.asarray(golden).astype(_np.float32)))
                if out_dtype_str in ('bfloat16', 'bf16'):
                    golden_t = golden_t.to(_torch.bfloat16).float()
                elif out_dtype_str in ('float16', 'fp16'):
                    golden_t = golden_t.to(_torch.float16).float()
                golden = golden_t.numpy().astype(_np.float32, copy=False)
        elif not use_torch:
            golden = golden.float().numpy().astype(_np.float32, copy=False)
        if multi_output_goldens is None:
            tc.golden_tensors = [golden.contiguous() if isinstance(golden, _torch.Tensor) else golden]
        else:
            converted_goldens = []
            for output_position, output_golden in enumerate(multi_output_goldens):
                if output_golden is None:
                    output_golden = _torch.zeros(tc.tensor_view_shapes[
                        tc.output_tensor_indexes[output_position]])
                if output_position < len(out_dtypes):
                    output_dtype = dtype_map.get(out_dtypes[output_position], None)
                    if output_dtype is not None and use_torch:
                        output_golden = output_golden.to(output_dtype)
                converted_goldens.append(output_golden.contiguous())
            tc.golden_tensors = converted_goldens[:len(tc.output_tensor_indexes)]
        # Format third_party to match golden (list of tensors/arrays)
        third_parties_list = None
        if third_party is not None:
            if isinstance(third_party, (list, tuple)):
                tp_items = list(third_party)
            else:
                tp_items = [third_party]
            if target_dtype is not None and use_torch:
                tp_items = [tp.to(target_dtype) if hasattr(tp, 'to') else tp for tp in tp_items]
            third_parties_list = [tp.contiguous() if isinstance(tp, _torch.Tensor) else tp for tp in tp_items]
        del rank_goldens[did]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({__fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(f"Multi-device: rank dev={did} comparison FAILED: {cr.precision} metrics={cr.metrics}")
            else:
                logging.info(f"Multi-device: rank dev={did} comparison PASSED")
        except Exception:
            logging.exception(f"Multi-device: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def __golden_all_reduce_compare(thread_contexts, device_ids, all_precision, world_size):
    """AllReduce-type ops: each rank computes locally, then all_reduce(sum).

    Handles:
      - MatmulAllReduce: matmul(x1, x2) + bias -> all_reduce(SUM)
      - MatmulAllReduceV2: matmul(x1, x2) + bias + x3 -> all_reduce(SUM)
      - WeightQuantMatmulAllReduce: dequant(int8_weight, scale) -> matmul(x1, weight_f) -> all_reduce(SUM)
      - QuantMatmulAllReduce: matmul(x1_i32, x2_i32) + bias) * dequant_scale -> all_reduce(SUM)
    """
    import torch

    first_ctx = next(iter(thread_contexts.values()))
    api_name = first_ctx.api_name
    attrs = first_ctx.attributes

    is_weight_quant = "WeightQuantMatmulAllReduce" in api_name
    is_quant_matmul = ("QuantMatmulAllReduce" in api_name
                       and "Weight" not in api_name)
    is_v2 = "AllReduceV2" in api_name or "AllReduceV3" in api_name

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]

        if is_weight_quant:
            x1_f = _to_torch_f32(x1)
            x2_f = _to_torch_f32(x2)
            antiquant_scale = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            antiquant_offset = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
            aq_scale_f = _to_torch_f32(antiquant_scale) if antiquant_scale is not None else None
            aq_offset_f = (_to_torch_f32(antiquant_offset)
                           if antiquant_offset is not None
                           and isinstance(antiquant_offset, torch.Tensor)
                           and antiquant_offset.numel() > 0 else None)
            group_size = int(attrs.get('antiquantGroupSize', 0))
            if group_size > 0 and aq_scale_f is not None:
                aq_scale_f = aq_scale_f.repeat_interleave(group_size, dim=0)
                if aq_offset_f is not None:
                    aq_offset_f = aq_offset_f.repeat_interleave(group_size, dim=0)
            if aq_offset_f is not None:
                weight_deq = (x2_f + aq_offset_f) * aq_scale_f
            elif aq_scale_f is not None:
                weight_deq = x2_f * aq_scale_f
            else:
                weight_deq = x2_f
            mm_out = torch.matmul(x1_f, weight_deq)
            bias = None
        elif is_quant_matmul:
            is_v4_v5 = "QuantMatmulAllReduceV4" in api_name or "QuantMatmulAllReduceV5" in api_name
            if is_v4_v5:
                x1_f = _to_torch_f32(x1)
                x2_f = _to_torch_f32(x2)
                mm_out = torch.matmul(x1_f, x2_f)
                x1scale = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                x2scale = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
                if x1scale is not None:
                    x1s_f = _to_torch_f32(x1scale)
                    if x1s_f.dim() == 1 and mm_out.dim() == 2:
                        x1s_f = x1s_f.unsqueeze(-1)
                    elif x1s_f.dim() == 1 and mm_out.dim() == 3:
                        x1s_f = x1s_f.unsqueeze(0).unsqueeze(-1)
                    mm_out = mm_out * x1s_f
                if x2scale is not None:
                    x2s_f = _to_torch_f32(x2scale)
                    if x2s_f.dim() == 1 and x2s_f.numel() == 1:
                        pass
                    elif x2s_f.dim() == 2 and x2s_f.shape[0] == 1:
                        pass
                    mm_out = mm_out * x2s_f
                ds = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
                if ds is not None:
                    ds_f = _to_torch_f32(ds)
                    if ds_f.dim() == 1 and mm_out.dim() >= 2:
                        ds_f = ds_f.unsqueeze(0)
                    mm_out = mm_out * ds_f
                x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                if x3 is not None and hasattr(x3, 'numel') and x3.numel() > 0:
                    mm_out = mm_out + _to_torch_f32(x3)
            else:
                x1_f = _to_torch_f32(x1)
                x2_f = _to_torch_f32(x2)
                mm_out = torch.matmul(x1_f, x2_f)
                is_v2_quant = "QuantMatmulAllReduceV2" in api_name
                if is_v2_quant:
                    ds = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                    if ds is not None:
                        ds_f = _to_torch_f32(ds)
                        mm_out = mm_out * ds_f
                    pt = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
                    if pt is not None and isinstance(pt, torch.Tensor) and pt.numel() > 0:
                        pt_f = _to_torch_f32(pt)
                        if pt_f.dim() == 1:
                            pt_f = pt_f.unsqueeze(1)
                        mm_out = mm_out * pt_f
                    x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                    if x3 is not None and isinstance(x3, torch.Tensor) and x3.numel() > 0:
                        mm_out = mm_out + _to_torch_f32(x3)
                else:
                    ds = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
                    if ds is not None:
                        ds_f = _to_torch_f32(ds)
                        mm_out = mm_out * ds_f
                    bias = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                    if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
                        bias = None
                    if bias is not None:
                        mm_out = mm_out + _to_torch_f32(bias)
                    if "V3" in api_name and len(tc.flatten_tensors) > 5:
                        pt = tc.flatten_tensors[5]
                        if pt is not None and isinstance(pt, torch.Tensor) and pt.numel() > 0:
                            pt_f = _to_torch_f32(pt)
                            if pt_f.dim() == 1:
                                pt_f = pt_f.unsqueeze(1)
                            mm_out = mm_out * pt_f
        else:
            bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
                bias = None
            x1_f = _to_torch_f32(x1)
            x2_f = _to_torch_f32(x2)
            mm_out = torch.matmul(x1_f, x2_f)
            if bias is not None:
                mm_out = mm_out + _to_torch_f32(bias)
            if is_v2:
                x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                if x3 is not None and isinstance(x3, torch.Tensor) and x3.numel() > 0:
                    mm_out = mm_out + _to_torch_f32(x3)

        x1_dtype = x1.dtype if hasattr(x1, 'dtype') else None
        if x1_dtype is not None and x1_dtype in (torch.bfloat16, torch.float16):
            mm_out = mm_out.to(x1_dtype).float()
        local_results[did] = mm_out

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    rank_third_parties = None
    try:
        from .hccl_cascade import run_matmul_allreduce_cascade
        t_x1 = False
        t_x2 = __get_transpose_flags(first_ctx)
        if is_weight_quant:
            t_x2 = False
        is_bias_flag = False
        for did in device_ids:
            tc = thread_contexts[did]
            if is_weight_quant:
                break
            if is_quant_matmul:
                b = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
            else:
                b = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            if b is not None and isinstance(b, torch.Tensor) and b.numel() > 0:
                is_bias_flag = True
                break
        cascade_outs = run_matmul_allreduce_cascade(
            thread_contexts, device_ids,
            transpose_x1=t_x1, transpose_x2=t_x2, is_bias=is_bias_flag)
        rank_third_parties = {did: [cascade_outs[did]['main']] for did in device_ids}
        logging.info("MatmulAllReduce: real HCCL cascade succeeded")
    except Exception:
        logging.exception("MatmulAllReduce: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    rank_goldens = {did: total for did in device_ids}
    __apply_goldens_and_compare(thread_contexts, device_ids, rank_goldens, all_precision,
                                rank_third_parties=rank_third_parties)


def __get_transpose_flags(first_ctx):
    attrs = first_ctx.attributes
    tx2 = attrs.get('transposeX2', attrs.get('isTransB', attrs.get('is_trans_b', None)))
    if tx2 is not None:
        return bool(tx2)
    remark = first_ctx.remark or ''
    for part in remark.split(','):
        kv = part.split('=', 1)
        if len(kv) == 2 and kv[0].strip() == 'is_trans_b':
            try:
                return bool(int(kv[1].strip()))
            except ValueError:
                pass
    return False


def __golden_reduce_scatter_compare(thread_contexts, device_ids, all_precision, world_size):
    import torch
    first_ctx = next(iter(thread_contexts.values()))

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        x1_f = _to_torch_f32(x1)
        x2_f = _to_torch_f32(x2)
        mm_out = torch.matmul(x1_f, x2_f)
        if bias is not None:
            mm_out = mm_out + _to_torch_f32(bias)
        local_results[did] = mm_out
    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    M = total.shape[0]
    chunk_m = M // world_size
    rank_goldens = {}
    for idx, did in enumerate(device_ids):
        rank_goldens[did] = total[idx * chunk_m : (idx + 1) * chunk_m, :].contiguous()
    del total

    # 真·小算子级联 third_party（参考 mc2_test get_hccl_mm）
    # matmul(x1, x2) -> reduce_scatter(SUM) -> output
    rank_third_parties = None
    try:
        from .hccl_cascade import run_matmul_reducescatter_cascade
        # 从 remark 解析 is_trans_b
        remark = first_ctx.remark or ''
        is_trans_b = False
        for part in remark.split(','):
            kv = part.split('=', 1)
            if len(kv) == 2 and kv[0].strip() == 'is_trans_b':
                is_trans_b = (kv[1].strip() == '1')
        cascade_outs = run_matmul_reducescatter_cascade(
            thread_contexts, device_ids, is_trans_b=is_trans_b)
        rank_third_parties = {did: [cascade_outs[did]] for did in device_ids}
        logging.info("MatmulReduceScatter: real HCCL cascade succeeded")
    except Exception:
        logging.exception("MatmulReduceScatter: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    __apply_goldens_and_compare(thread_contexts, device_ids, rank_goldens, all_precision,
                                rank_third_parties=rank_third_parties)


def __golden_all_gather_compare(thread_contexts, device_ids, all_precision, world_size):
    import torch
    import numpy as np
    first_ctx = next(iter(thread_contexts.values()))
    api_name = first_ctx.api_name or ''
    is_v2 = 'V2' in api_name or 'v2' in api_name

    # Detect V2 quant modes from remark/dtypes
    remark = first_ctx.remark or ''
    per_block_flag = False
    is_mxfp = False
    is_bias = False
    gather_output = len(first_ctx.output_tensor_indexes or ()) > 1
    is_trans_b = False
    for part in remark.split(','):
        kv = part.split('=', 1)
        if len(kv) == 2:
            k, v = kv[0].strip(), kv[1].strip()
            if k == 'per_block_flag':
                per_block_flag = (v.lower() in ('1', 'true'))
            elif k == 'is_trans_b':
                is_trans_b = (v == '1')
            elif k == 'is_mxfp':
                is_mxfp = (v.lower() in ('1', 'true'))
            elif k == 'is_bias':
                is_bias = (v == '1')
            elif k == 'gather_output':
                gather_output = (v == '1')

    # Determine quant mode from x1 dtype
    x1_dtype_str = (first_ctx.flat_tensor_dtypes[0] if first_ctx.flat_tensor_dtypes else '')
    is_quant = x1_dtype_str in ('fp8_e4m3fn', 'fp8_e5m2', 'hif8',
                                 'float8_e4m3fn', 'float8_e5m2', 'hifloat8')

    # Override mxfp detection by x1scale dtype (slot 3 in V2)
    if is_v2 and is_quant and len(first_ctx.flat_tensor_dtypes) > 3:
        x1s_dtype = first_ctx.flat_tensor_dtypes[3]
        if x1s_dtype in ('fp8_e8m0', 'float8_e8m0'):
            is_mxfp = True

    def _to_torch_f32(t):
        """Convert torch.Tensor or numpy.ndarray to float32 torch.Tensor."""
        if t is None:
            return None
        if isinstance(t, np.ndarray):
            return torch.from_numpy(t.astype(np.float32, copy=False))
        return t.float()

    def _to_torch_keep_dtype_npu_scale(t):
        """For mxfp/per_block e8m0 scales: keep as uint8 view (1 byte), NOT float32.
        npu_quant_matmul expects e8m0 scale (1 byte), float32 (4 byte) causes dtype mismatch.
        Mimics mc2_test common.py:332 — e8m0 via view(uint8)."""
        if t is None:
            return None
        if isinstance(t, np.ndarray):
            dtype_name = str(t.dtype)
            if 'e8m0' in dtype_name:
                return torch.from_numpy(t.view(np.uint8))
            return torch.from_numpy(t.astype(np.float32, copy=False))
        # torch.Tensor — e8m0 may already be uint8 view; keep dtype, don't .float()
        dtype_name = str(t.dtype)
        if 'e8m0' in dtype_name or (t.dtype == torch.uint8):
            return t
        return t.float()

    # Gather x1 across ranks
    all_x1 = []
    x2_per_rank = {}
    bias_per_rank = {}
    x1scale_per_rank = {}
    x2scale_per_rank = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        all_x1.append(_to_torch_f32(x1))
        x2_per_rank[did] = _to_torch_f32(x2)
        # Both AllGatherMatmul variants add the optional bias after matmul.
        if len(tc.flatten_tensors) > 2 and tc.flatten_tensors[2] is not None:
            bias_per_rank[did] = _to_torch_f32(tc.flatten_tensors[2])
        else:
            bias_per_rank[did] = None
        # V2 scales at slot 3 (x1Scale) and slot 4 (x2Scale)
        if is_v2 and is_quant:
            if len(tc.flatten_tensors) > 3 and tc.flatten_tensors[3] is not None:
                # mxfp/per_block: e8m0 scale must keep 1-byte dtype (uint8 view), not float32
                if is_mxfp or per_block_flag:
                    x1scale_per_rank[did] = _to_torch_keep_dtype_npu_scale(tc.flatten_tensors[3])
                else:
                    x1scale_per_rank[did] = _to_torch_f32(tc.flatten_tensors[3])
            if len(tc.flatten_tensors) > 4 and tc.flatten_tensors[4] is not None:
                if is_mxfp or per_block_flag:
                    x2scale_per_rank[did] = _to_torch_keep_dtype_npu_scale(tc.flatten_tensors[4])
                else:
                    x2scale_per_rank[did] = _to_torch_f32(tc.flatten_tensors[4])
    gathered = torch.cat(all_x1, dim=0)
    del all_x1

    # all_gather x1scale for per_block / mxfp (concat along dim 0)
    if is_v2 and is_quant and (per_block_flag or is_mxfp) and x1scale_per_rank:
        all_x1s = [x1scale_per_rank[did] for did in device_ids if did in x1scale_per_rank]
        if all_x1s:
            gathered_x1scale = torch.cat(all_x1s, dim=0)
        else:
            gathered_x1scale = None
    else:
        gathered_x1scale = None

    rank_goldens = {}
    for did in device_ids:
        x2_f = x2_per_rank[did]
        bias = bias_per_rank.get(did)

        if not is_quant:
            # Non-quant: matmul(gathered, x2) + bias
            golden = torch.matmul(gathered, x2_f)
            if bias is not None:
                golden = golden + bias
        else:
            x1s = x1scale_per_rank.get(did)
            x2s = x2scale_per_rank.get(did)
            if per_block_flag:
                # per_block: use gathered x1scale (all_gather along dim 0)
                gs = gathered_x1scale
                group_size = first_ctx.attributes.get('groupSize', 0)
                golden = __per_block_cpu_compute(group_size, gathered, x2_f, gs, x2s)
            elif is_mxfp:
                gs = gathered_x1scale
                # CSV convention: x2 is already [K, N] (pre-transposed for
                # trans_b=1, matching aclnn op_api which sees is_trans_b=false).
                # mxfp_cpu_compute expects x2 as [K, N] — use directly.
                x2_for_golden = x2_f
                golden = __mxfp_cpu_compute(gathered.numpy().astype(np.float32),
                                            x2_for_golden.numpy().astype(np.float32),
                                            gs.numpy().astype(np.float32),
                                            x2s.numpy().astype(np.float32))
                golden = torch.from_numpy(golden)
                if bias is not None:
                    golden = golden + bias
            else:
                # per_tensor: matmul then scale_generate(x1scale * x2scale)
                golden = torch.matmul(gathered, x2_f)
                if bias is not None:
                    golden = golden + bias
                double_scale = __scale_generate((x1s.numpy() * x2s.numpy()))
                double_scale_t = torch.unsqueeze(torch.from_numpy(double_scale), dim=1).float()
                golden = golden * double_scale_t
        rank_goldens[did] = {
            'main': golden.contiguous(),
            'gather': gathered.contiguous(),
        }
        del golden
    # NPU cascaded golden (third_party for cross_check): mimics mc2_test get_hccl_mm
    # — all_gather x1 then matmul.
    # Non-quant: bf16 matmul (dtype-native accumulation) + bias (CPU).
    # Quant: torch_npu.npu_quant_matmul on NPU (fp8 dequant + matmul precision).
    # For multi-output cases (gather_output=1), provide third_party for each output.
    rank_third_parties = {}
    out_idxs = first_ctx.output_tensor_indexes or (0,)
    # Gather x1 across ranks (all_gather equivalent)
    def _to_torch_keep_dtype(arr):
        """Convert numpy array to torch tensor, preserving fp8/hif8/e8m0 dtypes via view(uint8).
        Mimics mc2_test common.py:324-337."""
        if isinstance(arr, torch.Tensor):
            return arr
        dtype_name = str(arr.dtype)
        if 'e4m3' in dtype_name:
            return torch.from_numpy(arr.astype(np.float32)).to(torch.float8_e4m3fn)
        elif 'e5m2' in dtype_name:
            return torch.from_numpy(arr.astype(np.float32)).to(torch.float8_e5m2)
        elif 'e8m0' in dtype_name:
            return (torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e8m0)
                    if hasattr(torch, 'float8_e8m0') else torch.from_numpy(arr.astype(np.float32)))
        elif 'hifloat8' in dtype_name or 'hif8' in dtype_name:
            # hif8: view as uint8 (mc2_test common.py:332), keep as uint8 view for CPU golden
            return torch.from_numpy(arr.view(np.uint8))
        else:
            return torch.from_numpy(arr)
    all_x1_orig = [_to_torch_keep_dtype(thread_contexts[d].flatten_tensors[0]) for d in device_ids]
    gathered_orig = torch.cat(all_x1_orig, dim=0)
    del all_x1_orig
    tc0 = thread_contexts[device_ids[0]]
    x2_orig = tc0.flatten_tensors[1]
    bias_orig = tc0.flatten_tensors[2] if len(tc0.flatten_tensors) > 2 else None
    if bias_orig is not None and isinstance(bias_orig, torch.Tensor) and bias_orig.numel() == 0:
        bias_orig = None

    if not is_quant:
        # 真·小算子级联 third_party（参考 mc2_test get_hccl_mm）
        # all_gather(x1) -> matmul(gathered, x2) -> output
        # 仅非量化路径走真级联；量化路径保留 __npu_quant_matmul_cascade
        try:
            from .hccl_cascade import run_allgather_matmul_cascade
            cascade_outs = run_allgather_matmul_cascade(
                thread_contexts, device_ids,
                is_trans_b=is_trans_b, is_gather_output=gather_output)
            # cascade_outs[did] = {'main': tensor, 'gather': tensor|None}
            rank_third_parties = {}
            for did in device_ids:
                tp_list = [cascade_outs[did]['main']]
                if gather_output and cascade_outs[did].get('gather') is not None:
                    tp_list.append(cascade_outs[did]['gather'])
                rank_third_parties[did] = tp_list
            logging.info("AllGatherMatmul: real HCCL cascade succeeded")
        except Exception:
            logging.exception("AllGatherMatmul: real HCCL cascade failed, no third_party")
            rank_third_parties = None
    else:
        # Quant: 真HCCL all_gather(x1) + [all_gather(x1scale)] + npu_quant_matmul
        try:
            from .hccl_cascade import run_allgather_quant_matmul_v2_cascade
            cascade_outs = run_allgather_quant_matmul_v2_cascade(
                thread_contexts, device_ids,
                is_trans_b=is_trans_b, is_bias=(bias_orig is not None),
                is_mxfp=is_mxfp, per_block_flag=per_block_flag,
                is_gather_output=gather_output)
            rank_third_parties = {}
            for did in device_ids:
                tp_list = [cascade_outs[did]['main']]
                if gather_output and cascade_outs[did].get('gather') is not None:
                    tp_list.append(cascade_outs[did]['gather'])
                rank_third_parties[did] = tp_list
            logging.info("AllGatherMatmulV2 quant: real HCCL cascade succeeded")
        except Exception:
            logging.exception("AllGatherMatmulV2 quant: real HCCL cascade failed, no third_party")
            rank_third_parties = None
    del gathered_orig
    if 'cas_mm' in dir():
        del cas_mm
    del gathered, x2_per_rank
    __apply_goldens_and_compare(thread_contexts, device_ids, rank_goldens, all_precision,
                                rank_third_parties=rank_third_parties)


def __npu_quant_matmul_cascade(gathered_x1, x2, bias, x1scale_per_rank, x2scale_per_rank,
                                gathered_x1scale, device_ids, is_mxfp, per_block_flag,
                                out_dtypes, first_ctx):
    """Run npu_quant_matmul on NPU as cascaded golden (third_party for cross_check).

    Mimics mc2_test get_hccl_mm quant path: all_gather x1 then npu_quant_matmul.
    NPU device was reset after profiling; re-init with torch_npu here.
    """
    import torch
    import torch_npu
    import numpy as np
    import ctypes

    dev_id = device_ids[0]
    # NPU device was reset after profiling; re-init device context for npu_quant_matmul.
    # Mimics mc2_test common.py:98 — torch_npu.npu.set_device + ensure stream pool.
    try:
        acl_dll = ctypes.CDLL("libascendcl.so")
        acl_dll.aclrtSetDevice.argtypes = [ctypes.c_int32]
        acl_dll.aclrtSetDevice.restype = ctypes.c_int32
        acl_dll.aclrtSetDevice(ctypes.c_int32(dev_id))
    except Exception:
        pass
    try:
        torch.npu.set_device(dev_id)
        # Force torch_npu to (re-)initialize its stream pool for the new context,
        # otherwise npu_quant_matmul reuses a stale stream from the reset context.
        _ = torch.zeros(1, device=f'npu:{dev_id}')
    except Exception as e:
        logging.warning(f"npu re-init warmup failed: {e}")

    def _to_npu_fp8(arr):
        """Convert fp8/hif8 numpy array to NPU torch tensor, preserving dtype.
        Mimics mc2_test common.py:324-337 — fp8 via float32->torch dtype,
        e8m0/hif8 via view(uint8)->NPU."""
        if isinstance(arr, torch.Tensor):
            return arr.npu()
        dtype_name = str(arr.dtype)
        if 'e4m3' in dtype_name:
            t = torch.from_numpy(arr.astype(np.float32)).to(torch.float8_e4m3fn)
        elif 'e5m2' in dtype_name:
            t = torch.from_numpy(arr.astype(np.float32)).to(torch.float8_e5m2)
        elif 'e8m0' in dtype_name:
            # e8m0: view as uint8 -> NPU (mc2_test common.py:332)
            t = torch.from_numpy(arr.view(np.uint8)).npu()
            if hasattr(torch, 'float8_e8m0'):
                t = t.view(torch.float8_e8m0)
            return t
        elif 'hifloat8' in dtype_name or 'hif8' in dtype_name:
            # hif8: view as uint8 -> NPU (mc2_test common.py:332), keep uint8;
            # dtype enum (290) is passed via x1_dtype kwarg to npu_quant_matmul, not via tensor view.
            t = torch.from_numpy(arr.view(np.uint8)).npu()
            return t
        else:
            t = torch.from_numpy(arr.astype(np.float32))
        return t.npu()

    def _to_npu_scale(val, dtype_enum=None):
        """Convert scale to NPU tensor, preserving e8m0/fp32 dtype for npu_quant_matmul.
        For e8m0 (dtype_enum=293 or uint8 from _to_torch_keep_dtype_npu_scale), view as float8_e8m0."""
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            t = val.npu()
            # uint8 + e8m0 enum -> view as float8_e8m0 (mc2_test passes e8m0 tensor to npu_quant_matmul)
            if (dtype_enum == 293 or val.dtype == torch.uint8) and hasattr(torch, 'float8_e8m0'):
                t = t.view(torch.float8_e8m0)
            return t
        dtype_name = str(val.dtype)
        if 'e8m0' in dtype_name:
            t = torch.from_numpy(val.view(np.uint8)).npu()
            if hasattr(torch, 'float8_e8m0'):
                t = t.view(torch.float8_e8m0)
            return t
        return torch.from_numpy(val.astype(np.float32)).npu()

    # Resolve dtype enums for npu_quant_matmul, mimicking mc2_test DTYPE_ENUM (func.py:95).
    # Most dtypes map to None (op infers from tensor), only e8m0=293 / hif8=290 need explicit enum.
    dtype_enum_map = {
        'fp8_e8m0': 293, 'float8_e8m0': 293,
        'hif8': 290, 'hifloat8': 290,
    }
    flat_dtypes = list(first_ctx.flat_tensor_dtypes or [])
    def _enum_at(idx):
        if idx >= len(flat_dtypes):
            return None
        return dtype_enum_map.get(flat_dtypes[idx], None)
    # V2 flat order: [x1, x2, bias, x1scale, x2scale, ...]
    x1_dtype_enum = _enum_at(0)
    x2_dtype_enum = _enum_at(1)
    pertoken_scale_dtype_enum = _enum_at(3)
    scale_dtype_enum = _enum_at(4)

    # Determine output dtype
    out_dtype_str = out_dtypes[0] if out_dtypes else 'bfloat16'
    out_dtype_map = {
        'float16': torch.float16, 'fp16': torch.float16,
        'float32': torch.float32, 'fp32': torch.float32,
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
    }
    out_dtype = out_dtype_map.get(out_dtype_str, torch.bfloat16)

    # Prepare inputs on NPU (fp8 tensors)
    x1_npu = _to_npu_fp8(gathered_x1)
    x2_npu = _to_npu_fp8(x2)

    # Scales: per_tensor uses per-rank scale; mxfp/per_block use gathered scale
    did0 = device_ids[0]
    if is_mxfp or per_block_flag:
        x1s = gathered_x1scale
        x2s = x2scale_per_rank.get(did0)
    else:
        x1s = x1scale_per_rank.get(did0)
        x2s = x2scale_per_rank.get(did0)

    x1s_npu = _to_npu_scale(x1s, pertoken_scale_dtype_enum)
    x2s_npu = _to_npu_scale(x2s, scale_dtype_enum)
    bias_npu = _to_npu_scale(bias)

    try:
        logging.info(f"npu_quant_matmul cascade: x1={x1_npu.shape}/{x1_npu.dtype} x2={x2_npu.shape}/{x2_npu.dtype} "
                      f"x1s={x1s_npu.shape if x1s_npu is not None else None} "
                      f"x2s={x2s_npu.shape if x2s_npu is not None else None} "
                      f"bias={bias_npu.shape if bias_npu is not None else None} out_dtype={out_dtype}")
        # Build kwargs, only pass non-None dtype enums (mc2_test passes None for most dtypes).
        npu_kwargs = dict(
            scale=x2s_npu,
            pertoken_scale=x1s_npu,
            bias=bias_npu,
            output_dtype=out_dtype,
            offset=None,
            y_scale=None,
        )
        if x1_dtype_enum is not None:
            npu_kwargs['x1_dtype'] = x1_dtype_enum
        if x2_dtype_enum is not None:
            npu_kwargs['x2_dtype'] = x2_dtype_enum
        if pertoken_scale_dtype_enum is not None:
            npu_kwargs['pertoken_scale_dtype'] = pertoken_scale_dtype_enum
        if scale_dtype_enum is not None:
            npu_kwargs['scale_dtype'] = scale_dtype_enum
        if is_mxfp:
            # mc2_test passes group_sizes for mxfp (func.py:217)
            npu_kwargs['group_sizes'] = None
        cas_mm = torch_npu.npu_quant_matmul(x1_npu, x2_npu, **npu_kwargs)
        cas_mm = cas_mm.cpu().float()
    except Exception as e:
        logging.error(f"npu_quant_matmul cascade failed: {e}")
        cas_mm = torch.zeros(gathered_x1.shape[0], 1, dtype=torch.float32)

    del x1_npu, x2_npu, x1s_npu, x2s_npu, bias_npu
    torch.npu.empty_cache()
    return cas_mm


def __scale_generate(fp32_deq_scale):
    """Apply high-19-bit mask to fp32 scale (simulates hardware)."""
    import numpy as np
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    uint32_deq_scale &= 0xFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
    return fp32_deq_scale


def __unpack_group_size(group_size):
    """Unpack int64 group_size to (M, N, K) tuple."""
    if group_size == -1 or group_size == 0:
        return 0, 0, 0
    gsm = (group_size >> 32) & 0xFFFF
    gsn = (group_size >> 16) & 0xFFFF
    gsk = group_size & 0xFFFF
    return gsm, gsn, gsk


def __per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale):
    """Per-block quantized matmul golden (mirrors mc2_test func.per_block_cpu_compute).

    For 2D inputs only (M, K) x (K, N) with block scales (M/gsm, K/gsk) x (K/gsk, N/gsn).
    """
    import torch
    gsm, gsn, gsk = __unpack_group_size(group_size)
    if gsm == 0 or gsn == 0 or gsk == 0:
        # fallback: treat as per-tensor
        out = torch.matmul(x1, x2)
        if x1_scale is not None and x2_scale is not None:
            double_scale = __scale_generate((x1_scale.numpy() * x2_scale.numpy()))
            out = out * torch.from_numpy(double_scale).float()
        return out
    m = x1.shape[-2]
    k = x1.shape[-1]
    n = x2.shape[-1]
    out = torch.zeros(m, n)
    for m_idx in range((m + gsm - 1) // gsm):
        m_start = m_idx * gsm
        m_end = min((m_idx + 1) * gsm, m)
        for n_idx in range((n + gsn - 1) // gsn):
            n_start = n_idx * gsn
            n_end = min((n_idx + 1) * gsn, n)
            for k_idx in range((k + gsk - 1) // gsk):
                k_start = k_idx * gsk
                k_end = min((k_idx + 1) * gsk, k)
                block_out = torch.matmul(x1[m_start:m_end, k_start:k_end],
                                          x2[k_start:k_end, n_start:n_end]) * \
                            x1_scale[m_idx, k_idx] * x2_scale[k_idx, n_idx]
                out[m_start:m_end, n_start:n_end] += block_out
    return out


def __mxfp_cpu_compute(x1, x2, x1scale, x2scale):
    """MXFP quantized matmul golden (mirrors mc2_test common.mxfp_cpu_compute).

    x1scale/x2scale are 3D from ACLNN path. Flatten to 2D [M, K/32] / [K/32, N].
    x1scale 3D layout [M, K/64, 2] -> reshape(M, -1) = [M, K/32] (row-major
      over (k_block, pair), correct because M is leading axis).
    x2scale 3D layout depends on is_trans_b convention:
      - [K/64, N, 2] (CSV is_trans_b=false / pre-transposed trans_b=1):
        transpose(0, 2, 1) -> [K/64, 2, N] -> reshape(K/32, N) so that
        flat[k_block_3d*2 + pair, n] = scale_3d[k_block_3d, n, pair].
      - [N, K/64, 2] (alt layout): reshape(N, -1) = [N, K/32] then transpose.
    Then repeat each scale 32x along K axis to expand to full [M, K] / [K, N].
    """
    import numpy as np
    import torch
    # Flatten 3D scale [M, K/64, 2] -> [M, K/32]
    if x1scale.ndim == 3:
        x1scale = x1scale.reshape(x1scale.shape[0], -1)
    if x2scale.ndim == 3:
        # Determine layout by checking which dim equals N (x2.shape[-1]).
        n_dim = x2.shape[-1]
        if x2scale.shape[1] == n_dim:
            # [K/64, N, 2] layout: transpose (0,2,1) -> [K/64, 2, N] -> [K/32, N]
            x2scale = np.transpose(x2scale, (0, 2, 1)).reshape(-1, n_dim)
        else:
            # [N, K/64, 2] layout: reshape(N, -1) = [N, K/32]
            x2scale = x2scale.reshape(x2scale.shape[0], -1)
    # x2scale expected [K/32, N]; if shape is [N, K/32] (alt layout), transpose
    if x2scale.shape[0] != x1scale.shape[1] and x2scale.shape[1] == x1scale.shape[1]:
        x2scale = x2scale.T

    repeated_x1s = np.repeat(x1scale, 32, axis=-1)
    repeated_x2s = np.repeat(x2scale, 32, axis=-2)
    x1_pad_len = repeated_x1s.shape[-1] - x1.shape[-1]
    x2_pad_len = repeated_x2s.shape[-2] - x2.shape[-2]
    if x1_pad_len > 0:
        x1 = np.pad(x1, [(0, 0)] * (x1.ndim - 1) + [(0, x1_pad_len)], mode='constant')
    if x2_pad_len > 0:
        x2 = np.pad(x2, [(0, 0)] * (x2.ndim - 2) + [(0, x2_pad_len), (0, 0)], mode='constant')
    out = np.matmul(x1 * repeated_x1s, x2 * repeated_x2s)
    return out


def __grouped_matmul_cpu(gmm_x, gmm_weight, group_list):
    import torch
    import numpy as np
    B_list = list(torch.unbind(gmm_weight, dim=0))
    A_groups = torch.split(gmm_x, group_list, dim=0)
    results = []
    for i in range(len(group_list)):
        a = A_groups[i].numpy()
        b = B_list[i].numpy()
        results.append(torch.from_numpy(np.matmul(a, b)))
    return torch.cat(results, dim=0)


def __get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws):
    attrs = first_ctx.attributes
    exp_per_card = first_ctx.tensor_view_shapes[1][0] if len(first_ctx.tensor_view_shapes) > 1 else 1
    seed_val = 0
    remark = first_ctx.remark or ''
    for part in remark.split(','):
        kv = part.split('=', 1)
        if len(kv) == 2 and kv[0].strip() == 'seed':
            try:
                seed_val = int(kv[1].strip())
            except ValueError:
                pass
    bsk = first_ctx.tensor_view_shapes[0][0] if first_ctx.tensor_view_shapes else 0
    A_array = [bsk] * ep_ws
    return __generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)


def __get_gmm_send_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws):
    group_list = []
    for j in range(exp_per_card):
        total = sum(expTokenNums[rank_idx][r * exp_per_card + j] for r in range(ep_ws))
        group_list.append(total)
    return group_list


def __get_gmm_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws):
    group_list = []
    for j in range(exp_per_card):
        total = sum(expTokenNums[i][rank_idx * exp_per_card + j] for i in range(ep_ws))
        group_list.append(total)
    return group_list


def __unpermute_gmm_alltoallv(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    import numpy as np
    send_gl = __get_gmm_send_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws)
    expert_offsets = np.concatenate([[0], np.cumsum(send_gl[:-1])])
    my_row = expTokenNums[rank_idx]
    per_expert_cumsum = np.zeros((exp_per_card, ep_ws), dtype=np.int64)
    for j in range(exp_per_card):
        for r in range(ep_ws):
            per_expert_cumsum[j][r] = my_row[r * exp_per_card + j]
    per_expert_cumsum = np.cumsum(per_expert_cumsum, axis=1)
    all_indices = []
    for r in range(ep_ws):
        for j in range(exp_per_card):
            start = int(per_expert_cumsum[j][r - 1]) if r > 0 else 0
            end = int(per_expert_cumsum[j][r])
            all_indices.extend(range(int(expert_offsets[j]) + start, int(expert_offsets[j]) + end))
    if len(all_indices) == 0:
        return tokens.clone()
    idx_tensor = torch.tensor(all_indices, dtype=torch.long)
    return tokens.index_select(0, idx_tensor).to(tokens.dtype)


def __permute_alltoallv_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    import numpy as np
    indices = np.zeros((exp_per_card, ep_ws), dtype=np.int64)
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = int(expTokenNums[i][j + (exp_per_card * rank_idx)])
    trans = indices.T
    flaten = trans.reshape(-1)
    cumsum = np.cumsum(flaten)
    all_indices = []
    for e in range(exp_per_card):
        exp_token = []
        for r in range(ep_ws):
            flat_idx = e + r * exp_per_card
            start = int(cumsum[flat_idx - 1]) if flat_idx > 0 else 0
            end = int(cumsum[flat_idx])
            exp_token.extend(range(start, end))
        all_indices.extend(exp_token)
    if len(all_indices) == 0:
        return tokens.clone()
    idx_tensor = torch.tensor(all_indices, dtype=torch.long)
    return tokens.index_select(0, idx_tensor)


def __simulate_alltoallv(all_inputs, device_ids, send_counts_per_rank, recv_counts_per_rank, ep_ws, exp_per_card):
    import torch
    rank_outputs = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        received_chunks = []
        offset = 0
        for src_did in device_ids:
            src_idx = list(device_ids).index(src_did)
            src_data = all_inputs[src_did]
            src_send = send_counts_per_rank[src_did]
            chunk_start = offset
            chunk_size = sum(src_send[target_idx * exp_per_card:(target_idx + 1) * exp_per_card])
            chunk = src_data[chunk_start:chunk_start + chunk_size]
            received_chunks.append(chunk)
            offset += chunk_size
        rank_outputs[target_did] = (torch.cat(received_chunks, dim=0) if received_chunks
                                    else torch.zeros(0, src_data.shape[1] if src_data.dim() > 1 else 0))
    return rank_outputs


def __unpermute_mc2(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    import numpy as np
    empty_arr = np.zeros((ep_ws, exp_per_card), dtype=np.int64)
    for i in range(ep_ws):
        for j in range(exp_per_card):
            empty_arr[i][j] = int(expTokenNums[i][rank_idx * exp_per_card + j])
    tmp1 = empty_arr.T
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2 = np.cumsum(sum_list1)
    offsets = [0] + sum_list2[:-1].tolist()
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for ei in range(exp_per_card):
        tmp = []
        for j in range(ep_ws):
            if j == 0:
                tmp.append(list(map(lambda x: x + offsets[ei], list(range(0, sum_list[ei][j])))))
            else:
                tmp.append(list(map(lambda x: x + offsets[ei], list(range(sum_list[ei][j - 1], sum_list[ei][j])))))
        indices_list.append(tmp)
    selected = []
    for i in range(ep_ws):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long)
            selected.append(tokens.index_select(dim=0, index=indices))
    return torch.cat(selected, dim=0).to(tokens.dtype)


def __golden_gmm_alltoallv(thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card):
    import torch
    import numpy as np
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    trans_gmm_weight = attrs.get('transGmmWeight', False)
    trans_mm_weight = attrs.get('transMmWeight', False)

    gmm_x_qm = int(attrs.get('gmmXQuantMode', 0))
    gmm_w_qm = int(attrs.get('gmmWeightQuantMode', 0))
    is_mxfp = (gmm_x_qm == 6 and gmm_w_qm == 6)
    is_tt = (gmm_x_qm == 1 or gmm_w_qm == 1)
    is_quant = is_mxfp or is_tt

    all_gmm_out = {}
    all_unpermuted = {}
    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        gmm_x = _to_torch_f32(tc.flatten_tensors[0])
        gmm_weight = _to_torch_f32(tc.flatten_tensors[1])
        if trans_gmm_weight:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
        recv_gl = __get_gmm_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws)
        if is_quant:
            gmm_x_scale = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            gmm_w_scale = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            if is_mxfp:
                xs_np = _e8m0_to_f32(gmm_x_scale)
                ws_np = _e8m0_to_f32(gmm_w_scale)
                if isinstance(xs_np, np.ndarray):
                    xs_np = torch.from_numpy(xs_np)
                if isinstance(ws_np, np.ndarray):
                    ws_np = torch.from_numpy(ws_np)
                xs_np = xs_np.numpy()
                ws_np = ws_np.numpy()
                gmm_out = _quant_grouped_matmul_cpu(
                    gmm_x, gmm_weight, recv_gl,
                    xs_np, ws_np, is_mxfp=True, is_tt=False)
            else:
                xs_f = _to_torch_f32(gmm_x_scale)
                ws_f = _to_torch_f32(gmm_w_scale)
                gmm_out = _quant_grouped_matmul_cpu(
                    gmm_x, gmm_weight, recv_gl,
                    xs_f, ws_f, is_mxfp=False, is_tt=True)
        else:
            gmm_out = __grouped_matmul_cpu(gmm_x, gmm_weight, recv_gl)
        all_gmm_out[did] = gmm_out
        all_unpermuted[did] = __unpermute_mc2(gmm_out, exp_per_card, ep_ws, rank_idx, expTokenNums)

    rank_goldens = {}
    for target_did in device_ids:
        tc = thread_contexts[target_did]
        target_rank = list(device_ids).index(target_did)
        N = all_unpermuted[device_ids[0]].shape[1] if all_unpermuted[device_ids[0]].dim() > 1 else 1

        output_splits = []
        for i in range(ep_ws):
            output_splits.append(sum(expTokenNums[i][target_rank * exp_per_card:(target_rank + 1) * exp_per_card]))

        input_splits_map = {}
        for src_did in device_ids:
            src_rank = list(device_ids).index(src_did)
            is_list = []
            for t in range(ep_ws):
                is_list.append(sum(expTokenNums[src_rank][t * exp_per_card:(t + 1) * exp_per_card]))
            input_splits_map[src_did] = is_list

        output_chunks = []
        for src_did in device_ids:
            src_rank = list(device_ids).index(src_did)
            src_unpermuted = all_unpermuted[src_did]
            is_list = input_splits_map[src_did]
            offset = 0
            for t in range(ep_ws):
                if t == target_rank:
                    chunk = src_unpermuted[offset:offset + is_list[t]]
                    output_chunks.append(chunk.clone())
                offset += is_list[t]

        main_golden = torch.cat(output_chunks, dim=0) if output_chunks else torch.zeros(0, N)
        rank_goldens[target_did] = {'main': main_golden}
        del output_chunks, main_golden

        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_x_f = _to_torch_f32(mm_x)
            mm_weight_f = _to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            rank_goldens[target_did]['mm'] = torch.mm(mm_x_f, mm_weight_f)
            del mm_x_f, mm_weight_f
        else:
            rank_goldens[target_did]['mm'] = None

    del all_gmm_out, all_unpermuted
    return rank_goldens


def __golden_alltoallv_gmm_with_counts(thread_contexts, device_ids, expTokenNums,
                                          ep_ws, exp_per_card, send_counts_map, recv_counts_map):
    import torch
    import numpy as np
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    trans_gmm_weight = attrs.get('transGmmWeight', False)
    trans_mm_weight = attrs.get('transMmWeight', False)
    permute_out_flag = attrs.get('permuteOutFlag', False)

    gmm_x_qm = int(attrs.get('gmmx_quant_mode', attrs.get('gmmXQuantMode', 0)))
    gmm_w_qm = int(attrs.get('gmmweight_quant_mode', attrs.get('gmmWeightQuantMode', 0)))
    is_mxfp = (gmm_x_qm == 6 and gmm_w_qm == 6)
    is_tt = (gmm_x_qm == 1 or gmm_w_qm == 1)
    is_quant = is_mxfp or is_tt

    rank_goldens = {}
    all_src_segments = {}
    for src_did in device_ids:
        tc = thread_contexts[src_did]
        src_rank = list(device_ids).index(src_did)
        src_x = _to_torch_f32(tc.flatten_tensors[0])
        src_send = send_counts_map[src_did]
        segments = {}
        offset = 0
        for t in range(ep_ws):
            cs = sum(src_send[t * exp_per_card:(t + 1) * exp_per_card])
            segments[t] = src_x[offset:offset + cs]
            offset += cs
        all_src_segments[src_did] = segments

    for target_did in device_ids:
        tc = thread_contexts[target_did]
        target_rank = list(device_ids).index(target_did)
        gmm_weight = _to_torch_f32(tc.flatten_tensors[1])
        if trans_gmm_weight:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
        N = gmm_weight.shape[2] if not trans_gmm_weight else gmm_weight.shape[1]

        recv = expTokenNums[target_rank]
        recv_by_rank = [sum(recv[r * exp_per_card:(r + 1) * exp_per_card]) for r in range(ep_ws)]
        recv_cumsum = list(np.cumsum(recv_by_rank))
        recv_offsets = [0] + recv_cumsum[:-1]

        gathered = torch.zeros(sum(recv_by_rank), src_x.shape[1] if src_x.dim() > 1 else 1)
        for src_did in device_ids:
            src_rank = list(device_ids).index(src_did)
            src_segment = all_src_segments[src_did][target_rank]
            base = recv_offsets[src_rank]
            gathered[base:base + src_segment.shape[0]] = src_segment

        group_list = __get_gmm_group_list(expTokenNums, target_rank, exp_per_card, ep_ws)
        grouped_parts = []
        for j in range(exp_per_card):
            expert_rows = []
            for r in range(ep_ws):
                start = recv_offsets[r] + sum(recv[r * exp_per_card + k] for k in range(j))
                count = recv[r * exp_per_card + j]
                expert_rows.append(gathered[start:start + count])
            grouped_parts.append(torch.cat(expert_rows, dim=0))
        stacked = torch.cat(grouped_parts, dim=0)
        del gathered, grouped_parts

        if is_quant:
            gmm_x_scale = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            gmm_w_scale = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            if is_mxfp:
                xs_np = _e8m0_to_f32(gmm_x_scale)
                ws_np = _e8m0_to_f32(gmm_w_scale)
                if isinstance(xs_np, np.ndarray):
                    xs_np = torch.from_numpy(xs_np)
                if isinstance(ws_np, np.ndarray):
                    ws_np = torch.from_numpy(ws_np)
                xs_np = xs_np.numpy()
                ws_np = ws_np.numpy()
                gmm_out = _quant_grouped_matmul_cpu(
                    stacked, gmm_weight, group_list,
                    xs_np, ws_np, is_mxfp=True, is_tt=False)
            else:
                xs_f = _to_torch_f32(gmm_x_scale)
                ws_f = _to_torch_f32(gmm_w_scale)
                gmm_out = _quant_grouped_matmul_cpu(
                    stacked, gmm_weight, group_list,
                    xs_f, ws_f, is_mxfp=False, is_tt=True)
        else:
            gmm_out = __grouped_matmul_cpu(stacked, gmm_weight, group_list)
        del gmm_weight

        rank_goldens[target_did] = {}
        rank_goldens[target_did]['main'] = gmm_out
        rank_goldens[target_did]['permute'] = stacked if permute_out_flag else None
        del gmm_out, stacked

        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_x_f = _to_torch_f32(mm_x)
            mm_weight_f = _to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            rank_goldens[target_did]['mm'] = torch.mm(mm_x_f, mm_weight_f)
            del mm_x_f, mm_weight_f
        else:
            rank_goldens[target_did]['mm'] = None

    del all_src_segments
    return rank_goldens


def __permute_a2a_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    indices = torch.zeros(exp_per_card, ep_ws).long()
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = expTokenNums[i][j + exp_per_card * rank_idx]
    trans = indices.permute(1, 0)
    flaten = trans.reshape(-1)
    sum_list = torch.cumsum(flaten, dim=0)
    tmp = []
    for i in range(len(sum_list)):
        if i == 0:
            tmp.append(range(0, sum_list[i]))
        else:
            tmp.append(range(sum_list[i - 1], sum_list[i]))
    parts = []
    expert_sizes = []
    for e in range(exp_per_card):
        exp_token = []
        for r in range(ep_ws):
            exp_token += list(tmp[e + r * exp_per_card])
        combined = torch.tensor(exp_token)
        parts.append(tokens.index_select(0, combined))
        expert_sizes.append(len(exp_token))
    K = tokens.shape[1] if tokens.dim() > 1 else 1
    result = torch.zeros(sum(expert_sizes), K, dtype=tokens.dtype)
    offset = 0
    for e in range(exp_per_card):
        result[offset:offset + expert_sizes[e]] = parts[e]
        offset += expert_sizes[e]
    return result, expert_sizes


def __golden_alltoallv_gmm(thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card):
    import torch
    import numpy as np
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    trans_gmm_weight = attrs.get('transGmmWeight', False)
    trans_mm_weight = attrs.get('transMmWeight', False)
    permute_out_flag = attrs.get('permuteOutFlag', False)

    all_a2a_inputs = {}
    all_send_segments = {}
    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        src_x = _to_torch_f32(tc.flatten_tensors[0])
        all_a2a_inputs[did] = src_x
        my_row = expTokenNums[rank_idx]
        segments = []
        offset = 0
        for t in range(ep_ws):
            cs = sum(my_row[t * exp_per_card:(t + 1) * exp_per_card])
            segments.append(src_x[offset:offset + cs])
            offset += cs
        all_send_segments[did] = segments

    a2a_outputs = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        output_splits = [
            sum(expTokenNums[i][target_idx * exp_per_card:(target_idx + 1) * exp_per_card])
            for i in range(ep_ws)
        ]
        recv_by_src = output_splits
        recv_cumsum = list(np.cumsum(recv_by_src))
        recv_offsets = [0] + recv_cumsum[:-1]
        K = all_a2a_inputs[device_ids[0]].shape[1] if all_a2a_inputs[device_ids[0]].dim() > 1 else 1
        gathered = torch.zeros(sum(recv_by_src), K)
        for src_did in device_ids:
            src_idx = list(device_ids).index(src_did)
            chunk = all_send_segments[src_did][target_idx]
            base = recv_offsets[src_idx]
            gathered[base:base + chunk.shape[0]] = chunk
        a2a_outputs[target_did] = gathered

    rank_goldens = {}
    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        gmm_weight = _to_torch_f32(tc.flatten_tensors[1])
        if trans_gmm_weight:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()

        a2a_out = a2a_outputs[did]
        permuted, expert_sizes = __permute_a2a_gmm(a2a_out, exp_per_card, ep_ws, rank_idx, expTokenNums)

        gmm_out = __grouped_matmul_cpu(permuted, gmm_weight, expert_sizes)
        del gmm_weight

        rank_goldens[did] = {}
        rank_goldens[did]['main'] = gmm_out.contiguous()
        rank_goldens[did]['permute'] = permuted.contiguous() if permute_out_flag else None
        del gmm_out, permuted

        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_x_f = _to_torch_f32(mm_x)
            mm_weight_f = _to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            mm_golden = torch.mm(mm_x_f, mm_weight_f)
            rank_goldens[did]['mm'] = mm_golden
            del mm_x_f, mm_weight_f, mm_golden
        else:
            rank_goldens[did]['mm'] = None

    del all_a2a_inputs, all_send_segments, a2a_outputs
    return rank_goldens


def __apply_gmm_goldens(thread_contexts, device_ids, rank_goldens, all_precision,
                          rank_third_parties=None):
    import torch as _torch
    dtype_map = {
        'float16': _torch.float16, 'fp16': _torch.float16,
        'float32': _torch.float32, 'fp32': _torch.float32,
        'bfloat16': _torch.bfloat16, 'bf16': _torch.bfloat16,
    }
    for did in device_ids:
        tc = thread_contexts[did]
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        goldens = rank_goldens[did]
        third_party = rank_third_parties[did] if rank_third_parties else None
        golden_list = []
        third_parties_list = None
        for out_idx in tc.output_tensor_indexes:
            shape = tc.tensor_view_shapes[out_idx]
            if shape is None or any(s is None for s in shape if isinstance(shape, (list, tuple))):
                golden_list.append(None)
                continue
            if out_idx == tc.output_tensor_indexes[0]:
                g = goldens['main']
            elif len(tc.output_tensor_indexes) > 1 and out_idx == tc.output_tensor_indexes[1]:
                g = goldens.get('mm')
                if g is None:
                    g = goldens.get('permute')
            elif len(tc.output_tensor_indexes) > 2 and out_idx == tc.output_tensor_indexes[2]:
                g = goldens.get('permute')
            else:
                g = _torch.zeros(shape)
            if g is None:
                g = _torch.zeros(shape)
            dt_idx = list(tc.output_tensor_indexes).index(out_idx)
            if dt_idx < len(out_dtypes):
                target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                if target_dtype is not None:
                    g = g.to(target_dtype)
            if g.shape != _torch.Size(shape):
                g = g.reshape(shape)
            golden_list.append(g.contiguous())
        # 准备 third_parties：按 output_tensor_indexes 顺序对齐
        if third_party is not None:
            tp_items = third_party if isinstance(third_party, (list, tuple)) else [third_party]
            third_parties_list = []
            for oi_idx, out_idx in enumerate(tc.output_tensor_indexes):
                if oi_idx < len(tp_items):
                    tp = tp_items[oi_idx]
                    third_parties_list.append(tp.contiguous() if isinstance(tp, _torch.Tensor) else tp)
                else:
                    third_parties_list.append(None)
        tc.golden_tensors = golden_list
        del rank_goldens[did]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({__fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(f"Multi-device: rank dev={did} comparison FAILED: {cr.precision} metrics={cr.metrics}")
            else:
                logging.info(f"Multi-device: rank dev={did} comparison PASSED")
        except Exception:
            logging.exception(f"Multi-device: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def __golden_a2a_ag_bmm(thread_contexts, device_ids, all_precision, world_size):
    import torch
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    ep_ws = int(attrs.get('epWorldSize', 1))
    tp_ws = int(attrs.get('tpWorldSize', 1))
    shard_type = int(attrs.get('xShardType', 1))
    act_type = int(attrs.get('actType', 0))
    is_bias = bool(attrs.get('isBias', False))
    is_trans = bool(attrs.get('isTrans', False))
    need_ag_out = bool(attrs.get('needAllgatherOut', True))
    need_act_feat = bool(attrs.get('needActivationFeature', False))

    def _apply_act(x, act):
        if act == 0:
            return x
        elif act == 1:
            return torch.nn.functional.gelu(x)
        elif act == 2:
            return torch.nn.functional.silu(x)
        elif act == 3:
            return torch.nn.functional.relu(x)
        elif act == 4:
            return x / (1.0 + torch.exp(-1.702 * x))
        return x

    dtype_map = {
        'float16': torch.float16, 'fp16': torch.float16,
        'float32': torch.float32, 'fp32': torch.float32,
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
    }

    def _compare_rank(did, goldens, third_parties=None):
        tc = thread_contexts[did]
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        golden_list = []
        out_keys = ['main', 'allgather', 'bmm']
        for oi, out_idx in enumerate(tc.output_tensor_indexes):
            if oi < len(out_keys):
                g = goldens.get(out_keys[oi])
            else:
                g = None
            if g is None:
                g = torch.zeros(tc.tensor_view_shapes[out_idx])
            dt_idx = list(tc.output_tensor_indexes).index(out_idx)
            if dt_idx < len(out_dtypes):
                target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                if target_dtype is not None:
                    g = g.to(target_dtype)
            golden_list.append(g.contiguous())
        tc.golden_tensors = golden_list
        third_parties_list = None
        if third_parties is not None:
            tp_items = third_parties if isinstance(third_parties, (list, tuple)) else [third_parties]
            third_parties_list = []
            for oi_idx, out_idx in enumerate(tc.output_tensor_indexes):
                if oi_idx < len(tp_items):
                    tp = tp_items[oi_idx]
                    if isinstance(tp, torch.Tensor):
                        out_shape = tc.tensor_view_shapes[out_idx]
                        if tp.shape != torch.Size(out_shape):
                            tp = tp.reshape(out_shape)
                        dt_idx = list(tc.output_tensor_indexes).index(out_idx)
                        if dt_idx < len(out_dtypes):
                            target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                            if target_dtype is not None:
                                tp = tp.to(target_dtype)
                        third_parties_list.append(tp.contiguous())
                    else:
                        third_parties_list.append(None)
                else:
                    third_parties_list.append(None)
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            msg = f"rank{did}:{cr.passed}({__fmt_compare_result(cr)})"
            is_fail = cr.passed != "PASS"
        except Exception:
            logging.exception(f"A2A_AG_BMM: rank dev={did} comparison failure")
            msg = f"rank{did}:COMPARE_EXCEPTION"
            is_fail = True
        if is_fail:
            logging.error(f"A2A_AG_BMM: {msg}")
        else:
            logging.info(f"A2A_AG_BMM: {msg}")
        all_precision.append(msg)

    # 真·小算子级联 third_party（参考 mc2_test get_hccl_mm）
    # all_to_all(EP) -> all_gather(TP) -> bmm -> [bias + act] -> output
    rank_third_parties = None
    try:
        from .hccl_cascade import run_a2a_ag_bmm_cascade
        cascade_outs = run_a2a_ag_bmm_cascade(
            thread_contexts, device_ids, ep_ws=ep_ws, tp_ws=tp_ws,
            shard_type=shard_type, is_trans=is_trans, is_bias=is_bias,
            act_type=act_type, need_ag_out=need_ag_out, need_act_feat=need_act_feat)
        rank_third_parties = {}
        for did in device_ids:
            tp_list = [cascade_outs[did]['main']]
            if need_ag_out:
                tp_list.append(cascade_outs[did].get('allgather'))
            if need_act_feat:
                tp_list.append(cascade_outs[did].get('bmm'))
            rank_third_parties[did] = tp_list
        logging.info("AlltoAllAllGatherBatchMatMul: real HCCL cascade succeeded")
    except Exception:
        logging.exception("AlltoAllAllGatherBatchMatMul: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    x_shape = thread_contexts[device_ids[0]].flatten_tensors[0].shape
    E = x_shape[0]
    E_div_ep = E // ep_ws

    if shard_type == 0:
        C = x_shape[1]
        H_div_tp = x_shape[2]
    else:
        C_div_tp = x_shape[1]
        H = x_shape[2]

    n_ep_groups = world_size // ep_ws
    n_tp_groups = world_size // tp_ws

    ep_groups = [list(range(g * ep_ws, (g + 1) * ep_ws)) for g in range(n_ep_groups)]
    tp_groups = [[g + e * ep_ws for e in range(tp_ws)] for g in range(n_tp_groups)]

    all_a2a_dids = sorted(set(did for grp in ep_groups for did in grp))
    in_dtype = thread_contexts[device_ids[0]].flatten_tensors[0].dtype
    x_cache = {did: _to_torch_f32(thread_contexts[did].flatten_tensors[0]) for did in all_a2a_dids}

    a2a_per_rank = {}
    for group_dids in ep_groups:
        chunks_per_rank = {}
        for local_idx, did in enumerate(group_dids):
            chunks_per_rank[local_idx] = x_cache[did].chunk(ep_ws, dim=0)
        for target_local, target_did in enumerate(group_dids):
            result_chunks = [chunks_per_rank[src_local][target_local] for src_local in range(len(group_dids))]
            a2a_out = torch.cat(result_chunks, dim=0)
            if shard_type == 0:
                a2a_out = a2a_out.reshape(ep_ws, E_div_ep, C, H_div_tp).permute(1, 0, 2, 3).contiguous()
            else:
                a2a_out = a2a_out.reshape(ep_ws, E_div_ep, C_div_tp, H).permute(1, 0, 2, 3).contiguous()
            a2a_per_rank[target_did] = a2a_out
    del x_cache

    for tp_group_dids in tp_groups:
        all_parts = [a2a_per_rank[did] for did in tp_group_dids]
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

        for did in tp_group_dids:
            tc = thread_contexts[did]
            in_dtype = tc.flatten_tensors[0].dtype
            weight = _to_torch_f32(tc.flatten_tensors[1])
            # bmm in fp32 then truncate to in_dtype (matching NPU bf16 bmm output)
            bmm_out = torch.bmm(gathered.float(), weight)
            bmm_out = bmm_out.to(in_dtype).float()
            del weight
            if is_bias and len(tc.flatten_tensors) > 2:
                bias = tc.flatten_tensors[2]
                if bias.numel() > 0:
                    # NPU: bmm(bf16) -> cast fp32 -> + bias(fp32) -> cast bf16
                    bias_f = bias.float()
                    if bias.dim() == 2:
                        bias_f = bias_f.reshape(bias_f.shape[0], 1, bias_f.shape[1])
                    bmm_out = (bmm_out.float() + bias_f).to(in_dtype).float()
                del bias
            # activation: compute in fp32 on bf16-truncated input, truncate output to in_dtype
            activated = _apply_act(bmm_out, act_type).to(in_dtype).float()
            bmm_out = bmm_out.to(in_dtype).float()
            goldens = {'main': activated}
            if need_ag_out:
                goldens['allgather'] = gathered.to(in_dtype).float()
            if need_act_feat:
                goldens['bmm'] = bmm_out
            else:
                del bmm_out
            _compare_rank(did, goldens,
                         third_parties=rank_third_parties.get(did) if rank_third_parties else None)
            del goldens, activated

        del gathered

    del a2a_per_rank


def __golden_bmm_reduce_scatter_allto_all(thread_contexts, device_ids, all_precision, world_size):
    import torch
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    ep_ws = int(attrs.get('epWorldSize', 1))
    tp_ws = int(attrs.get('tpWorldSize', 1))
    shard_type = int(attrs.get('yShardType', 1))
    is_bias = bool(attrs.get('isBias', False))
    is_trans = bool(attrs.get('isTrans', False))

    dtype_map = {'float16': torch.float16, 'fp16': torch.float16,
                 'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
                 'float32': torch.float32, 'fp32': torch.float32}

    in_dtype = thread_contexts[device_ids[0]].flatten_tensors[0].dtype

    E_div_ep = thread_contexts[device_ids[0]].flatten_tensors[0].shape[0]
    x_dim1 = thread_contexts[device_ids[0]].flatten_tensors[0].shape[1]
    H = thread_contexts[device_ids[0]].flatten_tensors[1].shape[2]

    if shard_type == 0:
        C = x_dim1 // ep_ws
    else:
        C_div_tp = x_dim1 // ep_ws // tp_ws

    rs_per_rank = {}
    n_tp_groups = world_size // tp_ws
    for g in range(n_tp_groups):
        group_dids = [g + e * ep_ws for e in range(tp_ws)]
        all_parts = []
        for did in group_dids:
            tc = thread_contexts[did]
            x = _to_torch_f32(tc.flatten_tensors[0])
            weight = _to_torch_f32(tc.flatten_tensors[1])
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
        n_tp = len(group_dids)
        for local_idx, did in enumerate(group_dids):
            start = (local_idx + 1) % n_tp
            acc = all_parts[start][local_idx * E_div_ep:(local_idx + 1) * E_div_ep].clone().float()
            for step in range(1, n_tp):
                src_idx = (start + step) % n_tp
                src_chunk = all_parts[src_idx][local_idx * E_div_ep:(local_idx + 1) * E_div_ep]
                acc = (acc.to(in_dtype).float() + src_chunk.float())
            chunk = acc.to(in_dtype)
            tc = thread_contexts[did]
            if is_bias and len(tc.flatten_tensors) > 2:
                bias = _to_torch_f32(tc.flatten_tensors[2])
                if bias.numel() > 0:
                    if bias.dim() == 2:
                        bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
                    chunk = chunk.to(in_dtype).float() + bias
                    chunk = chunk.to(in_dtype)
                del bias
            rs_per_rank[did] = chunk
        del all_parts

    n_ep_groups = world_size // ep_ws

    # 真·小算子级联 third_party（参考 mc2_test get_hccl_mm）
    # bmm -> reduce_scatter(TP) -> all_to_all(EP) -> output
    rank_third_parties = None
    try:
        from .hccl_cascade import run_bmm_rs_a2a_cascade
        cascade_outs = run_bmm_rs_a2a_cascade(
            thread_contexts, device_ids, ep_ws=ep_ws, tp_ws=tp_ws,
            shard_type=shard_type, is_trans=is_trans, is_bias=is_bias)
        rank_third_parties = {did: [cascade_outs[did]['main']] for did in device_ids}
        logging.info("BatchMatMulReduceScatterAlltoAll: real HCCL cascade succeeded")
    except Exception:
        logging.exception("BatchMatMulReduceScatterAlltoAll: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    for g in range(n_ep_groups):
        group_dids = list(range(g * ep_ws, (g + 1) * ep_ws))
        for target_local, target_did in enumerate(group_dids):
            if shard_type == 0:
                all_chunks = []
                for src_local, src_did in enumerate(group_dids):
                    rs = rs_per_rank[src_did]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C, H // tp_ws)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C, H // tp_ws)
            else:
                all_chunks = []
                for src_local, src_did in enumerate(group_dids):
                    rs = rs_per_rank[src_did]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C_div_tp, H)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C_div_tp, H)
            del gathered, all_chunks

            tc = thread_contexts[target_did]
            out_idx = tc.output_tensor_indexes[0]
            out_shape = tc.tensor_view_shapes[out_idx]
            out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
            golden = out
            if golden.shape != torch.Size(out_shape):
                golden = golden.reshape(out_shape)
            if len(out_dtypes) > 0:
                target_dtype = dtype_map.get(out_dtypes[0], None)
                if target_dtype is not None:
                    golden = golden.to(target_dtype)
            tc.golden_tensors = [golden.contiguous()]
            del golden
            third_parties_list = None
            if rank_third_parties is not None:
                tp = rank_third_parties.get(target_did, [None])[0]
                if isinstance(tp, torch.Tensor):
                    if tp.shape != torch.Size(out_shape):
                        tp = tp.reshape(out_shape)
                    if len(out_dtypes) > 0:
                        target_dtype = dtype_map.get(out_dtypes[0], None)
                        if target_dtype is not None:
                            tp = tp.to(target_dtype)
                    third_parties_list = [tp.contiguous()]
            try:
                cr = Comparator(tc).compare(third_parties=third_parties_list)
                all_precision.append(f"rank{target_did}:{cr.passed}({__fmt_compare_result(cr)})")
                if cr.passed != "PASS":
                    logging.error(f"BMM_RS_A2A: rank dev={target_did} FAILED: {cr.precision}")
                else:
                    logging.info(f"BMM_RS_A2A: rank dev={target_did} PASSED")
            except Exception:
                logging.exception(f"BMM_RS_A2A: rank dev={target_did} comparison failure")
                all_precision.append(f"rank{target_did}:COMPARE_EXCEPTION")
            del out

    del rs_per_rank


def rank_idx_in_group(did, group_dids):
    for i, d in enumerate(group_dids):
        if d == did:
            return i
    return 0


def __golden_grouped_matmul_compare(thread_contexts, device_ids, all_precision, world_size):
    import torch
    first_ctx = next(iter(thread_contexts.values()))
    api_name = first_ctx.api_name

    is_alltoallv_gmm = "AlltoAllvGroupedMatMul" in api_name
    is_gmm_alltoallv = "GroupedMatMulAlltoAllv" in api_name

    attrs = first_ctx.attributes
    ep_ws = attrs.get('epWorldSize', world_size)
    exp_per_card = first_ctx.tensor_view_shapes[1][0] if len(first_ctx.tensor_view_shapes) > 1 else 1

    expTokenNums = __get_gmm_exp_token_nums(first_ctx, 0, ep_ws)

    if is_gmm_alltoallv:
        rank_goldens = __golden_gmm_alltoallv(
            thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card)
    elif is_alltoallv_gmm:
        rank_goldens = __golden_alltoallv_gmm(thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card)
    else:
        rank_goldens = {}
        for did in device_ids:
            tc = thread_contexts[did]
            out_idx = tc.pure_output_indexes[0] if tc.pure_output_indexes else 0
            out_shape = tc.tensor_view_shapes[out_idx] if out_idx < len(tc.tensor_view_shapes) else (1,)
            rank_goldens[did] = {'main': torch.zeros(out_shape)}

    # 真·小算子级联 third_party（参考 mc2_test get_hccl_mm）
    rank_third_parties = None
    trans_gmm_weight = bool(attrs.get('transGmmWeight', False))
    trans_mm_weight = bool(attrs.get('transMmWeight', False))
    permute_out_flag = bool(attrs.get('permuteOutFlag', False))
    mm_out_flag = 'mm' in rank_goldens.get(device_ids[0], {}) and rank_goldens[device_ids[0]].get('mm') is not None
    try:
        if is_alltoallv_gmm:
            from .hccl_cascade import run_alltoallv_gmm_cascade
            cascade_outs = run_alltoallv_gmm_cascade(
                thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card,
                trans_gmm_weight=trans_gmm_weight, trans_mm_weight=trans_mm_weight,
                permute_out_flag=permute_out_flag, mm_out_flag=mm_out_flag)
            logging.info("AlltoAllvGroupedMatMul: real HCCL cascade succeeded")
        elif is_gmm_alltoallv:
            from .hccl_cascade import run_gmm_alltoallv_cascade
            cascade_outs = run_gmm_alltoallv_cascade(
                thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card,
                trans_gmm_weight=trans_gmm_weight, trans_mm_weight=trans_mm_weight,
                mm_out_flag=mm_out_flag)
            logging.info("GroupedMatMulAlltoAllv: real HCCL cascade succeeded")
        else:
            cascade_outs = None
        if cascade_outs is not None:
            # cascade_outs[did] = {'main': tensor, 'permute': tensor|None, 'mm': tensor|None}
            rank_third_parties = {}
            for did in device_ids:
                tp_list = [cascade_outs[did]['main']]
                # 按 output_tensor_indexes 顺序对齐：main / mm / permute
                out_idxs = thread_contexts[did].output_tensor_indexes
                for oi in range(1, len(out_idxs)):
                    if oi == 1:
                        # 第二输出可能是 mm 或 permute
                        if cascade_outs[did].get('mm') is not None:
                            tp_list.append(cascade_outs[did]['mm'])
                        elif cascade_outs[did].get('permute') is not None:
                            tp_list.append(cascade_outs[did]['permute'])
                        else:
                            tp_list.append(None)
                    elif oi == 2:
                        tp_list.append(cascade_outs[did].get('permute'))
                    else:
                        tp_list.append(None)
                rank_third_parties[did] = tp_list
    except Exception:
        logging.exception(f"{api_name}: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    __apply_gmm_goldens(thread_contexts, device_ids, rank_goldens, all_precision,
                         rank_third_parties=rank_third_parties)


def __golden_matmul_allto_all(thread_contexts, device_ids, target_did,
                               x1, x2, bias, t_x1, t_x2, world_size,
                               x1scale=None, x2scale=None):
    """MatmulAlltoAll: matmul(x1, x2) -> [dequant scales] -> all_to_all -> output.

    Each rank does local matmul first, then all_to_all exchanges chunks.
    For quant variants, scales are applied after matmul, before all_to_all.
    """
    import torch
    input_mat = _to_torch_f32(x1)
    if t_x1:
        input_mat = input_mat.t().contiguous()
    weight_mat = _to_torch_f32(x2)
    if t_x2:
        weight_mat = weight_mat.t().contiguous()
    mm_out = torch.matmul(input_mat, weight_mat)
    if bias is not None:
        mm_out = mm_out + _to_torch_f32(bias)
    if x1scale is not None:
        x1s_f = _to_torch_f32(x1scale)
        if x1s_f.dim() == 1:
            x1s_f = x1s_f.unsqueeze(-1)
        mm_out = mm_out * x1s_f
    if x2scale is not None:
        x2s_f = _to_torch_f32(x2scale)
        if x2s_f.dim() == 1:
            x2s_f = x2s_f.unsqueeze(0)
        mm_out = mm_out * x2s_f

    M = mm_out.shape[0]
    N = mm_out.shape[1]
    chunk_n = N // world_size

    all_to_all_results = []
    for src_did in device_ids:
        src_tc = thread_contexts[src_did]
        src_x1 = src_tc.flatten_tensors[0]
        src_x2 = src_tc.flatten_tensors[1]
        src_bias = src_tc.flatten_tensors[2] if len(src_tc.flatten_tensors) > 2 else None
        if src_bias is not None and isinstance(src_bias, torch.Tensor) and src_bias.numel() == 0:
            src_bias = None
        s_input = _to_torch_f32(src_x1)
        if t_x1:
            s_input = s_input.t().contiguous()
        s_weight = _to_torch_f32(src_x2)
        if t_x2:
            s_weight = s_weight.t().contiguous()
        s_mm = torch.matmul(s_input, s_weight)
        if src_bias is not None:
            s_mm = s_mm + _to_torch_f32(src_bias)
        s_chunks = s_mm.view(M, world_size, chunk_n).permute(1, 0, 2).contiguous()
        s_chunks = s_chunks.view(world_size, M * chunk_n)
        send_chunks = s_chunks.chunk(world_size, dim=0)
        target_idx = list(device_ids).index(target_did)
        all_to_all_results.append(send_chunks[target_idx].clone())
        del s_mm, s_chunks, send_chunks

    received = torch.cat(all_to_all_results, dim=0)
    del all_to_all_results
    received = received.reshape(-1, chunk_n).contiguous()
    return {'main': received}


def __golden_allto_all_matmul(thread_contexts, device_ids, target_did,
                               x1, x2, bias, t_x1, t_x2, world_size):
    """AlltoAllMatmul: all_to_all(x1) -> matmul(a2a_out, x2) -> output.

    Flow per rank:
      1. input = x1 (or x1.t if transposeX1)
      2. input.reshape(ws, M_chunk, K) -> all_to_all -> [ws, M_chunk, K]
      3. permute(1,0,2).reshape(M_chunk, ws*K)
      4. matmul(result, weight) where weight = x2 (or x2.t)
    """
    import torch
    input_mat = _to_torch_f32(x1)
    if t_x1:
        input_mat = input_mat.t().contiguous()
    weight_mat = _to_torch_f32(x2)
    if t_x2:
        weight_mat = weight_mat.t().contiguous()
    M_total = input_mat.shape[0]
    K = input_mat.shape[1]
    M_chunk = M_total // world_size

    target_idx = list(device_ids).index(target_did)

    recv_chunks = []
    for src_did in device_ids:
        src_tc = thread_contexts[src_did]
        src_x1 = src_tc.flatten_tensors[0]
        s_input = _to_torch_f32(src_x1)
        if t_x1:
            s_input = s_input.t().contiguous()
        s_reshaped = s_input.view(world_size, M_chunk, K)
        recv_chunks.append(s_reshaped[target_idx])

    recv_tensor = torch.stack(recv_chunks, dim=0)
    a2a_out = recv_tensor.permute(1, 0, 2).reshape(M_chunk, world_size * K).contiguous()

    mm_out = torch.matmul(a2a_out, weight_mat)
    if bias is not None:
        mm_out = mm_out + _to_torch_f32(bias)
    return {'main': mm_out, 'alltoall': a2a_out}


def __simulate_moe_alltoallv(all_rank_inputs, device_ids, send_counts_per_rank):
    import torch
    rank_outputs = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        received_chunks = []
        for src_did in device_ids:
            src_data = all_rank_inputs[src_did]
            src_counts = send_counts_per_rank[src_did]
            offset = 0
            for dst_idx in range(len(device_ids)):
                if dst_idx == target_idx:
                    count = int(src_counts[dst_idx])
                    if count > 0:
                        received_chunks.append(src_data[offset:offset + count])
                    break
                offset += int(src_counts[dst_idx])
        if received_chunks:
            rank_outputs[target_did] = torch.cat(received_chunks, dim=0)
        else:
            h = all_rank_inputs[device_ids[0]].shape[-1] if all_rank_inputs[device_ids[0]].dim() > 1 else 0
            rank_outputs[target_did] = torch.zeros(0, h, dtype=all_rank_inputs[device_ids[0]].dtype)
    return rank_outputs


def __golden_moe_distribute_dispatch(thread_contexts, device_ids, all_precision, world_size):
    import torch
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    ep_ws = int(attrs.get('epWorldSize', world_size))
    moe_expert_num = int(attrs.get('moeExpertNum', 1))
    quant_mode = int(attrs.get('quantMode', 0))
    local_expert_num = moe_expert_num // ep_ws

    all_rank_expand_x = {}
    all_rank_expand_idx = {}
    all_rank_send_counts = {}
    all_rank_expert_token_nums = {}
    all_rank_dynamic_scales = {}
    all_rank_expand_scales = {}

    for did in device_ids:
        tc = thread_contexts[did]
        x = tc.flatten_tensors[0]
        expert_ids = tc.flatten_tensors[1]
        scales_tensor = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        expert_scales = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None

        bs = x.shape[0]
        h = x.shape[1]
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        send_counts = [0] * ep_ws
        token_groups = [[] for _ in range(ep_ws)]
        expand_idx_list = []
        dynamic_scales_list = []
        expand_scales_list = []

        for i in range(bs):
            for j in range(k):
                eid = int(expert_ids[i][j]) if expert_ids.dim() > 1 else int(expert_ids[i])
                dest_rank = eid // local_expert_num if local_expert_num > 0 else 0
                if dest_rank >= ep_ws:
                    dest_rank = ep_ws - 1
                token_groups[dest_rank].append(x[i])
                expand_idx_list.append(i * k + j)
                send_counts[dest_rank] += 1
                if scales_tensor is not None and scales_tensor.numel() > 0:
                    s = float(scales_tensor[i][j]) if scales_tensor.dim() > 1 else float(scales_tensor[i])
                    dynamic_scales_list.append(s)
                if expert_scales is not None and expert_scales.numel() > 0:
                    es = float(expert_scales[i][j]) if expert_scales.dim() > 1 else float(expert_scales[i])
                    expand_scales_list.append(es)

        sorted_tokens = []
        for r in range(ep_ws):
            sorted_tokens.extend(token_groups[r])

        if sorted_tokens:
            expand_x_local = torch.stack(sorted_tokens, dim=0).to(x.dtype)
        else:
            expand_x_local = torch.zeros(0, h, dtype=x.dtype)

        all_rank_expand_x[did] = expand_x_local
        all_rank_expand_idx[did] = torch.tensor(expand_idx_list, dtype=torch.int32)
        all_rank_send_counts[did] = send_counts
        all_rank_dynamic_scales[did] = (torch.tensor(dynamic_scales_list, dtype=torch.float32)
                                        if dynamic_scales_list else torch.zeros(0, dtype=torch.float32))
        all_rank_expand_scales[did] = (torch.tensor(expand_scales_list, dtype=torch.float32)
                                       if expand_scales_list else torch.zeros(0, dtype=torch.float32))

    for did in device_ids:
        rank_idx = list(device_ids).index(did)
        expert_token_nums = [0] * local_expert_num
        for src_did in device_ids:
            src_tc = thread_contexts[src_did]
            src_expert_ids = src_tc.flatten_tensors[1]
            src_bs = src_expert_ids.shape[0]
            src_k = src_expert_ids.shape[1] if src_expert_ids.dim() > 1 else 1
            for i in range(src_bs):
                for j in range(src_k):
                    eid = int(src_expert_ids[i][j]) if src_expert_ids.dim() > 1 else int(src_expert_ids[i])
                    dest_rank = eid // local_expert_num if local_expert_num > 0 else 0
                    if dest_rank >= ep_ws:
                        dest_rank = ep_ws - 1
                    if dest_rank == rank_idx:
                        local_eid = eid % local_expert_num if local_expert_num > 0 else 0
                        expert_token_nums[local_eid] += 1
        all_rank_expert_token_nums[did] = torch.tensor(expert_token_nums, dtype=torch.int64)

    alltoallv_out = __simulate_moe_alltoallv(all_rank_expand_x, device_ids, all_rank_send_counts)

    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        recv_x = alltoallv_out[did]
        num_recv = recv_x.shape[0]

        ep_recv_counts = []
        for src_did in device_ids:
            src_counts = all_rank_send_counts[src_did]
            ep_recv_counts.append(src_counts[rank_idx])
        ep_recv_counts_tensor = torch.tensor(ep_recv_counts, dtype=torch.int32)

        tp_ws = int(attrs.get('tpWorldSize', 0))
        tp_recv_counts_tensor = torch.zeros(max(tp_ws, 1), dtype=torch.int32)

        goldens = [
            recv_x.contiguous(),
            all_rank_dynamic_scales[did].contiguous(),
            all_rank_expand_idx[did].contiguous(),
            all_rank_expert_token_nums[did].contiguous(),
            ep_recv_counts_tensor.contiguous(),
            tp_recv_counts_tensor.contiguous(),
            all_rank_expand_scales[did].contiguous(),
        ]
        tc.golden_tensors = goldens
        try:
            cr = Comparator(tc).compare()
            all_precision.append(f"rank{did}:{cr.passed}({__fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(f"MoeDistributeDispatch: rank dev={did} FAILED: {cr.precision}")
            else:
                logging.info(f"MoeDistributeDispatch: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"MoeDistributeDispatch: rank dev={did} compare failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def __golden_moe_distribute_combine(thread_contexts, device_ids, all_precision, world_size):
    import torch
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    ep_ws = int(attrs.get('epWorldSize', world_size))
    moe_expert_num = int(attrs.get('moeExpertNum', 1))
    local_expert_num = moe_expert_num // ep_ws

    all_rank_expand_x = {}
    all_rank_send_counts = {}

    for did in device_ids:
        tc = thread_contexts[did]
        expand_x = tc.flatten_tensors[0]
        ep_send_counts = tc.flatten_tensors[3]

        num_tokens = expand_x.shape[0]
        h = expand_x.shape[1]

        send_counts = [0] * ep_ws
        if ep_send_counts is not None and ep_send_counts.numel() > 0:
            rank_idx = list(device_ids).index(did)
            total_send = 0
            for src_idx in range(ep_ws):
                start = src_idx * local_expert_num
                end = start + local_expert_num
                if end <= ep_send_counts.numel():
                    count = int(ep_send_counts[start:end].sum())
                else:
                    count = 0
                send_counts[src_idx] = count
                total_send += count
            if total_send != num_tokens and total_send > 0:
                send_counts = [num_tokens // ep_ws] * ep_ws
                remainder = num_tokens % ep_ws
                for i in range(remainder):
                    send_counts[i] += 1
        else:
            per_rank = num_tokens // ep_ws
            send_counts = [per_rank] * ep_ws
            remainder = num_tokens % ep_ws
            for i in range(remainder):
                send_counts[i] += 1

        all_rank_expand_x[did] = expand_x
        all_rank_send_counts[did] = send_counts

    alltoallv_out = __simulate_moe_alltoallv(all_rank_expand_x, device_ids, all_rank_send_counts)

    for did in device_ids:
        tc = thread_contexts[did]
        expand_x = tc.flatten_tensors[0]
        expert_ids = tc.flatten_tensors[1]
        expand_idx = tc.flatten_tensors[2]
        expert_scales = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None

        h = expand_x.shape[1]
        bs_k = expand_idx.shape[0] if expand_idx.dim() > 0 else 0
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1
        bs = bs_k // k if k > 0 else 0

        a2a_result = alltoallv_out[did]

        x_out = torch.zeros(bs, h, dtype=expand_x.dtype)
        if a2a_result.numel() > 0 and bs > 0:
            for idx_pos in range(min(bs_k, a2a_result.shape[0])):
                orig_idx = int(expand_idx[idx_pos]) // k if k > 0 else int(expand_idx[idx_pos])
                if orig_idx < bs:
                    scale = 1.0
                    if expert_scales is not None and expert_scales.numel() > 0:
                        j = int(expand_idx[idx_pos]) % k if k > 0 else 0
                        if expert_scales.dim() > 1 and orig_idx < expert_scales.shape[0] and j < expert_scales.shape[1]:
                            scale = float(expert_scales[orig_idx][j])
                        elif expert_scales.dim() == 1 and orig_idx < expert_scales.shape[0]:
                            scale = float(expert_scales[orig_idx])
                    x_out[orig_idx] = x_out[orig_idx] + a2a_result[idx_pos].float() * scale

        x_out = x_out.to(expand_x.dtype)
        tc.golden_tensors = [x_out.contiguous()]
        try:
            cr = Comparator(tc).compare()
            all_precision.append(f"rank{did}:{cr.passed}({__fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(f"MoeDistributeCombine: rank dev={did} FAILED: {cr.precision}")
            else:
                logging.info(f"MoeDistributeCombine: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"MoeDistributeCombine: rank dev={did} compare failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def __generate_exp_token_nums(exp_num, ep_world_size, bsk, seed):
    m = exp_num
    n = ep_world_size
    total = bsk
    sum_row = total * n // m
    sum_col = total
    if m * sum_row != n * sum_col:
        return [[total // m] * m for _ in range(n)]
    numpy.random.seed(seed)
    matrix = numpy.random.multinomial(sum_row - n, [1.0 / n] * n, size=m) + 1
    cur_col = matrix.sum(axis=0)
    target = numpy.full(n, sum_col)
    for _ in range(10000):
        if numpy.array_equal(cur_col, target):
            break
        j = int(numpy.argmax(cur_col - target))
        k = int(numpy.argmin(cur_col - target))
        for i in numpy.random.permutation(m):
            if matrix[i, j] > 1:
                matrix[i, j] -= 1
                matrix[i, k] += 1
                cur_col[j] -= 1
                cur_col[k] += 1
                break
    return [list(col) for col in zip(*matrix)]


def __generate_gmm_alltoallv_matrix(A_array_val, exp_per_card, seed):
    n = len(A_array_val)
    rng = numpy.random.default_rng(seed)
    total = sum(A_array_val)
    if total % n != 0:
        return [[total // n] * (exp_per_card * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in A_array_val:
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


def __patch_gmm_rank_attributes(ctx: 'TestcaseAclnn', rank_idx: int, world_size: int):
    api_name = ctx.api_name
    is_alltoallv_gmm = "AlltoAllvGroupedMatMul" in api_name
    is_gmm_alltoallv = "GroupedMatMulAlltoAllv" in api_name
    if not is_alltoallv_gmm and not is_gmm_alltoallv:
        return
    attrs = ctx.attributes
    ep_ws = attrs.get('epWorldSize', world_size)
    exp_per_card = ctx.tensor_view_shapes[1][0] if len(ctx.tensor_view_shapes) > 1 else 1
    seed_val = 0
    remark = ctx.remark or ''
    for part in remark.split(','):
        kv = part.split('=', 1)
        if len(kv) == 2 and kv[0].strip() == 'seed':
            try:
                seed_val = int(kv[1].strip())
            except ValueError:
                pass
    if is_alltoallv_gmm:
        bsk = ctx.tensor_view_shapes[0][0] if ctx.tensor_view_shapes else 0
        exp_num = exp_per_card * ep_ws
        A_array = [bsk] * ep_ws
        expTokenNums = __generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)
        send_counts = expTokenNums[rank_idx]
        recv_counts = []
        for i in range(ep_ws):
            recv_counts.extend(expTokenNums[i][rank_idx * exp_per_card:(rank_idx + 1) * exp_per_card])
        attrs['sendCounts'] = send_counts
        attrs['recvCounts'] = recv_counts
    elif is_gmm_alltoallv:
        M_per_rank = ctx.tensor_view_shapes[0][0] if ctx.tensor_view_shapes else 0
        exp_num = exp_per_card * ep_ws
        A_array = [M_per_rank] * ep_ws
        expTokenNums = __generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)
        recv_counts = expTokenNums[rank_idx]
        send_counts = []
        for i in range(ep_ws):
            send_counts.extend(expTokenNums[i][rank_idx * exp_per_card:(rank_idx + 1) * exp_per_card])
        attrs['sendCounts'] = send_counts
        attrs['recvCounts'] = recv_counts
        ctx._pure_attrs = None
        logging.info(f"[GMM patch] api={api_name} rank={rank_idx} ep_ws={ep_ws} "
                     f"seed={seed_val} send_counts={send_counts[:4]}... recv_counts={recv_counts[:4]}...")


def get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws):
    return __get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws)


def generate_gmm_alltoallv_matrix(A_array_val, exp_per_card, seed):
    return __generate_gmm_alltoallv_matrix(A_array_val, exp_per_card, seed)


def patch_gmm_rank_attributes(ctx, rank_idx, world_size):
    __patch_gmm_rank_attributes(ctx, rank_idx, world_size)


def patch_gmm_weight_transpose(ctx):
    pass
