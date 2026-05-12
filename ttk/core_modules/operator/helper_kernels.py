#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
helper_kernels
"""
import numpy
from tbe import tik
import tbe.common.platform as tbe_platform


def clear_ub(ipt: dict, full_soc_version: str, core_type: str,
             kernel_name: str = "clear_ub"):
    """
    clear UB data
    """
    tbe_platform.set_current_compile_soc_info(full_soc_version, core_type)
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")
    ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    dtype = ipt["dtype"]
    dtype_bytes = numpy.dtype(dtype).itemsize
    input_gm = tik_instance.Tensor(dtype, (32 // dtype_bytes,), name="input_gm", scope=tik.scope_gm)
    src_ub = tik_instance.Tensor(dtype, (ub_size // dtype_bytes,), name="src_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(dst=src_ub, src=input_gm, sid=0, nburst=1, burst=1,
                           src_stride=0, dst_stride=0)
    clean_val_scalar = tik_instance.Scalar(dtype=dtype, name="clean_val_scalar", init_value=src_ub[0])

    with tik_instance.for_range(0, core_num, block_num=core_num):
        _clear_ub_each_core(tik_instance, src_ub, clean_val_scalar, dtype_bytes)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_gm,), outputs=())
    return tik_instance


def test_clear_ub(output, full_soc_version: str, core_type: str,
                  clean_val: numpy.generic, kernel_name: str = "test_clear_ub"):
    """Test UB data after clear"""
    tbe_platform.set_current_compile_soc_info(full_soc_version, core_type)
    tik_instance = tik.Tik()
    dtype = clean_val.dtype.name
    dtype_bytes = clean_val.dtype.itemsize
    # copy out two blocks
    src_ub = tik_instance.Tensor(dtype, (2 * 32 // dtype_bytes,), name="src_ub", scope=tik.scope_ubuf)
    output_gm = tik_instance.Tensor(dtype, (2 * 32 // dtype_bytes,), name="output_gm", scope=tik.scope_gm)
    with tik_instance.for_range(0, 1, block_num=1):  # copy out only in one core
        tik_instance.data_move(dst=output_gm, src=src_ub, sid=0, nburst=1, burst=2,
                               src_stride=0, dst_stride=0)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(), outputs=(output_gm,))
    return tik_instance


def _clear_ub_each_core(tik_instance: tik.Tik, ub_to_clear: tik.Tensor, clear_value, dtype_bytes):
    block_elements = 32 // dtype_bytes
    mask_max = block_elements * 8
    repeat = ub_to_clear.size // mask_max
    repeat_tail = ub_to_clear.size % mask_max
    offset = 0
    while repeat > 255:
        tik_instance.vec_dup(mask=mask_max, dst=ub_to_clear[offset], scalar=clear_value,
                             repeat_times=255, dst_rep_stride=8)
        repeat = repeat - 255
        offset = offset + mask_max * 255
    if repeat > 0:
        tik_instance.vec_dup(mask=mask_max, dst=ub_to_clear[offset], scalar=clear_value,
                             repeat_times=repeat, dst_rep_stride=8)
        offset = offset + mask_max * repeat
    if repeat_tail > 0:
        tik_instance.vec_dup(mask=repeat_tail, dst=ub_to_clear[offset], scalar=clear_value,
                             repeat_times=1, dst_rep_stride=8)


def clear_l1(ipt: dict, full_soc_version: str, kernel_name: str = "clear_l1"):
    """
    clear L1 data
    """
    tbe_platform.set_current_compile_soc_info(full_soc_version, "AiCore")
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")

    l1_size = tbe_platform.get_soc_spec(tbe_platform.L1_SIZE)
    dtype = ipt["dtype"]
    dtype_bytes = numpy.dtype(dtype).itemsize
    # input will be always fixed as (128, 1024) bytes
    input_bytes = 128 * 1024
    input_gm = tik_instance.Tensor(dtype, (input_bytes // dtype_bytes,), name="input_gm", scope=tik.scope_gm)
    dst_l1 = tik_instance.Tensor(dtype, (l1_size // dtype_bytes,), name="dst_l1", scope=tik.scope_cbuf)

    with tik_instance.for_range(0, core_num, block_num=core_num):
        repeat = l1_size // input_bytes
        for idx in range(repeat):
            tik_instance.data_move(dst=dst_l1[idx * input_bytes // dtype_bytes], src=input_gm, sid=0,
                                   nburst=1, burst=input_bytes // 32,
                                   src_stride=0, dst_stride=0)
        repeat_tail = l1_size % input_bytes
        tail_burst = repeat_tail // 32
        if repeat_tail > 0 and tail_burst > 0:
            tik_instance.data_move(dst=dst_l1[repeat * input_bytes // dtype_bytes], src=input_gm, sid=0,
                                   nburst=1, burst=tail_burst,
                                   src_stride=0, dst_stride=0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(input_gm,), outputs=())
    return tik_instance


def warmup(full_soc_version: str, kernel_name: str = "warmup"):
    """
    a kernel without any action, but warmup all the cores, like: smmu/biu/tlb,
    to avoid cycles which is not belong to the kernel going to test.
    """
    tbe_platform.set_current_compile_soc_info(full_soc_version, "AiCore")
    tik_instance = tik.Tik()
    core_num = tbe_platform.get_soc_spec("CORE_NUM")

    with tik_instance.for_range(0, core_num, block_num=core_num):
        tik_instance.tik_return()
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=(), outputs=())
    return tik_instance
