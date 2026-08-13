#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -*- coding: UTF-8 -*-
"""
High Performance Transform Data with NumPy
Tips:
  - Chinese comments are not recommended.
  - The unified naming style is highly recommended.
  - Use "python type hints" if possible
  - Recommend to use methods in the standard library instead of reinventing-the-wheel.
"""
# Standard Packages
import copy
import inspect
from typing import Tuple, Union
# Third-Party Packages
import numpy
from ...utilities import ceil_div, align, lcm


BLOCK_SIZE = 16

PAD_C0_MAPPING = {
    "int64": 4,
    "uint64": 4,
    "float32": 16,
    "bfloat16": 16,
    "int32": 16,
    "uint32": 16,
    "float16": 16,
    "int16": 16,
    "uint16": 16,
    "int8": 32,
    "uint8": 32,
    "bool": 32,
    "uint1": 256,
}


def align_factor(dtype: str = "float16"):
    return PAD_C0_MAPPING.get(dtype, 16)


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def _calculate_group(cin, cout, groups, c0):
    cin_ori, cout_ori = cin // groups, cout // groups
    mag_factor0 = lcm(cin_ori, c0) // cin_ori
    mag_factor1 = lcm(cout_ori, BLOCK_SIZE) // cout_ori
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin_g = align(mag_factor * cin_ori, c0)
    cout_g = align(mag_factor * cout_ori, BLOCK_SIZE)

    group_dict = {
        "real_g": ceil_div(groups, mag_factor),
        "mag_factor": mag_factor,
        "cin_g": cin_g,
        "cin1_g": cin_g // c0,
        "cout_g": cout_g,
        "cout1_g": cout_g // BLOCK_SIZE,
        "groups": groups,
        "cin_ori": cin_ori,
        "cout_ori": cout_ori
    }
    print('cin:%d, cout:%d, groups:%d, group_dict:' % (cin, cout, groups),
          group_dict)
    return group_dict


def _get_ndchw_dim_and_transpose_axis(data: numpy.ndarray, data_format: str, is_ndchw: bool = False):
    data_shape = data.shape
    n, d, c, h, w = 1, 1, 1, 1, 1
    transpose_axis = []
    if "N" in data_format:
        transpose_axis.append(data_format.index("N"))
        n = data_shape[transpose_axis[-1]]
    if "C" in data_format:
        transpose_axis.append(data_format.index("C"))
        c = data_shape[transpose_axis[-1]]
    else:
        data = data.reshape(data.shape + (1,))
        transpose_axis.append(len(data.shape) - 1)
    if is_ndchw and "D" in data_format:
        transpose_axis.append(data_format.index("D"))
        d = data_shape[transpose_axis[-1]]
    if "H" in data_format:
        transpose_axis.append(data_format.index("H"))
        h = data_shape[transpose_axis[-1]]
    if "W" in data_format:
        transpose_axis.append(data_format.index("W"))
        w = data_shape[transpose_axis[-1]]
    if is_ndchw:
        return (n, d, c, h, w), transpose_axis
    else:
        return (n, c, h, w), transpose_axis


def determine_c0(dtype, target_shape: Union[list, tuple] = None) -> int:
    if not isinstance(dtype, str) and hasattr(dtype, 'name'):
        dtype = getattr(dtype, 'name')
    if target_shape and target_shape[-1] > 0:
        return target_shape[-1]
    else:
        return align_factor(dtype)


def is_nchw_like(shape, format_: str) -> bool:
    return len(shape) == 4 and len(set(format_)) == 4 and all([c in format_ for c in "NCHW"])


def is_ndchw_like(shape, format_: str) -> bool:
    return len(shape) == 5 and len(set(format_)) == 5 and all([c in format_ for c in "NDCHW"])


def nd_shape2fhd_shape(nd_shape, nd_format: str = "NCHW", dtype: str = "float16",
                       fhd_shape: Union[list, tuple] = None) -> Tuple:
    if not is_nchw_like(nd_shape, nd_format):
        raise RuntimeError(f"shape: {nd_shape} of format {nd_format} is not NCHW-like.")
    c0 = determine_c0(dtype, fhd_shape)
    n, c = nd_shape[nd_format.index("N")], nd_shape[nd_format.index("C")]
    h, w = nd_shape[nd_format.index("H")], nd_shape[nd_format.index("W")]
    c1 = ceil_div(c, c0)
    return n, c1, h, w, c0


def nd_shape2nz_shape(nd_shape: Union[list, tuple], dtype: str = "float16",
                      nz_shape: Union[list, tuple] = None) -> Tuple:
    m, n = nd_shape[-2:]
    m0 = 16
    n0 = determine_c0(dtype, nz_shape)
    m1 = ceil_div(m, m0)
    n1 = ceil_div(n, n0)
    return tuple(nd_shape[:-2] + (n1, m1, m0, n0))


def fhd2nd(data, target_shape, target_format: str = "NCHW"):
    if not is_nchw_like(target_shape, target_format):
        raise RuntimeError(f"shape: {target_shape} of format {target_format} is not NCHW-like.")
    fhd_shape = data.shape
    pad = 1 + (target_shape[target_format.index("C")] - 1) % fhd_shape[-1]
    main_block = data[:, :fhd_shape[1] - 1, :, :, :]  # main block
    tail_block = data[:, fhd_shape[1] - 1, :, :, :pad]  # tail block
    # NC1HWC0 -> NHWC1C0
    main_block = main_block.transpose((0, 2, 3, 1, 4))
    # NHWC1C0 -> NHWC
    main_block = main_block.reshape(main_block.shape[:3] + (-1,))
    # concatenate
    nhwc = numpy.concatenate((main_block, tail_block), axis=-1)
    if target_format != "NHWC":
        return nhwc.transpose(("NHWC".index(target_format[0]), "NHWC".index(target_format[1]),
                               "NHWC".index(target_format[2]), "NHWC".index(target_format[3])))
    return nhwc


def shd2nd(data, target_shape, target_format: str = "NDCHW"):
    if not is_ndchw_like(target_shape, target_format):
        raise RuntimeError(f"shape: {target_shape} of format {target_format} is not NDCHW-like.")
    shd_shape = data.shape
    pad = 1 + (target_shape[target_format.index("C")] - 1) % shd_shape[-1]
    main_block = data[:, :, :shd_shape[2] - 1, :, :, :]
    tail_block = data[:, :, shd_shape[2] - 1, :, :, :pad]
    # NDC1HWC0 -> NDHWC1C0
    main_block = main_block.transpose((0, 1, 3, 4, 2, 5))
    # NDHWC1C0 -> NDHWC
    main_block = main_block.reshape(main_block.shape[:4] + (-1,))
    # concatenate
    ndhwc = numpy.concatenate((main_block, tail_block), axis=-1)
    if target_format != "NDHWC":
        return ndhwc.transpose(("NDHWC".index(target_format[0]), "NDHWC".index(target_format[1]),
                                "NDHWC".index(target_format[2]), "NDHWC".index(target_format[3]),
                                "NDHWC".index(target_format[4])))
    return ndhwc


def nz2nd(data, target_shape):
    """
    Convert FRACTAL_NZ format to ND format
    (A0, A1, A2, ..., An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    (A, M1, M0, N1, N0) -> (A, [0, M1-2], M0, [0, N1-2], N0) == (A, (M1-1), M0, (N1-1), N0)
                                                             -> (A, (M1-1) *M0, (N1-1) *N0)
                        -> (A, [M1-1], [0, pad_m-1], [0, N1-2], N0) == (A, 1, pad_m, (N1-1), N0)
                                                                    -> (A, pad_m, (N1-1) *N0)
                        -> (A, [0, M1-2], M0, [N1-1], [0, pad_n-1]) == (A, (M1-1), M0, 1, pad_n)
                                                                    -> (A, (M1-1) *M0, pad_n)
                        -> (A, [M1-1], [0, pad_m-1], [N1-1], [0, pad_n-1]) == (A, 1, pad_m, 1, pad_n)
                                                                           -> (A, pad_m, pad_n)
    M = (M1-1) *M0 + pad_m
    N = (N1-1) *N0 + pad_n
    (A, (M1-1) *M0, (N1-1) *N0) + (A, pad_m, (N1-1) *N0) -> (A, (M1-1) *M0 + pad_m, (N1-1) *N0) -> (A, M, (N1-1) *N0)
    (A, (M1-1) *M0, pad_n) + (A, pad_m, pad_n) -> (A, (M1-1) *M0 + pad_m, pad_n) -> (A, M, pad_n)
    (A, M, (N1-1) *N0) + (A, M, pad_n) -> (A, M, (N1-1) *N0 + pad_n) -> (A, M, N)
    (A, M, N) -> (A0, A1, A2, ..., An, M, N)
    """
    if len(data.shape) == 4:
        data = numpy.reshape(data, (1,) + data.shape)
    nd_shape = (1,) + tuple(target_shape)
    data_shape = data.shape
    m, n = nd_shape[-2:]
    N1, M1 = data_shape[-4:-2]
    M0, N0 = data_shape[-2:]
    pad_m = 1 + (m - 1) % M0
    pad_n = 1 + (n - 1) % N0
    # (A0, A1, A2, ... , An, N1, M1, M0, N0) -> (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    data = numpy.reshape(data, (numpy.prod(data_shape[:-4]),) + data_shape[-4:]).transpose((0, 2, 3, 1, 4))
    main_block = data[:, :M1 - 1, :, :N1 - 1, :]  # main block
    part_1 = data[:, M1 - 1, :pad_m, :N1 - 1, :]  # part 1
    part_2 = data[:, :M1 - 1, :, N1 - 1, :pad_n]  # part 2
    tail_block = data[:, M1 - 1, :pad_m, N1 - 1, :pad_n]  # tail_block
    # Reshape
    A = data.shape[0]
    main_block = numpy.reshape(main_block, (A, (M1 - 1) * M0, (N1 - 1) * N0))
    part_1 = numpy.reshape(part_1, (A, pad_m, (N1 - 1) * N0))
    part_2 = numpy.reshape(part_2, (A, (M1 - 1) * M0, pad_n))
    tail_block = numpy.reshape(tail_block, (A, pad_m, pad_n))
    # Concatenate
    main_concat_part1 = numpy.concatenate((main_block, part_1), axis=1)  # (A, M, (N1-1) *N0)
    part_2_concat_tail = numpy.concatenate((part_2, tail_block), axis=1)  # (A, M, pad_n)
    nd = numpy.concatenate((main_concat_part1, part_2_concat_tail), axis=-1)
    # Reshape
    nd = numpy.reshape(nd, data_shape[:-4]+(m, n))
    nd = numpy.reshape(nd, target_shape)

    return nd


def to_fractal_z(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    data_shape = data.shape
    if not is_nchw_like(data_shape, ori_format):
        raise RuntimeError(f"shape: {data_shape} of format {ori_format} is not NCHW-like.")
    # data_format:NCHW or NHWC
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    if groups is None:
        groups = 1
    c_in = c * groups
    c_out = n
    c0 = determine_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(c_in, c_out, groups, c0)
    G = group_dict["real_g"]
    ci_ori = group_dict["cin_ori"]
    co_ori = group_dict["cout_ori"]
    cin1_g = group_dict["cin1_g"]
    cou1_g = group_dict["cout1_g"]
    E = group_dict["mag_factor"]
    # Initialization
    out = numpy.zeros([G * cou1_g * BLOCK_SIZE, cin1_g * c0, h, w], dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"), ori_format.index("H"), ori_format.index("W")])
    for m in range(groups):
        for k in range(co_ori):
            for l in range(0, ci_ori):
                i = m // E
                j = m % E
                out[i * E * co_ori + j * co_ori + k, j * ci_ori + l, :, :] = \
                    data[i * E * co_ori + j * co_ori + k, l, :, :]
    # nchw->FRACTAL_Z
    out = out.reshape((G, cou1_g * BLOCK_SIZE, cin1_g, c0, h, w)).transpose(0, 2, 4, 5, 1, 3)
    out = out.reshape(G * cin1_g * h * w, cou1_g, BLOCK_SIZE, c0)
    return out


def to_fractal_z_c04(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    data_shape = data.shape
    # data_format: NCHW or NHWC
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    if groups is None:
        groups = 1
    c_in = c * groups
    c_out = n
    c0 = determine_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(c_in, c_out, groups, c0)
    G = group_dict["real_g"]
    ci_ori = group_dict["cin_ori"]
    co_ori = group_dict["cout_ori"]
    cin1_g = group_dict["cin1_g"]
    cou1_g = group_dict["cout1_g"]
    # Initialization
    out = numpy.zeros([G * cou1_g * BLOCK_SIZE, cin1_g * c0, h, w], dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"), ori_format.index("H"), ori_format.index("W")])
    # NCHW->FractalZ_C04
    for k in range(co_ori):
        for l in range(0, ci_ori):
            out[k, l, :, :] = data[k, l, :, :]
    # NCHW->Fractal_Z
    out = out.reshape((G, cou1_g * BLOCK_SIZE, cin1_g, c0, h, w)).transpose(0, 2, 4, 5, 1, 3)
    out = out.reshape(G * cin1_g * h * w, cou1_g, BLOCK_SIZE, c0)
    out_pad = numpy.zeros([align(G * cin1_g * h * w, 4), cou1_g, BLOCK_SIZE, c0], dtype=data.dtype)
    for i in range(G * cin1_g * h * w):
        out_pad[i, :, :, :] = out[i, :, :, :]
    cin_outer = ceil_div(G * ceil_div(c_in, 4) * 4 * h * w, BLOCK_SIZE)
    fractal_z_c04_res = numpy.zeros([cin_outer, cou1_g, BLOCK_SIZE, c0], dtype=data.dtype)
    for k in range(cin_outer):
        for cin0 in range(c0):
            fractal_z_c04_res[k, :, :, cin0] = out_pad[k * 4 + cin0 // 4, :, :, cin0 % 4]
    return fractal_z_c04_res


def to_fractal_z_3d(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None, groups=None):
    data_shape = data.shape
    # data_format: 'NCDHW' or 'NDHWC'
    n, c = data_shape[ori_format.index("N")], data_shape[ori_format.index("C")]
    h, w = data_shape[ori_format.index("H")], data_shape[ori_format.index("W")]
    d = data_shape[ori_format.index("D")]
    if groups is None:
        groups = 1
    fmap_c = c * groups
    out_c = n
    c0 = determine_c0(data.dtype.name, target_shape)
    group_dict = _calculate_group(fmap_c, out_c, groups, c0)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    cout_g = group_dict["cout_g"]
    mag_factor = group_dict["mag_factor"]
    cout1_g = group_dict["cout1_g"]
    weight_group = numpy.zeros((real_g, d, cin1_g, h, w, cout_g, c0), dtype=data.dtype)
    data = data.transpose([ori_format.index("N"), ori_format.index("C"),
                           ori_format.index("D"), ori_format.index("H"), ori_format.index("W")])
    for g in range(groups):
        for ci in range(c):
            for co in range(n // groups):
                try:
                    e = g % mag_factor
                    dst_cin = e * c + ci
                    dst_cout = e * (n // groups) + co
                    src_cout = g * (n // groups) + co
                    weight_group[g // mag_factor, :, dst_cin // c0, :, :, dst_cout, dst_cin % c0] = \
                        data[src_cout, ci, :, :, :]
                except:
                    e = g % mag_factor
                    dst_cin = e * c + ci
                    dst_cout = e * (n // groups) + co
                    src_cout = g * (n // groups) + co
                    print(
                        "================================== Error Detected ======================================="
                    )
                    print("weight_group shape:", weight_group.shape)
                    print("Weight Shape : ", data.shape)
                    print("co:", co)
                    print("e : ", e)
                    print("dst_cin :", dst_cin)
                    print("dst_cout : ", dst_cout)
                    print("src_cout and Ci", src_cout, "", ci)
                    print("mag_factor : ", mag_factor)
                    raise
    weight_group = weight_group.reshape([real_g * d * cin1_g * h * w, cout1_g, BLOCK_SIZE, c0])
    return weight_group


def from_fractal_z_3d(data: numpy.ndarray, target_shape: Union[list, tuple] = None,
                      target_format: str = "NCDHW", groups=None):
    if groups is None:
        groups = 1
    n = target_shape[target_format.index("N")]
    c_in = target_shape[target_format.index("C")]
    d = target_shape[target_format.index("D")]
    h = target_shape[target_format.index("H")]
    w = target_shape[target_format.index("W")]
    c0 = data.shape[-1]
    n0 = BLOCK_SIZE
    cin_ori = c_in
    cout_ori = n // groups

    group_dict = _calculate_group(c_in * groups, n, groups, c0)
    real_g = group_dict["real_g"]
    cin1_g = group_dict["cin1_g"]
    mag_factor = group_dict["mag_factor"]
    cout1_g = group_dict["cout1_g"]

    data = numpy.ascontiguousarray(data)
    data_reshaped = data.reshape((real_g, d, cin1_g, h, w, cout1_g, n0, c0))

    result = numpy.zeros((n, c_in, d, h, w), dtype=data.dtype)
    for g in range(groups):
        for ci in range(cin_ori):
            for co in range(cout_ori):
                e = g % mag_factor
                dst_cin = e * cin_ori + ci
                dst_cout = e * cout_ori + co
                src_cout = g * cout_ori + co
                result[src_cout, ci, :, :, :] = data_reshaped[
                    g // mag_factor, :, dst_cin // c0, :, :,
                    dst_cout // n0, dst_cout % n0, dst_cin % c0
                ]

    if target_format == "NDHWC":
        return result.transpose(0, 2, 3, 4, 1)
    elif target_format == "DHWCN":
        return result.transpose(2, 3, 4, 1, 0)
    return result


def to_NC1HWC0(data: numpy.ndarray, ori_format: str,
               target_shape: Union[list, tuple] = None):
    ori_shape = data.shape
    if len(ori_shape) > 4:
        raise RuntimeError("Please check original format and original shape: NC1HWC0 transformer doesn't support"
                           f"{len(ori_shape)}D shape")
    c0 = determine_c0(data.dtype.name, target_shape)
    nchw, transpose_axis = _get_ndchw_dim_and_transpose_axis(data, ori_format)
    n, c, h, w = nchw
    c1 = ceil_div(c, c0)
    data = data.transpose(transpose_axis)
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = numpy.zeros((n, num_2_padding_in_cin, h, w), dtype=data.dtype)
    data = numpy.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, h, w)).transpose(0, 1, 3, 4, 2)
    return data


def to_NDC1HWC0(data: numpy.ndarray, ori_format: str, target_shape: Union[list, tuple] = None):
    ori_shape = data.shape
    if len(ori_shape) > 5:
        raise RuntimeError("Please check original format and original shape: NDC1HWC0 transformer doesn't support"
                           f"{len(ori_shape)}D shape")
    c0 = determine_c0(data.dtype.name, target_shape)
    ndchw, transpose_axis = _get_ndchw_dim_and_transpose_axis(data, ori_format, is_ndchw=True)
    n, d, c, h, w = ndchw
    c1 = ceil_div(c, c0)
    data = data.transpose(transpose_axis)
    num_2_padding_in_cin = c1 * c0 - c
    zero_padding_array = numpy.zeros((n, num_2_padding_in_cin, d, h, w), dtype=data.dtype)
    data = numpy.concatenate((data, zero_padding_array), axis=1)
    data = data.reshape((n, c1, c0, d, h, w)).transpose(0, 3, 1, 4, 5, 2)
    return data


def nd_to_fractal_nz(data: numpy.ndarray, target_shape: Union[list, tuple] = None):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    m0, n0 = 16, determine_c0(data.dtype.name, target_shape)
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = numpy.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [2, 0, 1, 3])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def nd_to_fractal_z(data: numpy.ndarray, target_shape: Union[list, tuple] = None):
    ori_shape = data.shape
    m_ori, n_ori = ori_shape[-2:]
    batch_ori = ori_shape[:-2]
    batch_num = len(batch_ori)
    batch_padding = ((0, 0),) * batch_num
    m0, n0 = determine_c0(data.dtype.name, target_shape), 16
    m1, n1 = ceil_div(m_ori, m0), ceil_div(n_ori, n0)
    padding_m = m1 * m0 - m_ori
    padding_n = n1 * n0 - n_ori
    data = numpy.pad(data, (batch_padding + ((0, padding_m), (0, padding_n))), 'constant')
    array_trans = gen_axes_for_transpose(len(data.shape) - 2, [0, 2, 3, 1])
    data = data.reshape(batch_ori + (m1, m0, n1, n0)).transpose(*array_trans)
    return data


def is_transformable(ori_format, target_format):
    if ori_format in format_transformation_map:
        if target_format in format_transformation_map[ori_format]:
            return True
    return False


def transform(data, ori_format,
              target_format, target_shape: Union[list, tuple] = None,
              groups=None):
    if is_transformable(ori_format, target_format):
        transform_func = format_transformation_map[ori_format][target_format]
        params = inspect.signature(transform_func).parameters
        kwargs = {"target_shape": target_shape}
        if "ori_format" in params:
            kwargs.update({"ori_format": ori_format})
        if "target_format" in params:
            kwargs.update({"target_format": target_format})
        if "target_shape" in params:
            kwargs.update({"target_shape": target_shape})
        if "groups" in params:
            kwargs.update({"groups": groups})
        return format_transformation_map[ori_format][target_format](data, **kwargs)
    return None


format_transformation_map = {
    "NHWC": {
        "NC1HWC0": to_NC1HWC0,
        "FRACTAL_Z": to_fractal_z,
        "FRACTAL_Z_C04": to_fractal_z_c04
    },
    "NCHW": {
        "NC1HWC0": to_NC1HWC0,
        "FRACTAL_Z": to_fractal_z,
        "FRACTAL_Z_C04": to_fractal_z_c04
    },
    "HWCN": {
        "NC1HWC0": to_NC1HWC0,
        "FRACTAL_Z": to_fractal_z,
        "FRACTAL_Z_C04": to_fractal_z_c04
    },
    "NDHWC": {
        "NDC1HWC0": to_NDC1HWC0,
        "FRACTAL_Z_3D": to_fractal_z_3d
    },
    "NCDHW": {
        "NDC1HWC0": to_NDC1HWC0,
        "FRACTAL_Z_3D": to_fractal_z_3d
    },
    "DHWCN": {
        "NC1HWC0": to_NDC1HWC0,
        "FRACTAL_Z_3D": to_fractal_z_3d
    },
    "ND": {
        "FRACTAL_NZ": nd_to_fractal_nz,
        "FRACTAL_Z": nd_to_fractal_z,
        "FRACTAL_ZN_RNN": nd_to_fractal_z
    },
    # revert
    "NC1HWC0": {
        "NHWC": fhd2nd,
        "NCHW": fhd2nd,
        "HWCN": fhd2nd,
    },
    "NDC1HWC0": {
        "NDHWC": shd2nd,
        "NCDHW": shd2nd,
        "DHWCN": shd2nd,
    },
    "FRACTAL_NZ": {
        "ND": nz2nd,
    },
    "FRACTAL_Z_3D": {
        "NCDHW": from_fractal_z_3d,
        "NDHWC": from_fractal_z_3d,
        "DHWCN": from_fractal_z_3d,
    },
}

