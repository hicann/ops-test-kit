#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
File utils
"""

__all__ = [
    "get_file_md5",
    "load_numpy_data",
    "dump_to_file",
    "read_file",
    "get_loaded_so_path",
    "delete_files_in_folder",
    "extract_csv_cells",
    "smart_extract",
]


import csv
import hashlib
import os
import shutil
import zipfile
from ctypes import CDLL
from subprocess import check_output
from typing import TYPE_CHECKING, Optional, Union

import numpy

from .container_utils import shape_product
from .dtypes import is_4bit_dtype, numpy_to_torch_tensor, pack_4bits, torch_to_numpy_tensor, unpack_4bits

if TYPE_CHECKING:
    import torch


def get_file_md5(file_path: str):
    block_size = 2**20
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while True:
            chunk = f.read(block_size)
            if chunk:
                file_hash.update(chunk)
            else:
                break
    return file_hash.hexdigest()


def _resolve_load_dtype(dtype):
    """Map special dtypes to a numpy-loadable dtype for raw file reading."""
    if dtype == "uint1":
        return "uint8"
    if dtype == "complex32":
        return "float16"
    if is_4bit_dtype(dtype):
        return "uint8"
    return dtype


def _postprocess_4bit(array, ori_dtype, shape):
    """Unpack or view 4-bit data after loading."""
    if array.dtype.name in ("int8", "uint8"):
        size = shape_product(shape)
        if array.size == size:
            array = array.view(ori_dtype)
        else:
            array = unpack_4bits(array, ori_dtype)
            array = array[:size]
    else:
        array = array.view(ori_dtype)
    return array


def _postprocess_special(array, ori_dtype):
    """View-cast special dtypes loaded from .npy (stored as uint8/int8)."""
    for sd in ("bfloat16", "float8_e4m3fn", "float8_e8m0", "hifloat8"):
        if sd in str(ori_dtype) and array.dtype.name != sd:
            array = array.view(ori_dtype)
            break
    return array


def load_numpy_data(file_name: str, dtype: str, shape: Union[list, tuple]):
    if not isinstance(dtype, str) and hasattr(dtype, "name"):
        if dtype.name in ("float8_e5m2",) and file_name.endswith(".npy"):
            raise RuntimeError(
                f".npy file is not supported to be loaded for dtype: {dtype.name}. "
                f"ValueError [descr is not a valid dtype descriptor: '<f1'] will be threw."
            )
    ori_dtype = dtype
    dtype = _resolve_load_dtype(dtype)

    if file_name.endswith(".pt"):
        import torch

        array = torch_to_numpy_tensor(torch.load(file_name, map_location="cpu"))
    elif file_name.endswith(".npy"):
        array = numpy.load(file_name, allow_pickle=True)
    else:
        array = numpy.fromfile(file_name, dtype=dtype)

    if is_4bit_dtype(ori_dtype):
        array = _postprocess_4bit(array, ori_dtype, shape)
    array = _postprocess_special(array, ori_dtype)
    if ori_dtype == "int1":
        shape = list(shape[:-1]) + [shape[-1] // 8]
    elif ori_dtype == "complex32":
        shape = list(shape) + [2]
    try:
        return array.reshape(shape)
    except ValueError:
        # in manual output scenario:
        # output shape may be wrong (needs infer-shaping by golden)
        return array


# TODO: abstract BaseDumperClass


def _dump_array_npy(data, npy_file_abspath):
    if isinstance(data, numpy.ndarray):
        numpy.save(npy_file_abspath, data)
    elif "torch.Tensor" in str(type(data)):
        numpy.save(npy_file_abspath, torch_to_numpy_tensor(data))


def _dump_array_pt(data, pt_file_abspath):
    import torch

    if isinstance(data, numpy.ndarray):
        torch.save(numpy_to_torch_tensor(data), pt_file_abspath)
    elif "torch.Tensor" in str(type(data)):
        torch.save(data, pt_file_abspath)


def _dump_array_print(data):
    print(data)


def _decode_binary_to_array(data, dtype):
    """Decode raw bytes to numpy array based on dtype."""
    if dtype is None:
        return None
    if "uint1" == str(dtype):
        return numpy.frombuffer(data, dtype="uint8")
    if not is_4bit_dtype(dtype):
        dtype = "float16" if str(dtype) == "complex32" else dtype
        return numpy.frombuffer(data, dtype=dtype), True
    return unpack_4bits(numpy.frombuffer(data, dtype=numpy.uint8), dtype)


def _dump_binary_data(data, dtype, file_format, npy_file_abspath, pt_file_abspath):
    """Dump raw bytes data to npy/pt/print."""
    if dtype is None:
        if file_format in ("npy", "pt"):
            with open(npy_file_abspath if file_format == "npy" else pt_file_abspath, "wb+") as f:
                f.write(data)
        else:
            print(data)
        return

    result = _decode_binary_to_array(data, dtype)
    support_pt = False
    if isinstance(result, tuple):
        np_array, support_pt = result
    else:
        np_array = result

    if file_format == "npy" or (file_format == "pt" and not support_pt):
        numpy.save(npy_file_abspath, np_array)
    elif file_format == "pt":
        import torch

        torch.save(numpy_to_torch_tensor(np_array), pt_file_abspath)
    else:
        print(np_array)


def _dump_bin(data, bin_file_abspath):
    """Dump numpy/torch/bytes data to binary file."""
    if "torch.Tensor" in str(type(data)):
        data = torch_to_numpy_tensor(data)
    if isinstance(data, numpy.ndarray):
        if is_4bit_dtype(data.dtype):
            compressed = pack_4bits(data)
            compressed.tofile(bin_file_abspath)
        else:
            data.tofile(bin_file_abspath)
    elif not isinstance(data, str) and data is not None:
        with open(bin_file_abspath, "wb+") as f:
            f.write(data)


def dump_to_file(
    data: Union[numpy.ndarray, bytes, "torch.Tensor"],
    file_path: str,
    file_name: str,
    file_format: str = "bin",
    dtype: Optional[str] = None,
):
    bin_file_abspath, npy_file_abspath, pt_file_abspath = "", "", ""
    if file_format != "print":
        bin_file_abspath = os.path.abspath(os.path.join(file_path, f"{file_name}.bin"))
        npy_file_abspath = os.path.abspath(os.path.join(file_path, f"{file_name}.npy"))
        pt_file_abspath = os.path.abspath(os.path.join(file_path, f"{file_name}.pt"))
    if file_format == "pt":
        pass
    if file_format in ("npy", "print", "pt"):
        if isinstance(data, (numpy.ndarray,)) or "torch.Tensor" in str(type(data)):
            if file_format == "npy":
                _dump_array_npy(data, npy_file_abspath)
            elif file_format == "pt":
                _dump_array_pt(data, pt_file_abspath)
            else:
                _dump_array_print(data)
        elif not isinstance(data, str) and data is not None:
            _dump_binary_data(data, dtype, file_format, npy_file_abspath, pt_file_abspath)
    else:
        _dump_bin(data, bin_file_abspath)


def read_file(file_path: str, size_limit: int = 1024 * 1024 * 1024) -> bytes:
    """
    :param file_path: Path to the file
    :param size_limit: Raise an Exception if the file is too large
    :return: binary object
    """
    file_size = os.stat(file_path).st_size
    if file_size > size_limit:
        raise OSError(f"File is too large! Size of {file_path} exceeds the limit: {size_limit}")
    with open(file_path, "rb") as file:
        file_content = file.read()
    return file_content


def get_loaded_so_path(loaded_cdll: CDLL) -> str:
    """
    :param loaded_cdll:
    :return:
    """
    try:
        # noinspection PyProtectedMember
        results = [
            o.split(" ")[-1]
            for o in check_output(["lsof", "-p", str(os.getpid())], encoding="UTF-8").split("\n")
            if loaded_cdll._name in o
        ]
    except (FileNotFoundError, RuntimeError):
        return "UNKNOWN"
    else:
        return "|".join(results) if len(results) > 0 else "UNKNOWN"


def delete_files_in_folder(folder_path: str, include_fn=None, exclude_fn=None):
    # only delete files, but keep folders.
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return
    for root, _, files in os.walk(folder_path, topdown=False):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                if exclude_fn and exclude_fn(file_path):
                    continue
                if include_fn:
                    if include_fn(file_path):
                        os.unlink(file_path)
                else:
                    os.unlink(file_path)
            except Exception:
                pass


def extract_csv_cells(filename, extract_cols, col_name_mapping: dict = None, cmp=None) -> list:
    """
    Extract columns of each row.
    multi-columns fills to a dict.
    multi-rows performs a list.
    """
    results = []
    if col_name_mapping is None:
        col_name_mapping = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if cmp is None or cmp(row):
                data_dict = {}
                for _, c in enumerate(extract_cols):
                    map_name = col_name_mapping.get(c, c)
                    try:
                        data_dict.update({map_name: float(row[c])})
                    except ValueError:
                        data_dict.update({map_name: row[c]})
                results.append(data_dict)
    return results


def move_contents(source_dir, target_dir):
    """
    move files under @source_dir to @target_dir
    """
    os.makedirs(target_dir, mode=0o700, exist_ok=True)

    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        if os.path.exists(target_path):
            if os.path.isdir(source_path):
                move_contents(source_path, target_path)
            else:
                shutil.move(source_path, target_path)
        else:
            shutil.move(source_path, target_path)


def smart_extract(zip_file, extract_to, remove_top_level=False):
    if not remove_top_level:
        with zipfile.ZipFile(zip_file, mode="r") as zf:
            zf.extractall(extract_to)
    else:
        tmp_dir = f"{extract_to}_tmp"
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(tmp_dir)
        tmp_contents = os.listdir(tmp_dir)
        if len(tmp_contents) == 1:
            top_level_path = os.path.join(tmp_dir, tmp_contents[0])
            if os.path.isdir(top_level_path):
                move_contents(top_level_path, extract_to)
            else:
                shutil.move(top_level_path, extract_to)
        else:
            move_contents(tmp_dir, extract_to)
        shutil.rmtree(tmp_dir)
