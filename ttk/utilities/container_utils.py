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
Precious Container Related Utilities
"""
# Standard Packages
import math
import logging
import numpy
from typing import Any, Optional, Tuple, Sequence, Union
from functools import reduce

# Third-Party Packages
from .classes import SWITCHES

# Global Storage
global_storage: Optional[SWITCHES] = SWITCHES()


def get(container: Sequence, idx: int, out_of_range=None):
    """
    If a container contains only one element, return that element for whatever the idx is
    :param container: A container
    :param idx: index
    :param out_of_range: Out of Range return
    :return: element
    """
    if len(container) == 1:
        return container[-1]
    else:
        if idx >= len(container) or (idx < 0 and abs(idx) > len(container)):
            if out_of_range is not None:
                return out_of_range
            else:
                logging.warning("Detected out of range access to container %s, "
                                "access index is %d", str(container), idx)
        return container[idx]


def get_global_storage() -> SWITCHES:
    """
    Return global storage structure
    :return:
    """
    return global_storage


def set_global_storage(new_storage: SWITCHES):
    """
    Set global storage structure
    :return:
    """
    global global_storage
    global_storage = new_storage


def shape_product(shape: Tuple[int, ...], initial=1):
    """
    Get shape dimension product
    :param shape: A shape_like container
    :param initial: Initial value if shape is empty
    :return: int
    """
    return reduce(lambda x, y: x * y, shape, initial)


def shape_product_with_strides(shape: Union[list, tuple], strides: Union[list, tuple]):
    numel = 1
    for idx, s in enumerate(shape):
        if s == 0:
            return 0
        numel += (s - 1) * strides[idx]
    return numel


def check_equal_length(*containers: Sequence) -> bool:
    """
    Check if containers all in the same length
    :param containers: multiple containers
    :return: bool
    """
    if len(containers) == 0:
        return True
    all_length = tuple(len(container) == len(containers[0]) for container in containers)
    return all(all_length)


def is_shape(shape: tuple, allowed_type: tuple = (int,)) -> bool:
    """
    Check if input args contains shapes only
    :param shape: a container
    :param allowed_type: a container
    :return: bool
    """
    if not isinstance(shape, Sequence):
        return False
    for dim in shape:
        if not isinstance(dim, allowed_type):
            return False
    return True


def eliminate_scalar_shapes(args: tuple) -> tuple:
    """
    Replace all empty tuple like () with (1,)
    :param args: shape_like containers
    :return:
    """
    if args is None:
        return ()
    return tuple(None if i is None else i if len(i) > 0 else (1,) for i in args)


def shape_stride(shape: Union[list, tuple]):
    if not shape:  # ()
        return ()
    return tuple([shape_product(tuple(list(shape)[idx+1:])) for idx in range(len(shape))])


def sum_len_of_sequences(*args) -> int:
    """
    Get sum of the length of all sequences
    :param args:
    :return:
    """
    result = 0
    for s in args:
        result += len(s)
    return result


def parse_tiling_data(tiling_data: Any) -> Tuple[Optional[bytes], Optional[tuple]]:
    """
    Parse tiling data
    :param tiling_data:
    :return:
    """
    # little-endian int32 Tiling data
    le_int32 = numpy.dtype(numpy.int32)
    le_int32 = le_int32.newbyteorder('<')
    if isinstance(tiling_data, tuple):
        tiling_data_np_array = numpy.array(tiling_data, dtype=le_int32)
        tiling_data_bytes = tiling_data_np_array.tobytes()
        tuple_tiling_data = tuple(tiling_data_np_array)
    elif isinstance(tiling_data, bytes):
        tiling_data_bytes = tiling_data
        if len(tiling_data) % 4 == 0:
            tiling_data_np_array = numpy.frombuffer(tiling_data, dtype=le_int32)
            tuple_tiling_data = tuple(tiling_data_np_array)
        else:
            tuple_tiling_data = ("TILING_NOT_4BYTE_ALIGNED",)
    elif tiling_data is None:
        return None, None
    else:
        raise TypeError("Tiling data parsing error, received unknown object: " + str(tiling_data))
    return tiling_data_bytes, tuple_tiling_data


def get_str_tiling_data(dyn_tuple_tiling_data: tuple, dyn_compile_info: dict, dyn_tiling_key: int):
    """
    Get string tiling data
    :param dyn_tuple_tiling_data:
    :param dyn_compile_info:
    :param dyn_tiling_key:
    :return:
    """
    if dyn_tuple_tiling_data is None:
        return dyn_tiling_key, False
    if len(dyn_tuple_tiling_data) > 0:
        if "_vars" in dyn_compile_info:
            _vars = {int(k): v for k, v in dyn_compile_info["_vars"].items()}
            tiling_data_indexes = _vars.get(dyn_tiling_key) or _vars.get(str(dyn_tiling_key))
            if tiling_data_indexes is not None:
                if len(tiling_data_indexes) == len(dyn_tuple_tiling_data):
                    dict_tiling_data = dict(zip(tiling_data_indexes, dyn_tuple_tiling_data))
                    dyn_str_tiling_data = str(dict_tiling_data)
                else:
                    even_is_zero = [data == 0 for idx, data in enumerate(dyn_tuple_tiling_data) if idx % 2 == 1]
                    if len(tiling_data_indexes) * 2 == len(dyn_tuple_tiling_data) and all(even_is_zero):
                        odd_data = [data for idx, data in enumerate(dyn_tuple_tiling_data) if idx % 2 == 0]
                        dict_tiling_data = dict(zip(tiling_data_indexes, odd_data))
                        dyn_str_tiling_data = str(dict_tiling_data)
                    else:
                        dyn_str_tiling_data = f"Tiling data not match with compile_info _vars: {dyn_tuple_tiling_data}"
                        logging.warning(f"Tiling data not match with sub-keys in compile_info _vars: "
                                        f"{dyn_tuple_tiling_data} VS {tiling_data_indexes}")
            else:
                dyn_str_tiling_data = "Tiling_key %s not found in _vars key %s, treat as Tik operator" \
                                      % (str(dyn_tiling_key),
                                         str(tuple(_vars.keys())))
        else:
            dyn_str_tiling_data = f"{dyn_tuple_tiling_data}"
    else:
        dyn_str_tiling_data = "Tiling data is empty"
    return dyn_str_tiling_data


def adapt_pickup_by_names(params: dict, signatures: tuple) -> dict:
    """Pick up params matching signatures and adapt key names.

    For each param key, checks if it (or its transformed form) matches a
    signature name. Matching keys are renamed to the signature form.
    Non-matching keys are removed.

    Transformations checked (same order as pickup_attr_by_names):
        1. axis/axes alias:  axis ↔ axes
        2. _in__ suffix:     param → param_in__
        3. _in__ strip:      param_in__ → param

    Args:
        params: Parameter dict (e.g. operator attributes).
        signatures: Tuple of function parameter names.

    Returns:
        Dict with keys renamed to match signatures.
    """
    params = params.copy()
    k_r = {"axis": "axes", "axes": "axis"}
    for param in tuple(params.keys()):
        if param in k_r and k_r[param] in signatures and k_r[param] not in params:
            params[k_r[param]] = params[param]
            del params[param]
            continue
        if param in k_r:
            k_r_with_tail = f"{k_r[param]}_in__"
            if k_r_with_tail in signatures and k_r_with_tail not in params:
                params[k_r_with_tail] = params[param]
                del params[param]
                continue
        param_with_tail = f"{param}_in__"
        if param_with_tail in signatures and param_with_tail not in params:
            params[param_with_tail] = params[param]
            del params[param]
            continue
        if param.endswith('_in__'):
            param_wo_tail = param[:-5]
            if param_wo_tail in signatures and param_wo_tail not in params:
                params[param_wo_tail] = params[param]
                del params[param]
                continue
        if param not in signatures:
            del params[param]
    return params


# Backward compatible alias
param_transformation = adapt_pickup_by_names


def pickup_by_names(attrs: dict, input_names) -> dict:
    """Extract attributes whose keys match input parameter names.

    Uses the same mapping logic as pickup_param_by_signatures: direct name match,
    _in__ suffix, and axis/axes alias. Keys are preserved in their original form
    preserved in their original form (no renaming).

    Args:
        attrs: Operator attributes dict (e.g. testcase.attributes).
        input_names: Sequence of input parameter names (e.g. from op_info["inputs"]).

    Returns:
        Dict of attributes that match input names. Caller can derive the
        remainder via `{k: v for k, v in attrs.items() if k not in result}`.
    """
    input_names = set(input_names)
    _ALIAS = {"axis": "axes", "axes": "axis"}
    result = {}
    for key, value in attrs.items():
        if key in input_names:
            result[key] = value
        elif key.endswith("_in__"):
            base = key[:-5]
            if base in input_names or (base in _ALIAS and _ALIAS[base] in input_names):
                result[key] = value
        elif f"{key}_in__" in input_names:
            result[key] = value
        elif key in _ALIAS and _ALIAS[key] in input_names:
            result[key] = value
    return result


def deep_flatten(sequence: Sequence) -> tuple:
    """Recursively flatten all tuple/list/set nesting.

    For atomic elements (ndarray, str, scalar): the element itself is never
    traversed into. Only tuple/list/set containers are recursively expanded.

    ((arr1, arr2), arr3) → (arr1, arr2, arr3)
    (('f32', 'f32'), 'int32') → ('f32', 'f32', 'int32')
    (arr1, arr2) → (arr1, arr2)  # already flat
    """
    result = []
    for ele in sequence:
        if isinstance(ele, (tuple, list, set)):
            for true_ele in deep_flatten(ele):
                result.append(true_ele)
        else:
            result.append(ele)
    return tuple(result)


def shape_like_flatten(nested) -> tuple:
    """Flatten one level of tuple-of-tuples nesting, preserving shape tuples.

    Unlike deep_flatten which recursively flattens everything, this preserves
    leaf tuples (like (3,3)) by only flattening when element[0] is also a tuple.
    Designed for shape/range structures.

    (((3,3),(3,2)),(3,5)) → ((3,3),(3,2),(3,5))
    ((8,16),(32,)) → ((8,16),(32,))  # no nested groups, preserved
    None values are preserved.
    """
    flat = []
    for element in nested:
        if element is None:
            flat.append(None)
        elif isinstance(element, (tuple, list)) and len(element) > 0:
            if isinstance(element[0], (tuple, list)):
                flat.extend(element)
            else:
                flat.append(element)
        else:
            flat.append(element)
    return tuple(flat)


# backward compat aliases
tuple_flatten = deep_flatten
flatten_nested_sequence = shape_like_flatten


def infer_list_distribution_from_nesting(nested):
    """Infer tensor_list_distribution from nested structure.

    Each top-level element:
      - tuple/list of shapes (tuple-of-tuples) → TensorList, count = len
      - single shape (tuple-of-ints) or None → single tensor, count = 0

    Example: (((3,3),(3,2)),(3,5)) → (2, 0)
    """
    distribution = []
    for element in nested:
        if element is None:
            distribution.append(0)
        elif isinstance(element, (tuple, list)) and len(element) > 0:
            if isinstance(element[0], (tuple, list)):
                distribution.append(len(element))
            else:
                distribution.append(0)
        else:
            distribution.append(0)
    return tuple(distribution)


def apply_as_list(inputs: Sequence, as_list_distribution: Sequence):
    """ fold case's input & output """
    if as_list_distribution:
        result = []
        last_num = 0
        for num in as_list_distribution:
            if num == 0:
                result.append(inputs[last_num])
                last_num += 1
            else:
                result.append(inputs[last_num:last_num + num])
                last_num += num
        if last_num < len(inputs):
            result.extend(inputs[last_num:])
    else:
        result = inputs
    return result


def split_fused_tensors(fused_tensors: Sequence, flatten_inputs_count: int,
                        is_dynamic: bool = True) -> Tuple[list, list]:
    input_tensor_count = 0
    folded_inputs, folded_outputs = [], []
    for i in fused_tensors:
        if input_tensor_count == flatten_inputs_count:
            folded_outputs.append(i)
        elif input_tensor_count > flatten_inputs_count:
            mode = "dyn" if is_dynamic else "cst"
            raise RuntimeError(f"tensor_list_distribution is invalid, exceeded size of {mode}_inputs.")
        else:
            folded_inputs.append(i)
            if isinstance(i, Sequence):
                input_tensor_count += len(i)
            else:
                input_tensor_count += 1
    return folded_inputs, folded_outputs


def input_apply_as_list(inputs: Sequence, as_list_distribution: Sequence):
    """ fold case's input only """
    if as_list_distribution:
        result = []
        last_num = 0
        for num in as_list_distribution:
            if last_num >= len(inputs):
                break
            if num == 0:
                result.append(inputs[last_num])
                last_num += 1
            else:
                result.append(inputs[last_num:last_num + num])
                last_num += num
        if last_num < len(inputs):
            result.extend(inputs[last_num:])
    else:
        result = inputs
    return result


def output_apply_as_list(outputs: Sequence, as_list_distribution: Sequence, input_count: int):
    """ fold case's output only """
    if as_list_distribution:
        result = []
        last_num = 0
        for num in as_list_distribution:
            if last_num < input_count:
                last_num += (1 if num == 0 else num)
                continue
            if last_num >= input_count + len(outputs):
                break
            if num == 0:
                result.append(outputs[last_num - input_count])
                last_num += 1
            else:
                result.append(outputs[(last_num - input_count):(last_num - input_count + num)])
                last_num += num
        if last_num < input_count + len(outputs):
            result.extend(outputs[(last_num - input_count):])
    else:
        result = outputs
    return result


def table_print(data: Sequence[Tuple[str, ...]]):
    result = ""
    cells_width = [0]
    sub_row_size = []
    minimum_table_width = 0
    # Determine Table Width
    for row in data:
        sub_row_size.append(0)
        if len(row) == 1:
            sub_rows = row[0].split("\n")
            for sub_row in sub_rows:
                if minimum_table_width < len(sub_row) + 2:
                    minimum_table_width = len(sub_row) + 2
            sub_row_size[-1] = len(sub_rows)
        else:
            for idx, column in enumerate(row):
                if len(cells_width) <= idx:
                    cells_width.append(0)
                sub_rows = str(column).split("\n")
                for sub_row in sub_rows:
                    if cells_width[idx] < len(sub_row) + 2:
                        cells_width[idx] = len(sub_row) + 2
                if sub_row_size[-1] < len(sub_rows):
                    sub_row_size[-1] = len(sub_rows)

    # Check for Minimum Table Width Requirements
    if sum(cells_width) < minimum_table_width:
        for idx, cell_width in enumerate(cells_width):
            if cell_width < minimum_table_width // len(cells_width):
                cells_width[idx] = math.ceil(minimum_table_width / len(cells_width))
    char_length = sum(cells_width) + len(cells_width) * 2 - 1
    for idx, row in enumerate(data):
        length = len(row)
        result += '+' + char_length * '-' + '+'
        result += '\n'
        sub_result = []
        for sub_row_idx in range(sub_row_size[idx]):
            if length == 1:
                sub_result.append('| ' + str(row[0]).split('\n')[sub_row_idx].ljust(
                    sum(cells_width) + 2 * len(cells_width) - 2) + '|')
            else:
                sub_row = [str(column).split('\n')[sub_row_idx] if sub_row_idx < len(str(column).split('\n')) else ""
                           for column in row]
                sub_result.append('|' + '|'.join(' ' + sub_row[i].ljust(cells_width[i])
                                                 for i in range(length)) + '|')
        result += '\n'.join(sub_result)
        result += '\n'
    result += '+' + char_length * '-' + '+'
    return result


def frameless_table_print(data: list) -> str:
    """
    print table like this:
    ----------------- ------------- ------------- ---------- -------------
                 Name           avg           max        min    # of Calls
    ----------------- ------------- ------------- ---------- -------------
                item1           1.1           2.2        3.3             3
                item2           2.2           3.3        4.4             3
    ----------------------------------------------------------------------
    """
    min_cell_width = 10
    cols = len(data[0])
    cols_width = [min_cell_width] * cols
    for row in data:
        for idx, col in enumerate(row):
            cols_width[idx] = max(cols_width[idx], len(str(col)))
    result = '\n' + ' '.join([w * '-' for w in cols_width])
    result += '\n' + ' '.join([col.rjust(cols_width[idx])
                               for idx, col in enumerate(data[0])])
    result += '\n' + ' '.join([w * '-' for w in cols_width])
    for row in data[1:]:
        result += '\n' + ' '.join([str(col).rjust(cols_width[idx])
                                   for idx, col in enumerate(row)])
    result += '\n' + '-'.join([w * '-' for w in cols_width])
    return result


def list_append_union(lst1, lst2):
    for ele in lst2:
        if ele not in lst1:
            lst1.append(ele)


def list_exclude(lst1, lst2):
    for ele in lst1:
        if ele in lst2:
            lst1.remove(ele)
