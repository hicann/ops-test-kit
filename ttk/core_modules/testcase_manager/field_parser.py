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
Parser for each csv field
"""


__all__ = ["process_bool", "process_string",
           "process_dynamic_shapelike", "process_dynamic_inferable_shapelike",
           "shapelike_stc", "shapelike_stc_ex",
           "shapelike_float", "shapelike_float_signed",
           "rangelike", "shape_stride",
           "shapelike_stc_nested", "shapelike_stc_ex_nested",
           "shapelike_float_signed_nested", "shapelike_float_nested",
           "string_container",
           "scalar_nested",
           "int_container",
           "process_eval",
           "process_int", "process_float", "process_dict"]


# Standard Packages
import logging
import re
from typing import Union

# Third-Party Packages
from ...utilities import is_shape


def _is_inference(value: str) -> bool:
    # Check if inference is needed
    if value in ["ELEWISE", "REDUCE"]:
        return True
    # Check if limited inference is needed
    if "ELEWISE" in value and len(value) > 7:
        if isinstance(eval(value[7:]), (tuple, list)):
            return True
        else:
            raise ValueError("%s is not a valid limited inference value!" % value)
    if "REDUCE" in value and len(value) > 6:
        if isinstance(eval(value[6:]), (tuple, list)):
            return True
        else:
            raise ValueError("%s is not a valid limited inference value!" % value)
    return False


def process_bool(value: str) -> bool:
    value = value.upper()
    if value == "TRUE":
        value = "True"
    elif value == "FALSE":
        value = "False"
    try:
        parsed = eval(value)
    except:
        raise TypeError("%s is not a valid boolean value!" % value)
    if parsed:
        return True
    return False


def process_string(value) -> str:
    # noinspection PyBroadException
    try:
        result = eval(value)
    except:
        return value
    else:
        if isinstance(result, type(None)):
            return result
        else:
            return value


def _shapelike(value: str,
               allow_inference=False, positive_only=False,
               allow_float=False, allow_sub_value_none=False,
               allow_none=False, allow_empty_tensor=False,
               allow_inf=False) -> Union[tuple, str, type(None)]:
    if allow_float:
        # convert -0.0 or -0 to -float(0) for powf & atan operator
        value = value.replace('\'', '').replace('\"', '')
        value = re.sub(r'(-0+\.0*|-0+\.?)(?!(\d|\.))', r'-float(0)', value)
        if allow_inf:
            # convert inf to float(inf)
            value = re.sub(r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))',
                           lambda match: f'float("{match.group(1)}")',
                           value)
    if allow_inference:
        if _is_inference(value):
            return value
    parsed = eval(value)
    # No inference needed, treat as normal
    if parsed is None and allow_none:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid shapelike value!" % value)
    # Make sure all shapelike container is a tuple
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed
    # Iterate through all sub_values, set allowed_type
    allowed_type = (int,)
    if allow_float:
        allowed_type += (float,)
    if allow_sub_value_none:
        allowed_type += (type(None),)
    if all(isinstance(sub_value, allowed_type) for sub_value in parsed):
        # sub_value is single value, convert to tuple and return
        if positive_only and not (all((i > 0 for i in parsed if i is not None)) or
                                  (any([i == 0 for i in parsed]) if allow_empty_tensor else False)):
            raise ValueError("shapelike value should not have %s dim %s" %
                            ("non-positive" if not allow_empty_tensor else "negative", value))
        return parsed,
    for sub_value in parsed:
        if isinstance(sub_value, (tuple, list)):
            if not is_shape(sub_value, allowed_type):
                raise TypeError("%s is not a valid shape in type %s" % (str(sub_value), str(allowed_type)))
            if not all((isinstance(i, allowed_type) for i in sub_value)):
                raise ValueError("shapelike value should not have invalid dim %s" % value)
            if positive_only and not (all((i > 0 for i in sub_value)) or
                                      (any([i == 0 for i in sub_value]) if allow_empty_tensor else False)):
                raise ValueError("shapelike value should not have %s dim %s" %
                                ("non-positive" if not allow_empty_tensor else "negative", value))
        elif isinstance(sub_value, type(None)):
            pass
        else:
            raise TypeError("%s of %s is not a valid shapelike value for its corresponding field" % (str(sub_value),
                                                                                                      value))
    new_parsed = tuple(tuple(element) if element is not None else None for element in parsed)
    return new_parsed


def process_dynamic_shapelike(value: str) -> tuple:
    """
    For shapelike (1, 7, -1)
    :param value:
    :return:
    """
    return _shapelike(value, allow_none=True)


def process_dynamic_inferable_shapelike(value: str):
    """
    For shapelike REDUCE or (1, 7, -1)
    :param value:
    :return:
    """
    return _shapelike(value, allow_inference=True, allow_sub_value_none=True, allow_none=True)


def shapelike_stc(value: str):
    """
    For shapelike (34, 16, 16)
    :param value:
    :return:
    """
    return _shapelike(value, positive_only=True, allow_empty_tensor=True)


def shapelike_stc_ex(value: str):
    """
    For shapelike ELEWISE or (34, 16, 16)
    :param value:
    :return:
    """
    return _shapelike(value, allow_inference=True, positive_only=True,
                      allow_sub_value_none=True, allow_empty_tensor=True)


def shapelike_float(value: str):
    """
    For shapelike (1.1001, 3.263)
    :param value:
    :return:
    """
    return _shapelike(value, positive_only=True, allow_float=True,
                      allow_none=True, allow_empty_tensor=True)


def shapelike_float_signed(value: str):
    """
    For shapelike (None, -1.129) or (1.236, None)
    :param value:
    :return:
    """
    return _shapelike(value, allow_float=True, allow_sub_value_none=True, allow_inf=True)


def shape_stride(value: str):
    return _shapelike(value, positive_only=True, allow_none=True)


def shapelike_stc_nested(value: str):
    """Parse shape with TensorList nesting support.

    Supports 3-level nesting where:
      - ((3,3),(3,5))         → flat: two single tensors
      - (((3,3),(3,2)),(3,5)) → nested: TensorList of 2 + single tensor
      - None elements are allowed

    Returns the nested structure as-is for later flattening.
    """
    parsed = eval(value)
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid shapelike value!" % value)
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    if all(isinstance(e, int) for e in parsed):
        if not all(d >= 0 for d in parsed):
            raise ValueError("shapelike value should not have negative dim %s" % str(parsed))
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_shapes = []
                for sub in element:
                    if sub is None:
                        sub_shapes.append(None)
                    elif isinstance(sub, (tuple, list)):
                        if not is_shape(sub, (int,)):
                            raise TypeError("%s is not a valid shape in %s" % (str(sub), value))
                        if not all(d >= 0 for d in sub):
                            raise ValueError("shapelike value should not have negative dim %s" % str(sub))
                        sub_shapes.append(tuple(sub))
                    else:
                        raise TypeError("%s is not a valid shape in %s" % (str(sub), value))
                result.append(tuple(sub_shapes))
            else:
                if not is_shape(element, (int,)):
                    raise TypeError("%s is not a valid shape in %s" % (str(element), value))
                if not all(d >= 0 for d in element):
                    raise ValueError("shapelike value should not have negative dim %s" % str(element))
                result.append(tuple(element))
        else:
            raise TypeError("%s of %s is not a valid shapelike value" % (str(element), value))
    return tuple(result)


def shapelike_stc_ex_nested(value: str):
    """shapelike_stc_nested + string inference support (ELEWISE/REDUCE)."""
    if _is_inference(value):
        return value
    return shapelike_stc_nested(value)


def shapelike_float_signed_nested(value: str):
    """Parse signed float shapelike with TensorList nesting support.

    Supports same nesting as shapelike_stc_nested but allows float and None values.
    e.g. ((None, 1.0), (-1.0, 1.0))                → flat
         (((None, 1.0), (None, 1.0)), (-1.0, 1.0))  → nested TensorList
    """
    value = re.sub(r'(-0+\.0*|-0+\.?)(?!(\d|\.))', r'-float(0)', value)
    value = re.sub(r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))',
                   lambda match: f'float("{match.group(1)}")',
                   value)
    allowed = (int, float, type(None))
    parsed = eval(value)
    if parsed is None:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid shapelike value!" % value)
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    if all(isinstance(e, allowed) for e in parsed):
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_items = []
                for sub in element:
                    if sub is None:
                        sub_items.append(None)
                    elif isinstance(sub, (tuple, list)):
                        for v in sub:
                            if not isinstance(v, allowed):
                                raise TypeError("%s is not a valid float value in %s" % (str(v), value))
                        sub_items.append(tuple(sub))
                    else:
                        raise TypeError("%s is not a valid float range in %s" % (str(sub), value))
                result.append(tuple(sub_items))
            else:
                for v in element:
                    if not isinstance(v, allowed):
                        raise TypeError("%s is not a valid float value in %s" % (str(v), value))
                result.append(tuple(element))
        else:
            raise TypeError("%s of %s is not a valid float shapelike value" % (str(element), value))
    return tuple(result)


def shapelike_float_nested(value: str):
    """Parse float shapelike with TensorList nesting support (positive only).

    Same nesting as shapelike_float_signed_nested but enforces positive values.
    e.g. ((0.001, 0.001), (0.001, 0.001))               → flat
         (((0.001, 0.001), (0.001, 0.001)), (0.001, 0.001))  → nested TensorList
    """
    parsed = eval(value)
    if parsed is None:
        return None
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid shapelike value!" % value)
    parsed = tuple(parsed)
    if len(parsed) == 0:
        return parsed

    allowed = (int, float)
    if all(isinstance(e, allowed) for e in parsed):
        if not all(d >= 0 for d in parsed if isinstance(d, (int, float))):
            logging.warning("shapelike value should not have negative dim %s" % str(parsed))
        return (parsed,)

    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if len(element) == 0:
                result.append(tuple(element))
            elif isinstance(element[0], (tuple, list)):
                sub_items = []
                for sub in element:
                    if sub is None:
                        sub_items.append(None)
                    elif isinstance(sub, (tuple, list)):
                        for v in sub:
                            if not isinstance(v, allowed):
                                raise TypeError("%s is not a valid float value in %s" % (str(v), value))
                            if v < 0:
                                logging.warning("shapelike value should not have negative dim %s" % str(sub))
                        sub_items.append(tuple(sub))
                    else:
                        raise TypeError("%s is not a valid float range in %s" % (str(sub), value))
                result.append(tuple(sub_items))
            else:
                for v in element:
                    if not isinstance(v, allowed):
                        raise TypeError("%s is not a valid float value in %s" % (str(v), value))
                    if v < 0:
                        logging.warning("shapelike value should not have negative dim %s" % str(element))
                result.append(tuple(element))
        else:
            raise TypeError("%s of %s is not a valid float shapelike value" % (str(element), value))
    return tuple(result)


def rangelike(value: str):
    """
    For multiple shapelike ((None, 3), (55, None)
    :param value:
    :return:
    """
    parsed = eval(value)
    if not isinstance(parsed, (tuple, list)):
        raise TypeError("%s is not a valid rangelike value." % value)
    result = []
    for _range in parsed:
        result.append(_shapelike(str(_range), allow_none=True, allow_sub_value_none=True))
    return tuple(result)


def _container(value: str, allowed_type: Union[type, tuple, list]):
    result = eval(value)
    for t in allowed_type:
        if isinstance(result, t):
            if result is None:
                return result
            return (result,)
    result = tuple(result)
    for element in result:
        if allowed_type and not isinstance(element, allowed_type):
            raise TypeError("Received type %s for element %s instead of %s"
                            % (str(type(element)), str(element), str(allowed_type)))
    return result


def string_container(value: str) -> tuple:
    """
    Container for multiple string
    :param value:
    :return:
    """
    # noinspection PyBroadException
    try:
        result = _container(value, (str, type(None)))
    except:
        result = (value,)
    return result


def scalar_nested(value: str, allowed_type=None) -> tuple:
    """Scalar field with TensorList nesting support.

    Handles string, float, and int element types.
    'float32'                              → ('float32',)
    1e-08                                  → (1e-08,)
    ('float32', 'float32')                 → ('float32', 'float32')
    (1e-08, 1e-08)                         → (1e-08, 1e-08)
    (('float32','float32'), 'float32')     → (('float32','float32'), 'float32')
    ''                                     → ()

    :param allowed_type: type or tuple of types for element validation.
                         None means no validation.
    """
    if value == "":
        return ()
    try:
        parsed = eval(value)
    except Exception:
        raise TypeError("%s is not a valid scalar value!" % value)
    if not isinstance(parsed, (tuple, list)):
        if allowed_type and not isinstance(parsed, allowed_type):
            raise TypeError(
                "Value %r (type %s) is not %s" %
                (parsed, type(parsed).__name__, allowed_type))
        return (parsed,)
    result = []
    for element in parsed:
        if element is None:
            result.append(None)
        elif isinstance(element, (tuple, list)):
            if allowed_type:
                for e in element:
                    if not isinstance(e, allowed_type):
                        raise TypeError(
                            "Element %r (type %s) in nested group is not %s" %
                            (e, type(e).__name__, allowed_type))
            result.append(tuple(element))
        else:
            if allowed_type and not isinstance(element, allowed_type):
                raise TypeError(
                    "Element %r (type %s) is not %s" %
                    (element, type(element).__name__, allowed_type))
            result.append(element)
    return tuple(result)


def process_eval(value: str):
    return eval(value)


def int_container(value: str) -> tuple:
    """
    Container for multiple integer
    :param value:
    :return:
    """
    if value == "":
        return ()
    try:
        result = _container(value, (int, type(None)))
    except TypeError as terr:
        raise TypeError(("Invalid value %s for int_container: " + str(terr.args))
                        % value)
    except Exception as e:
        raise ValueError(("Invalid value %s for int_container: " + str(e.args))
                         % value)
    else:
        return result



    return tuple(result)


def process_int(value: str) -> int:
    """
    Integer
    :param value:
    :return:
    """
    try:
        result = eval(value)
    except:
        raise TypeError(f"Invalid value [{value}] for int")
    else:
        if isinstance(result, int):
            return result
        else:
            raise TypeError(f"Invalid value [{value}] for int")


def process_float(value: str) -> float:
    """
    Integer
    :param value:
    :return:
    """
    try:
        result = float(value)
    except:
        raise TypeError(("Invalid value %s for float: " + value)
                        % value)
    else:
        return result


def process_dict(value: str) -> dict:
    """
    dictionary
    :param value:
    :return:
    """
    try:
        # convert -0.0 or -0 to -float(0) for powf & atan operator
        value = re.sub(r'(-0+\.0*|-0+\.?)(?!(\d|\.))', r'-float(0)', value)
        # convert inf to float(inf), nan to float(nan)
        # but skip when nan/inf is inside single/double quotes (dict key)
        def _replace_nan_inf(m):
            start = m.start()
            if start > 0 and value[start - 1] in ("'", '"'):
                return m.group(0)
            return f'float("{m.group(1)}")'
        value = re.sub(r'(?<!float\(["\'])\b(inf|nan)\b(?!["\']\))',
                       _replace_nan_inf,
                       value)
        result = eval(value)
    except:
        raise ValueError("Invalid value %s for dict" % value)
    else:
        if isinstance(result, dict):
            return result
        elif isinstance(result, type(None)):
            return result
        else:
            raise TypeError("Value %s is not a dict" % value)
