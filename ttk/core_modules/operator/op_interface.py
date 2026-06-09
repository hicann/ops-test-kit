#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/env python3
"""
Operator Compilation Interface
"""
# Standard Packages
import copy
import functools
import inspect
import json
import logging
import numpy
import os
import re
import time
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
from collections import OrderedDict

from typing import Any, Optional, Sequence, Tuple, Union


# Third-Party Packages
from .tbe_interface import Opc
from .op_info_keeper import OpInfoKeeper
from ..testcase_manager import TestcaseOp
from ...utilities import BinaryCompilationResult, DynamicCompilationResult, Singleton
from ...utilities import ceil_div, get, get_global_storage, lcm
from ...utilities import param_transformation, read_file, tuple_flatten, extract_plog_errors
from ...utilities import get_dtype_width, resolve_custom_numpy_dtypes




class CaseNotSupportedError(Exception):
    pass


class OperatorNotFoundError(Exception):
    pass


# noinspection PyBroadException
class OperatorInterface(metaclass=Singleton):
    """
    Class Interface for Operator Definition and its Compilation
    """

    def __init__(self):
        self._opc = Opc()

    @staticmethod
    def prepare_operator_parameters(testcase: TestcaseOp,
                                    mode: str) -> Tuple[tuple, tuple]:
        """
        This method is intended to construct operator input / output dict
        """
        if mode.lower() in ("dyn",):
            ipt, opt = testcase.dyn_tensor_dict
            op_name = testcase.op_name
        elif mode.lower() in ("bin",):
            ipt, opt = testcase.bin_tensor_dict
            op_name = testcase.op_name
        else:
            raise RuntimeError(f"mode [{mode}] is not supported.")
        if OpInfoKeeper().op_output_defined(op_name):  # for ApplyAdamWV2
            return ipt, opt
        else:  # for ApplyAdamWV2
            return ipt, ()

    @staticmethod
    def _build_tensor_dict_group(shapes, dtypes, formats, ori_shapes, ori_formats, ranges):
        """Build a tuple of dicts for a TensorList position (one dict per sub-tensor)."""
        group = []
        for sub_idx, sub_shape in enumerate(shapes):
            if sub_shape is None:
                group.append(None)
            else:
                sub_dtype = dtypes[sub_idx] if isinstance(dtypes, (tuple, list)) else dtypes
                sub_fmt = formats[sub_idx] if isinstance(formats, (tuple, list)) else formats
                sub_ori = ori_shapes[sub_idx] if isinstance(ori_shapes, (tuple, list)) else ori_shapes
                sub_ori_fmt = ori_formats[sub_idx] if isinstance(ori_formats, (tuple, list)) else ori_formats
                sub_range = ranges[sub_idx] if isinstance(ranges, (tuple, list)) else ranges
                group.append({"shape": sub_shape,
                              "ori_shape": sub_ori,
                              "range": sub_range,
                              "dtype": sub_dtype,
                              "format": sub_fmt,
                              "ori_format": sub_ori_fmt})
        return tuple(group)

    @staticmethod
    def _build_tensor_dict(shape, ori_shape, dtype, fmt, ori_fmt, range_):
        return {"shape": shape, "ori_shape": ori_shape, "range": range_,
                "dtype": dtype, "format": fmt, "ori_format": ori_fmt}

    def _resolve_const_value(self, ip_n, dyn_func_params, possible_const_input_dict,
                             attributes, stc_dtypes):
        key = dyn_func_params[ip_n]
        if key in possible_const_input_dict:
            value = possible_const_input_dict[key]
        elif key in attributes:
            value = attributes[key]
        else:
            return None
        if not isinstance(value, Sequence):
            value = (value,)
        my_dtype = get(stc_dtypes, ip_n)
        if isinstance(my_dtype, (tuple, list)):
            my_dtype = my_dtype[0]
        my_value = numpy.array(value, dtype=resolve_custom_numpy_dtypes([my_dtype])[0])
        return {"shape": tuple(map(int, my_value.shape)),
                "ori_shape": tuple(map(int, my_value.shape)),
                "range": get(self._stc_input_ranges, ip_n),
                "dtype": my_dtype,
                "format": get(self._input_formats, ip_n),
                "ori_format": get(self._input_ori_formats, ip_n),
                "name": key,
                "const_value": tuple_flatten(my_value.tolist())}

    def prepare_operator_parameters_const(self, testcase: TestcaseOp) -> Tuple[tuple, tuple]:
        """
        This method is intended to construct operator dict inputs for const.
        TensorList positions produce a tuple of dicts (matching dyn_tensor_dict format).
        """
        _elim = TestcaseOp._eliminate_scalar_shapes_nested
        self._input_dtypes = testcase.input_dtypes
        self._input_formats = testcase.input_formats
        self._input_ori_formats = testcase.input_ori_formats
        self._stc_input_ranges = testcase.dyn_input_ranges
        stc_shapes = _elim(testcase.input_shapes)
        stc_ori_shapes = _elim(testcase.input_ori_shapes)
        input_dist = testcase.input_distribution

        operator = self.get_dyn_operator(testcase)
        if operator is None:
            raise OperatorNotFoundError(f"Operator {testcase.op_name} not found.")
        dyn_func_params = self.get_op_func_params(operator)
        possible_const_input_dict = param_transformation(testcase.spec_tensors, dyn_func_params)
        const_indexes = testcase.const_input_indexes

        ipt = []
        for ip_n, stc_shape in enumerate(stc_shapes):
            if stc_shape is None:
                ipt.append(None)
                continue
            is_tl = bool(input_dist) and ip_n < len(input_dist) and input_dist[ip_n] > 0
            # try const path first
            if ip_n in const_indexes:
                const_dict = self._resolve_const_value(
                    ip_n, dyn_func_params, possible_const_input_dict,
                    testcase.attributes, self._input_dtypes)
                if const_dict is not None:
                    ipt.append(const_dict)
                    continue
            # normal input path
            if is_tl:
                ipt.append(self._build_tensor_dict_group(
                    stc_shape,
                    get(self._input_dtypes, ip_n),
                    get(self._input_formats, ip_n),
                    get(stc_ori_shapes, ip_n),
                    get(self._input_ori_formats, ip_n),
                    get(self._stc_input_ranges, ip_n)))
            else:
                ipt.append(self._build_tensor_dict(
                    stc_shape,
                    get(stc_ori_shapes, ip_n),
                    get(self._input_dtypes, ip_n),
                    get(self._input_formats, ip_n),
                    get(self._input_ori_formats, ip_n),
                    get(self._stc_input_ranges, ip_n)))

        # outputs
        opt = []
        if OpInfoKeeper().op_output_defined(testcase.op_name):
            stc_out_shapes = testcase.output_shapes
            stc_out_ori_shapes = _elim(testcase.output_ori_shapes)
            output_dist = testcase.output_distribution
            for op_n, shape in enumerate(stc_out_shapes):
                if shape is None:
                    opt.append(None)
                    continue
                is_tl = bool(output_dist) and op_n < len(output_dist) and output_dist[op_n] > 0
                if is_tl:
                    opt.append(self._build_tensor_dict_group(
                        shape,
                        get(testcase.output_dtypes, op_n),
                        get(testcase.output_formats, op_n),
                        get(stc_out_ori_shapes, op_n),
                        get(testcase.output_ori_formats, op_n),
                        get(testcase.dyn_output_ranges, op_n)))
                else:
                    opt.append(self._build_tensor_dict(
                        shape,
                        get(stc_out_ori_shapes, op_n),
                        get(testcase.output_dtypes, op_n),
                        get(testcase.output_formats, op_n),
                        get(testcase.output_ori_formats, op_n),
                        get(testcase.dyn_output_ranges, op_n)))
        return tuple(ipt), tuple(opt)

    @staticmethod
    def _remove_range_keys(tensor_list):
        """Remove 'range' from dicts, handling TensorList tuple-of-dicts."""
        for pt in tensor_list:
            if pt is None:
                continue
            if isinstance(pt, (tuple, list)):
                for ti in pt:
                    if ti is not None and "range" in ti:
                        del ti["range"]
            else:
                if "range" in pt:
                    del pt["range"]

    def prepare_tiling_params(self, testcase: TestcaseOp) -> Tuple[tuple, tuple, tuple]:
        attrs = self.construct_optiling_attrs(testcase.op_name, testcase.attributes or {})
        ipt, opt = self.prepare_operator_parameters_const(testcase)
        self._remove_range_keys(ipt)
        self._remove_range_keys(opt)
        return tuple(ipt), tuple(opt), tuple(attrs)

    def get_op_generalize_func(self, op_type: str):
        return None if not op_type else self._opc.get_param_generalization(op_type)

    def with_core_type(self, core_type: str):
        self._opc.core_type = core_type
        return self

    def compile_dynamic_shape(self, dyn_params: tuple,
                              testcase: TestcaseOp,
                              kernel_name: str,
                              mode: str = "Dyn") -> Union[None, Tuple[str, dict, float, Tuple[str], str, str, str]]:
        """
        Dynamic shape operator compilation
        """
        use_static_context = True if mode == 'Cst' else False
        operator_func = self.get_dyn_operator(testcase)
        if operator_func is None:
            return None
        # Fix kwargs — use spec_attrs (already excludes input-matching attrs)
        op_func_parameters = self.get_op_func_params(operator_func, testcase.op_name)
        op_kwargs = param_transformation(testcase.spec_attrs, op_func_parameters)
        op_kwargs["kernel_name"] = kernel_name
        # dyn_params is ipt + opt, already nested from dyn_tensor_dict
        tensor_list_list = dyn_params

        def _compile_dynamic_shape():
            int64_shape_enable = self._enable_shape_int64(tensor_list_list)
            with self._opc.api_config.bit_width_64() if int64_shape_enable \
                    else self._opc.api_config.bit_width_32():
                with self._opc.op_context.OpContext("dynamic" if not use_static_context else "static") as cxt:
                    self.set_dynamic_compile_context(cxt, testcase, operator_func, kernel_name, dyn_params)
                    compile_time = self._compile_op(mode, testcase.op_name,
                                                    operator_func, op_func_parameters,
                                                    tensor_list_list, op_kwargs)
                    compile_info = self._opc.get_compile_info()
                    tiling_op_type = self._opc.get_tiling_op_type()
                    logging.debug("Received op_type from operator context: %s" % tiling_op_type)
            return (str(tiling_op_type), compile_info, compile_time,
                    tuple(op_func_parameters))

        # Call function
        return _compile_dynamic_shape()

    def call_const_op_tiling(self,
                             compile_result: Union[DynamicCompilationResult, BinaryCompilationResult],
                             testcase: TestcaseOp) -> dict:
        """
        Dynamic shape op_tiling
        """
        # Op Type initialization
        tiling_op_type = compile_result.tiling_op_type
        self.add_compile_info_to_op_context(compile_result.compile_info, testcase)

        final_inputs, final_outputs, attrs = self.prepare_tiling_params(testcase)
        # Call do_op_tiling
        logging.debug("Calling Optiling with arguments: %s" % str((tiling_op_type,
                                                                   json.dumps(compile_result.compile_info),
                                                                   final_inputs,
                                                                   final_outputs,
                                                                   attrs)))
        tiling_time = []
        build_cfg = self._build_compile_cfg()
        adapter_before_tiling(testcase, compile_result, final_inputs, final_outputs)
        with self._opc.build_config(**build_cfg):
            for i in range(get_global_storage().tiling_run_time):
                tiling_time_temp = []
                try:
                    tiling_result = self._opc.do_op_tiling(tiling_op_type,
                                                           compile_result.compile_info,
                                                           final_inputs,
                                                           final_outputs,
                                                           timer=tiling_time_temp,
                                                           attrs=attrs)
                except Exception as e:
                    if 'undefined symbol' in str(e):
                        raise e
                    time.sleep(0.5)
                    error_logs = extract_plog_errors()
                    raise RuntimeError(f"OPTILING_FAILURE: \n"
                                       f"***************************************************************************\n"
                                       f"{os.linesep.join(error_logs)}\n"
                                       f"***************************************************************************") \
                        from None
                else:
                    tiling_time.extend(tiling_time_temp[:1])
        tiling_result["tiling_time"] = tuple(tiling_time)
        return tiling_result

    def _construct_compile_context_op_info(self, operator_func: Optional[Callable], op_name: str,
                                           kernel_name: str, attrs: dict):
        op_type = OpInfoKeeper().op_type_of(op_name) or \
                  self.get_op_type_from_source_code(operator_func)
        if not op_type:
            logging.warning("OpInfo not registered with @register_operator "
                            "or configured in aic-*-ops-info.ini, add unknown opinfo")
            op_type = "UNKNOWN"
        op_info = self._opc.op_info.OpInfo(op_type, op_type)
        op_info.kernel_name = kernel_name  # for RL bank search
        if "impl_mode" in attrs:
            op_info.precision_mode = attrs["impl_mode"]
        return op_info

    @staticmethod
    def _build_compile_cfg():
        return dict(get_global_storage().compile_options)

    def _compile_op(self, mode: str, op_name: str,
                    op_func: Union[Callable, str], op_func_parameters: tuple,
                    tensor_list: list, op_kwargs: dict) -> float:
        op_impl_type = "dynamic"
        try:
            logging.debug("Calling %s operator: %s(%s, %s)" % (op_impl_type, op_name,
                                                               str(tensor_list)[1:-1], str(op_kwargs)[1:-1]))
            before_compile = time.time()
            if isinstance(op_func, Callable):
                build_cfg = self._build_compile_cfg()
                with self._opc.build_config(**build_cfg):
                    op_func(*copy.deepcopy(tensor_list), **copy.deepcopy(op_kwargs))
            else:
                raise RuntimeError(f"Operator [{op_name}] implement function is not callable: {type(op_func)}")
            after_compile = time.time()
        except:
            param_print = self.print_func_params(op_func_parameters, op_kwargs, tensor_list)
            logging.error(("%s operator compile failure, mode: %s\n" % (op_impl_type, mode)) +
                          ("Operator: %s\n" % op_name) + "\n".join(param_print))
            raise
        else:
            return after_compile - before_compile

    def _switch_opc(self, opc_type: str):
        if opc_type in ('tbe', 'asc'):
            self._opc.switch_opc(opc_type)

    @staticmethod
    def _enable_shape_int64(tensors: Union[list, tuple]):
        def _exceed_int32_max(_t: dict):
            shape = _t["shape"]
            dtype = _t["dtype"]
            shape_prod = functools.reduce(lambda x, y: x * y,
                                          shape, 1)
            int32_max = numpy.iinfo(numpy.int32).max
            if shape_prod <= 0:
                return False
            if shape_prod > int32_max:
                return True
            dtype_bytes = get_dtype_width(dtype)
            if shape_prod * dtype_bytes > int32_max:
                return True
            return

        for t in tensors:
            if t is None:
                continue
            if isinstance(t, (tuple, list)):
                for ti in t:
                    if _exceed_int32_max(ti):
                        return True
            else:
                if _exceed_int32_max(t):
                    return True
        return False

    @staticmethod
    def print_func_params(dynamic_shape_func_parameters, op_kwargs, tensor_list_list):
        """
        :param dynamic_shape_func_parameters:
        :param op_kwargs:
        :param tensor_list_list:
        :return:
        """
        op_param_distribution = {}
        param_idx = 0
        for param in dynamic_shape_func_parameters:
            if param in op_kwargs:
                param_idx += 1
                op_param_distribution[param] = op_kwargs[param]
            else:
                if param_idx < len(tensor_list_list):
                    op_param_distribution[param] = tensor_list_list[param_idx]
                else:
                    op_param_distribution[param] = "UNKNOWN"
                param_idx += 1
        param_idx = 0
        for param in dynamic_shape_func_parameters:
            if param_idx < len(tensor_list_list):
                op_param_distribution[param] = tensor_list_list[param_idx]
            param_idx += 1
        param_print = ["***Params***"]
        for param in op_param_distribution:
            param_print.append("%s %s:\n%s" % (param, str(type(op_param_distribution[param])),
                                               str(op_param_distribution[param])))
        param_print.append("***Params***")
        return param_print

    @staticmethod
    def construct_optiling_attrs(dyn_op_name: str, attr_dictionary: dict) -> tuple:
        def detect_type_of_sequence(_sequence: Any):
            supported_types = (bool, float, int, str)
            _result = None
            if isinstance(_sequence, Sequence) and not isinstance(_sequence, str):
                _element_types = set()
                for element in _sequence:
                    _element_types.add(detect_type_of_sequence(element))
                if len(_element_types) == 1:
                    _element_type = _element_types.pop()
                    if _element_type is not None:
                        _result = "list_" + _element_type
            else:
                for _type in supported_types:
                    if isinstance(_sequence, _type):
                        _result = _type.__name__
                        break
            return _result

        def construct_attr_info(key: str, val: Any) -> Optional[dict]:
            attr_type = detect_type_of_sequence(val)
            if attr_type is None:
                raise RuntimeError(f"optiling dtype detection for attr {key} value {val} type {type(val)} failed.")
            return {"name": key, "dtype": attr_type, "value": val}

        result = []
        op_cfg_info = OpInfoKeeper().info_of(dyn_op_name)
        if op_cfg_info is not None:
            attr_name_lst = tuple([a["name"] for a in op_cfg_info["attr"]])
            possible_attrs = param_transformation(attr_dictionary, attr_name_lst)
            for attr in op_cfg_info["attr"]:
                k, typ = attr["name"], attr["type"]
                if k in possible_attrs:
                    v = possible_attrs[k]
                else:
                    if attr["defaultValue"] is None:
                        raise RuntimeError(f"Required attribute [{k}] is not configured "
                                           f"in attributes.")
                    v = attr["defaultValue"]
                detected_type = detect_type_of_sequence(v)  # in case attr supports both listInt and Int
                if detected_type:
                    detected_type = detected_type.split('_')[:-1] + [typ.lower().replace("list", "")]
                    detected_type = "_".join(detected_type)
                else:  # in case default is [] or [[]], which will detect fail.
                    detected_type = typ.lower().replace("list", "list_")
                result.append({"name": k, "value": v, "dtype": detected_type})
        else:
            for k in attr_dictionary:
                if str(k).startswith("!") or str(k).startswith("#") or str(k).startswith("@"):
                    continue
                ret = construct_attr_info(k, attr_dictionary[k])
                if ret is not None:
                    result.append(ret)
        # add private attributes
        for k in attr_dictionary:
            if str(k).startswith("@"):
                ret = construct_attr_info(k[1:], attr_dictionary[k])
                if ret is not None:
                    result.append(ret)
        return tuple(result)

    @staticmethod
    def add_compile_info_to_op_context(cxt, testcase: TestcaseOp):
        other_x_params = testcase.attributes or {}
        if isinstance(cxt, dict):
            cxt["_sgt_cube_vector_core_type"] = testcase.core_type
            for param in other_x_params:
                if param.startswith("#"):
                    cxt[param[1:]] = other_x_params[param]
        else:
            cxt.add_compile_info("_sgt_cube_vector_core_type", testcase.core_type)
            for param in other_x_params:
                if param.startswith("#"):
                    cxt.add_compile_info(param[1:], other_x_params[param])

    @staticmethod
    def add_addition_to_op_context(cxt, attrs: dict, op_info):
        cxt.add_addition("op_name", op_info.op_name)  # for RL bank search
        for param in attrs:
            if param.startswith("!"):
                cxt.add_addition(param[1:], attrs[param])

    @staticmethod
    def add_private_attr_to_op_info(tiling_attr: tuple, attributes: dict, op_info):
        if not hasattr(op_info, "private_attrs"):
            return
        private_attrs = {}
        for ta in tiling_attr:
            name = ta["name"]
            if f"@{name}" in attributes:
                private_attrs.update(ta)
        op_info.private_attrs = private_attrs

    def set_common_compile_context(self, cxt, testcase: TestcaseOp,
                                   operator_func: Optional[Callable], kernel_name: str):
        cxt.add_addition("master_pid", testcase.kb_pid)
        attrs = testcase.attributes or {}
        op_info = self._construct_compile_context_op_info(
            operator_func, testcase.op_name,
            kernel_name, attrs)
        cxt.add_op_info(op_info)
        self.add_addition_to_op_context(cxt, attrs, op_info)

    def set_dynamic_compile_context(self, cxt, testcase: TestcaseOp,
                                    operator_func: Optional[Callable], kernel_name: str,
                                    dyn_params):
        self.set_common_compile_context(cxt, testcase, operator_func, kernel_name)
        if cxt.get_op_mode() == "static":
            attrs = testcase.attributes
            tiling_attrs = self.construct_optiling_attrs(testcase.op_name, attrs)
            op_info = cxt.get_op_info()[-1]
            op_info.inputs = dyn_params[:len(testcase.dyn_inputs)]
            op_info.outputs = dyn_params[len(testcase.dyn_inputs):]
            op_info.attrs = tiling_attrs
            self.add_private_attr_to_op_info(tiling_attrs, attrs, op_info)
            self.add_compile_info_to_op_context(cxt, testcase)

    def get_dyn_operator(self, testcase: TestcaseOp):
        op_name = testcase.op_name
        operator_func = OpInfoKeeper().get_operator_function(op_name)
        if operator_func is None:
            logging.warning(f'Get dynamic impl function for operator [{op_name}] failed.')
        self._switch_opc(OpInfoKeeper().impl_type_of(op_name))
        return operator_func

    @staticmethod
    def get_op_type_from_source_code(operator_func: Callable) -> Optional[str]:
        op_type = None
        if not isinstance(operator_func, Callable):
            return None
        try:
            src_lines = inspect.getsource(operator_func).split("\n")
            for line in src_lines:
                if "register_operator" in line:
                    op_type = eval(line[line.index("register_operator") + 18:-1].split(",")[0])
                    break
        except OSError:
            return None
        return op_type

    @staticmethod
    def get_op_func_params(operator_func=None, op_name: str = None) -> tuple:
        if isinstance(operator_func, Callable):
            return tuple(inspect.signature(operator_func).parameters)
        elif op_name:
            op_cfg_info = OpInfoKeeper().info_of(op_name)
            if not op_cfg_info:
                raise RuntimeError(f"Operator {op_name} is not configured in aic-**-ops-info.json")
            return tuple([pi["name"] for pi in op_cfg_info["inputs"]] +
                         [po["name"] for po in op_cfg_info["outputs"]] +
                         [attr["name"] for attr in op_cfg_info["attr"]])
        else:
            return ()

    @staticmethod
    def get_op_func_parameter_dict(operator_func=None, op_name: str = None):
        if isinstance(operator_func, Callable):
            return inspect.signature(operator_func).parameters
        elif op_name:
            op_cfg_info = OpInfoKeeper().info_of(op_name)
            if not op_cfg_info:
                raise RuntimeError(f"Operator {op_name} is not configured in aic-**-ops-info.json")
            parameters = OrderedDict()
            for pi in op_cfg_info["inputs"]:
                k, param_type = pi["name"], pi["paramType"]
                if param_type == "optional":
                    parameters[k] = inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                      default=None, annotation=dict)
                else:
                    parameters[k] = inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                      annotation=dict)
            for po in op_cfg_info["outputs"]:
                k = po["name"]
                parameters[k] = inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                  annotation=dict)
            for attr in op_cfg_info["attr"]:
                k, attr_type = attr["name"], OperatorInterface.dtype_str_to_type(attr["type"])
                if attr["defaultValue"] is None:
                    parameters[k] = inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                      annotation=attr_type)
                else:
                    parameters[k] = inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                      default=attr["defaultValue"], annotation=attr_type)
            return parameters
        else:
            return None

    @staticmethod
    def dtype_str_to_type(s):
        return list if s.startswith("list") else eval(s)

def adapter_before_tiling(testcase: TestcaseOp,
                          compile_result: Union[DynamicCompilationResult, BinaryCompilationResult],
                          final_inputs: tuple, final_outputs: tuple):
    if compile_result.compile_info.setdefault("tiling_type") == 'binary' and \
            compile_result.compile_info.setdefault('op_type_list') == ['Conv2D']:
        groups = testcase.attributes.get('groups')
        if final_inputs[1].setdefault('format') == 'NC1HWC0':
            inputs_filter_shape = final_inputs[1].setdefault('shape')
            Cout, C1, H, W, C0 = inputs_filter_shape
            Cout = (Cout + C0) // 16 * 16
            final_inputs[1]["shape"] = (C1 * H * W, Cout // 16, 16, 16)
            final_inputs[1]["format"] = 'FRACTAL_Z'
        elif final_inputs[1].setdefault('format') == 'NCHW':
            CUBE_K = 16
            CUBE_N = 16
            inputs_filter_shape = final_inputs[1].setdefault('shape')
            Cout, Cin, H, W = inputs_filter_shape
            Cin_ori = Cin
            Cout_ori = Cout // groups
            A = lcm(Cin_ori, CUBE_K) // Cin_ori
            B = lcm(Cout_ori, CUBE_N) // Cout_ori
            C = lcm(A, B)
            enlarge = min(C, groups)
            Cin_opt = ceil_div(enlarge * Cin_ori, CUBE_K) * CUBE_K
            Cout_opt = ceil_div(enlarge * Cout_ori, CUBE_N) * CUBE_N
            group_opt = ceil_div(groups, enlarge)
            final_inputs[1]["shape"] = (group_opt * (Cin_opt // CUBE_K) * H * W, Cout_opt // CUBE_N, CUBE_N, CUBE_K)
            final_inputs[1]["format"] = 'FRACTAL_Z'
