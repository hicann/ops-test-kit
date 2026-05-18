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
output generation method for Universal testcases
"""


__all__ = ["GoldenGenerator"]


# Standard Packages
import contextlib
import copy
import gc
import inspect
import logging
import numpy
import os
from typing import Sequence
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

# Third-party Packages
from ...plugin_loader import get_plugin_function
from ...testcase_manager import TestcaseAclnn
from ...aclnn import OpApiInfoKeeper, OpApiInfo
from ....utilities import get_global_storage
from ....utilities import camel_to_snake, acl_to_torch_dtype


class GoldenGenerator:
    def __init__(self, context: TestcaseAclnn):
        self._ctx = context
        self._switch = get_global_storage()

    def gen(self):
        if self._switch.golden_mode == "Disable":
            golden_arrays = ("SUPPRESSED",)
        elif self._ctx.manual_golden_binaries:
            golden_arrays = self._load_golden_from_file()
        else:
            golden_arrays = self._generate_golden()

        # golden_arrays now is flatten.
        # maybe torch or numpy arrays.
        self._ctx.golden_tensors = golden_arrays[:len(self._ctx.output_tensor_indexes)]

    def _load_golden_from_file(self):
        # TODO
        logging.warning("Using manually configured golden data")
        return None

    def _call_builtin_golden(self, gf):
        # NOTE: DO NOT COMMENT this below.
        import torch
        if isinstance(gf, str):
            gf = eval(gf)
        # noinspection PyBroadException
        try:
            golden_parameters = inspect.signature(gf).parameters
        except:
            golden_parameters = []
        with self._golden_mode():
            if "context" in golden_parameters:
                results = gf(self._ctx)
            else:
                # construct golden params without outputs.
                tensors = self._package_golden_tensors()
                kwargs = self._package_golden_torch_kwargs()
                results = gf(*tensors, **kwargs)
        return results

    def _call_custom_golden(self, gf):
        plan = self._ctx.get_param_plan()
        args = plan.build_args(self._ctx.tensors, self._ctx.scalars,
                               self._ctx.attributes)
        kwargs = {
            'short_soc_version': self._switch.short_soc_version,
            'testcase_name': self._ctx.testcase_name,
            'tensor_dtypes': self._ctx.tensor_dtypes,
            'tensor_formats': self._ctx.tensor_formats,
            'scalar_dtypes': self._ctx.scalar_dtypes,
            'use_torch': self._ctx.is_torch_dtype_support(),
        }
        results = gf(*args, **kwargs)
        return results

    def _generate_golden(self):
        golden_func, src = self._import_golden_funcs()
        disable_builtin = int(os.getenv("TTK_DISABLE_BUILTIN", "0"))
        if golden_func is None:
            logging.warning(f"Golden data generation method for [{self._ctx.api_name}] "
                            f"is not registered!")
            golden_arrays = ["UNSUPPORTED"]
        elif src == 'builtin' and disable_builtin:
            logging.warning(f"builtin golden for [{self._ctx.api_name}] is disabled!")
            golden_arrays = ["SUPPRESSED"]
        else:
            # noinspection PyBroadException
            try:
                if src == 'builtin':
                    golden_results = self._call_builtin_golden(golden_func)
                else:
                    golden_results = self._call_custom_golden(golden_func)
            except TimeoutError:
                logging.exception("Golden generation timeout")
                golden_results = ["GOLDEN_TIMEOUT"]
            except Exception as e:
                logging.exception(f"Golden generation failure: {e}")
                golden_results = ["GOLDEN_FAILURE"]
            finally:
                gc.collect()
            golden_arrays = self._golden_reshape(golden_results)
        return golden_arrays

    def _import_golden_funcs(self):
        golden_func, src = get_plugin_function(
            self._ctx.api_name, 'golden', 'aclnn', self._switch.plugin_path)
        if golden_func:
            return golden_func, src
        else:
            snake_api_name = camel_to_snake(self._ctx.api_name[5:])
            func = self._auto_import_from_torch(snake_api_name)
            # TD: skip searching next time. optimize later.
            return func, 'builtin'

    def _find_torch_api(self, torch_module, snake_name: str):
        # find torch.snake_name
        try:
            torch_api = getattr(torch_module, snake_name, None)
        except RuntimeError:
            # older torch version, 1.11.0 for exp,
            # will throw RuntimeError when absent.
            torch_api = None
        if torch_api and isinstance(torch_api, Callable):
            logging.debug(f"Using function [{getattr(torch_module, '__name__')}.{snake_name}] "
                            f"for api {self._ctx.api_name} to generate golden.")
            return torch_api
        else:
            if snake_name.startswith("inplace_"):
                return self._find_torch_api(torch_module, snake_name[len("inplace_"):])
            elif snake_name.endswith("_scalar"):
                return self._find_torch_api(torch_module, snake_name[:-len("_scalar")])
            elif snake_name.endswith("_tensor"):
                return self._find_torch_api(torch_module, snake_name[:-len("_tensor")])
            elif snake_name.endswith("_v2"):
                return self._find_torch_api(torch_module, snake_name[:-len("_v2")])
            elif snake_name.endswith("s"):
                return self._find_torch_api(torch_module, snake_name[:-len("s")])
            elif not snake_name.startswith("_"):  # _add_relu
                return self._find_torch_api(torch_module, "_" + snake_name)
            else:
                return None

    def _auto_import_from_torch(self, snake_name: str):
        import torch
        # torch api first
        torch_api = self._find_torch_api(torch, snake_name)
        if torch_api is not None:
            return torch_api
        torch_nn_api = self._find_torch_api(torch.nn.functional, snake_name)
        if torch_nn_api is not None:
            return torch_nn_api
        return self._find_torch_api(torch.ops.aten, snake_name)

    @contextlib.contextmanager
    def _golden_mode(self):
        if self._switch.golden_mode != "Promote":
            yield
        else:
            yield from self._promote_dtype()

    def _promote_dtype(self):
        # TODO
        yield

    def _package_golden_torch_kwargs(self):
        DEL_KWARG = ('impl_mode', 'cube_math_type')
        # torch api both support keepdims & keepdim.
        # but dim / dims diffs.
        RENAME_KWARG = {
            'keep_dim': 'keepdim',
            'keep_dims': 'keepdims',
            'keepDims': 'keepdims',
            'keepDim': 'keepdim',
        }
        kwargs_camel = copy.deepcopy(self._ctx.attributes)
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self._ctx.api_name)
        for idx, s in enumerate(self._ctx.flatten_scalars or ()):
            s_name = op_api_info.scalars[idx]
            kwargs_camel.update({s_name: s})
        # camel -> snake
        kwargs_snake = {}
        for k, v in kwargs_camel.items():
            k_snake = camel_to_snake(k)
            kwargs_snake.update({k_snake: v})
        for k, v in RENAME_KWARG.items():
            if k in kwargs_snake:
                kwargs_snake[v] = kwargs_snake[k]
                del kwargs_snake[k]
        for ele in DEL_KWARG:
            if  ele in kwargs_snake:
                del kwargs_snake[ele]
        # dtype from aclDType to torchDtype automatically
        if 'dtype' in kwargs_snake:
            if isinstance(kwargs_snake['dtype'], int):
                kwargs_snake['dtype'] = acl_to_torch_dtype([kwargs_snake['dtype']])[0]
        if 'dim' in kwargs_snake:
            if isinstance(kwargs_snake['dim'], list):
                kwargs_snake['dim'] = tuple(kwargs_snake['dim'])
        if 'dims' in kwargs_snake:
            if isinstance(kwargs_snake['dims'], list):
                kwargs_snake['dims'] = tuple(kwargs_snake['dims'])
        return kwargs_snake

    def _package_golden_tensors(self) -> list:
        golden_tensors = []
        case_tensors = self._ctx.tensors
        real_idx = 0
        for ct in case_tensors:
            if real_idx in self._ctx.pure_output_indexes:
                continue
            else:
                golden_tensors.append(ct)
            real_idx += len(ct) if isinstance(ct, (list, tuple)) else 1
        return golden_tensors

    @staticmethod
    def _golden_reshape(golden_results):
        golden_arrays = []
        if not isinstance(golden_results, Sequence):
            golden_results = [golden_results]
        for golden_result in golden_results:
            if isinstance(golden_result, numpy.generic):
                golden_arrays.append(numpy.array((golden_result,)))
            else:
                golden_arrays.append(golden_result)
        return golden_arrays
