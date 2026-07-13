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
Resolve api_name string to callable object.
Handles both module functions (torch.add) and Tensor methods (torch.Tensor.relu_).
"""

_MODULE_ALIAS = {}


def resolve_api(api_name: str):
    """
    Resolve api_name to (callable_or_method_name, is_tensor_method).

    Examples:
        'torch.add'                 -> (torch.add, False)
        'torch.nn.functional.relu'  -> (torch.nn.functional.relu, False)
        'torch_npu.npu_conv2d'      -> (torch_npu.npu_conv2d, False)
        'torch.Tensor.relu_'        -> ('relu_', True)
        'torch.Tensor.npu_scatter_' -> ('npu_scatter_', True)
    """
    parts = api_name.split(".")

    if len(parts) < 2:
        raise ValueError(f"Invalid api_name: {api_name}, expected format: module.func")

    # torch.Tensor.xxx -> Tensor method
    if len(parts) >= 3 and parts[1] == "Tensor":
        method_name = ".".join(parts[2:])
        return method_name, True

    # Module function: import root module (with alias support), traverse attributes
    root_name = _MODULE_ALIAS.get(parts[0], parts[0])
    try:
        obj = __import__(root_name)
        for p in parts[1:]:
            obj = getattr(obj, p)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Cannot resolve api_name '{api_name}': {e}") from e
    return obj, False