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
GraphNetwork — generic torch.nn.Module wrapper for torch.compile graph mode testing.
Also provides split_params for custom Module parameter routing (__init__ vs forward).
"""

import inspect

import torch


class GraphNetwork(torch.nn.Module):
    """
    Generic torch.nn.Module wrapper for torch.compile graph mode testing.

    api_func is always a callable — either the resolved API function directly,
    or a closure that handles tensor method dispatch.
    """

    def __init__(self, api_func):
        super().__init__()
        self.api_func = api_func

    def forward(self, *args, **kwargs):
        return self.api_func(*args, **kwargs)


def split_params(module_cls, overload_params, args, kwargs):
    """Split args/kwargs into __init__ and forward kwargs based on module signatures.

    Uses overload_params (ParamInfo list) to map positional args to parameter names,
    then inspects module_cls.__init__ and .forward to split them.

    For tensor methods, overload_params[0] is "self" (the input tensor).
    It is mapped to the first forward parameter automatically.

    Args:
        module_cls: torch.nn.Module subclass
        overload_params: list of ParamInfo from plan.overload_params (positional order)
        args: positional args from prepare_device_args
        kwargs: keyword args from prepare_device_args

    Returns:
        (init_kwargs, fwd_kwargs) — dicts for __init__ and forward respectively
    """
    full_kwargs = dict(kwargs)
    positional_param_names = [p.name for p in overload_params if not p.is_keyword_only]
    for i, val in enumerate(args):
        if i < len(positional_param_names):
            full_kwargs[positional_param_names[i]] = val

    init_params = set(inspect.signature(module_cls.__init__).parameters.keys()) - {"self"}
    fwd_params = set(inspect.signature(module_cls.forward).parameters.keys()) - {"self"}

    if "self" in full_kwargs:
        fwd_ordered = [
            p.name
            for p in inspect.signature(module_cls.forward).parameters.values()
            if p.name != "self" and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        for name in fwd_ordered:
            if name not in init_params and name not in full_kwargs:
                full_kwargs[name] = full_kwargs.pop("self")
                break

    fwd_has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(module_cls.forward).parameters.values()
    )

    init_kwargs = {k: v for k, v in full_kwargs.items() if k in init_params}
    if fwd_has_var_kw:
        fwd_kwargs = {k: v for k, v in full_kwargs.items() if k not in init_params}
    else:
        fwd_kwargs = {k: v for k, v in full_kwargs.items() if k in fwd_params}

    return init_kwargs, fwd_kwargs
