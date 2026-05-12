#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""
FrameworkApiInfoKeeper — cached API parameter info for torch/torch_npu.

Uses simple_param_extractor for auto-parsing with manual override support.
Validates testcase parameters against API signatures.
"""

import logging
from typing import Optional, Dict

from ttk.utilities import Singleton
from ttk.utilities.simple_param_extractor import (
    APIParamInfo, get_api_params, register_api_params, ParamInfo
)


class FrameworkApiInfoKeeper(metaclass=Singleton):

    def __init__(self):
        self._cache: Dict[str, Optional[APIParamInfo]] = {}

    def get(self, api_name: str) -> Optional[APIParamInfo]:
        if api_name in self._cache:
            return self._cache[api_name]
        try:
            info = get_api_params(api_name)
        except Exception as e:
            logging.warning(f"Parse {api_name} signature failed: {type(e).__name__}: {e}")
            info = None
        self._cache[api_name] = info
        if info:
            logging.debug(f"Parsed {api_name}: {len(info.params)} params from {info.source}")
        else:
            logging.debug(f"Could not parse {api_name}")
        return info

    def register(self, api_name: str, params, source="manual"):
        if isinstance(params, APIParamInfo):
            self._cache[api_name] = params
        elif isinstance(params, list) and params and isinstance(params[0], list):
            info = APIParamInfo(api_name=api_name, overloads=params, source=source)
            self._cache[api_name] = info
        else:
            register_api_params(api_name, params, source)
            self._cache[api_name] = get_api_params(api_name)

    def validate_testcase_params(self, api_name: str, tensor_count: int,
                                  scalar_count: int = 0) -> Optional[str]:
        info = self.get(api_name)
        if info is None:
            return None
        api_tensor_count = info.tensor_count
        api_scalar_count = info.scalar_count
        if tensor_count != api_tensor_count:
            return (f"API [{api_name}] has {api_tensor_count} tensor parameters, "
                    f"but testcase configured {tensor_count}. "
                    f"(source: {info.source})")
        return None

    def get_tensor_distribution(self, api_name: str) -> tuple:
        info = self.get(api_name)
        if info is None:
            return ()
        dist = []
        for p in info.tensors:
            if p.is_tensor_list:
                dist.append(-1)
            else:
                dist.append(0)
        return tuple(dist)

    def clear_cache(self):
        self._cache.clear()
