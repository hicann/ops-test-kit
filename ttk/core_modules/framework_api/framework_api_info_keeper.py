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
FrameworkApiInfoKeeper — cached API parameter info for torch/torch_npu/tf.

Uses simple_param_extractor for torch/torch_npu and tf_param_extractor for TF.
Validates testcase parameters against API signatures.
"""

import logging
from typing import Dict, Optional

from ttk.utilities import Singleton
from ttk.utilities.simple_param_extractor import APIParamInfo, _resolve_function, get_api_params, register_api_params
from ttk.utilities.torch_ops_package_loader import TorchOpsPackageLoader

from .framework_detector import TF_API_PREFIXES


class FrameworkApiInfoKeeper(metaclass=Singleton):
    def __init__(self):
        self._cache: Dict[str, Optional[APIParamInfo]] = {}

    def get(self, api_name: str) -> Optional[APIParamInfo]:
        if api_name in self._cache:
            return self._cache[api_name]
        try:
            if api_name.startswith(TF_API_PREFIXES):
                from ttk.utilities.tf_param_extractor import extract_tf_params

                info = extract_tf_params(api_name)
            else:
                TorchOpsPackageLoader.ensure_registered(api_name)
                info = get_api_params(api_name)
                if info is None:
                    # Normal extraction probes may suppress import errors; recover
                    # the original exception only after every signature fallback failed.
                    obj = _resolve_function(api_name, raise_on_error=True)
                    if obj is None:
                        raise AttributeError(f"API callable not found: {api_name}")
                    raise ValueError(f"No supported API signature could be extracted: {api_name}")
        except Exception as e:
            logging.exception("API signature parsing failed: %s: %s: %s", api_name, type(e).__name__, e)
            info = None
        self._cache[api_name] = info
        if info:
            logging.debug(f"Parsed {api_name}: {len(info.params)} params from {info.source}")
        elif info is None:
            logging.debug("Could not parse %s; see API signature parsing errors above", api_name)
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

    def validate_testcase_params(self, api_name: str, tensor_count: int, scalar_count: int = 0) -> Optional[str]:
        info = self.get(api_name)
        if info is None:
            return None
        api_tensor_count = info.tensor_count
        api_scalar_count = info.scalar_count
        if tensor_count != api_tensor_count:
            return (
                f"API [{api_name}] has {api_tensor_count} tensor parameters, "
                f"but testcase configured {tensor_count}. "
                f"(source: {info.source})"
            )
        if scalar_count != api_scalar_count:
            return (
                f"API [{api_name}] has {api_scalar_count} scalar parameters, "
                f"but testcase configured {scalar_count}. "
                f"(source: {info.source})"
            )
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
