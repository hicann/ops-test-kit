#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
op api info keeper.
"""

__all__ = ["OpApiInfoKeeper", "OpApiInfo"]


# Standard Packages
import re
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib import util as importlib_util
from typing import Dict, List

# Third-Party Packages
from ...utilities import Singleton, get_ascend_scene_info


@dataclass
class OpApiInfo:
    params: OrderedDict
    # tensor/tensor-list names including input/output
    tensors: List[str] = field(default_factory=list, init=False)
    # scalar/scalar-list names
    scalars: List[str] = field(default_factory=list, init=False)
    # other param names except tensor/scalar/tensor-list/scalar-list
    others: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        for k, v in self.params.items():
            if v["type"] in ("aclTensor*", "aclTensorList*"):
                self.tensors.append(k)
            elif v["type"] in ("aclScalar*", "aclScalarList*"):
                self.scalars.append(k)
            else:
                self.others.append(k)


class OpApiInfoKeeper(metaclass=Singleton):
    """
    Singleton Class to keep op api info
    """

    FUNC_PATTERN = r'\b([A-Za-z0-9_]+)\s+\b([A-Za-z0-9_]+)\s*\(([^){]*?)\)(?=\s*\;)'
    PARAM_PATTERN = r'(\w+[\s\*\s]*)(\w+)(\s*=\s*(.*?))?(,|$)'

    def __init__(self):
        self._api_info = None
        try:
            _op_api_path = ""
            _op_api_module = importlib_util.find_spec("op_api")
            if _op_api_module is not None:
                _op_api_path = list(_op_api_module.submodule_search_locations)[0]
            self._api_hdr_path = os.path.join(_op_api_path, "include", "aclnnop")
            if not os.path.isdir(self._api_hdr_path):
                # new version.
                opp_path = os.getenv('ASCEND_OPP_PATH', '')
                os_scene, os_arch = get_ascend_scene_info(opp_path)
                if not os_scene:
                    raise RuntimeError("File 'scene.info' is missing.")
                self._api_hdr_path = os.path.abspath(
                    os.path.join(opp_path, "..", f"{os_arch}-{os_scene}",
                                 "include", "aclnnop"))
            if not os.path.isdir(self._api_hdr_path):
                raise RuntimeError(f"Directory aclnnop not exit")
            logging.debug("op api header files locates: %s" % self._api_hdr_path)
        except (AttributeError, RuntimeError):
            raise RuntimeError(f"OpApi header file path does not exist. "
                               f"Make sure opp-kernel package has been install !!!")

    @property
    def api_info(self) -> dict:
        if self._api_info is None:
            self._api_info = self._parse_op_api_header_files()
        return self._api_info

    def has_api(self, api_name: str) -> bool:
        return api_name in self.api_info

    def info_of(self, api_name: str) -> OpApiInfo:
        return self.api_info.get(api_name, None)

    def _parse_op_api_header_files(self) -> dict:
        files = tuple([f for f in os.listdir(self._api_hdr_path)
                       if os.path.isfile(os.path.join(self._api_hdr_path, f))])
        api_dict: Dict[str, OpApiInfo] = {}
        for f in files:
            with open(os.path.join(self._api_hdr_path, f), 'r') as hf:
                content = hf.read()
            for match in re.finditer(self.FUNC_PATTERN, content):
                func_name = match.group(2).strip()
                if not (func_name.startswith('aclnn') and
                        func_name.endswith('GetWorkspaceSize')):
                    continue
                api_name = func_name[:-16]
                func_params = re.sub(r'\bconst\b(?!_)', '', match.group(3).strip())
                params = OrderedDict({})
                for param_match in re.finditer(self.PARAM_PATTERN, func_params):
                    param_type = param_match.group(1).strip().replace(' ', '')
                    param_name = param_match.group(2).strip()
                    if param_name in ('workspaceSize', 'executor'):
                        break
                    default = param_match.group(4)
                    params.update({param_name: {"type": param_type, "default": default}})
                api_dict[api_name] = OpApiInfo(params)
        return api_dict
