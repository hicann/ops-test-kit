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
Golden data generator functions
"""

# Standard Packages
import logging
from typing import Sequence
from functools import wraps

# Third-party Packages


golden_funcs = {
    "aclnnInplaceOne": "torch.ops.aten.ones_like",
    "aclnnInplaceZero": "torch.ops.aten.zeros_like",
    "aclnnMaxDim": "torch.ops.aten.max",
    "aclnnMax": "torch.ops.aten.amax",
    "aclnnVarCorrection": "torch.ops.aten.var",
    "aclnnStdMeanCorrection": "torch.ops.aten.std_mean",
    "aclnnIsFinite": "torch.ops.aten.isfinite",
    "aclnnIsInf": "torch.ops.aten.isinf",
    "aclnnIsClose": "torch.ops.aten.isclose",
    "aclnnMatmul": "torch.mm",
    "aclnnBatchMatMul": "torch.bmm",
    "aclnnArgMax": "torch.argmax",
    "aclnnAminmaxAll": "torch.aminmax",
    "aclnnAminmaxDim": "torch.aminmax",
    "aclnnLogSumExp": "torch.logsumexp",
}


# dma_copy ops don't need to transfer INF/NAN in overflow_mode=0 scenario.
user_specified_dma_copy_ops = set()


def register_golden(api_names: Sequence[str], dma_copy_op: bool = False):
    """Register golden function"""
    if not isinstance(api_names, (list, tuple)):
        raise TypeError("Register function for golden funcs must receive a list or tuple, not %s"
                        % str(api_names))

    def __inner_golden_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for api in api_names:
            if api in golden_funcs:
                logging.warning("golden function of %s has already been registered!" % api)
            golden_funcs[api] = __wrapper
            if dma_copy_op:
                user_specified_dma_copy_ops.add(api)
        return __wrapper

    return __inner_golden_registry

