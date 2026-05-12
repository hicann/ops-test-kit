
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Standard Packages
import gc

import numpy as np
from typing import List, Tuple, Union

# Third-party Packages
from .registry import ComparisonRegister
from ...utilities import get


__all__ = ["compare"]


def compare(outputs: Union[Tuple[np.ndarray], List[np.ndarray]],
            goldens: Union[Tuple[np.ndarray], List[np.ndarray]],
            output_dtypes: tuple,
            methods: Union[str, list, tuple] = 'close',
            options: dict = None) -> (str, str, bool):
    from .binary_equal import BinaryComparison
    from .cosine_similarity import CosineSimilarityComparison
    from .is_close import CloseComparison
    from .re_quantize import ReQuantizeComparison

    if not outputs:
        return ("UNKNOWN",
                "Output or Golden data is empty, compare result UNKNOWN\n",
                False)

    if not isinstance(methods, (list, tuple)):
        methods = (methods,)

    total_precision, total_pass, total_log = [], [], ""
    for idx, data_pair in enumerate(zip(outputs, goldens)):
        output, golden = data_pair
        if isinstance(output, str):
            total_precision.append(f"{output}")
            total_pass.append(_filter_fake_fail(output))
            continue
        if isinstance(golden, str):
            total_precision.append(f"{golden}")
            total_pass.append(_filter_fake_fail(golden))
            continue
        if golden is None:
            total_precision.append("SUPPRESSED")
            total_pass.append(True)
            continue

        method = get(methods, idx)
        dtype_str = str(output.dtype).split('.')[-1]
        if (dtype_str in ('float8_e5m2', 'float8_e4m3fn', 'hifloat8') and
                method not in ('bin', 'binary', 'requant')):
            method = 'requant'
        elif dtype_str in ('float4_e2m1', 'float4_e1m2', 'float8_e8m0'):
            method = 'bin'
        comparison = ComparisonRegister.registry.get(method.lower())
        if comparison is None:
            raise ValueError(f"Comparison method [{method}] is not recognized.")
        c = comparison(output, golden, idx,
                       get(output_dtypes, idx), options)

        precision, log, is_pass = c.compare()
        total_precision.append(precision)
        total_pass.append(is_pass)
        total_log += log
        gc.collect()

    return ','.join(total_precision), total_log, all(total_pass)


def _filter_fake_fail(output: str):
    if output in ("DYN_OFF", "STC_OFF", "CST_OFF", "BIN_OFF",
                  "DYN_UNSUPPORTED", "STC_UNSUPPORTED", "CST_UNSUPPORTED", "BIN_UNSUPPORTED",
                  "DYN_OPERATOR_NOT_FOUND", "STC_OPERATOR_NOT_FOUND",
                  "CST_OPERATOR_NOT_FOUND", "BIN_OPERATOR_NOT_FOUND",
                  "SUPPRESSED", "DYN_INPUT_MISSING"):
        return True
    else:
        return False
