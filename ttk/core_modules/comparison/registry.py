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
comparison method register
"""

# Standard Packages
import gc
import logging
import numpy as np
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass
class EachCompareResult:
    precision: Union[str, float, int]
    diff_index: Optional[np.ndarray] = None
    is_pass: bool = False
    log: str = ""
    standard: str = ""
    metrics: dict = field(default_factory=dict)
    error_info: Optional[str] = None  # custom compare 接口失败时填；compare() 打 ERROR 日志


def _to_numpy(arr):
    """compare() 入口把 torch 统一转 numpy（numpy 直通）。
    复用 utilities.dtypes.torch_to_numpy_tensor（处理 bf16/fp8/complex32/普通）。"""
    if not hasattr(arr, "numel"):
        return arr
    from ...utilities.dtypes import torch_to_numpy_tensor

    return torch_to_numpy_tensor(arr)


class ComparisonBase(metaclass=ABCMeta):
    # Max diff entries printed by _log_diff_output (worst-first). One place to tune.
    MAX_DIFF_OUTPUT = 100

    def __init__(
        self,
        output: Union[np.ndarray, np.ndarray],
        golden: Union[np.ndarray, np.ndarray],
        output_idx: int,
        output_dtype: str,
        options: dict = None,
        third_party=None,
    ):
        if not options:
            options = {}
        self.tol_options = options
        self.output = output.reshape([-1])
        self.golden = golden.reshape([-1])
        # third_party 跟 output/golden 同款 flatten；None 位置跳过 reshape（cross_check → GOLDEN_FAILURE）
        self.third_party = third_party.reshape([-1]) if third_party is not None else None
        self._pad_int4_if_need()
        self.output_idx = output_idx
        self.output_dtype = output_dtype
        self.__post_init__()
        self.standard = getattr(type(self), "STANDARD_NAME", "")

    def __post_init__(self):
        pass

    @abstractmethod
    def compare_impl(self) -> EachCompareResult:
        pass

    def _check_empty(self) -> Optional[EachCompareResult]:
        """都空→PASS；一方空→FAIL；都非空→None（继续 compare_impl 主体）。"""
        out_empty = self.output.size == 0
        gold_empty = self.golden.size == 0
        if out_empty and gold_empty:
            return EachCompareResult(
                1, is_pass=True, standard=self.standard, metrics={"standard": self.standard, "pass": True}
            )
        if out_empty or gold_empty:
            return EachCompareResult(
                0, is_pass=False, standard=self.standard, metrics={"standard": self.standard, "pass": False}
            )
        return None

    def compare(self):
        # 入口统一 numpy（torch 输入转 numpy；比对类内部纯 numpy）
        self.output = _to_numpy(self.output)
        self.golden = _to_numpy(self.golden)
        self.third_party = _to_numpy(self.third_party) if self.third_party is not None else None
        # 空数组统一短路（唯一点）：都空→PASS、一方空→FAIL；compare_impl 只在非空时跑
        compare_result = self._check_empty()
        if compare_result is None:
            compare_result = self.compare_impl()
        if compare_result.error_info:  # 失败原因→ERROR 日志（空不打；不进 metrics）
            logging.error("Output %d %s", self.output_idx, compare_result.error_info)
        if isinstance(compare_result.precision, str):
            precision = compare_result.precision
        else:
            precision = f"{compare_result.precision * 100}%"
        compare_result.log += "Output %d Precision is %s\n" % (self.output_idx, precision)
        compare_result.log += self._log_diff_output(compare_result.diff_index)
        gc.collect()
        return precision, compare_result.log, compare_result.is_pass, compare_result.metrics

    def _pad_int4_if_need(self):
        # only numpy dtype has `name` attribute. Op api output will always be torch.Tensor.
        from ...utilities.dtypes import is_4bit_dtype

        if hasattr(self.output.dtype, "name") and is_4bit_dtype(self.output.dtype):
            # output will always be 2*n
            if self.output.size >= 2 and self.golden.size == self.output.size - 1:
                self.golden = np.pad(self.golden, (0, 1), mode="constant")

    def _log_diff_output(self, diff_index) -> str:
        log = ""
        if diff_index is not None:
            diff_index_size = diff_index.size
            golden_size = self.golden.size
            log += "Output %d Compare Difference length %d\n" % (self.output_idx, diff_index_size)
            for idx in range(diff_index_size):
                real_index = diff_index[idx]
                golden_v = self.golden[real_index] if golden_size > 0 else 0
                actual_v = self.output[real_index]
                if golden_size <= 0:
                    log += "Index: %03d RealIndex: %06d Expected: 0 Actual: %-14.18f, Diff: inf\n" % (
                        idx,
                        real_index,
                        actual_v,
                    )
                elif any([s in str(self.golden.dtype) for s in ("float8", "float4", "hifloat8")]):
                    # out-of-range number will be nan/inf in
                    # `- (golden_v - actual_v) / golden_v`. convert to float.
                    tmp_array = np.array([float(golden_v), float(actual_v)], dtype=np.float32)
                    log += "Index: %03d RealIndex: %06d Expected: %-14.18f Actual: %-14.18f Diff: %-14.06f\n" % (
                        idx,
                        real_index,
                        golden_v,
                        actual_v,
                        -(tmp_array[0] - tmp_array[1]) / tmp_array[0],
                    )
                elif "bool" not in str(self.golden.dtype):
                    # (golden_v - actual_v) is different with (actual_v - golden_v) in BFP16 vs FP32-Golden
                    log += "Index: %03d RealIndex: %06d Expected: %-14.18f Actual: %-14.18f Diff: %-14.06f\n" % (
                        idx,
                        real_index,
                        golden_v,
                        actual_v,
                        -(golden_v - actual_v) / golden_v,
                    )
                else:
                    log += "Index: %03d RealIndex: %06d Expected: %s Actual: %s Diff: FAIL\n" % (
                        idx,
                        real_index,
                        str(golden_v),
                        str(actual_v),
                    )
                if idx == self.MAX_DIFF_OUTPUT:
                    break
        return log


class ComparisonRegister:
    registry: Dict[str, type] = {}  # token(lower) -> 比对类；只此一份，无 metadata dict


# 失败 reason 模板（PASS 无 reason）；带 {占位符} 的由各比对类 .format() 填实值
FAIL_REASONS = {
    "nan_inf_mismatch": "NaN/Inf mismatch",
    "threshold_exceeded": "threshold exceeded: {metrics}",
    "bitwise_mismatch": "bitwise mismatch",
    "cross_dtype_uncomparable": "cross-dtype not comparable",
    "tolerance_exceeded": "exceeded atol/rtol",
    "similarity_below_threshold": "cosine similarity below threshold",
    "precision_exceeded": "exceeded ptol",
    "third_party_count_mismatch": "third_party count != outputs",
    "third_party_unavailable": "third_party unavailable",
    "ratio_exceeded": "ratio exceeded: {exceeded}",
    "small_value_exceeded": "small value ErrorCount ratio exceeded: {ratio}",
}


def register_comparison(methods):
    if not isinstance(methods, (list, tuple, str)):
        raise TypeError(f"Register class for comparison must receive a list / tuple / str, not {methods}")
    if isinstance(methods, str):
        methods = [methods]

    def decorator(cls):
        for method in methods:
            key = method.lower()
            if key in ComparisonRegister.registry:
                logging.warning(f"comparison class of [{key}] has already been registered!")
            ComparisonRegister.registry[key] = cls
        return cls

    return decorator
