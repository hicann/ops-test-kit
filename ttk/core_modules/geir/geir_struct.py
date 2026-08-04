#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class GeirReturnStructure:
    precision: str = ""
    passed: bool = False
    precision_status: str = "UNKNOWN"
    log: str = ""
    cst_precision: str = ""
    dyn_precision: str = ""
    cst_perf_us: Optional[float] = None
    dyn_perf_us: Optional[float] = None
    xpu_metrics: Dict[str, Any] = field(default_factory=dict)
    deterministic_status: Optional[str] = None

    @staticmethod
    def get_titles(is_binary: bool = False) -> Tuple[str, ...]:
        cst_perf = "cst_bin_perf_us" if is_binary else "cst_perf_us"
        dyn_perf = "dyn_bin_perf_us" if is_binary else "dyn_perf_us"
        cst_gold = "cst_bin_precision" if is_binary else "cst_precision"
        dyn_gold = "dyn_bin_precision" if is_binary else "dyn_precision"
        return (
            "testcase_name",
            "op_name",
            "precision",
            "precision_status",
            cst_perf,
            dyn_perf,
            cst_gold,
            dyn_gold,
            "xpu_metrics",
            "deterministic_status",
            "log",
        )

    @staticmethod
    def get_title_indices():
        titles = GeirReturnStructure.get_titles()
        return {t: i for i, t in enumerate(titles)}

    def pick_data(self, case_result_title: tuple) -> tuple:
        data = []
        for title in case_result_title:
            if title == "testcase_name":
                data.append("")
            elif title == "op_name":
                data.append("")
            elif title == "precision":
                data.append(self.precision)
            elif title == "precision_status":
                data.append(self.precision_status)
            elif title in ("cst_perf_us", "cst_bin_perf_us"):
                data.append(self.cst_perf_us)
            elif title in ("dyn_perf_us", "dyn_bin_perf_us"):
                data.append(self.dyn_perf_us)
            elif title in ("cst_precision", "cst_bin_precision"):
                data.append(self.cst_precision)
            elif title in ("dyn_precision", "dyn_bin_precision"):
                data.append(self.dyn_precision)
            elif title == "xpu_metrics":
                data.append(self.xpu_metrics)
            elif title == "deterministic_status":
                data.append(self.deterministic_status)
            elif title == "log":
                data.append(self.log)
            else:
                data.append("")
        return tuple(data)
