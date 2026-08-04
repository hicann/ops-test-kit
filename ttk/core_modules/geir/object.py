#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from typing import Any, Set

from ttk.core_modules.infra.profile_object import ProfileObject
from ttk.core_modules.infra.task import TaskA, TaskType
from ttk.core_modules.tbe_multiprocessing import SimpleCommandProcess
from ttk.core_modules.testcase_manager.testcase_base import TestcaseBase

from .geir_struct import GeirReturnStructure
from .profiling import geir_profile_process


class GeirProfileObject(ProfileObject):
    def __init__(self, task_keeper, mp_context):
        super().__init__(task_keeper, mp_context)

    def setup(self):
        pass

    def possible_result_titles(self) -> tuple:
        is_binary = getattr(self.switches, "geir_binary", False)
        return GeirReturnStructure.get_titles(is_binary=is_binary)

    def init_tasks(self, testcases: Set[TestcaseBase]):
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices

        # GEIR mode: gen + compile + profile in one task (like E2E)
        for testcase in testcases:
            if not testcase.is_valid:
                self.skipped_cases += 1
                continue
            task = TaskA(
                testcase,
                geir_profile_process,
                (testcase, grant_events, granted_indices),
                TaskType.PROFILE,
            )
            self.task_keeper.append(task)

    def apply_profile_success_result(self, testcase: TestcaseBase, result: Any) -> tuple:
        if isinstance(result, GeirReturnStructure):
            titles = self.possible_result_titles()
            data = {}
            data["testcase_name"] = testcase.testcase_name
            data["op_name"] = testcase.op_name
            data["precision"] = result.precision
            data["precision_status"] = result.precision_status
            data["cst_perf_us"] = result.cst_perf_us if result.cst_perf_us is not None else "CST_OFF"
            data["cst_bin_perf_us"] = result.cst_perf_us if result.cst_perf_us is not None else "CST_OFF"
            data["dyn_perf_us"] = result.dyn_perf_us if result.dyn_perf_us is not None else "DYN_OFF"
            data["dyn_bin_perf_us"] = result.dyn_perf_us if result.dyn_perf_us is not None else "DYN_OFF"
            data["cst_precision"] = result.cst_precision
            data["cst_bin_precision"] = result.cst_precision
            data["dyn_precision"] = result.dyn_precision
            data["dyn_bin_precision"] = result.dyn_precision
            data["xpu_metrics"] = result.xpu_metrics
            data["deterministic_status"] = result.deterministic_status
            data["log"] = result.log
            row = tuple(data.get(t, "") for t in titles)
            return row, False
        fallback = (str(result),) + ("",) * (len(self.possible_result_titles()) - 1)
        return fallback, False
