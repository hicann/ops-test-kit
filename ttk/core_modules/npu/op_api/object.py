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
OP API profile object
"""

__all__ = ["ApiProfileObject"]


# Standard Packages
from multiprocessing.context import BaseContext
from typing import Any, Iterable, Optional

from ...comparison.compare_log import (
    compare_log_size,
    print_compare_log_failures,
    read_compare_log_failures,
)
from ...infra import ProfileObject, TaskA, TaskKeeper, TaskType
from ...tbe_multiprocessing import SimpleCommandProcess
from ...testcase_manager import TestcaseAclnn, TestcaseBase
from .profiling import profile_process

# Third-Party Packages
from .profiling_structure import ApiProfilingReturnStructure


class ApiProfileObject(ProfileObject):
    def __init__(self, task_keeper: TaskKeeper, mp_context: BaseContext):
        super().__init__(task_keeper, mp_context)
        self._compare_log_read_size: int = 0

    def setup(self):
        # Record offset of ttk-compare.log before any worker writes, so the
        # main process only reads mismatches produced by this run.
        self._compare_log_read_size = compare_log_size()

    def _print_new_compare_failures(self, testcase_name: Optional[str] = None):
        # Read mismatches appended since the last check and print them, so
        # failures surface as each case completes instead of only at the end.
        diff_lines, end_size = read_compare_log_failures(self._compare_log_read_size)
        if end_size > self._compare_log_read_size:
            self._compare_log_read_size = end_size
        print_compare_log_failures(diff_lines, testcase_name)

    def pre_exit(self):
        # Flush any residual mismatch lines not yet printed (e.g. from cases
        # that errored before returning a normal result).
        self._print_new_compare_failures()

    def possible_result_titles(self) -> tuple:
        """return all possible result titles"""
        return ApiProfilingReturnStructure.get_titles()

    def init_tasks(self, testcases: Iterable[TestcaseBase]):
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        sorted_cases = sorted(testcases, key=lambda t: getattr(t, "_csv_row_index", -1))
        for t in sorted_cases:
            device_ids = list(t.device_ids) if hasattr(t, "device_ids") and t.device_ids else None
            self.task_keeper.append(
                TaskA(t, profile_process, (t, grant_events, granted_indices), TaskType.PROFILE, device_ids=device_ids)
            )

    def apply_profile_success_result(self, testcase: TestcaseAclnn, result: Any) -> tuple:
        if not isinstance(result, ApiProfilingReturnStructure):
            raise RuntimeError(f"Only ApiProfilingReturnStructure is valid. But got {type(result)}")
        self._print_new_compare_failures(testcase.testcase_name)
        return result.pick_data(self.case_result_title), False
