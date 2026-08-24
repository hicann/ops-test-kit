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
FrameworkApiProfileObject — ProfileObject implementation for framework_api tests.
"""
from typing import Iterable, Any, Optional

from ttk.core_modules.infra.profile_object import ProfileObject
from ttk.core_modules.infra.task import TaskA, TaskType
from ttk.core_modules.testcase_manager.testcase_base import TestcaseBase
from ttk.core_modules.tbe_multiprocessing import SimpleCommandProcess
from ttk.core_modules.comparison.compare_log import (
    compare_log_size,
    read_compare_log_failures,
    print_compare_log_failures,
)

from .profiling import profile_process
from .result import FrameworkApiReturnStructure


class FrameworkApiProfileObject(ProfileObject):
    """ProfileObject for framework-level API testing."""

    def __init__(self, task_keeper, mp_context, backend=None):
        super().__init__(task_keeper, mp_context)
        self.backend = backend
        self._compare_log_read_size: int = 0

    def setup(self):
        # Record offset of ttk-compare.log before any worker writes, so the
        # main process only reads mismatches produced by this run.
        self._compare_log_read_size = compare_log_size()

    def possible_result_titles(self) -> tuple:
        return FrameworkApiReturnStructure.get_titles()

    def init_tasks(self, testcases: Iterable[TestcaseBase]):
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        sorted_cases = sorted(testcases, key=lambda t: getattr(t, '_csv_row_index', -1))
        for testcase in sorted_cases:
            device_ids = list(testcase.device_ids) if testcase.is_multi_device() else None
            task = TaskA(
                testcase,
                profile_process,
                (testcase, grant_events, granted_indices),
                TaskType.PROFILE,
                device_ids=device_ids,
            )
            self.task_keeper.append(task)

    def pre_exit(self):
        # Flush any residual mismatch lines not yet printed (e.g. from cases
        # that errored before returning a normal result).
        self._print_new_compare_failures()

    def apply_profile_success_result(self, testcase: TestcaseBase, result: Any) -> tuple:
        self._print_new_compare_failures(testcase.testcase_name)
        if isinstance(result, FrameworkApiReturnStructure):
            results = result.pick_data(self.possible_result_titles())
            return results, False
        return (str(result),) + (None,) * (len(self.possible_result_titles()) - 1), False

    def _print_new_compare_failures(self, testcase_name: Optional[str] = None):
        # Read mismatches appended since the last check and print them, so
        # failures surface as each case completes instead of only at the end.
        diff_lines, end_size = read_compare_log_failures(self._compare_log_read_size)
        if end_size > self._compare_log_read_size:
            self._compare_log_read_size = end_size
        print_compare_log_failures(diff_lines, testcase_name)
