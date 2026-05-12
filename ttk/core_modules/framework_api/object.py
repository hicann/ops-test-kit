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
from typing import Set, Any

from ttk.core_modules.infra.profile_object import ProfileObject
from ttk.core_modules.infra.task import TaskA, TaskType
from ttk.core_modules.testcase_manager.testcase_base import TestcaseBase
from ttk.core_modules.tbe_multiprocessing import SimpleCommandProcess

from .profiling import profile_process
from .result import FrameworkApiReturnStructure


class FrameworkApiProfileObject(ProfileObject):
    """ProfileObject for framework-level API testing."""

    def __init__(self, task_keeper, mp_context, backend=None):
        super().__init__(task_keeper, mp_context)
        self.backend = backend

    def setup(self):
        pass

    def possible_result_titles(self) -> tuple:
        return FrameworkApiReturnStructure.get_titles()

    def init_tasks(self, testcases: Set[TestcaseBase]):
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        for testcase in testcases:
            task = TaskA(
                testcase,
                profile_process,
                (testcase, grant_events, granted_indices),
                TaskType.PROFILE,
            )
            self.task_keeper.append(task)

    def apply_profile_success_result(self, testcase: TestcaseBase, result: Any) -> tuple:
        if isinstance(result, FrameworkApiReturnStructure):
            results = result.pick_data(self.possible_result_titles())
            return results, False
        return (str(result),) + (None,) * (len(self.possible_result_titles()) - 1), False