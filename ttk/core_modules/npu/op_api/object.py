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
import os
from multiprocessing.context import BaseContext
from typing import Set, Any

# Third-Party Packages
from .profiling_structure import ApiProfilingReturnStructure
from .profiling import profile_process
from ...testcase_manager import TestcaseBase, TestcaseAclnn
from ...tbe_multiprocessing import SimpleCommandProcess
from ...infra import TaskA, TaskType, TaskKeeper, ProfileObject


class ApiProfileObject(ProfileObject):
    def __init__(self, task_keeper: TaskKeeper, mp_context: BaseContext):
        super().__init__(task_keeper, mp_context)
        if os.getenv("ASCEND_OPP_KERNEL_PATH") is None:
            raise RuntimeError(f"Please install opp_kernel first.")

    def setup(self):
        pass

    def possible_result_titles(self) -> tuple:
        """ return all possible result titles """
        return ApiProfilingReturnStructure.get_titles()

    def init_tasks(self, testcases: Set[TestcaseBase]):
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        for t in testcases:
            self.task_keeper.append(TaskA(t, profile_process,
                                          (t, grant_events, granted_indices),
                                          TaskType.PROFILE))

    def apply_profile_success_result(self, testcase: TestcaseAclnn, result: Any) -> tuple:
        if not isinstance(result, ApiProfilingReturnStructure):
            raise RuntimeError(f"Only ApiProfilingReturnStructure is valid. "
                               f"But got {type(result)}")
        return result.pick_data(self.case_result_title), False
