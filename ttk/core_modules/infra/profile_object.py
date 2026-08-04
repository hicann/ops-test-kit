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
Abstract Interface for profile objects.
"""


__all__ = ["ProfileObject"]


# Standard Packages
import copy
import logging
from abc import ABCMeta, abstractmethod
from multiprocessing.context import BaseContext
from typing import Optional, Set, Tuple, Any

# Third-Party Packages
from ...utilities import get_global_storage, list_exclude
from ..testcase_manager import TestcaseBase
from .task import TaskA, TaskKeeper, TaskType


class ProfileObject(metaclass=ABCMeta):
    def __init__(self, task_keeper: TaskKeeper,
                 mp_context: BaseContext):
        self.task_keeper = task_keeper
        self.mp_context = mp_context
        self.switches = get_global_storage()
        self.case_result_title: Tuple[str] = tuple()
        self.case_input_title: Tuple[str] = tuple()
        self.skipped_cases: int = 0
        self._front_input_count: int = 0

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def possible_result_titles(self) -> tuple:
        """ return all possible result titles """
        pass

    @abstractmethod
    def init_tasks(self, testcases: Set[TestcaseBase]):
        pass

    @abstractmethod
    def apply_profile_success_result(self, testcase: TestcaseBase, result: Any) -> tuple:
        pass

    def result_titles(self) -> tuple:
        """ return all result titles as per current command options """
        return self.possible_result_titles()

    def pre_exit(self):
        pass

    def apply_compile_fail_result(self, testcase: TestcaseBase,
                                  crash_info: str, task_sub_type: str):
        pass

    def apply_compile_success_result(self, testcase: TestcaseBase, result: Any):
        pass

    def compile_done(self, testcase: TestcaseBase) -> Optional[tuple]:
        return None

    def handle_task_result_system_error(self, task: TaskA, result: SystemError,
                                        last_stage: str, pid: int) -> Optional[tuple]:
        """ return output content to csv """
        if task.type == TaskType.COMPILE:
            self._compile_crash(task, result, last_stage, pid)
            return self.compile_done(task.testcase)
        else:
            return self._profile_crash(task, result, last_stage, pid)

    def handle_task_result_runtime_error(self, task: TaskA, result: RuntimeError,
                                         pid: int) -> Optional[tuple]:
        """ return output content to csv """
        if task.type == TaskType.COMPILE:
            self._compile_fail(task, result, pid)
            return self.compile_done(task.testcase)
        else:
            return self._profile_fail(task, result, pid)

    def handle_task_result_complete(self, task: TaskA, result: object) -> Optional[tuple]:
        """ return output content to csv & whether kill current process."""
        if task.type == TaskType.COMPILE:
            self._compile_normal_complete(task, result)
            return self.compile_done(task.testcase), False
        else:
            return self._profile_normal_complete(task, result)

    def handle_task_result_none(self, task) -> Optional[tuple]:
        pass

    def output_titles(self, testcase: TestcaseBase, case_original_headers: list) -> tuple:
        if self.switches.preserve_original_csv:
            input_titles = copy.deepcopy(case_original_headers)
            possible_result_titles = self.possible_result_titles()
            list_exclude(input_titles, possible_result_titles)
        else:
            input_titles = testcase.get_all_visible_headers()

        result_titles = self.result_titles()
        if self.switches.custom_columns:
            result_titles = tuple(title for title in result_titles
                                  if title in self.switches.custom_columns)
            input_titles = tuple(title for title in input_titles
                                 if title in self.switches.custom_columns)
        input_titles = self._reorder_input_titles(input_titles)
        self.case_input_title = tuple(input_titles)
        self.case_result_title = tuple(result_titles)
        return self._compose_header()

    _FRONT_IDENTITY_HEADERS = ("testcase_name", "op_name", "api_name")

    def _reorder_input_titles(self, input_titles: tuple) -> tuple:
        """Move identity headers (testcase_name, op_name/api_name) to the front,
        preserving the relative order of the remaining titles. Returns a tuple;
        caller reads ``self._front_input_count`` for the count of moved headers."""
        self._front_input_count = 0
        if self.switches.preserve_original_csv or self.switches.custom_columns:
            return input_titles
        if not input_titles or input_titles[0] != "testcase_name":
            return input_titles
        front = ["testcase_name"]
        rest = list(input_titles[1:])
        for h in self._FRONT_IDENTITY_HEADERS[1:]:
            if h in rest:
                rest.remove(h)
                front.append(h)
        self._front_input_count = len(front)
        return tuple(front + rest)

    def _compose_header(self) -> tuple:
        return self._assemble_row(self.case_input_title, self.case_result_title)

    def _assemble_row(self, inputs: tuple, results: tuple) -> tuple:
        n = self._front_input_count
        if n > 0 and len(inputs) > n:
            return (*inputs[:n], *results, *inputs[n:])
        return (*inputs, *results)

    def _compile_crash(self, task: TaskA, result: SystemError, last_stage: str, pid: int):
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. "
                               f"But got {task.testcase.__class__}")
        logging.fatal(f"Compilation process of mode {task.sub_type} crashed at stage {last_stage} "
                      f"for testcase {task.testcase.testcase_name} with pid {pid}. "
                      f"System error: {result}")
        crash_info = f"Crashed at stage {last_stage}: {result}"
        self.apply_compile_fail_result(task.testcase, crash_info, task.sub_type)

    def _compile_fail(self, task: TaskA, result: RuntimeError, pid: int):
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. "
                               f"But got {task.testcase.__class__}")
        exception_print: str = result.args[0]
        reason = "COMPILE_FAILURE"
        logging.error(f"Compilation process of mode {task.sub_type} failed "
                      f"for testcase {task.testcase.testcase_name} with pid {pid}, "
                      f"fail reason: \n{exception_print}")
        self.apply_compile_fail_result(task.testcase, reason, task.sub_type)

    def _compile_normal_complete(self, task: TaskA, result: object):
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. "
                               f"But got {task.testcase.__class__}")
        self.apply_compile_success_result(task.testcase, result)

    def _profile_fail_result(self, main: str, details: str) -> tuple:
        results = [main] * len(self.case_result_title)
        if len(results) > 0:
            results[0] = details  # store details.
        return tuple(results)

    def _profile_crash(self, task: TaskA, result: SystemError, last_stage: str, pid: int) -> tuple:
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. "
                               f"But got {task.testcase.__class__}")
        logging.fatal(f"Profile process crashed at stage {last_stage} "
                      f"for testcase {task.testcase.testcase_name} with pid {pid}. "
                      f"System error: {result}")
        results = self._profile_fail_result(f"PROFILE_CRASH",
                                             f"Crashed at profiling stage: {result}")
        inputs = task.testcase.pick_data(self.case_input_title)
        return self._assemble_row(inputs, results)

    def _profile_fail(self, task: TaskA, result: RuntimeError, pid: int) -> tuple:
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. "
                               f"But got {task.testcase.__class__}")
        exception = result.args[0]
        logging.error(f"Profiling process failed "
                      f"for testcase {task.testcase.testcase_name} with pid {pid}, "
                      f"fail reason:\n{exception}")
        results = self._profile_fail_result(f"FAILURE", exception)
        inputs = task.testcase.pick_data(self.case_input_title)
        return self._assemble_row(inputs, results)

    def _profile_normal_complete(self, task: TaskA, result: object) -> tuple:
        if not issubclass(task.testcase.__class__, TestcaseBase):
            raise RuntimeError(f"Only subclass of TestcaseBase instance is valid. "
                               f"But got {task.testcase.__class__}")
        results, kill_Proc = self.apply_profile_success_result(task.testcase, result)
        inputs = task.testcase.pick_data(self.case_input_title)
        return self._assemble_row(inputs, results), kill_Proc
