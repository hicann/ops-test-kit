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
OP profile object
"""

__all__ = ["OpProfileObject"]


# Standard Packages
import logging
import time
from multiprocessing.context import BaseContext
from typing import Any, Iterable, Optional

from ....utilities import (
    BaseCompilationResult,
    compilation_result,
    construct_crash_compilation_result,
)
from ....utilities.proc import msdebug_runtime_injection_enabled
from ...comparison.compare_log import (
    compare_log_size,
    print_compare_log_failures,
    read_compare_log_failures,
)
from ...infra import ProfileObject, TaskA, TaskKeeper, TaskType
from ...operator import knowledge_base_sequence
from ...tbe_multiprocessing import SimpleCommandProcess
from ...testcase_manager import TestcaseOp

# Third-Party Packages
from .compilation import compilation_process
from .profiling import ProfilingReturnStructure, profile_process


class OpProfileObject(ProfileObject):
    def __init__(self, task_keeper: TaskKeeper, mp_context: BaseContext):
        super().__init__(task_keeper, mp_context)
        self.kb: Optional[SimpleCommandProcess] = None
        self._compare_log_read_size: int = 0

    def setup(self):
        self._compare_log_read_size = compare_log_size()
        if msdebug_runtime_injection_enabled():
            logging.warning("Running under msdebug, skipping KnowledgeBase server startup")
            return
        self._launch_knowledge_server(self.mp_context)

    def possible_result_titles(self) -> tuple:
        """return all possible result titles"""
        return ProfilingReturnStructure.get_titles()

    def result_titles(self) -> tuple:
        """return all result titles as per current command options"""
        return ProfilingReturnStructure.get_titles()

    def init_tasks(self, testcases: Iterable[TestcaseOp]):
        for case in testcases:
            case.kb_pid = self.kb.get_pid() if self.kb else 0
        grouped_testcases = TestcaseOp.hash_cases_to_groups(testcases)
        for cases in grouped_testcases.values():
            is_first = True
            for t in cases:
                compile_tasks = []
                for mode in ("Dyn", "Cst", "Bin"):
                    switch = getattr(self.switches, f"{mode.lower()}_switches")
                    if switch.enabled:
                        compile_tasks.append(TaskA(t, compilation_process, (t, mode), TaskType.COMPILE, mode))
                    else:
                        result = compilation_result(mode)
                        result.all_set(f"{mode.upper()}_OFF")
                        self.apply_compile_success_result(t, result)
                if not compile_tasks:
                    self.skipped_cases += 1
                    continue
                if is_first:
                    self.task_keeper.insert(compile_tasks)
                    is_first = False
                else:
                    self.task_keeper.append(compile_tasks)

    def pre_exit(self):
        # Flush any residual mismatch lines not yet printed (e.g. from cases
        # that errored before returning a normal result).
        self._print_new_compare_failures()
        if self.kb:
            self.kb.data["switch"] = False
            # 与 _launch_knowledge_server 同理: 固定 sleep(1) 轮询让**收尾**也平白多等数秒。
            # 这段在每次跑批结束时都会执行, 属与用例数无关的固定开销。
            poll_s, waited = 0.01, 0.0
            while self.kb.status == self.kb.status.RUNNING:
                self.kb.update()
                time.sleep(poll_s)
                waited += poll_s
                poll_s = min(poll_s * 1.5, 0.2)
            logging.info("OPT_MARK kb_stop_wait=%.2f", waited)  # TEMP
            self.kb.close()

    def apply_compile_fail_result(self, testcase: TestcaseOp, fail_info: str, task_sub_type: str):
        result = construct_crash_compilation_result(fail_info, task_sub_type)
        testcase.apply_compile_result(result)

    def apply_compile_success_result(self, testcase: TestcaseOp, result: Any):
        if not isinstance(result, BaseCompilationResult):
            raise RuntimeError(f"Only subtype of BaseCompilationResult is valid. But got {type(result)}")
        testcase.apply_compile_result(result)

    def apply_profile_success_result(self, testcase: TestcaseOp, result: Any) -> tuple:
        if not isinstance(result, ProfilingReturnStructure):
            raise RuntimeError(f"Only ProfilingReturnStructure is valid. But got {type(result)}")
        self._print_new_compare_failures(testcase.testcase_name)
        # if profiling fail, check to restart process to clear ErrorMessage
        return result.pick_data(self.case_result_title), result.kernel_execute_failed()

    def _print_new_compare_failures(self, testcase_name: Optional[str] = None):
        # Read mismatches appended since the last check and print them, so
        # failures surface as each case completes instead of only at the end.
        diff_lines, end_size = read_compare_log_failures(self._compare_log_read_size)
        self._compare_log_read_size = max(self._compare_log_read_size, end_size)
        print_compare_log_failures(diff_lines, testcase_name)

    def compile_done(self, testcase: TestcaseOp):
        if testcase.ready_for_profile():
            self._send_to_profiling(testcase)

    def handle_task_result_none(self, task) -> Optional[tuple]:
        if task.type == TaskType.COMPILE:
            self._compile_invalid_case(task)
            return self.compile_done(task.testcase)
        raise RuntimeError("Profile result is None which should not happen. Maybe it is a BUG of TTK !!!")

    def _launch_knowledge_server(self, mp_context: BaseContext):
        logging.info("Launching knowledge base Server process")
        self.kb = SimpleCommandProcess(mp_context, name="KBS")
        self.kb.data["switch"] = True
        self.kb.send_action(knowledge_base_sequence, (), {})
        # 等它就绪。原先是固定 sleep(1) 轮询: 进程实测几十毫秒就 RUNNING, 却要等到下一个
        # 整秒才被发现 —— 实测空等 4~5 秒, 且**每次跑批都付**(与用例数无关的固定开销)。
        # 改为短间隔起步、逐步放大到 0.2s 封顶: 就绪快时几乎零等待, 就绪慢时也不会忙等烧 CPU。
        # 日志仍按约 1 秒节流打印, 保留原有的可观测性。
        poll_s, next_log = 0.01, 0.0
        waited = 0.0
        while self.kb.status != self.kb.status.RUNNING:
            if waited >= next_log:
                logging.info(f"Process KnowledgeBaseServer status is {self.kb.status} !!! Update ...")
                next_log = waited + 1.0
            self.kb.update()
            if self.kb.is_dead():
                raise RuntimeError(
                    "Process KnowledgeBaseServer is DEAD. Please check exception raised by KnowledgeBaseServer."
                )
            time.sleep(poll_s)
            waited += poll_s
            poll_s = min(poll_s * 1.5, 0.2)
        logging.info(f"Knowledge base Server Pid: {self.kb.get_pid()} (ready in {waited:.2f}s)")
        logging.info("OPT_MARK kb_launch_wait=%.2f", waited)  # TEMP

    @staticmethod
    def _compile_invalid_case(task: TaskA):
        # only when testcase is invalid
        if not isinstance(task.testcase, TestcaseOp):
            raise RuntimeError(f"Only TestcaseOp instance is valid. But got {type(task.testcase)}")
        testcase: TestcaseOp = task.testcase
        reason = testcase.fail_reason
        logging.warning(
            f"Compilation process of mode {task.sub_type} skipped for "
            f"testcase {testcase.testcase_name} because of {reason}"
        )
        result = construct_crash_compilation_result(reason, task.sub_type)
        testcase.apply_compile_result(result)

    def _send_to_profiling(self, testcase: TestcaseOp):
        grant_events = SimpleCommandProcess._device_grant_events
        granted_indices = SimpleCommandProcess._device_granted_indices
        self.task_keeper.append(
            TaskA(testcase, profile_process, (testcase, grant_events, granted_indices), TaskType.PROFILE)
        )
