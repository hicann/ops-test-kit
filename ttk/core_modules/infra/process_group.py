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
Process Group
"""

__all__ = ["ProcessGroup"]


# Standard Packages
import logging
import time
from multiprocessing.context import BaseContext
from typing import Dict

from ..tbe_multiprocessing import SimpleCommandProcess

# Third-Party Packages
from .task import TaskA, TaskType


class ProcessGroup:
    def __init__(self, dev_id: int, process_per_device: int, mp_context: BaseContext, timeout: int = 0):
        self.dev_id = dev_id
        self.processes = tuple(
            SimpleCommandProcess(mp_context, name=f"D{dev_id}P{i}", timeout=timeout) for i in range(process_per_device)
        )
        self.process_to_task: Dict[SimpleCommandProcess, TaskA] = {}

    def is_ready(self):
        return all(proc.is_ready() for proc in self.processes)

    def update(self):
        for proc in self.processes:
            proc.update()

    def idle_count(self):
        return sum(proc.is_idle() for proc in self.processes)

    def close_idles(self):
        for proc in self.processes:
            if proc.is_idle():
                proc.close()

    def close_all(self):
        for proc in self.processes:
            proc.close()

    def push(self, task: TaskA, is_multi_device: bool = False, rank_dev_id: int = None):
        for proc in self.processes:
            if proc.is_idle():
                logging.debug(
                    f"Sending {task.type.name} task {task.sub_type} "
                    f"of testcase {task.testcase.testcase_name} "
                    f"to process pid {proc.get_pid()}"
                )
                self.process_to_task[proc] = task
                kwargs = {"dev_id": rank_dev_id if rank_dev_id is not None else self.dev_id}
                if is_multi_device:
                    kwargs["is_multi_device"] = True
                    kwargs["device_ids"] = task.device_ids
                proc.send_action(task.func, task.params, kwargs)
                break
        else:
            raise RuntimeError(f"[BUG] no idle process in device{self.dev_id}. It maybe a BUG of TTK.")

    def has_prof_tasks(self):
        for proc in self.processes:
            if proc.is_idle():
                continue
            if proc not in self.process_to_task:
                continue
            task = self.process_to_task[proc]
            if task.type == TaskType.PROFILE:
                return True
        return False

    def completed_process(self):
        ret = []
        for proc in self.processes:
            if proc.is_completed() and proc in self.process_to_task:
                ret.append((proc, self.process_to_task[proc]))
                del self.process_to_task[proc]
        return tuple(ret)

    def info(self) -> str:
        return "\n".join(
            f"{proc.get_pid()} "
            f"{proc.name.ljust(16) if len(proc.name) < 16 else proc.name[-16:]} "
            f"{proc.current_stage().ljust(20)} "
            f"{proc.status.name.ljust(8)} "
            f"{int(time.time() - proc.process_status_timestamp)}s"
            for proc in self.processes
            if not proc.is_dead()
        )
