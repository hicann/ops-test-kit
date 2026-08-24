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
Profiling Instance Base Class
"""


__all__ = ["InstanceBase"]


# Standard Packages
import csv
import io
import logging
import multiprocessing
import shutil
import subprocess

import numpy
import os
import time
import zipfile
from abc import ABCMeta, abstractmethod
from multiprocessing.context import BaseContext
from typing import Optional, IO, Any, Dict, List, Tuple

# Third-Party Packages
from .process_group import ProcessGroup
from .task import TaskA, TaskType, TaskKeeper
from .profile_object import ProfileObject
from ..tbe_multiprocessing import SimpleCommandProcess
from ..testcase_manager import UniversalTestcaseFactory, TestcaseBase
from ...utilities import get_global_storage, VERSION, table_print, list_append_union
from ...utilities import cpu_count


class InstanceBase(metaclass=ABCMeta):
    """
    Profiling Instance Base Class
    """

    def __init__(self):
        self.switches = get_global_storage()
        self.flatten_testcases: List[TestcaseBase] = []
        self.total_case_count = 0
        self.completed_case_count = 0
        # Final summary counters (driven by precision_status in result rows)
        self.pass_count = 0
        self.fail_count = 0
        self.other_count = 0
        self._header_flushed = False
        self._precision_status_idx: Optional[int] = None
        # Test Result CSV Storage
        self.result_csv_writer = None
        self.result_csv_file: Optional[IO[Any]] = None
        self.result_path: str = ""
        # Testcase
        self.case_original_headers: List[str] = []
        self.case_result_titles: Tuple[str] = tuple()
        # Multiprocessing
        self.mp_context: BaseContext = multiprocessing.get_context("forkserver")
        self.mp_manager = None
        self.device_locks = ()
        self.used_device = []
        self.process_groups: Dict[int, ProcessGroup] = {}
        # Tasks
        self.task_keeper: TaskKeeper = TaskKeeper()
        # Multi-device task tracking
        self._multi_device_completed: Dict[str, dict] = {}
        self._multi_device_running = False
        # DFX
        self.start_timestamp = time.time()
        self.last_print_timestamp = time.time() - 20
        self.print_cycle = 10
        # detail profile target
        self.profile_object: Optional[ProfileObject] = None
        if self.switches.random_seed:
            numpy.random.seed(self.switches.random_seed)
        self._commit_id: Optional[str] = None
        self.heartbeat_manager = None  # HeartbeatManager or None
        self.collected_results: list = []

    @staticmethod
    def _read_existing_header(path: str):
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                return next(reader, None)
        except (UnicodeDecodeError, csv.Error, OSError):
            return None

    @abstractmethod
    def env_prepare(self):
        pass

    @abstractmethod
    def get_device_count(self):
        pass

    @abstractmethod
    def get_device_platform(self):
        pass

    @abstractmethod
    def setup_profile_object(self):
        pass

    @abstractmethod
    def device_info(self, dev_id: int) -> str:
        pass

    def profile(self):
        logging.info("TTK Profiling Start!")
        logging.info(f"Mode: {self.switches.mode.name}")
        self.env_prepare()
        self.get_device_platform()
        logging.info(f"Device Platform: {self.switches.dev_plat}")
        if self.switches.validate_only:
            self._validate_only()
            return
        self.get_device_count()
        if self.switches.device_count <= 0:
            raise RuntimeError(f"Device count is invalid: {self.switches.device_count}")
        logging.info(f"Device Count: {self.switches.device_count}")
        self._parse_testcases()
        self._start_heartbeat_process()  # after validate_only early-return, before setup_profile_object
        self.setup_profile_object()
        self.profile_object.setup()
        self._open_result_file()
        self.prepare_subprocesses()
        self._prepare_tasks()
        # loop and push processes
        while True:
            self._supervise_heartbeat()  # respawn HB if it died; None-safe + throttled ~1s
            self._update_processes()
            self._handle_completed_process()
            self._push_task_to_process()
            self._close_idle_processes()
            self._summary_print(self.print_cycle)
            if self.total_case_count == self.completed_case_count:
                logging.info(f"ttk Profiling complete")
                break
            time.sleep(0.01)  # pacing before next iteration; avoids busy-spin 100% CPU on one core. Placed after the break-check so we don't sleep on the exiting iteration.
        # close all processes
        self.close_subprocesses()
        # batch consistency post-processing (level=2)
        self._post_process_batch_consistency()
        # clean up
        self._pre_exit()

    def prepare_subprocesses(self):
        self._prepare_device_locks()
        # Create process for every usable device
        if self.switches.process_per_device is None:
            # Use 80% of total cpu cores, not exceed 4
            self.switches.process_per_device = min(max(int(cpu_count() * 0.8) //
                                                       len(self.used_device), 1),
                                                   4)
        # Not exceed testcase count
        self.switches.process_per_device = min(self.switches.process_per_device,
                                               len(self.flatten_testcases))
        self.used_device = self.used_device[:len(self.flatten_testcases)]
        logging.info(f"Process per device: {self.switches.process_per_device}")
        # Prepare SubProcesses
        logging.info("Preparing Task Executors...")
        for dev_id in self.used_device:
            self.process_groups[dev_id] = ProcessGroup(dev_id,
                                                       self.switches.process_per_device,
                                                       self.mp_context,
                                                       timeout=self.switches.proc_timeout)
        os.environ["TTK_LOAD_TF"] = "1"
        for pg in self.process_groups.values():
            while not pg.is_ready():
                pg.update()

    def _start_heartbeat_process(self) -> None:
        """Spawn heartbeat subprocess if remote execution is configured.

        Called from profile() before setup_profile_object. Heartbeat starts
        sending GET /v1/heartbeat (merged health+detect+register) to all
        configured endpoints; writes health state to TTK_XPU_HEALTH_PATH.
        """
        try:
            from ttk.remote import is_remote_configured, get_tenant_id
            from ttk.remote.config import get_remote_config
            from ttk.remote.heartbeat import heartbeat_loop
            from ttk.remote.heartbeat_manager import HeartbeatManager
        except ImportError:
            return  # ttk.remote not available

        if not is_remote_configured():
            return

        config = get_remote_config()
        if not config or not config.endpoints:
            return

        # Shared TLS module (same source as dispatcher): tls_from_config does the
        # cert/key pair check (raises on mismatch, fail-loud); returns {} when no
        # TLS configured — build_tls_connection treats {} as plain HTTP.
        # NB: tls_from_config call is OUTSIDE the ImportError try/except above so a
        # cert/key mismatch raises as a loud startup failure (not swallowed).
        from ttk.remote.tls import tls_from_config
        tls = tls_from_config(config)
        self.heartbeat_manager = HeartbeatManager(
            heartbeat_target=heartbeat_loop,
            root_path=self.switches.root_path,
            tenant_id=get_tenant_id(),
            endpoints=config.endpoints,
            tls=tls,
        )
        self.heartbeat_manager.start()

    def _stop_heartbeat_process(self) -> None:
        """Terminate heartbeat subprocess and cleanup state.

        Called from close_subprocesses(). The subprocess sends DELETE to all
        endpoints before exiting, cleaning up tenant state on the server.
        """
        if self.heartbeat_manager is not None:
            self.heartbeat_manager.stop()
            self.heartbeat_manager = None

    def _supervise_heartbeat(self) -> None:
        """Respawn HB if it died. None-safe + throttled (~1s, not every 10ms loop tick).

        Called at the top of the profile() main loop before _update_processes.
        heartbeat_manager is None when remote execution is off — guarded.
        """
        if self.heartbeat_manager is None:
            return
        now = time.time()
        if now - getattr(self, "_last_supervise_ts", 0.0) < 1.0:
            return
        self._last_supervise_ts = now
        self.heartbeat_manager.supervise()

    def close_subprocesses(self):
        self._stop_heartbeat_process()
        for pg in self.process_groups.values():
            pg.close_all()

    def testcase_complete(self):
        self.completed_case_count += 1

    def _summary_print(self, print_cycle: int = 0):
        now = time.time()
        if now - self.last_print_timestamp > print_cycle:
            self.last_print_timestamp = now
            self._output_progress()
            if self.switches.summary_print:
                self._get_head_commit_id()
                try:
                    percentage = int(self.completed_case_count /
                                     self.total_case_count * 100)
                except:
                    percentage = "?"
                title = (f"Version: {VERSION} "
                         f"Summary (Device Total: {self.switches.device_count}) "
                         f"Progress: {percentage}% "
                         f"{self.completed_case_count} / {self.total_case_count} "
                         f"ET: {int(now - self.start_timestamp)}s "
                         f"Rev: {self._commit_id}",)

                loop_count = len(self.used_device) // 2 \
                    if self.used_device else self.switches.device_count // 2
                remain_count = len(self.used_device) % 2 \
                    if self.used_device else self.switches.device_count % 2
                lines = [title]
                for loop in range(loop_count):
                    if self.used_device:
                        dev_id_0 = self.used_device[loop * 2]
                        dev_id_1 = self.used_device[loop * 2 + 1]
                    else:
                        dev_id_0, dev_id_1 = loop * 2, loop * 2 + 1
                    lines.append((*(self.device_info(dev_id_0),
                                    self.process_groups[dev_id_0].info()
                                    if dev_id_0 in self.process_groups else ''),
                                  *(self.device_info(dev_id_1),
                                    self.process_groups[dev_id_1].info()
                                    if dev_id_1 in self.process_groups else '')
                                  ))
                if remain_count:
                    if self.used_device:
                        dev_id = self.used_device[-1]
                    else:
                        dev_id = loop_count * 2
                    lines.append((*(self.device_info(dev_id),
                                    self.process_groups[dev_id].info()
                                    if dev_id in self.process_groups else ''),
                                  ))
                logging.info("\n" + table_print(lines))

    def _pre_exit(self):
        self.profile_object.pre_exit()
        if self.switches.progress_output:
            self._output_progress()
        if self.result_csv_file:
            self.result_csv_file.close()
        self._print_final_summary()

    def _print_final_summary(self):
        total = self.pass_count + self.fail_count + self.other_count
        if total <= 0:
            return
        pass_rate = (self.pass_count / total * 100) if total else 0.0
        summary = (
            "\n==================== TTK Test Summary ===================\n"
            f"  Total    : {total}\n"
            f"  PASS     : {self.pass_count}\n"
            f"  FAIL     : {self.fail_count}\n"
            f"  SKIP/NA  : {self.other_count}\n"
            f"  PassRate : {pass_rate:.2f}%  ({self.pass_count}/{total})\n"
            "========================================================="
        )
        logging.info(summary)

    def _post_process_batch_consistency(self):
        """Level=3: cross-testcase batch consistency comparison."""
        if self.switches.deterministic_level != 3:
            return
        if not self.collected_results:
            return
        from ttk.core_modules.comparison.batch_consistency import compare_batch_consistency

        results = compare_batch_consistency(self.collected_results)
        if not results:
            logging.info("No batch consistency groups found (testcases lack batch_seed)")
            return
        total = len(results)
        passed = sum(1 for r in results if r["pass"])
        logging.info(f"Batch consistency: {passed}/{total} groups passed")
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            names = [m["testcase"] for m in r["members"]]
            logging.info(f"  [{status}] group={r['batch_consistency_id'][:32]}... "
                         f"members={names}")

    def _prepare_device_locks(self):
        # TODO: device-id not start from 0 to max-count in docker.
        # Prepare device locks
        self.device_locks = self._initialize_device_lock()
        for dev_id, dev_lock in enumerate(self.device_locks):
            if dev_lock is not None:
                self.used_device.append(dev_id)
        if len(self.used_device) <= 0:
            raise RuntimeError("Available device count is zero, aborting.")

    def _parse_testcases(self):
        logging.info("Preparing testcases ...")
        for case_file in self.switches.input_files:
            logging.info(f"Reading testcase file {case_file} ...")
            self._load_cases(case_file)
        if len(self.flatten_testcases) <= 0:
            raise RuntimeError("Case Number is 0!!!!")
        logging.info("Checking testcase name...")
        self._check_duplicate_case()
        self.total_case_count = len(self.flatten_testcases)
        logging.info(f"Case number: {self.total_case_count}")

    def _validate_only(self):
        self._parse_testcases()
        valid_count = 0
        invalid_cases = []
        reason_counts = {}
        for tc in sorted(self.flatten_testcases, key=lambda t: t.testcase_name):
            if tc.is_valid:
                valid_count += 1
            else:
                reason = tc.fail_reason or "UNKNOWN"
                invalid_cases.append((tc.testcase_name, getattr(tc, 'api_name', ''), reason))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        invalid_count = len(invalid_cases)
        total = valid_count + invalid_count
        logging.info(f"Validate Complete: {total} total, {valid_count} valid, {invalid_count} invalid")
        if reason_counts:
            logging.info("Invalid reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                logging.info(f"  {reason}: {count}")
        if invalid_cases:
            logging.info("Invalid cases:")
            for name, api, reason in invalid_cases:
                api_str = f" ({api})" if api else ""
                logging.info(f"  {name}{api_str}: {reason}")
        if self.switches.output_file_name:
            self._write_validate_result(invalid_cases, valid_count, invalid_count)
        if invalid_count > 0:
            logging.warning(f"{invalid_count} testcase(s) have validation errors")
        else:
            logging.info("All testcases passed validation")

    def _write_validate_result(self, invalid_cases, valid_count, invalid_count):
        import pathlib
        result_path = self.switches.output_file_name
        if not result_path.endswith('.csv'):
            result_path += '.csv'
        with open(result_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(("testcase_name", "api_name", "status", "fail_reason"))
            for tc in sorted(self.flatten_testcases, key=lambda t: t.testcase_name):
                api = getattr(tc, 'api_name', '')
                if tc.is_valid:
                    writer.writerow((tc.testcase_name, api, "VALID", ""))
                else:
                    writer.writerow((tc.testcase_name, api, "INVALID", tc.fail_reason or ""))
        logging.info(f"Validate result written to {result_path}")

    def _load_cases(self, testcase_file: str):
        use_csv_mode = False
        try:
            self._load_case_from_zip(testcase_file)
        except zipfile.BadZipFile:
            use_csv_mode = True
        if use_csv_mode:
            logging.info("Input testcase file is not a valid zip file, "
                         "switch to normal csv mode...")
            self._load_case_from_csv(testcase_file)

    def _check_duplicate_case(self):
        testcase_names = set()
        duplicates = set()
        for t in self.flatten_testcases:
            if t.testcase_name in testcase_names:
                duplicates.add(t)
            else:
                testcase_names.add(t.testcase_name)
        for t in duplicates:
            logging.warning(f"Testcase duplicate: {t.testcase_name}. Skip it ...")
            self.flatten_testcases.remove(t)

    def _open_result_file(self):
        logging.info("Initialize output csv file...")
        self.result_path = self.switches.output_file_name
        if not self.switches.input_files:
            raise RuntimeError("Please specify input csv files !!!")
        if not self.result_path:
            split_input_path = self.switches.input_files[0].split(".")
            split_input_path[-2] += "_result"
            self.result_path = '.'.join(split_input_path)
            logging.info(f"Output csv file is not specified. "
                         f"It will be set as {self.result_path}")
        if not self.result_path.endswith('.csv'):
            self.result_path += '.csv'
        parent_dir = os.path.dirname(self.result_path)
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logging.info(f"Created output directory: {parent_dir}")
        append_mode = self.switches.append_mode
        file_exists = os.path.exists(self.result_path) and os.path.getsize(self.result_path) > 0
        if append_mode and file_exists:
            self._prepare_output_titles()
            existing_header = self._read_existing_header(self.result_path)
            header_match = existing_header is not None and tuple(existing_header) == self.case_result_titles
            if header_match:
                self.result_csv_file = open(self.result_path, newline='', mode='a+')
                self.result_csv_writer = csv.writer(self.result_csv_file)
                self._header_flushed = True
                self._precision_status_idx = self._resolve_precision_status_idx(self.case_result_titles)
                logging.info(f"Append mode: appending to existing {self.result_path}")
            else:
                logging.warning(f"Append mode: existing file header does not match "
                                f"(file has {len(existing_header) if existing_header else 0} columns, "
                                f"current expects {len(self.case_result_titles)}). "
                                f"Overwriting {self.result_path}")
                self.result_csv_file = open(self.result_path, newline='', mode='w+')
                self.result_csv_writer = csv.writer(self.result_csv_file)
                self._flush(self.case_result_titles)
        else:
            self.result_csv_file = open(self.result_path, newline='', mode='w+')
            self.result_csv_writer = csv.writer(self.result_csv_file)
            self._prepare_output_titles()
            self._flush(self.case_result_titles)

    def _prepare_output_titles(self):
        first_testcase = next(iter(self.flatten_testcases))
        self.case_result_titles = self.profile_object.output_titles(first_testcase,
                                                                    self.case_original_headers)

    def _initialize_device_lock(self) -> tuple:
        self.mp_manager = self.mp_context.Manager()
        if self.switches.device_whitelist:
            blacklist = list(set([i for i in range(self.switches.device_count)]) -
                             set(self.switches.device_whitelist))
            self.switches.device_blacklist = list(set(blacklist).union(
                                                  set(self.switches.device_blacklist)))
        # Print Device blacklist info
        if self.switches.device_blacklist:
            logging.info(f"Device {self.switches.device_blacklist} "
                         f"has been blacklisted, removing...")
        available = tuple(True if n not in self.switches.device_blacklist else None
                          for n in range(self.switches.device_count))
        available_devices = [i for i, v in enumerate(available) if v is not None]
        SimpleCommandProcess.initialize_device_locks(available_devices, self.mp_manager)
        return available

    def _prepare_tasks(self):
        self.profile_object.init_tasks(self.flatten_testcases)
        self.completed_case_count += self.profile_object.skipped_cases

    def _update_processes(self):
        for pg in self.process_groups.values():
            pg.update()

    def _get_most_idle_process_group(self):
        max_idle, max_pg = 0, None
        for pg in self.process_groups.values():
            idles = pg.idle_count()
            if idles > max_idle:
                max_idle, max_pg = idles, pg
        return max_pg

    def _close_idle_processes(self):
        if self.task_keeper.empty():
            for pg in self.process_groups.values():
                pg.close_idles()

    def _output_progress(self):
        if self.switches.progress_output:
            now = time.time()
            with open(self.switches.progress_output, 'w') as f:
                f.write(f"StartAt:{str(self.start_timestamp)}\n"
                        f"ElapsedTime:{str(now - self.start_timestamp)}\n"
                        f"TotalCases:{str(self.total_case_count)}\n"
                        f"CompletedCases:{str(self.completed_case_count)}")

    def _push_at_least_one_prof_task_to_process(self):
        if self._multi_device_running:
            return
        for pg in self.process_groups.values():
            idles = pg.idle_count()
            if idles <= 0:
                continue
            if pg.has_prof_tasks():
                continue
            task = self.task_keeper.pop(TaskType.PROFILE)
            if task is None:
                break
            if task.is_multi_device():
                if self._try_push_multi_device_task(task):
                    self._multi_device_running = True
                    return
                else:
                    self.task_keeper.insert(task)
                    break
            pg.push(task)

    def _try_push_multi_device_task(self, task: TaskA) -> bool:
        primary_dev = task.device_ids[0]
        pg = self.process_groups.get(primary_dev)
        if pg is None or pg.idle_count() <= 0:
            return False
        pg.push(task, is_multi_device=True, rank_dev_id=primary_dev)
        return True

    def _push_task_to_process(self):
        if self._multi_device_running:
            return
        self._push_at_least_one_prof_task_to_process()
        if self._multi_device_running:
            return
        while True:
            max_pg = self._get_most_idle_process_group()
            if not max_pg:
                return
            task = self.task_keeper.pop()
            if task:
                if task.is_multi_device():
                    if self._try_push_multi_device_task(task):
                        self._multi_device_running = True
                        return
                    else:
                        self.task_keeper.insert(task)
                        break
                max_pg.push(task)
            else:
                break

    def _handle_completed_process(self):
        completed_process = []
        for pg in self.process_groups.values():
            completed_process.extend(pg.completed_process())
        for pt_pair in completed_process:
            proc, task = pt_pair
            if task.is_multi_device():
                self._multi_device_running = False
                self._handle_multi_device_task_result(task, proc)
            else:
                self._handle_task_result(task, proc)
                if task.type == TaskType.PROFILE:
                    self.testcase_complete()

    def _handle_multi_device_task_result(self, task: TaskA, proc: SimpleCommandProcess):
        case_name = task.testcase.testcase_name
        result = proc.get_result()
        pid = proc.get_pid()
        dev_id = None
        for pg_dev_id, pg in self.process_groups.items():
            if proc in pg.process_to_task:
                dev_id = pg_dev_id
                break
        if isinstance(result, (SystemError, RuntimeError)):
            proc.resurrect()
            if isinstance(result, SystemError):
                output_data = self.profile_object.handle_task_result_system_error(
                    task, result, "MultiDevice", pid)
            else:
                output_data = self.profile_object.handle_task_result_runtime_error(
                    task, result, pid)
        elif result is not None:
            output_data, kill_proc = self.profile_object.handle_task_result_complete(task, result)
            if self.switches.proc_no_reuse or kill_proc or task.is_multi_device():
                proc.close()
                proc.resurrect()
        else:
            output_data = self.profile_object.handle_task_result_none(task)
        if output_data:
            self._flush(output_data)
        self.testcase_complete()

    def _handle_task_result(self, task: TaskA, proc: SimpleCommandProcess):
        result = proc.get_result()
        pid = proc.get_pid()
        if result is None:
            output_data = self.profile_object.handle_task_result_none(task)
        elif isinstance(result, SystemError):
            proc.resurrect()
            output_data = self.profile_object.handle_task_result_system_error(task, result,
                                                                              proc.current_stage(),
                                                                              pid)
        elif isinstance(result, RuntimeError):
            proc.resurrect()
            output_data = self.profile_object.handle_task_result_runtime_error(task, result, pid)
        else:
            output_data, kill_proc = self.profile_object.handle_task_result_complete(task, result)
            if self.switches.deterministic_level > 0:
                self.collected_results.append((task.testcase, result))
            if self.switches.proc_no_reuse or kill_proc:
                proc.close()
                proc.resurrect()
        if output_data:
            self._flush(output_data)

    def _flush(self, row: tuple):
        self.result_csv_writer.writerow(row)
        self.result_csv_file.flush()
        if not self._header_flushed:
            self._header_flushed = True
            self._precision_status_idx = self._resolve_precision_status_idx(row)
            return
        self._update_summary(row)

    def _resolve_precision_status_idx(self, header: tuple) -> Optional[int]:
        try:
            return header.index("precision_status")
        except ValueError:
            return None

    def _update_summary(self, row: tuple):
        idx = self._precision_status_idx
        if idx is None or idx >= len(row):
            return
        status = str(row[idx]).upper()
        if status == "PASS":
            self.pass_count += 1
        elif status == "FAIL":
            self.fail_count += 1
        else:
            self.other_count += 1

    def _load_case_from_zip(self, testcase_path: str):
        with zipfile.ZipFile(testcase_path) as zipped_file:
            logging.info("Reading zipped testcases...")
            test_result = zipped_file.testzip()
            if test_result:
                raise RuntimeError(f"Zipfile corrupted on file: {test_result}")
            all_files = zipped_file.infolist()
            for file in all_files:
                if not file.filename.endswith(".csv"):
                    logging.warning(f"Skipped zipped non-testcase file {file.filename}")
                    continue
                logging.info(f"Reading zipped testcase file {file.filename} ...")
                with zipped_file.open(file) as real_file:
                    testcase_manager = UniversalTestcaseFactory(
                        io.TextIOWrapper(real_file, encoding="UTF-8", newline=''))
                    self.flatten_testcases.extend(testcase_manager.get())
                    list_append_union(self.case_original_headers, testcase_manager.header)

    def _load_case_from_csv(self, testcase_path: str):
        logging.info("Reading normal csv testcases...")
        with open(testcase_path, newline='', encoding='utf-8') as file:
            testcase_manager = UniversalTestcaseFactory(file)
            self.flatten_testcases.extend(testcase_manager.get())
            list_append_union(self.case_original_headers, testcase_manager.header)

    def _get_head_commit_id(self):
        if self._commit_id is not None:
            return
        if shutil.which('git') is None:
            self._commit_id = "UNKNOWN"
            return

        try:
            result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True,
                                    shell=False, check=False)
            if result.returncode != 0:
                self._commit_id = "UNKNOWN"
            else:
                self._commit_id = result.stdout.split('\n')[0]
        finally:
            if self._commit_id is None:
                self._commit_id = "UNKNOWN"
        return
