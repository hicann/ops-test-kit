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
Custom Pool Classes
"""


__all__ = ["get_process_context", "SimpleCommandProcess", "DeviceLock"]


# Standard Packages
import atexit
import gc
import os
import sys
import time
import logging
import traceback
import re
import subprocess
import importlib
import multiprocessing as mp
import numpy
from enum import Enum, auto
from queue import SimpleQueue
from typing import Any, Dict, List, Optional, NoReturn, Callable

# Third-Party Packages
import psutil

from ...utilities import set_process_name, set_thread_name, signal_registered
from ...utilities import set_global_storage, get_global_storage
from ..tbe_logging import default_logging_config


class PROCESS_STATUS_CODE(Enum):
    CREATED = 0
    LAUNCHED = 114
    IDLE = 1919
    RUNNING = 810
    WAITING = 1551
    DEAD = 1145141919810


class PROCESS_RPC(Enum):
    EXECUTE_FUNCTION = auto()
    FUNCTION_RETURN = auto()
    ACQUIRE_SEMAPHORE = auto()
    SET_SEMAPHORE = auto()
    GET_SEMAPHORE = auto()
    RELEASE_SEMAPHORE = auto()
    GET_ACQUIRED_SEMAPHORES = auto()
    STORE_DATA = auto()
    GET_DATA = auto()
    CHANGE_NAME = auto()
    SUICIDE = auto()
    SEMAPHORE_DEAD_SEQUENCE = auto()
    GET_LOCK = auto()
    RELEASE_LOCK = auto()


process_context = None


class DeviceLockManager:
    """Parent-side device lock state manager."""
    lock_holders: Dict[int, "SimpleCommandProcess"] = {}
    pending: Dict[int, List["SimpleCommandProcess"]] = {}

    @classmethod
    def try_grant(cls, proc, device_id, lock_id, grant_event, granted_idx):
        if cls.lock_holders.get(device_id) is None:
            cls.lock_holders[device_id] = proc
            granted_idx.value = lock_id
            grant_event.set()
        else:
            cls.pending.setdefault(device_id, []).append((proc, lock_id))

    @classmethod
    def release(cls, proc, device_id, grant_event, granted_idx):
        if cls.lock_holders.get(device_id) is proc:
            cls.lock_holders[device_id] = None
            cls._grant_next(device_id, grant_event, granted_idx)

    @classmethod
    def on_process_dead(cls, proc, grant_events, granted_indices):
        for dev_id, holder in list(cls.lock_holders.items()):
            if holder is proc:
                cls.lock_holders[dev_id] = None
                cls._grant_next(dev_id, grant_events[dev_id], granted_indices[dev_id])
        for dev_id in list(cls.pending.keys()):
            cls.pending[dev_id] = [(p, lid) for p, lid in cls.pending[dev_id] if p is not proc]

    @classmethod
    def _grant_next(cls, device_id, grant_event, granted_idx):
        waiters = cls.pending.get(device_id, [])
        if waiters:
            nxt_proc, nxt_lock_id = waiters.pop(0)
            cls.lock_holders[device_id] = nxt_proc
            granted_idx.value = nxt_lock_id
            grant_event.set()

    @classmethod
    def initialize(cls, available_devices):
        cls.lock_holders = {dev_id: None for dev_id in available_devices}
        cls.pending = {dev_id: [] for dev_id in available_devices}


class DeviceLock:
    """
    Context manager for child-side device lock.

    Args:
        process_ctx: ProcessContext instance (child-side)
        device_id: which device to lock
        use_device: if False, skip lock entirely (no-device / CPU mode)
        grant_event: Manager().Event() for grant signal
        granted_idx: Manager().Value('i', -1) for grant identity verification
    """

    def __init__(self, process_ctx, device_id, use_device=True,
                 grant_event=None, granted_idx=None):
        self.process_ctx = process_ctx
        self.device_id = device_id
        self.use_device = use_device
        self.grant_event = grant_event
        self.granted_idx = granted_idx
        self._lock_id = os.getpid()

    def __enter__(self):
        if not self.use_device:
            return self
        self.process_ctx.get_lock(self.device_id, self._lock_id,
                                  self.grant_event, self.granted_idx)
        self.grant_event.wait()
        while self.granted_idx.value != self._lock_id:
            self.grant_event.wait()
        self.grant_event.clear()
        return self

    def __exit__(self, *args):
        if not self.use_device:
            return
        self.process_ctx.release_lock(self.device_id)


def get_process_context() -> "Optional[ProcessContext]":
    return process_context


def wait_attach():
    while True:
        time.sleep(0.1)
        with open(f"/proc/{os.getpid()}/status") as f:
            res = f.readlines()
        pattern = re.compile(r"TracerPid:\t(\d*)\n")
        final = [re.match(pattern, r) for r in res]
        for f in final:
            if f and f.group(1).isnumeric() and int(f.group(1)) > 0:
                return


def on_exit():
    if atexit._ncallbacks() > 1:  # 1 for Python internal callback
        atexit._run_exitfuncs()


def _preload_plugin_frameworks(plugin_path):
    """Pre-import heavy frameworks (tensorflow/torch) declared in plugin files.

    CANN tbe native libs conflict with TensorFlow C extensions when tbe is
    loaded first (SIGSEGV). Importing TF/torch BEFORE tbe avoids the clash.
    Scan plugin .py for import statements (any scope) and pre-load matching frameworks.
    """
    import ast
    from pathlib import Path

    framework_names = {"tensorflow", "torch"}
    to_load = set()
    if not plugin_path:
        return
    paths = plugin_path if isinstance(plugin_path, (list, tuple)) else (plugin_path,)
    for p in paths:
        p = Path(p) if not isinstance(p, Path) else p
        files = [p] if p.is_file() else (list(p.rglob("*.py")) if p.is_dir() else [])
        for py_file in files:
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    root = name.split(".")[0]
                    if root in framework_names:
                        to_load.add(root)
    for fw in to_load:
        try:
            importlib.import_module(fw)
            logging.debug(f"Pre-imported {fw} before tbe to avoid native lib conflict")
        except Exception as e:
            logging.warning(f"Pre-import {fw} failed (will try lazy import later): {e}")


def worker_bootstrap(global_storage):
    """Worker 进程初始化：设全局存储 + 加载配置 + 日志 + 随机种子。

    从 intermediate_func 的 prologue 提取，可独立单测（intermediate_func 是
    while True RPC 循环，无法直接测）。load_config 让 forkserver worker
    拿到 --config 的配置（含 TLS）——本 spec 的核心修复。
    """
    set_global_storage(global_storage)
    from ttk.config.loader import load_config  # lazy import（函数体内，避循环依赖）
    load_config(global_storage.config_path)
    default_logging_config(file_handler=global_storage.logging_to_file)
    if global_storage.random_seed:
        numpy.random.seed(global_storage.random_seed)
    _preload_plugin_frameworks(getattr(global_storage, "plugin_path", None))


def intermediate_func(pipe: "mp.connection.Connection", global_storage) -> NoReturn:
    global process_context
    process_context = ProcessContext(pipe)
    process_context.report_status(PROCESS_STATUS_CODE.LAUNCHED)  # 0x0114 -> Launched
    worker_bootstrap(global_storage)
    while True:
        process_context.report_status(PROCESS_STATUS_CODE.IDLE)  # 0x1919 -> Ready for Command
        # noinspection PyBroadException
        try:
            message = pipe.recv()
        except:
            # Silently suicide
            on_exit()
            raise RuntimeError("Subprocess is still alive but main process is dead.")
        if isinstance(message, tuple):
            rpc_command = message[0]
            if rpc_command == PROCESS_RPC.EXECUTE_FUNCTION:
                rpc_args = message[1]
                process_context.report_status(PROCESS_STATUS_CODE.RUNNING)
                func: Callable = rpc_args[0]
                func_args: tuple = rpc_args[1]
                func_kwargs: dict = rpc_args[2]
                function_return = None
                # noinspection PyBroadException
                try:
                    function_return = func(*func_args, **func_kwargs)
                    process_context.rpc_call((PROCESS_RPC.FUNCTION_RETURN, (function_return,)))
                except:
                    process_context.notify_status("OnExceptionReport")
                    function_return = RuntimeError(traceback.format_exc())
                    process_context.rpc_call((PROCESS_RPC.FUNCTION_RETURN, (function_return,)))
                    process_context.semaphore_dead_process(function_return)
                    process_context.report_status(PROCESS_STATUS_CODE.DEAD)
                    pipe.close()
                    on_exit()
                    return
                finally:
                    del func, func_args, func_kwargs, function_return
                    gc.collect()
            elif rpc_command == PROCESS_RPC.SUICIDE:
                on_exit()
                if signal_registered(34):
                    os.kill(os.getpid(), 34)  # send signal to HDT for code coverage.
                sys.exit(-1)
            else:
                logging.warning(f"SimpleCommandProcess command pipe received invalid rpc call, ignored: {message}")
        else:
            logging.warning(f"SimpleCommandProcess command pipe received invalid command, ignored: {message}")


class ProcessContext:
    def __init__(self, pipe):
        self.pipe: "mp.connection.Connection" = pipe
        self.storage = {}

    def report_status(self, status: PROCESS_STATUS_CODE):
        self.pipe.send(status)

    def rpc_call(self, args):
        self.pipe.send(args)

    def acquire_semaphore(self, name) -> bool:
        self.pipe.send((PROCESS_RPC.ACQUIRE_SEMAPHORE, (name,)))
        return self.pipe.recv()

    def set_semaphore(self, name, value) -> NoReturn:
        self.pipe.send((PROCESS_RPC.SET_SEMAPHORE, (name, value)))

    def get_semaphore(self, name):
        self.pipe.send((PROCESS_RPC.GET_SEMAPHORE, (name,)))
        return self.pipe.recv()

    def get_acquired_semaphore(self) -> tuple:
        self.pipe.send((PROCESS_RPC.GET_ACQUIRED_SEMAPHORES, ()))
        return self.pipe.recv()

    def semaphore_dead_process(self, value):
        self.pipe.send((PROCESS_RPC.SEMAPHORE_DEAD_SEQUENCE, (value,)))

    def send_data(self, name, value) -> NoReturn:
        self.pipe.send((PROCESS_RPC.STORE_DATA, (name, value)))

    def get_data(self, name):
        self.pipe.send((PROCESS_RPC.GET_DATA, (name,)))
        return self.pipe.recv()

    def change_name(self, name: str) -> NoReturn:
        set_process_name(name)
        self.pipe.send((PROCESS_RPC.CHANGE_NAME, (name,)))

    def get_lock(self, device_id, lock_id, grant_event, granted_idx):
        self.pipe.send((PROCESS_RPC.GET_LOCK, (device_id, lock_id, grant_event, granted_idx)))

    def release_lock(self, device_id):
        self.pipe.send((PROCESS_RPC.RELEASE_LOCK, (device_id,)))

    def notify_status(self, status: str):
        set_thread_name(status)
        self.send_data("stage", status)


class SimpleCommandProcess:
    semaphore_to_holder: Dict[Any, "SimpleCommandProcess"] = {}
    holder_to_semaphores: Dict["SimpleCommandProcess", List[Any]] = {}
    semaphores: Dict[Any, Any] = {}
    all_processes: List["SimpleCommandProcess"] = []
    # Parent-managed device lock state
    _device_grant_events: Dict[int, Any] = {}
    _device_granted_indices: Dict[int, Any] = {}

    def __init__(self, context=mp, name="TBESimpleCommandProcess", daemon=None,
                 timeout: int = 0):
        self.original_input_params = (context, name, daemon, timeout)
        self.status: PROCESS_STATUS_CODE = PROCESS_STATUS_CODE.CREATED
        self.rpc_queue: SimpleQueue = SimpleQueue()
        self.rpc_results: SimpleQueue = SimpleQueue()
        self.locks: list = []
        self.process_status_timestamp = time.time()
        self.data: Dict[str, Any] = {}
        self.parent_pipe, self.child_pipe = context.Pipe()
        self.name = name
        self.timeout = timeout
        self.parent = context.Process(target=intermediate_func, name=name, args=(self.child_pipe,
                                                                                 get_global_storage()),
                                      daemon=daemon)
        self.all_processes.append(self)
        self.parent.start()
        logging.debug(f"Process created with name {self.parent.name}")

    @classmethod
    def initialize_device_locks(cls, available_devices, mp_manager):
        """Create Event+Value per device and initialize DeviceLockManager."""
        for dev_id in available_devices:
            cls._device_grant_events[dev_id] = mp_manager.Event()
            cls._device_granted_indices[dev_id] = mp_manager.Value('i', -1)
        DeviceLockManager.initialize(available_devices)

    @staticmethod
    def _handle_locks():
        lock = None
        if isinstance(lock, mp.synchronize.Semaphore):
            raise RuntimeError("Subprocess dead while holding Semaphore")
        elif isinstance(lock, mp.synchronize.Lock):
            # noinspection PyBroadException
            try:
                lock.release()
            except:
                logging.exception("Lock releasing failure:")
        elif isinstance(lock, mp.synchronize.RLock):
            raise RuntimeError("Subprocess dead while holding RLock")

    def _update(self):
        while self.parent_pipe.poll():
            message = self.parent_pipe.recv()
            if isinstance(message, PROCESS_STATUS_CODE):
                self.status = message
                self.process_status_timestamp = time.time()
                if message == PROCESS_STATUS_CODE.DEAD:
                    return
            elif isinstance(message, tuple):
                rpc_command = message[0]
                rpc_args = message[1]
                self._rpc_call(rpc_command, rpc_args)
            else:
                raise RuntimeError(f"SimpleCommandProcess received invalid command: {message}")
        if self.get_exitcode() is not None:
            raise EOFError()
        if (
                0 < self.timeout < int(time.time() - self.process_status_timestamp) and
                (
                    'Profiling' in self.current_stage() or
                    'Compilation' in self.current_stage() or
                    'Gen' in self.current_stage()
                )
        ):
            raise TimeoutError()

    def _parent_send_rpc(self, rpc_command: PROCESS_RPC, rpc_args: tuple):
        self.parent_pipe.send((rpc_command, rpc_args))

    def _clear_data(self):
        self.data = {}

    def _rpc_call(self, rpc_command: PROCESS_RPC, rpc_args: tuple):
        if rpc_command == PROCESS_RPC.EXECUTE_FUNCTION:
            func = rpc_args[0]
            func_args = rpc_args[1]
            func_kwargs = rpc_args[2]
            function_return = func(*func_args, **func_kwargs)
            self._parent_send_rpc(PROCESS_RPC.FUNCTION_RETURN, (function_return,))
        elif rpc_command == PROCESS_RPC.FUNCTION_RETURN:
            function_return = rpc_args[0]
            self.rpc_results.put(function_return)
            self._clear_data()
            self.name = self.original_input_params[1]
            self.parent.name = self.name
        elif rpc_command == PROCESS_RPC.STORE_DATA:
            name: str = rpc_args[0]
            value = rpc_args[1]
            self.data[name] = value
            if name == 'stage':
                self._update_timestamp()
        elif rpc_command == PROCESS_RPC.GET_DATA:
            name: str = rpc_args[0]
            if name in self.data:
                self.parent_pipe.send(self.data[name])
            else:
                self.parent_pipe.send(None)
        elif rpc_command == PROCESS_RPC.CHANGE_NAME:
            name: str = rpc_args[0]
            self.name = name
            self.parent.name = name
        elif rpc_command == PROCESS_RPC.ACQUIRE_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphore_to_holder:
                self.parent_pipe.send(False)
            else:
                self.semaphore_to_holder[name] = self
                self.holder_to_semaphores.setdefault(self, []).append(name)
                self.semaphores[name] = None
                self.parent_pipe.send(True)
        elif rpc_command == PROCESS_RPC.SET_SEMAPHORE:
            name = rpc_args[0]
            value = rpc_args[1]
            if self.semaphore_to_holder[name] == self:
                self.semaphores[name] = value
            else:
                logging.warning(f"{self.name} trying to access semaphore of another process: {name}")
        elif rpc_command == PROCESS_RPC.GET_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphores:
                value = self.semaphores[name]
            else:
                value = None
            self.parent_pipe.send(value)
        elif rpc_command == PROCESS_RPC.RELEASE_SEMAPHORE:
            name = rpc_args[0]
            if name in self.semaphores and self.semaphore_to_holder[name] == self:
                del self.semaphores[name]
                del self.semaphore_to_holder[name]
                self.holder_to_semaphores[self].remove(name)
            else:
                logging.warning(f"{self.name} trying to release invalid semaphore: {name}")
        elif rpc_command == PROCESS_RPC.GET_ACQUIRED_SEMAPHORES:
            if self in self.holder_to_semaphores:
                self.parent_pipe.send(self.holder_to_semaphores[self])
            else:
                self.parent_pipe.send(())
        elif rpc_command == PROCESS_RPC.SEMAPHORE_DEAD_SEQUENCE:
            value = rpc_args[0]
            self._semaphore_dead_sequence(value)
        elif rpc_command == PROCESS_RPC.GET_LOCK:
            device_id, lock_id, grant_event, granted_idx = rpc_args
            logging.debug(f"Get lock for device {device_id}")
            DeviceLockManager.try_grant(self, device_id, lock_id, grant_event, granted_idx)
        elif rpc_command == PROCESS_RPC.RELEASE_LOCK:
            device_id = rpc_args[0]
            logging.debug(f"Release lock for device {device_id}")
            DeviceLockManager.release(self, device_id,
                                      SimpleCommandProcess._device_grant_events[device_id],
                                      SimpleCommandProcess._device_granted_indices[device_id])
        else:
            raise NotImplementedError(f"SimpleCommandProcess Master received invalid rpc call: "
                                      f"{rpc_command, rpc_args}")

    def _semaphore_dead_sequence(self, value):
        if self in self.holder_to_semaphores:
            for sem in self.holder_to_semaphores[self]:
                if self.semaphores[sem] is None:
                    self.semaphores[sem] = value

    def _update_timestamp(self):
        self.process_status_timestamp = time.time()

    def get_pid(self):
        try:
            return self.parent.pid
        except ValueError:
            return None

    def get_exitcode(self):
        return self.parent.exitcode

    def get_result(self):
        return self.rpc_results.get()

    def get_memory_usage_percent(self):
        return psutil.Process(self.get_pid()).memory_percent()

    def current_stage(self) -> str:
        return self.data.get("stage", "UNKNOWN")

    def resurrect(self):
        self.__init__(*self.original_input_params)

    def close(self, no_update=False):
        if not no_update:
            self.update()
        if not self.status == PROCESS_STATUS_CODE.DEAD:
            # noinspection PyBroadException
            try:
                self._parent_send_rpc(PROCESS_RPC.SUICIDE, ())
                self.parent.join(3)
            finally:
                if self.get_exitcode() is None:
                    self.parent.terminate()
                    self.parent.join(3)
                    self.kill()
                try:
                    self.parent.close()
                except ValueError:
                    self.parent.terminate()
                    self.parent.join()
                    self.parent.close()
                self.status = PROCESS_STATUS_CODE.DEAD

    def kill(self):
        self.parent.kill()

    def update(self) -> NoReturn:
        if self.status == PROCESS_STATUS_CODE.DEAD:
            return
        try:
            self._update()
        except (EOFError, TimeoutError) as e:
            if isinstance(e, EOFError):
                if self.get_exitcode() is not None:
                    logging.warning(f"Process {self.name} exited unexpectedly with code {self.get_exitcode()}")
                    exception = SystemError(f"Process {self.name} exited unexpectedly with code {self.get_exitcode()}")
                    self.rpc_results.put(exception)
                else:
                    logging.warning(f"Process {self.name} lost connection with the parent process")
                    exception = SystemError(f"Process {self.name} lost connection with the parent process")
                    self.rpc_results.put(exception)
                    self.kill()
            else:
                logging.warning(f"Process {self.name} Timeout")
                exception = SystemError(f"Process Timeout")
                self.rpc_results.put(exception)
            for lock in self.locks:
                # noinspection PyBroadException
                try:
                    lock.release()
                except:
                    logging.exception("Release lock failed")
            # Release device locks held by dead process
            DeviceLockManager.on_process_dead(
                self,
                SimpleCommandProcess._device_grant_events,
                SimpleCommandProcess._device_granted_indices)
            self._semaphore_dead_sequence(exception)
            self.close(True)
        except:
            self.close(True)
            raise
        if self.status == PROCESS_STATUS_CODE.IDLE and not self.rpc_queue.empty():
            func_call = self.rpc_queue.get()
            self._parent_send_rpc(*func_call)
            self.status = PROCESS_STATUS_CODE.WAITING

    def send_action(self, target: Callable, args: tuple, kwargs: dict):
        self.rpc_queue.put((PROCESS_RPC.EXECUTE_FUNCTION, (target, args, kwargs)))

    def is_ready(self):
        return self.status == PROCESS_STATUS_CODE.IDLE

    def is_idle(self):
        return self.status == PROCESS_STATUS_CODE.IDLE and self.rpc_queue.empty()

    def is_dead(self):
        return self.status == PROCESS_STATUS_CODE.DEAD

    def is_completed(self):
        return not self.rpc_results.empty()
