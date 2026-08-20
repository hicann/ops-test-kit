# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.infra.task: TaskType / TaskA / TaskKeeper."""

from unittest.mock import MagicMock

from ttk.core_modules.infra.task import TaskA, TaskKeeper, TaskType


def _make_task(task_type=TaskType.PROFILE, sub_type="dyn"):
    return TaskA(
        testcase=MagicMock(testcase_name="t0"),
        func=lambda: None,
        params=(),
        type=task_type,
        sub_type=sub_type,
    )


def test_task_type_enum_has_compile_and_profile():
    assert TaskType.COMPILE.value != TaskType.PROFILE.value
    assert {TaskType.COMPILE, TaskType.PROFILE} == set(TaskType)


def test_task_a_defaults():
    t = TaskA(testcase=MagicMock(), func=int, params=(1,))
    assert t.type is TaskType.PROFILE
    assert t.sub_type == ""
    assert t.proc_options == {}


def test_keeper_append_then_pop_fifo():
    keeper = TaskKeeper()
    t1, t2 = _make_task(sub_type="a"), _make_task(sub_type="b")
    keeper.append([t1, t2])
    assert keeper.pop() is t1
    assert keeper.pop() is t2


def test_keeper_append_single_task():
    keeper = TaskKeeper()
    t = _make_task()
    keeper.append(t)
    assert keeper.pop() is t


def test_keeper_insert_prepends():
    keeper = TaskKeeper()
    t1, t2 = _make_task(sub_type="old"), _make_task(sub_type="new")
    keeper.append(t1)
    keeper.insert(t2)
    assert keeper.pop() is t2
    assert keeper.pop() is t1


def test_keeper_insert_batch_preserves_order_at_front():
    keeper = TaskKeeper()
    base = _make_task(sub_type="base")
    keeper.append(base)
    a, b = _make_task(sub_type="a"), _make_task(sub_type="b")
    keeper.insert([a, b])
    assert keeper.pop() is a
    assert keeper.pop() is b
    assert keeper.pop() is base


def test_keeper_pop_by_type():
    keeper = TaskKeeper()
    compile_task = _make_task(TaskType.COMPILE, "stc")
    profile_task = _make_task(TaskType.PROFILE, "dyn")
    keeper.append(compile_task)
    keeper.append(profile_task)
    assert keeper.pop(TaskType.PROFILE) is profile_task
    assert keeper.pop(TaskType.COMPILE) is compile_task


def test_keeper_pop_default_priority_compile_first():
    keeper = TaskKeeper()
    profile_task = _make_task(TaskType.PROFILE, "dyn")
    compile_task = _make_task(TaskType.COMPILE, "stc")
    keeper.append(profile_task)
    keeper.append(compile_task)
    assert keeper.pop() is compile_task
    assert keeper.pop() is profile_task


def test_keeper_empty():
    keeper = TaskKeeper()
    assert keeper.empty()
    keeper.append(_make_task())
    assert not keeper.empty()
    keeper.pop()
    assert keeper.empty()


def test_keeper_pop_returns_none_when_empty():
    keeper = TaskKeeper()
    assert keeper.pop() is None
    assert keeper.pop(TaskType.COMPILE) is None
