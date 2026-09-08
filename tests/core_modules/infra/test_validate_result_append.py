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
Regression tests for --validate --append (issue #120): validate output must
append to an existing CSV with matching header, not silently overwrite it.
"""

import csv
from types import SimpleNamespace

HEADER = ["testcase_name", "api_name", "status", "fail_reason"]
SENTINEL = ["SENTINEL", "old.api", "VALID", "keep-me"]


def _make_instance(tmp_path, append, existing_rows=None):
    from ttk.core_modules.npu.instance_refactor import NpuInstance

    ins = NpuInstance()
    ins.switches.append_mode = append
    ins.switches.output_file_name = str(tmp_path / "validate_result.csv")
    ins.flatten_testcases = [
        SimpleNamespace(testcase_name="case_a", api_name="api.add", is_valid=True, fail_reason=""),
        SimpleNamespace(testcase_name="case_b", api_name="api.sub", is_valid=False, fail_reason="bad shape"),
    ]
    if existing_rows is not None:
        with open(ins.switches.output_file_name, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(existing_rows)
    return ins


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


class TestValidateAppend:
    def test_append_matching_header_keeps_sentinel(self, tmp_path):
        ins = _make_instance(tmp_path, append=True, existing_rows=[HEADER, SENTINEL])
        ins._write_validate_result([], 1, 1)
        rows = _read_rows(ins.switches.output_file_name)
        assert rows[0] == HEADER  # header written once
        assert SENTINEL in rows  # history preserved
        assert rows.index(SENTINEL) < len(rows) - 2  # new rows appended after sentinel
        assert rows[-2] == ["case_a", "api.add", "VALID", ""]
        assert rows[-1] == ["case_b", "api.sub", "INVALID", "bad shape"]

    def test_append_mismatched_header_warns_and_overwrites(self, tmp_path, caplog):
        ins = _make_instance(tmp_path, append=True, existing_rows=[["wrong", "header"], SENTINEL])
        with caplog.at_level("WARNING"):
            ins._write_validate_result([], 1, 1)
        rows = _read_rows(ins.switches.output_file_name)
        assert rows[0] == HEADER
        assert SENTINEL not in rows
        assert any("header does not match" in r.message for r in caplog.records)

    def test_append_empty_file_writes_header(self, tmp_path):
        ins = _make_instance(tmp_path, append=True, existing_rows=[])
        ins._write_validate_result([], 1, 1)
        rows = _read_rows(ins.switches.output_file_name)
        assert rows[0] == HEADER
        assert len(rows) == 3

    def test_no_append_overwrites(self, tmp_path):
        ins = _make_instance(tmp_path, append=False, existing_rows=[HEADER, SENTINEL])
        ins._write_validate_result([], 1, 1)
        rows = _read_rows(ins.switches.output_file_name)
        assert SENTINEL not in rows
        assert len(rows) == 3
