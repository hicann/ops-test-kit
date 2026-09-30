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
Universal testcase manager for csv support
"""

__all__ = ["UniversalTestcaseFactory"]


# Standard Packages
import collections
import logging
import random
from typing import Any, Dict, List, Optional, Set, TextIO

from ...utilities import get_global_storage, set_process_name, set_thread_name
from ...utilities.table_reader import read_csv_rows, read_table

# Third-Party Packages
from .testcase_base import TestcaseBase


class PLACEHOLDER:
    """
    Simple Placeholder
    """


# 载入阶段被丢弃的用例统计（喂入 − 载入）。解析与最终汇总都在主进程, 不跨进程共享。
# 为什么要有它: 用例被跳过只按 reason 各自 debug 一行, 默认日志级看不见, 而 Summary 的
# Total 是**载入条数**不是**喂入条数** —— 喂 200 跑 181 也会报 PassRate 98.34%,
# 分母悄悄变小且毫无提示（2026-09-26 实测 19 例 is_enabled=FALSE 被静默丢弃）。
LOAD_STATS: "collections.Counter" = collections.Counter()

_SKIP_REASON_DESC = {
    "disabled": "is_enabled=FALSE",
    "soc": "当前 soc 不在 soc_series 内",
    "name_filter": "不在 --testcase-name 选择范围",
    "index_filter": "不在 --testcase-index 选择范围",
    "op_filter": "不在 --op-name 选择范围",
    "priority_filter": "不在 --priority 选择范围",
    "rerun_filter": "不在 rerun 范围",
}


class UniversalTestcaseFactory:
    """
    Universal Testcase Factory
    """

    __slots__ = [
        "raw_data",
        "header",
        "real_header_indexes",
        "testcase_instance",
        "testcases",
        "skip_stats",
        "skip_names",
    ]

    def __init__(self, file: TextIO):
        """
        Store the whole Testcases in the csv file into memory
        """
        self._init_common()
        header, rows = read_csv_rows(file)
        self._init_from_rows(header, rows)

    @classmethod
    def from_path(cls, path: str, sheet: Optional[str] = None):
        """
        Load testcases from a CSV or XLSX file by path. XLSX uses openpyxl
        and honors ``sheet`` (default: first worksheet); CSV ignores it.
        """
        self = cls.__new__(cls)
        self._init_common()
        header, rows = read_table(path, sheet)
        self._init_from_rows(header, rows)
        return self

    def _init_common(self):
        # Raw rows
        self.raw_data: List[List[str]] = []
        # Headers
        self.header: List[str] = []
        # header index
        self.real_header_indexes: Dict[str, int] = {}
        # testcase instance
        self.testcase_instance: Optional[TestcaseBase] = None
        # testcase.
        self.testcases: List[TestcaseBase] = []
        # 各跳过原因的计数 + 被跳过的用例名（用于载入后一次性汇报, 而不是每条 debug 一行）
        self.skip_stats: collections.Counter = collections.Counter()
        self.skip_names: Dict[str, List[str]] = {}

        set_process_name("TestcaseManager")
        set_thread_name("Initialization")

    def _init_from_rows(self, header: List[str], rows: List[List[str]]):
        self.header = list(header)
        self.raw_data = [list(row) for row in rows]
        self._testcase_hdr_check()
        self._parse_testcase()
        set_process_name()
        set_thread_name()

    def get(self) -> List[TestcaseBase]:
        """
        Get all testcases
        :return:
        """
        return self.testcases

    @staticmethod
    def validate_testcases(parsed_testcases: List[TestcaseBase]) -> None:
        if not parsed_testcases:
            return
        for testcase_struct in parsed_testcases:
            set_thread_name(testcase_struct.testcase_name)
            try:
                testcase_struct.validate()
            except Exception as err:
                raise RuntimeError(f"Failed parsing testcase {testcase_struct.testcase_name}") from err

    @staticmethod
    def set_case_default_value(testcases: List[TestcaseBase]) -> None:
        def __process(
            _result: Dict[str, Any], _case: TestcaseBase, result_keys: list = None, apply_default: bool = False
        ):
            changed_ = False
            placeholder_queue = []
            keys = result_keys if result_keys else _result
            for header in keys:
                column_value = _result[header]
                # Parsed line
                if not isinstance(column_value, PLACEHOLDER):
                    continue
                value = getattr(_case, header)
                # Header not exist or value is empty, check for equivalent and default value
                if not value:
                    if _case.has_equivalent_header(header) and not apply_default:
                        # Check for equivalent
                        resolved = False
                        value = None
                        for equivalent_header in _case.get_equivalent_headers(header):
                            equivalent_header_value = _result[equivalent_header]
                            if isinstance(equivalent_header_value, PLACEHOLDER):
                                # Equivalent not exist either, check for next equivalent
                                continue
                            value = equivalent_header_value
                            resolved = True
                            break
                        if not resolved:
                            placeholder_queue.append(header)
                            continue
                    elif _case.has_default_value(header):
                        # Check for default value
                        changed_ = True
                        default_value = str(_case.get_default_value(header))
                        _result[header] = _case.get_header_func(header)(default_value)
                        continue
                    else:
                        placeholder_queue.append(header)
                        continue
                try:
                    changed_ = True
                    if isinstance(value, str):
                        _result[header] = _case.get_header_func(header)(value)
                    else:
                        _result[header] = value
                except:
                    logging.exception(
                        f"Failed to process header {header} of case {_case.testcase_name} with value {value}"
                    )
                    raise
            return changed_, placeholder_queue

        if not testcases:
            return
        headers = testcases[0].get_all_legit_headers()
        for case in testcases:
            # Initialize full testcase fields
            result = dict(zip(headers, [PLACEHOLDER() for _ in headers]))
            changed, queue = __process(result, case)
            while changed:
                changed, queue = __process(result, case, queue)
            if queue:
                changed, queue = __process(result, case, queue, True)
            while changed:
                changed, queue = __process(result, case, queue)
            if queue:
                raise RuntimeError(f"Could not determine value of missing fields: {queue}")
            for hdr in headers:
                setattr(case, hdr, result[hdr])

    @staticmethod
    def _check_testcase_rerun(testcase_struct: TestcaseBase) -> bool:
        re_run_titles = testcase_struct.supported_rerun_title()
        if get_global_storage().rerun_targets:
            for title in get_global_storage().rerun_targets:
                if title in re_run_titles:
                    original_result = testcase_struct.original_dict[title]
                    try:
                        float(original_result)
                    except Exception:
                        if original_result not in ("PASS",):
                            return True
                else:
                    raise RuntimeError(f"Unsupported rerun targets: {title}")
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_enabled(testcase_struct: TestcaseBase) -> Optional[str]:
        """返回 None 表示保留；返回字符串表示跳过原因（键见 _SKIP_REASON_DESC）。"""
        if not testcase_struct.is_enabled:
            logging.debug(f"Testcase {testcase_struct.testcase_name} skipped bcz it's disabled")
            return "disabled"
        # Skip testcase if it is disabled in current soc
        current_soc = get_global_storage().short_soc_version
        if testcase_struct.soc_series and current_soc:
            enabled_soc = [s for s in testcase_struct.soc_series if not s.startswith("-")]
            disabled_soc = [s[1:] for s in testcase_struct.soc_series if s.startswith("-")]
            if not enabled_soc and not disabled_soc:  # both enabled_soc and disabled_soc are empty
                enabled = True
            elif not disabled_soc:  # only enabled_soc
                enabled = current_soc in enabled_soc
            elif not enabled_soc:  # only disabled_soc
                enabled = current_soc not in disabled_soc
            else:  # enabled_soc and disabled_soc all fill with options
                enabled = current_soc in enabled_soc and current_soc not in disabled_soc
            if not enabled:
                logging.debug(
                    f"Testcase {testcase_struct.testcase_name} skipped bcz it's disabled in current soc {current_soc}."
                )
                return "soc"
        return None

    @staticmethod
    def _check_testcase_name_selection(testcase_name: str) -> bool:
        if get_global_storage().selected_testcases:
            return testcase_name in list(get_global_storage().selected_testcases)
        return True

    @staticmethod
    def _check_testcase_indexes_selection(testcase_idx: int) -> bool:
        if get_global_storage().selected_testcase_indexes:
            return testcase_idx in get_global_storage().selected_testcase_indexes
        return True

    @staticmethod
    def _check_testcase_operator_selection(testcase_op_name: str) -> bool:
        if get_global_storage().selected_operators:
            return testcase_op_name in get_global_storage().selected_operators
        if get_global_storage().excluded_operators:
            return testcase_op_name not in get_global_storage().excluded_operators
        return True

    @staticmethod
    def _check_testcase_priority_selection(priority: int) -> bool:
        if not get_global_storage().priorities:
            return True
        return any(p[0] <= priority <= p[1] for p in get_global_storage().priorities)

    @staticmethod
    def _rename_duplicate_case_name(ori_name: str, op_name: str, conflict_names: set):
        i = 0
        while True:
            new_name = f"{ori_name}_{op_name}_dup{i}"
            if new_name in conflict_names:
                i = i + 1
                continue
            logging.warning(f"Detected duplicate testcase name: {ori_name}. Rename it to {new_name}")
            return new_name

    def _testcase_hdr_check(self):
        set_thread_name("HeaderCheckTestcaseName")
        # Testcase name generation
        if "testcase_name" not in self.header:
            logging.warning(
                "Testcase name not found! It is important to add a testcase_name in order to identify your testcases"
            )
            self.header.append("testcase_name")
            for idx, row in enumerate(self.raw_data):
                row.append(f"auto_testcase_name_{idx + 1}")

        test_mode = get_global_storage().test_mode
        if test_mode == "aclnn":
            from .testcase_aclnn import TestcaseAclnn

            self.testcase_instance = TestcaseAclnn()
        elif test_mode == "framework-api":
            from .testcase_e2e import TestcaseE2e

            self.testcase_instance = TestcaseE2e()
        elif test_mode == "geir":
            from ttk.core_modules.geir.testcase import GeirTestcase

            self.testcase_instance = GeirTestcase()
        else:
            from .testcase_op import TestcaseOp

            self.testcase_instance = TestcaseOp()

        set_thread_name("HeaderCheckUnidentifiedHeaders")
        ignored_headers = []
        for actual_header in self.header:
            if not self.testcase_instance.is_legit_header(actual_header):
                ignored_headers.append(actual_header)
        if ignored_headers:
            logging.warning(f"Detected unidentified headers and will be ignored: {ignored_headers}")

        set_thread_name("HeaderCheckDuplicateHeaders")
        # Check duplicates
        header_check_set = set()
        for actual_header in self.header:
            if actual_header in header_check_set and actual_header not in ignored_headers:
                logging.error(f"Detected duplicate header: {actual_header}")
                raise RuntimeError(f"Detected duplicate header: {actual_header}")
            header_check_set.add(actual_header)

    def _parse_testcase(self):
        set_thread_name("HeaderParseMapping")
        # Get idx of required testcase header in real header
        for header_name in self.testcase_instance.get_all_legit_headers():
            header_idx = self._get_idx_of_header(header_name)
            self.real_header_indexes[header_name] = header_idx

        set_thread_name("TestcasePreParsing")
        raw_testcases = []

        # Process each line
        for idx, line in enumerate(self.raw_data):
            try:
                sub_result = self._process(line, idx)
            except:
                logging.exception(f"Failed to process row index [{idx}]. Whole row is: {','.join(line)}")
                raise
            raw_testcases.append(tuple(sub_result.values()))
        logging.debug(f"Processed {len(raw_testcases)} raw testcases")
        set_thread_name("FinalTestcaseStructureFormation")
        self._parse(raw_testcases)

    def _get_idx_of_header(self, header_name, searched_tag=None) -> Optional[int]:
        """
        Get header position in actual header sequence
        :param header_name: string name of the header
        :param searched_tag: DO NOT USE
        :return: index of the header or its equivalent
        """
        result = None
        if header_name in self.header:
            result = self.header.index(header_name)
        else:
            equivalents = self.testcase_instance.get_equivalent_headers(header_name)
            # Header not found, search for equivalent
            if not equivalents:
                return result
            # Initialize recursion variable
            if searched_tag is None:
                searched_tag = []
            for equivalent in equivalents:
                if equivalent not in searched_tag:
                    searched_tag.append(header_name)
                    result = self._get_idx_of_header(equivalent, searched_tag)
                    if result is not None:
                        break
        return result

    def _process(self, row: list, row_index: int) -> Dict[str, Any]:
        # Initialize full testcase fields
        result = dict(
            zip(
                self.testcase_instance.get_all_legit_headers(),
                [PLACEHOLDER() for _ in self.testcase_instance.get_all_legit_headers()],
            )
        )
        changed, queue = self._process_over_result(result, row, row_index)
        while changed:
            changed, queue = self._process_over_result(result, row, row_index, queue)
        if queue:
            changed, queue = self._process_over_result(result, row, row_index, queue, True)
        while changed:
            changed, queue = self._process_over_result(result, row, row_index, queue)
        if queue:
            raise RuntimeError(f"Could not determine value of missing fields of row {row_index}: {queue}")
        return result

    def _process_over_result(self, result, row, row_index, result_keys=None, apply_default=False):
        changed = False
        placeholder_queue = []
        keys = result_keys if result_keys else result
        for current_header_name in keys:
            column_value = result[current_header_name]
            # Parsed line
            if not isinstance(column_value, PLACEHOLDER):
                continue
            header_raw_idx = self.real_header_indexes[current_header_name]
            # Header not exist or value is empty, check for equivalent and default value
            if header_raw_idx is None or header_raw_idx >= len(row) or not row[header_raw_idx]:
                if self.testcase_instance.has_equivalent_header(current_header_name) and not apply_default:
                    # Check for equivalent
                    resolved = False
                    value = None
                    for equivalent_header in self.testcase_instance.get_equivalent_headers(current_header_name):
                        equivalent_header_value = result[equivalent_header]
                        if isinstance(equivalent_header_value, PLACEHOLDER):
                            # Equivalent not exist either, check for next equivalent
                            continue
                        value = equivalent_header_value
                        resolved = True
                        break
                    if not resolved:
                        placeholder_queue.append(current_header_name)
                        continue
                elif self.testcase_instance.has_default_value(current_header_name):
                    # Check for default value
                    changed = True
                    default_value = str(self.testcase_instance.get_default_value(current_header_name))
                    result[current_header_name] = self.testcase_instance.get_header_func(current_header_name)(
                        default_value
                    )
                    continue
                else:
                    placeholder_queue.append(current_header_name)
                    continue
            else:
                value = row[header_raw_idx]
            try:
                changed = True
                if isinstance(value, str):
                    result[current_header_name] = self.testcase_instance.get_header_func(current_header_name)(value)
                else:
                    result[current_header_name] = value
            except:
                logging.exception(
                    f"Failed to process header [{current_header_name}] of row [{row_index}] with value [{value}]"
                )
                raise
        return changed, placeholder_queue

    # noinspection PyBroadException
    def _parse(self, testcases: list):
        testcase_names: Set[str] = set()
        for testcase_idx, testcase in enumerate(testcases):
            # Construct Testcase Structure
            testcase_struct = type(self.testcase_instance)()
            testcase_struct.original_line = self.raw_data[testcase_idx]
            testcase_struct.original_dict = dict(zip(self.header, self.raw_data[testcase_idx]))
            testcase_struct._csv_row_index = testcase_idx
            headers = testcase_struct.get_all_legit_headers()
            unidentified_headers = []
            for header_idx, header in enumerate(headers):
                if hasattr(testcase_struct, header):
                    setattr(testcase_struct, header, testcase[header_idx])
                else:
                    unidentified_headers.append(header)
            if unidentified_headers:
                raise KeyError(f"TestcaseManager header not match. Report Bug to us: {unidentified_headers}")
            # 逐条记下"为什么没跑"，载入后一次性汇报。判定顺序与取值保持原样：
            # 各 lambda 惰性求值，第一个判否即短路，不会多跑后面的过滤器。
            reason = self._check_testcase_enabled(testcase_struct)
            if reason is None:
                # lambda 用默认参数**绑定**当轮的 struct/idx: 闭包按引用捕获循环变量的话,
                # 惰性求值时拿到的是最后一轮的值(ruff B023)。
                for key, keep in (
                    ("name_filter", lambda ts=testcase_struct: self._check_testcase_name_selection(ts.testcase_name)),
                    ("index_filter", lambda ti=testcase_idx: self._check_testcase_indexes_selection(ti)),
                    ("op_filter", lambda ts=testcase_struct: self._check_testcase_operator_selection(ts.op_name)),
                    (
                        "priority_filter",
                        lambda ts=testcase_struct: self._check_testcase_priority_selection(ts.priority),
                    ),
                    ("rerun_filter", lambda ts=testcase_struct: self._check_testcase_rerun(ts)),
                ):
                    if not keep():
                        reason = key
                        break
            if reason is not None:
                self.skip_stats[reason] += 1
                self.skip_names.setdefault(reason, []).append(testcase_struct.testcase_name)
                continue
            set_thread_name(testcase_struct.testcase_name)
            testcase_struct.validate()
            if testcase_struct not in self.testcases:
                self.testcases.append(testcase_struct)
                if testcase_struct.testcase_name in testcase_names:
                    testcase_struct.testcase_name = self._rename_duplicate_case_name(
                        testcase_struct.testcase_name, testcase_struct.op_name, testcase_names
                    )
                testcase_names.add(testcase_struct.testcase_name)
            else:
                logging.warning(f"Duplicate testcase: {testcase_struct.testcase_name}")
        # For testcase_count selector
        if 0 < get_global_storage().selected_testcase_count < len(self.testcases):
            logging.info(f"Selecting {get_global_storage().selected_testcase_count} cases from all testcases")
            all_indexes = random.sample(
                tuple(range(len(self.testcases))), k=get_global_storage().selected_testcase_count
            )
            sampled = len(self.testcases) - get_global_storage().selected_testcase_count
            self.testcases = [testcase for idx, testcase in enumerate(self.testcases) if idx in all_indexes]
            self.skip_stats["count_selector"] += sampled
        self._report_load_stats()

    def _report_load_stats(self):
        """把"喂入 − 载入"的差额在 INFO 级讲清楚，并交给最终 Summary 复述一遍。

        原来每种跳过只在 DEBUG 打一行：默认日志级下, 喂 200 跑 181 与喂 181 跑 181
        在输出上**完全一样**, 覆盖被砍掉毫无提示。分母不可信是最难发现的一类问题
        （同类坑还有 CSV 逗号未转义丢例、跑批 OOM 静默中断）。
        """
        LOAD_STATS.clear()
        LOAD_STATS["fed"] = len(self.raw_data)
        LOAD_STATS["loaded"] = len(self.testcases)
        dropped = len(self.raw_data) - len(self.testcases)
        if dropped <= 0:
            return
        LOAD_STATS["dropped"] = dropped
        for reason, cnt in self.skip_stats.items():
            LOAD_STATS[reason] = cnt
        detail = "; ".join(f"{_SKIP_REASON_DESC.get(r, r)}: {c}" for r, c in sorted(self.skip_stats.items()) if c)
        logging.info(
            f"Testcase load: fed {len(self.raw_data)} → loaded {len(self.testcases)}, dropped {dropped} ({detail})"
        )
        for reason, names in sorted(self.skip_names.items()):
            head = ", ".join(names[:5]) + (f" ... (+{len(names) - 5})" if len(names) > 5 else "")
            logging.info(f"  dropped[{_SKIP_REASON_DESC.get(reason, reason)}]: {head}")
