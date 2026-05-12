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
import csv
import random
import logging
from typing import Any, Dict, List, Optional, Set, TextIO

# Third-Party Packages
from .testcase_base import TestcaseBase
from ...utilities import set_process_name, set_thread_name, get_global_storage


class PLACEHOLDER:
    """
    Simple Placeholder
    """
    pass


class UniversalTestcaseFactory:
    """
    Universal Testcase Factory
    """

    __slots__ = ["raw_data",
                 "header",
                 "real_header_indexes",
                 "testcase_instance",
                 "testcases",
                 "_skip_validate"]

    def __init__(self, file: TextIO, skip_validate=False):
        """
        Store the whole Testcases in the csv file into memory
        """
        # Raw rows
        self.raw_data: List[List[str]] = []
        # Headers
        self.header: List[str] = []
        # header index
        self.real_header_indexes: Dict[str, int] = {}
        # testcase instance
        self.testcase_instance: Optional[TestcaseBase] = None
        # testcase.
        self.testcases: Set[TestcaseBase] = set()
        self._skip_validate = skip_validate

        set_process_name("TestcaseManager")
        set_thread_name("Initialization")
        # read csv file.
        self._read_csv(file)
        self._testcase_hdr_check()
        self._parse_testcase()
        set_process_name()
        set_thread_name()

    def get(self) -> Set[TestcaseBase]:
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
            except:
                raise RuntimeError(f"Failed parsing testcase {testcase_struct.testcase_name}")

    @staticmethod
    def set_case_default_value(testcases: List[TestcaseBase]) -> None:
        def __process(_result: Dict[str, Any], _case: TestcaseBase,
                      result_keys: list = None, apply_default: bool = False):
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
                            else:
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
                    logging.exception(f"Failed to process header {header} of case {_case.testcase_name} with "
                                      f"value {value}")
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
                    except:
                        if original_result not in ("PASS",):
                            return True
                else:
                    raise RuntimeError(f"Unsupported rerun targets: {title}")
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_enabled(testcase_struct: TestcaseBase) -> bool:
        if not testcase_struct.is_enabled:
            logging.debug("Testcase %s skipped bcz it's disabled" % testcase_struct.testcase_name)
            return False
        # Skip testcase if it is disabled in current soc
        current_soc = get_global_storage().short_soc_version
        if testcase_struct.soc_series and current_soc:
            enabled_soc = [s for s in testcase_struct.soc_series if not s.startswith('-')]
            disabled_soc = [s[1:] for s in testcase_struct.soc_series if s.startswith('-')]
            if not enabled_soc and not disabled_soc:  # both enabled_soc and disabled_soc are empty
                enabled = True
            elif not disabled_soc:  # only enabled_soc
                enabled = True if current_soc in enabled_soc else False
            elif not enabled_soc:  # only disabled_soc
                enabled = True if current_soc not in disabled_soc else False
            else:  # enabled_soc and disabled_soc all fill with options
                enabled = True if current_soc in enabled_soc and current_soc not in disabled_soc else False
            if not enabled:
                logging.debug(f"Testcase {testcase_struct.testcase_name} skipped "
                              f"bcz it's disabled in current soc {current_soc}.")
            return enabled
        return True

    @staticmethod
    def _check_testcase_name_selection(testcase_name: str) -> bool:
        if get_global_storage().selected_testcases:
            if testcase_name in [s for s in get_global_storage().selected_testcases]:
                return True
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_indexes_selection(testcase_idx: int) -> bool:
        if get_global_storage().selected_testcase_indexes:
            if testcase_idx in get_global_storage().selected_testcase_indexes:
                return True
        else:
            return True
        return False

    @staticmethod
    def _check_testcase_operator_selection(testcase_op_name: str) -> bool:
        if get_global_storage().selected_operators:
            if testcase_op_name in get_global_storage().selected_operators:
                return True
            else:
                return False
        elif get_global_storage().excluded_operators:
            if testcase_op_name in get_global_storage().excluded_operators:
                return False
            else:
                return True
        else:
            return True

    @staticmethod
    def _check_testcase_priority_selection(priority: int) -> bool:
        if not get_global_storage().priorities:
            return True
        for p in get_global_storage().priorities:
            if p[0] <= priority <= p[1]:
                return True
        return False

    @staticmethod
    def _rename_duplicate_case_name(ori_name: str, op_name: str, conflict_names: set):
        i = 0
        while True:
            new_name = f"{ori_name}_{op_name}_dup{i}"
            if new_name in conflict_names:
                i = i + 1
                continue
            else:
                logging.warning(f"Detected duplicate testcase name: {ori_name}. "
                                f"Rename it to {new_name}")
                return new_name

    def _read_csv(self, file: TextIO):
        csv_reader = csv.reader(file)
        for row in csv_reader:
            row = [column.strip() for column in row]
            if row:
                self.raw_data.append(row)
        # First line should always be the title
        try:
            self.header = self.raw_data[0]
        except IndexError:
            logging.error("Testcase initialization received IndexError, csv file might be empty!")
            raise
        del self.raw_data[0]

    def _testcase_hdr_check(self):
        set_thread_name("HeaderCheckTestcaseName")
        # Testcase name generation
        if "testcase_name" not in self.header:
            logging.warning("Testcase name not found!"
                            " It is important to add a testcase_name in order to identify your testcases")
            self.header.append("testcase_name")
            for idx, row in enumerate(self.raw_data):
                row.append("auto_testcase_name_%d" % (idx + 1))

        if 'api_name' in self.header:
            # Auto-detect from api_name values: aclnnXxx -> aclnn, others -> framework-api
            first_api = None
            api_idx = self.header.index('api_name')
            for row in self.raw_data:
                if api_idx < len(row) and row[api_idx]:
                    first_api = row[api_idx].strip()
                    break

            if first_api and not first_api.lower().startswith('aclnn'):
                from .testcase_framework_api import FrameworkApiTestcaseStructure
                self.testcase_instance = FrameworkApiTestcaseStructure()
            else:
                from .testcase_api import ApiTestcaseStructure
                self.testcase_instance = ApiTestcaseStructure()
        else:
            from .testcase_op import UniversalTestcaseStructure
            self.testcase_instance = UniversalTestcaseStructure()

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
                logging.error("Detected duplicate header: %s" % actual_header)
                raise RuntimeError("Detected duplicate header: %s" % actual_header)
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
                logging.exception(f"Failed to process row index [{idx}]. "
                                  f"Whole row is: {','.join(line)}")
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
                    result = self._get_idx_of_header(equivalent,
                                                     searched_tag)
                    if result is not None:
                        break
        return result

    def _process(self, row: list, row_index: int) -> Dict[str, Any]:
        # Initialize full testcase fields
        result = dict(zip(self.testcase_instance.get_all_legit_headers(),
                          [PLACEHOLDER() for _ in self.testcase_instance.get_all_legit_headers()]))
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
                        else:
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
                    result[current_header_name] = \
                        self.testcase_instance.get_header_func(current_header_name)(default_value)
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
                logging.exception(f"Failed to process header [{current_header_name}] of row [{row_index}] with "
                                  f"value [{value}]")
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
            headers = testcase_struct.get_all_legit_headers()
            unidentified_headers = []
            for header_idx, header in enumerate(headers):
                if hasattr(testcase_struct, header):
                    setattr(testcase_struct, header, testcase[header_idx])
                else:
                    unidentified_headers.append(header)
            if unidentified_headers:
                raise KeyError(f"TestcaseManager header not match. "
                               f"Report Bug to us: {unidentified_headers}")
            # Skip testcase if it is disabled
            if not self._check_testcase_enabled(testcase_struct):
                continue
            # Skip testcase if it is not in global testcase_name selector range
            if not self._check_testcase_name_selection(testcase_struct.testcase_name):
                continue
            # Skip testcase if it is not in global testcase_index selector range
            if not self._check_testcase_indexes_selection(testcase_idx):
                continue
            # Skip testcase if it is not in global testcase_op_name selector range
            if not self._check_testcase_operator_selection(testcase_struct.op_name):
                continue
            # Skip testcase if it is not in specified priority
            if not self._check_testcase_priority_selection(testcase_struct.priority):
                continue
            # Skip testcase if it is not in rerun range
            if not self._check_testcase_rerun(testcase_struct):
                continue
            set_thread_name(testcase_struct.testcase_name)
            if not self._skip_validate:
                testcase_struct.validate()
            if testcase_struct not in self.testcases:
                self.testcases.add(testcase_struct)
                if testcase_struct.testcase_name in testcase_names:
                    testcase_struct.testcase_name = self._rename_duplicate_case_name(testcase_struct.testcase_name,
                                                                                     testcase_struct.op_name,
                                                                                     testcase_names)
                testcase_names.add(testcase_struct.testcase_name)
            else:
                logging.warning("Duplicate testcase: %s" % testcase_struct.testcase_name)
        # For testcase_count selector
        if 0 < get_global_storage().selected_testcase_count < len(self.testcases):
            logging.info("Selecting %d cases from all testcases" % get_global_storage().selected_testcase_count)
            all_indexes = random.sample(tuple(range(len(self.testcases))),
                                        k=get_global_storage().selected_testcase_count)
            self.testcases = set([testcase for idx, testcase in enumerate(self.testcases) if idx in all_indexes])
