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
Logging module
"""

__all__ = ["default_logging_config", "build_single_log_dir"]


# Standard Packages
import logging.handlers
import os
import sys
import warnings

# Third-party Packages
import numpy

_MODE_LOG_DIR = {
    "op": "kernel",
    "aclnn": "aclnn",
    "geir": "geir",
    "framework-api": "e2e",
}


def build_single_log_dir(test_mode, op_or_api_name, root_path):
    """Construct per-op single-log directory: {root}/log/{mode}/{op_or_api_name}.

    Mirrors msprof path layout so logs are grouped by operator under each mode.
    Creates the directory if missing. Returns the absolute path.
    """
    mode_dir = _MODE_LOG_DIR.get(test_mode, test_mode or "unknown")
    op_or_api_name = op_or_api_name or "unknown"
    log_dir = os.path.join(root_path, "log", mode_dir, op_or_api_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


class MyFilter:
    """
    This Filter is used to make logging module to print only message of specific level
    """

    def __init__(self, level):
        self.__level = level

    def filter(self, record):
        """
        filter for printing only message of specific level
        """
        return record.levelno == self.__level


def attach_handler(_handler, _level=None, _filter=None, _formatter=None):
    """
    :param _handler:
    :param _level:
    :param _formatter:
    :param _filter:
    :return:
    """
    if _level is not None:
        _handler.setLevel(_level)
    if _formatter is not None:
        _handler.setFormatter(_formatter)
    else:
        log_format = (
            "%(asctime)s [%(levelname)s] "
            "[%(process)d %(processName)s %(threadName)s] [%(filename)s:%(lineno)d]: "
            "%(message)s "
        )
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
    if _filter is not None:
        _handler.addFilter(_filter)
    logging.getLogger().addHandler(_handler)
    # noinspection PyUnresolvedReferences
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.addHandler(_handler)


def add_level(name: str, visible_name: str, level: int):
    """Add a new logging level"""

    def _logging_method(self, message, *args, **kwargs):
        if self.isEnabledFor(level):
            self._log(level, message, args, **kwargs)

    def _logging_method_root(message, *args, **kwargs):
        logging.log(level, message, *args, **kwargs)

    logging.addLevelName(level, visible_name)
    setattr(logging, name, _logging_method_root)
    setattr(logging.getLoggerClass(), name, _logging_method)


def default_logging_config(file_handler: bool = False, testcase_name: str = None, log_dir: str = None):
    add_level("debugc", "COMPARE", logging.DEBUG - 5)
    for handler in logging.getLogger().handlers:
        logging.getLogger().removeHandler(handler)
    # noinspection PyUnresolvedReferences
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    # Ignore numpy RuntimeWarnings
    numpy.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    # Attach logging handler for internal logging levels
    if file_handler:
        suffix = f"-{testcase_name}" if testcase_name else ""
        base = log_dir if log_dir else "."
        # single-log mode (testcase_name set): overwrite; default: append
        file_mode = "w" if testcase_name else "a"
        attach_handler(logging.FileHandler(os.path.join(base, f"ttk-debug{suffix}.log"), mode=file_mode), logging.DEBUG)
        attach_handler(
            logging.FileHandler(os.path.join(base, f"ttk-info{suffix}.log"), mode=file_mode),
            logging.INFO,
            MyFilter(logging.INFO),
        )
        attach_handler(
            logging.FileHandler(os.path.join(base, f"ttk-warning{suffix}.log"), mode=file_mode),
            logging.WARNING,
            MyFilter(logging.WARNING),
        )
        attach_handler(
            logging.FileHandler(os.path.join(base, f"ttk-critical{suffix}.log"), mode=file_mode),
            logging.CRITICAL,
            MyFilter(logging.CRITICAL),
        )
        attach_handler(
            logging.FileHandler(os.path.join(base, f"ttk-error{suffix}.log"), mode=file_mode),
            logging.ERROR,
            MyFilter(logging.ERROR),
        )
        attach_handler(
            logging.FileHandler(os.path.join(base, f"ttk-compare{suffix}.log"), mode=file_mode),
            logging.DEBUG - 5,
            MyFilter(logging.DEBUG - 5),
        )
        attach_handler(logging.StreamHandler(sys.stdout), logging.INFO)
    else:
        attach_handler(logging.StreamHandler(sys.stdout), logging.NOTSET)
    # Enable all logging
    logging.getLogger().setLevel(logging.NOTSET)
    if testcase_name:
        logging.debug(f"Trying to log single testcase log for testcase {testcase_name}")
