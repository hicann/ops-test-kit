#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Load installed packages that register custom ``torch.ops`` namespaces."""

import importlib
import os
import pathlib
import sys
import threading


class TorchOpsPackageRegistrationError(RuntimeError):
    """An installed custom-op package could not be discovered or imported."""


class TorchOpsPackageLoader:
    """Register known custom ``torch.ops`` namespaces before parsing or execution."""

    NAMESPACE_PACKAGES = {
        "cann_ops_transformer": "cann_ops_transformer",
        "cann_ops_nn": "cann_ops_nn",
    }
    CANN_ROOT_ENV_VARS = (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_AICPU_PATH",
    )
    LOCK = threading.RLock()

    @classmethod
    def package_for_api(cls, api_name):
        parts = api_name.split(".")
        if len(parts) < 4 or parts[0:2] != ["torch", "ops"]:
            return None
        return cls.NAMESPACE_PACKAGES.get(parts[2])

    @classmethod
    def site_package_candidates(cls):
        candidates = []
        seen = set()
        for env_name in cls.CANN_ROOT_ENV_VARS:
            root = os.environ.get(env_name)
            if not root:
                continue
            candidate = pathlib.Path(root).expanduser() / "python" / "site-packages"
            candidate = candidate.resolve()
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
        return candidates

    @classmethod
    def environment_summary(cls):
        return ", ".join(
            f"{name}={os.environ.get(name) or '<unset>'}"
            for name in cls.CANN_ROOT_ENV_VARS
        )

    @classmethod
    def ensure_registered(cls, api_name):
        package_name = cls.package_for_api(api_name)
        if package_name is None:
            return

        with cls.LOCK:
            candidates = cls.site_package_candidates()
            inserted = []
            for candidate in reversed(candidates):
                candidate_str = str(candidate)
                if candidate.is_dir() and candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                    inserted.append(candidate_str)
            importlib.invalidate_caches()

            try:
                if package_name not in sys.modules:
                    importlib.import_module(package_name)
            except Exception as error:
                for candidate_str in inserted:
                    if candidate_str in sys.path:
                        sys.path.remove(candidate_str)
                checked = ", ".join(str(path) for path in candidates) or "<none>"
                raise TorchOpsPackageRegistrationError(
                    f"Cannot register {api_name!r}: importing installed package "
                    f"{package_name!r} failed with {type(error).__name__}: {error}. "
                    f"CANN environment: {cls.environment_summary()}. "
                    f"Checked site-packages: {checked}. Source the target CANN "
                    "environment and install the matching transformer package."
                ) from error
