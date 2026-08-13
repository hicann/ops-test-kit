#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Optional runtime data exposed to explicitly opted-in TestSpec hooks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True)
class RuntimeKernelProfile:
    """One device kernel collected by an explicitly profiled hook operation."""

    name: str
    device_us: float
    calls: int
    avg_us: float
    max_us: float
    min_us: float

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "device_us": self.device_us,
            "calls": self.calls,
            "avg_us": self.avg_us,
            "max_us": self.max_us,
            "min_us": self.min_us,
        }


@dataclass(frozen=True)
class RuntimeProfile:
    """Profiling summary returned by ``TtkContext.run_profiled``."""

    enabled: bool
    repeat_count: int
    elapsed_us: float
    result_path: Optional[Path]
    kernels: Sequence[RuntimeKernelProfile] = ()

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "repeat_count": self.repeat_count,
            "elapsed_us": self.elapsed_us,
            "result_path": (
                str(self.result_path) if self.result_path is not None else None
            ),
            "kernels": [kernel.as_dict() for kernel in self.kernels],
        }


@dataclass
class TtkContext:
    """Share runtime data across input, pre-NPU, Golden, and compare hooks.

    Hooks receive this object only when they explicitly declare a keyword-capable
    ``context`` parameter typed as ``TtkContext`` or its optional form. ``state``
    is process-local. Operators may use ``manual_case_dir`` for their own
    cross-process files, whose names and contents remain outside TTK's contract.
    """

    api_name: str
    testcase_name: str
    case_type: str
    input_tensors: Any
    input_scalars: Any
    attributes: Mapping[str, Any]
    csv_fields: Mapping[str, Any]
    options: Mapping[str, Any]
    manual_data_mode: Optional[str]
    manual_data_writes_goldens: bool
    manual_data_dirs: Sequence[Path]
    manual_case_dir: Optional[Path] = None
    manual_data_format: Optional[str] = None
    state: MutableMapping[str, Any] = field(default_factory=dict)
    _aclnn_runner: Optional[Callable[..., Any]] = field(
        default=None, init=False, repr=False
    )
    _profile_runner: Optional[Callable[..., RuntimeProfile]] = field(
        default=None, init=False, repr=False
    )

    def run_aclnn(
        self,
        api_name: str,
        *,
        tensors: Mapping[str, Any],
        attributes: Mapping[str, Any],
        output_names: Sequence[str],
        tensor_formats: Optional[Mapping[str, str]] = None,
        storage_shapes: Optional[Mapping[str, Sequence[int]]] = None,
        scalars: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Run an auxiliary ACLNN API during an opted-in pre-NPU hook."""
        if self._aclnn_runner is None:
            raise RuntimeError(
                "run_aclnn is only available while an ACLNN pre-NPU hook is executing"
            )
        self._aclnn_runner(
            api_name,
            tensors=tensors,
            attributes=attributes,
            output_names=output_names,
            tensor_formats=tensor_formats,
            storage_shapes=storage_shapes,
            scalars=scalars,
        )

    def run_profiled(
        self, stage_name: str, operation: Callable[[], Any]
    ) -> RuntimeProfile:
        """Run one hook-owned operation with the current profiling switches."""
        if self._profile_runner is None:
            raise RuntimeError(
                "run_profiled is only available while a pre-NPU hook is executing"
            )
        return self._profile_runner(stage_name, operation)
