#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Framework detection from api_name prefix."""

from typing import Optional


def detect_framework(api_name: str) -> str:
    """Detect framework ('torch' or 'tf') from api_name prefix.

    Routing rule:
        "tf." / "tensorflow." prefix -> "tf"
        everything else (torch., torch_npu., torch.ops., torch.Tensor.) -> "torch"
    """
    if not api_name:
        return "torch"
    if api_name.startswith(("tf.", "tensorflow.")):
        return "tf"
    return "torch"


def is_inplace_tensor_method(api_name: str, framework: Optional[str] = None) -> bool:
    """Check if api_name is an inplace tensor method.

    torch: torch.Tensor.xxx_ (trailing underscore)
    tf: never (TF has no inplace tensor methods)
    """
    if framework is None:
        framework = detect_framework(api_name)
    if framework == "tf":
        return False
    if not api_name:
        return False
    parts = api_name.split(".")
    return len(parts) >= 3 and parts[0] == "torch" and parts[1] == "Tensor" and parts[-1].endswith("_")
