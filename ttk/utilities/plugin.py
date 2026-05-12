#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
from typing import Sequence


class CommonPlugin:
    """Single plugin"""

    def __init__(self, name: str) -> None:
        self._dict = {}
        self._name = name

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error("[%s] is not registered for %s functions", key, self._name)
            raise e

    def __setitem__(self, key, value):
        if key in self._dict:
            logging.warning("%s function for [%s] has already been registered! Replace it.", self._name, key)
        self._dict[key] = value

    def __iter__(self):
        return iter(self._dict.items())

    def register(self, keys: Sequence[str]):
        """Register plugin function
           use it like: @CustomizePlugins.xxxx.register(["key1"])
           With Definition of CustomizePlugins like:
                class CustomizePlugins:
                    xxxx = Plugin("xxxx")
        """
        if not isinstance(keys, (list, tuple)):
            raise TypeError("Register function for %s functions must receive a list or tuple, "
                            "rather than %s" % (self._name, str(keys)))

        def __inner_registry(func):
            for k in keys:
                self[k] = func
            return func

        return __inner_registry
