#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#!/usr/bin/env python3
"""
Tbe Interface
"""

__all__ = ["Opc"]


# Standard Packages
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Optional

# Third-Party Packages
from ...utilities import Singleton, get_global_storage
from ...utilities.platform import get_npu_hw_info


class NullApiConfig:
    def __getattr__(self, name):
        class DummyContext:
            def __enter__(self, *args):
                return self

            def __exit__(self, *args):
                return False

        return DummyContext


class IOpc(metaclass=ABCMeta):
    @abstractmethod
    def is_initialized(self) -> bool: ...

    @abstractmethod
    def set_compile_soc_info(self, soc_version, core_type="AiCore"): ...

    @abstractmethod
    def set_platform_info_res(self, device_id, res: dict): ...

    @abstractmethod
    def get_soc_spec(self, key: str): ...

    @abstractmethod
    def get_context(self): ...

    @abstractmethod
    def get_compile_info(self): ...

    @abstractmethod
    def get_param_generalization(self, op_type: str): ...

    @abstractmethod
    def do_op_tiling(self, *args, **kwargs): ...

    @abstractmethod
    def build_config(self, **kwargs): ...

    @abstractmethod
    def get_tiling_op_type(self): ...

    @property
    @abstractmethod
    def op_context(self): ...

    @property
    @abstractmethod
    def op_info(self): ...

    @property
    def api_config(self):
        return NullApiConfig()

    def get_computes(self):
        return None


class TbeOpc(IOpc):
    """
    Class Interface for via `tbe` module
    """

    def __init__(self):
        self._initialized = False
        try:
            self._tbe = __import__("tbe")
            logging.debug(f"Using tbe module from {self._tbe.__file__}")
        except ModuleNotFoundError as e:
            logging.info(f"Import `tbe` failed. Maybe it has been removed: {e}")
        else:
            self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def set_platform_info_res(self, device_id, res: dict):
        self._tbe.common.platform.platform_info.set_platform_info_res(0, res)

    def get_soc_spec(self, key: str):
        return self._tbe.common.platform.platform_info.get_soc_spec(key)

    def get_context(self):
        return self._tbe.dsl.base.operation.get_context()

    def get_computes(self):
        return self.get_context().get_computes()

    def get_tiling_op_type(self):
        return self.get_context().get_op_type()

    def get_compile_info(self):
        return self._tbe.dsl.base.operation.get_compile_info()

    def set_compile_soc_info(self, soc_version, core_type="AiCore"):
        self._tbe.common.platform.platform_info.set_current_compile_soc_info(soc_version, core_type)

    def get_param_generalization(self, op_type: str):
        return self._tbe.common.register.get_param_generalization(op_type)

    def do_op_tiling(self, *args, **kwargs):
        return self._tbe.common.utils.op_tiling.do_op_tiling(*args, **kwargs)

    def build_config(self, **kwargs):
        return self._tbe.common.buildcfg.build_config(**kwargs)

    @property
    def op_context(self):
        return self._tbe.common.context.op_context

    @property
    def op_info(self):
        return self._tbe.common.context.op_info

    @property
    def api_config(self):
        return self._tbe.tvm.api_config


class AscOpc(IOpc):
    """
    Class Interface for via `asc_op_compile_base` module
    """

    def __init__(self):
        self._initialized = False
        try:
            self._asc = __import__("asc_op_compile_base")
            logging.debug(f"Using asc_op_compile_base module from {self._asc.__file__}")
        except ModuleNotFoundError as e:
            logging.info(f"Import `asc_op_compile_base` failed. Maybe it is not activated: {e}")
        else:
            self._initialized = True
            try:
                self._asc_common = __import__("asc_op_compile_base.common", fromlist=["register"])
            except BaseException as e:
                logging.critical(f"Module `AscOpc` init failed: {e}")
                raise e

    def is_initialized(self) -> bool:
        return self._initialized

    def set_compile_soc_info(self, soc_version, core_type="AiCore"):
        self._asc.common.platform.platform_info.set_current_compile_soc_info(soc_version, core_type)

    def set_platform_info_res(self, device_id, res: dict):
        self._asc.common.platform.platform_info.set_platform_info_res(device_id, res)

    def get_soc_spec(self, key: str):
        return self._asc.common.platform.platform_info.get_soc_spec(key)

    def get_context(self):
        return self._asc.common.context.op_context.get_context()

    def get_compile_info(self):
        return {}

    def get_param_generalization(self, op_type: str):
        return self._asc_common.register.get_param_generalization(op_type)

    def do_op_tiling(self, *args, **kwargs):
        return self._asc.asc_op_compiler.op_tiling.do_op_tiling(*args, **kwargs)

    def build_config(self, **kwargs):
        return self._asc.common.buildcfg.build_config(**kwargs)

    def get_tiling_op_type(self):
        return self.get_context().get_addition("op_name")

    @property
    def op_context(self):
        return self._asc.common.context.op_context

    @property
    def op_info(self):
        return self._asc.common.context.op_info


class Opc(metaclass=Singleton):
    OpcImplement: dict = {"tbe": TbeOpc, "asc": AscOpc}

    def __init__(self):
        self._core_type = None
        self._opc = None

        for k, v in self.OpcImplement.items():
            setattr(self, f"_{k}_opc", v())

        try:
            self.core_type = None
        except BaseException as e:
            logging.critical(f"Opc init failed: {e}")
            raise e

    def __getattribute__(self, item):
        if item in (
            "get_compile_info",
            "get_tiling_op_type",
            "get_param_generalization",
            "get_soc_spec",
            "do_op_tiling",
            "build_config",
            "op_context",
            "op_info",
            "api_config",
        ):
            return getattr(self._opc, item)
        else:
            return super().__getattribute__(item)

    @property
    def core_type(self) -> str:
        return self._core_type

    @core_type.setter
    def core_type(self, val: Optional[str]):
        self._core_type = val or "AiCore"
        self._core_type = self._core_type or "AiCore"
        dev_plat = get_global_storage().dev_plat
        logging.debug(f"Setting soc version to {dev_plat} for {self._core_type}")
        self._all_opc_invoke("set_compile_soc_info", dev_plat, self._core_type)

    def switch_opc(self, opc_type: str):
        if opc_type not in self.OpcImplement.keys():
            raise ValueError(f"Invalid opc type: {opc_type}")
        self._opc = getattr(self, f"_{opc_type}_opc")

    def compile_ub_clear(self):
        switches = get_global_storage()
        opc = self._tbe_opc
        if switches.force_clear_ub is not None:
            obj_file = os.path.join(switches.kernel_meta, "clear_ub.o")
            if os.path.exists(obj_file):
                os.remove(obj_file)
            full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
            core_type = "VectorCore" if get_npu_hw_info(full_soc_version).get("cv_split") else "AiCore"
            clean_val = switches.force_clear_ub
            from .helper_kernels import clear_ub

            with opc.op_context.OpContext("pre-static") as cxt:
                tensor = {"shape": (1,), "range": ((1, None),), "dtype": clean_val.dtype.name, "format": "ND"}
                attrs = {"full_soc_version": full_soc_version, "core_type": core_type, "kernel_name": "clear_ub"}
                op_info = opc.op_info.OpInfo("ClearUB", "ClearUB")
                cxt.add_op_info(op_info)
                cxt.add_addition("op_name", "ClearUB")
                clear_ub(tensor, **attrs)
            """ Uncomment this to test clear result.
            from .helper_kernels import test_clear_ub
            with opc.op_context.OpContext("pre-static") as cxt:
                tensor = {"shape": (1,), "range": ((1, None),), "dtype": clean_val.dtype.name, "format": "ND"}
                attrs = {"full_soc_version": full_soc_version, "core_type": core_type,
                         "clean_val": clean_val, "kernel_name": "test_clear_ub"}
                op_info = opc.op_info.OpInfo("TestClearUB", "TestClearUB")
                cxt.add_op_info(op_info)
                cxt.add_addition("op_name", "TestClearUB")
                test_clear_ub(tensor, **attrs)
            """
            if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "clear_ub.json")):
                raise RuntimeError(
                    f"Compile clear_ub failed. kernel or json file does not exist in {switches.kernel_meta}"
                )

    def compile_l1_clear(self):
        switches = get_global_storage()
        opc = self._tbe_opc
        if switches.force_clear_l1 is not None:
            obj_file = os.path.join(switches.kernel_meta, "clear_l1.o")
            if os.path.exists(obj_file):
                os.remove(obj_file)
            full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
            clean_val = switches.force_clear_l1
            from .helper_kernels import clear_l1

            with opc.op_context.OpContext("pre-static") as cxt:
                tensor = {"shape": (1,), "range": ((1, None),), "dtype": clean_val.dtype.name, "format": "ND"}
                attrs = {"full_soc_version": full_soc_version, "kernel_name": "clear_l1"}
                op_info = opc.op_info.OpInfo("ClearL1", "ClearL1")
                cxt.add_op_info(op_info)
                cxt.add_addition("op_name", "ClearL1")
                clear_l1(tensor, **attrs)
            if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "clear_l1.json")):
                raise RuntimeError(
                    f"Compile clear_l1 failed. kernel or json file does not exist in {switches.kernel_meta}"
                )

    def compile_warmup_kernel(self):
        switches = get_global_storage()
        opc = self._tbe_opc
        if switches.warmup:
            obj_file = os.path.join(switches.kernel_meta, "warmup.o")
            if os.path.exists(obj_file):
                os.remove(obj_file)
            full_soc_version = opc.get_soc_spec("FULL_SOC_VERSION")
            from .helper_kernels import warmup

            with opc.op_context.OpContext("pre-static") as cxt:
                build_cfg = {"op_debug_config": "dump_cce"}
                with opc.build_config(**build_cfg):
                    attrs = {"full_soc_version": full_soc_version, "kernel_name": "warmup"}
                    op_info = opc.op_info.OpInfo("Warmup", "Warmup")
                    cxt.add_op_info(op_info)
                    cxt.add_addition("op_name", "Warmup")
                    warmup(**attrs)
            if not os.path.exists(obj_file) or not os.path.exists(os.path.join(switches.kernel_meta, "warmup.json")):
                raise RuntimeError(
                    f"Compile warmup kernel failed. kernel or json file does not exist in {switches.kernel_meta}"
                )

    def _get_any_opc(self) -> Optional[IOpc]:
        for k in self.OpcImplement.keys():
            opc = getattr(self, f"_{k}_opc")
            if opc.is_initialized():
                return opc
        return None

    def _all_opc_invoke(self, func, *args, **kwargs):
        for k in self.OpcImplement.keys():
            opc: IOpc = getattr(self, f"_{k}_opc")
            if not opc.is_initialized():
                continue
            getattr(opc, func)(*args, **kwargs)
