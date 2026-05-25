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
Interface for acl
"""


__all__ = ["AclInterface"]


# Standard Packages
import atexit
import ctypes
import logging
import time

import numpy
import numpy as np
from typing import List, Optional, Union, Set, Dict, Tuple

# Third-Party Packages
from ...utilities import (
    DATA_TYPE_DICT, FORMAT_DICT,
    torch_to_numpy_tensor
)
from ..runtime import RTSInterface


ACLNN_ERROR_DESC_DICT = {
    0: "ACLNN_SUCCESS",
    161001: "ACLNN_ERR_PARAM_NULLPTR",
    161002: "ACLNN_ERR_PARAM_INVALID",
    361001: "ACLNN_ERR_RUNTIME_ERROR",
    561000: "ACLNN_ERR_INNER",
    561001: "ACLNN_ERR_INNER_INFERSHAPE_ERROR",
    561002: "ACLNN_ERR_INNER_TILING_ERROR",
    561003: "ACLNN_ERR_INNER_FIND_KERNEL_ERROR",
    561101: "ACLNN_ERR_INNER_CREATE_EXECUTOR",
    561102: "ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR",
    561103: "ACLNN_ERR_INNER_NULLPTR",
    561104: "ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE",
    561105: "ACLNN_ERR_INNER_KEY_CONFLICT",
    561106: "ACLNN_ERR_INNER_INVALID_IMPL_MODE",
    561107: "ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND",
    561108: "ACLNN_ERR_INNER_LOAD_JSON_FAILED",
    561109: "ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND",
    561110: "ACLNN_ERR_INNER_JSON_FORMAT_INVALID",
    561111: "ACLNN_ERR_INNER_JSON_DTYPE_INVALID",
    561112: "ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND",
    561113: "ACLNN_ERR_INNER_OP_FILE_INVALID",
    561114: "ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND",
    561115: "ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH",
    561116: "ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE",
    561117: "ACLNN_ERR_INNER_INPUT_JSON_IS_NULL",
    561118: "ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID",
    561119: "ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID",
}


class AclInterface:
    """
    libascendcl.so & libop_api.so Wrappers
    """

    AclArrayDict: dict = {
        "Bool": ctypes.c_bool,
        "Int": ctypes.c_int64,
        "Float": ctypes.c_float,
    }

    def __init__(self, short_soc_version: Optional[str] = None,
                 camodel: Optional[str] = None):
        # memories need to be clean up when reset.
        self._acl_tensors: Set[int] = set()
        self._acl_tensor_lists: Dict[int, Tuple[ctypes.c_void_p]] = {}
        self._acl_scalars: Set[int] = set()
        self._acl_scalar_lists: Set[int] = set()
        self._acl_bool_arrays: Set[int] = set()
        self._acl_int_arrays: Set[int] = set()
        self._acl_float_arrays: Set[int] = set()
        # map aclTensor* -> device_mem_addr only
        self._acl_tensor_to_device_mem: Dict[int, ctypes.c_void_p] = {}
        self._acl_inited: bool = False
        self._device_id = None
        self._rts_interface = RTSInterface(camodel=camodel,
                                           short_soc_version=short_soc_version)
        self._acl_dll = ctypes.CDLL(f"libascendcl.so")
        self._opbase_dll = ctypes.CDLL(f"libnnopbase.so")
        atexit.register(self._on_exit)

    def __del__(self):
        self._on_exit()

    @property
    def device_id(self):
        return self._device_id

    @staticmethod
    def parse_error(acl_ret: ctypes.c_uint64, api_name: str, extra_info: str):
        # Convert aclnnStatus to int if received ctypes object
        if isinstance(acl_ret, ctypes.c_uint64):
            acl_ret = acl_ret.value
        elif isinstance(acl_ret, int):
            pass
        else:
            raise TypeError("Invalid acl_ret type %s for %s" % (str(type(acl_ret)), str(acl_ret)))
        # Success
        if acl_ret == 0x0:
            logging.debug(f"Acl API Call {api_name}() Success, {extra_info}")
            return
        # Fail
        desc = ACLNN_ERROR_DESC_DICT.get(acl_ret, str(acl_ret))
        raise RuntimeError(f"Acl API Call {api_name}() Failed: {desc}, {extra_info}")

    def is_model(self) -> bool:
        return self._rts_interface.is_model()

    def create_context(self, context_mode: str = "RT_CTX_NORMAL_MODE") -> ctypes.c_void_p:
        return self._rts_interface.create_context(context_mode)

    def destroy_context(self, c_context: ctypes.c_void_p = None):
        self._rts_interface.destroy_context(c_context)

    def create_stream(self, priority=0) -> ctypes.c_void_p:
        return self._rts_interface.create_stream(priority)

    def destroy_stream(self, stream: ctypes.c_void_p):
        self._rts_interface.destroy_stream(stream)

    def destroy_stream_force(self, stream: ctypes.c_void_p):
        self._rts_interface.destroy_stream_force(stream)

    def warmup(self, switches: "ttk.utilities.SWITCHES"):
        self._rts_interface.warmup(switches)

    def clear_l1(self, switches: "ttk.utilities.SWITCHES"):
        self._rts_interface.clear_l1(switches)

    def clear_ub(self, switches: "ttk.utilities.SWITCHES"):
        self._rts_interface.clear_ub(switches)

    def set_device(self, device_id: int) -> None:
        """
        Set device_id for current thread and initialize acl & rts_interface
        """
        self._acl_init()
        self._acl_set_device(device_id)
        self._rts_interface.set_device(device_id, skip_rt_intf=True)

    def reset(self) -> None:
        self._release_acl_memory()
        if hasattr(self, "_rts_interface") and self._rts_interface is not None:
            self._rts_interface.reset(skip_rt_intf=True)
        self._acl_reset_device()

    def set_float_overflow_mode(self, mode: int):
        self._rts_interface.set_float_overflow_mode(mode)

    def get_device_mem_addr(self, acl_tensor: Union[ctypes.c_void_p, int]) \
            -> Union[ctypes.c_void_p, List[ctypes.c_void_p]]:
        """
        Get device memory address through aclTensor* ptr or aclTensorList* ptr
        """
        ptr = acl_tensor.value if isinstance(acl_tensor, ctypes.c_void_p) else acl_tensor
        return self._acl_tensor_to_device_mem.get(ptr, None)

    def get_data_from_hbm(self, c_memory_p: Union[ctypes.c_void_p, int], data_size: int):
        return self._rts_interface.get_data_from_hbm(c_memory_p, data_size)

    def create_acl_tensor(self, tensor, view_format: str,
                          storage_shape: Optional[Union[tuple, list]] = None) -> ctypes.c_void_p:
        """
        Create aclTensor via torch.Tensor or numpy.ndarray
        aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                   const int64_t *stride, int64_t offset, aclFormat format,
                                   const int64_t *storageDims, uint64_t storageDimsNum,
                                   void *tensorData);
        """
        if isinstance(tensor, numpy.ndarray):
            return self._create_acl_tensor_from_numpy(tensor, view_format, storage_shape)
        else:
            return self._create_acl_tensor_from_torch(tensor, view_format, storage_shape)

    def _create_acl_tensor_from_torch(self, torch_tensor, view_format: str,
                                       storage_shape: Optional[Union[tuple, list]] = None) -> ctypes.c_void_p:
        """Create aclTensor from torch.Tensor"""
        view_dims = torch_tensor.shape
        c_view_dims = (ctypes.c_int64 * len(view_dims))(*view_dims)
        c_view_dims_num = ctypes.c_uint64(len(view_dims))
        dtype_str = str(torch_tensor.dtype).split('.')[-1]
        c_acl_dtype = ctypes.c_int32(DATA_TYPE_DICT[dtype_str])
        c_format = ctypes.c_int32(FORMAT_DICT[view_format])

        if not storage_shape or (
                tuple(view_dims) == tuple(storage_shape) and torch_tensor.is_contiguous()):
            stride = 0
            c_stride = None
            c_offset = ctypes.c_int64(0)
            c_storage_dims = (ctypes.c_int64 * len(view_dims))(*view_dims)
            c_storage_dims_num = ctypes.c_uint64(len(view_dims))
        else:
            stride = torch_tensor.stride()
            c_stride = (ctypes.c_int64 * len(stride))(*stride)
            c_offset = ctypes.c_int64(torch_tensor.storage_offset())
            c_storage_dims = (ctypes.c_int64 * len(storage_shape))(*storage_shape)
            c_storage_dims_num = ctypes.c_uint64(len(storage_shape))
        device_mem_addr = self._rts_interface.copy_torch_tensor_to_hbm(torch_tensor)

        c_ptr = self._opbase_api_call_with_ptr_return("aclCreateTensor",
                                                      f"Args: view_shape={view_dims}, dtype={dtype_str}, "
                                                      f"format={view_format}, stride={stride}, "
                                                      f"offset={torch_tensor.storage_offset()}, "
                                                      f"storage_shape={storage_shape}",
                                                      c_view_dims, c_view_dims_num, c_acl_dtype,
                                                      c_stride, c_offset, c_format,
                                                      c_storage_dims, c_storage_dims_num,
                                                      device_mem_addr)
        self._acl_tensor_to_device_mem[c_ptr.value] = device_mem_addr
        self._acl_tensors.add(c_ptr.value)
        return c_ptr

    def _create_acl_tensor_from_numpy(self, np_tensor: numpy.ndarray, view_format: str,
                                       storage_shape: Optional[Union[tuple, list]] = None) -> ctypes.c_void_p:
        """Create aclTensor from numpy.ndarray（可能是 as_strided 后的 view）"""
        view_dims = np_tensor.shape
        c_view_dims = (ctypes.c_int64 * len(view_dims))(*view_dims)
        c_view_dims_num = ctypes.c_uint64(len(view_dims))
        dtype_str = np_tensor.dtype.name
        c_acl_dtype = ctypes.c_int32(DATA_TYPE_DICT[dtype_str])
        c_format = ctypes.c_int32(FORMAT_DICT[view_format])

        # 获取原始 storage（as_strided 之前的连续数组）
        # as_strided 的 base 可能是 DummyArray，需要递归找到真正的 ndarray
        np_storage = np_tensor
        while np_storage.base is not None:
            if isinstance(np_storage.base, numpy.ndarray):
                np_storage = np_storage.base
                break
            # as_strided 产生的 DummyArray，取其内部的 ndarray
            if hasattr(np_storage.base, '__array_interface__'):
                np_storage = numpy.asarray(np_storage.base)
                break
            np_storage = np_storage.base

        # numpy stride 单位是字节，转为元素单位
        elem_strides = tuple(s // np_tensor.itemsize for s in np_tensor.strides)
        is_contiguous = np_tensor.flags['C_CONTIGUOUS']

        if not storage_shape or (
                tuple(view_dims) == tuple(storage_shape) and is_contiguous):
            stride = 0
            c_stride = None
            c_offset = ctypes.c_int64(0)
            c_storage_dims = (ctypes.c_int64 * len(view_dims))(*view_dims)
            c_storage_dims_num = ctypes.c_uint64(len(view_dims))
            offset = 0
        else:
            stride = elem_strides
            c_stride = (ctypes.c_int64 * len(stride))(*stride)
            # storage_offset：view 起始地址相对 storage base 的元素偏移
            base_addr = np_storage.__array_interface__['data'][0]
            view_addr = np_tensor.__array_interface__['data'][0]
            offset = (view_addr - base_addr) // np_tensor.itemsize
            c_offset = ctypes.c_int64(offset)
            c_storage_dims = (ctypes.c_int64 * len(storage_shape))(*storage_shape)
            c_storage_dims_num = ctypes.c_uint64(len(storage_shape))

        # 拷贝原始 storage（连续内存）到 HBM
        device_mem_addr = self._rts_interface.copy_nparray_to_hbm(np_storage)

        c_ptr = self._opbase_api_call_with_ptr_return("aclCreateTensor",
                                                      f"Args: view_shape={view_dims}, dtype={dtype_str}, "
                                                      f"format={view_format}, stride={stride}, "
                                                      f"offset={offset}, "
                                                      f"storage_shape={storage_shape}",
                                                      c_view_dims, c_view_dims_num, c_acl_dtype,
                                                      c_stride, c_offset, c_format,
                                                      c_storage_dims, c_storage_dims_num,
                                                      device_mem_addr)
        self._acl_tensor_to_device_mem[c_ptr.value] = device_mem_addr
        self._acl_tensors.add(c_ptr.value)
        return c_ptr

    def create_acl_tensor_list(self, acl_tensor_ptr_lst: Union[tuple, list]) -> ctypes.c_void_p:
        """
        Create aclTensorList
        aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size);
        """
        cnt = len(acl_tensor_ptr_lst)
        c_size = ctypes.c_uint64(cnt)
        c_value = (ctypes.c_void_p * cnt)(*acl_tensor_ptr_lst)
        c_ptr = self._opbase_api_call_with_ptr_return("aclCreateTensorList",
                                                      f"Args: value={acl_tensor_ptr_lst}, size={cnt}",
                                                      c_value, c_size)
        self._acl_tensor_lists.update({c_ptr.value: tuple(acl_tensor_ptr_lst)})
        # aclTensor ptr now is managed by aclTensorList ptr
        for t in acl_tensor_ptr_lst:
            if t is None:
                continue
            if isinstance(t, ctypes.c_void_p):
                t = t.value
            self._acl_tensors.remove(t)
        return c_ptr

    def create_acl_scalar(self, val: Union[numpy.ndarray, "torch.Tensor"]) -> ctypes.c_void_p:
        """
        Create aclScalar.
        NOTE:
        aclScalar *aclCreateScalar(void *value, aclDataType dataType)
        """
        if isinstance(val, numpy.ndarray):
            # numpy.ndarray with shape [1]
            c_val_ptr = ctypes.c_void_p(val.__array_interface__['data'][0])
            scalar_dtype = val.dtype.name
        else:
            import torch
            if not isinstance(val, torch.Tensor):
                raise RuntimeError(f"Only numpy.ndarray or torch.Tensor is supported. "
                                   f"But got {type(val)}")
            # torch.Tensor with shape [1]
            scalar_dtype = str(val.dtype).split(".")[1]
            np_val = torch_to_numpy_tensor(val)
            c_val_ptr = ctypes.c_void_p(np_val.__array_interface__['data'][0])

        c_ptr = self._opbase_api_call_with_ptr_return("aclCreateScalar",
                                                      f"Args: value={val}, dtype={scalar_dtype}",
                                                      c_val_ptr, DATA_TYPE_DICT[scalar_dtype])
        self._acl_scalars.add(c_ptr.value)
        return c_ptr

    def create_acl_scalar_list(self, acl_scalar_ptr_lst: Union[tuple, list]) -> ctypes.c_void_p:
        """
        Create aclScalarList
        aclScalarList *aclCreateScalarList(const aclScalar *const *value, uint64_t size)
        """
        cnt = len(acl_scalar_ptr_lst)
        c_size = ctypes.c_uint64(cnt)
        c_value = (ctypes.c_void_p * cnt)(*acl_scalar_ptr_lst)
        c_ptr = self._opbase_api_call_with_ptr_return("aclCreateScalarList",
                                                      f"Args: value={acl_scalar_ptr_lst}, size={cnt}",
                                                      c_value, c_size)
        self._acl_scalar_lists.add(c_ptr.value)
        # aclScalar ptr now is managed by aclScalarList ptr
        for s in acl_scalar_ptr_lst:
            if s is None:
                continue
            if isinstance(s, ctypes.c_void_p):
                s = s.value
            self._acl_scalars.remove(s)
        return c_ptr

    def create_acl_array(self, val_lst: Union[tuple, list], typ: str) -> ctypes.c_void_p:
        """
        aclBoolArray *aclCreateBoolArray(const bool *value, uint64_t size)
        """
        cnt = len(val_lst)
        c_type = self.AclArrayDict[typ]
        c_size = ctypes.c_uint64(cnt)
        if cnt == 0:
            c_value = None
        else:
            c_value = (c_type * cnt)(*val_lst)
        c_ptr = self._opbase_api_call_with_ptr_return(f"aclCreate{typ}Array",
                                                      f"Args: value={val_lst}, size={cnt}",
                                                      c_value, c_size)
        getattr(self, f"_acl_{typ.lower()}_arrays").add(c_ptr.value)
        return c_ptr

    def get_view_shape(self, tensor: ctypes.c_void_p) -> tuple:
        c_view_dims_ptr = ctypes.POINTER(ctypes.c_int64)()
        c_view_dims_ptr_ptr = ctypes.byref(c_view_dims_ptr)
        c_view_dims_num = ctypes.c_uint64()
        c_view_dims_num_ptr = ctypes.byref(c_view_dims_num)
        self._opbase_api_call("aclGetViewShape", None,
                              tensor, c_view_dims_ptr_ptr, c_view_dims_num_ptr)
        view_shape = tuple([c_view_dims_ptr[i] 
                            for i in range(c_view_dims_num.value)])
        delete_array_func = self._get_opbase_dll('_ZdaPv')
        if delete_array_func:
            delete_array_func.argtypes = [ctypes.c_void_p]
            delete_array_func.restype = None
            delete_array_func(c_view_dims_ptr)
        return view_shape

    def acl_get_workspace(self, api_name: str, args: list) -> tuple:
        func_name = f"{api_name}GetWorkspaceSize"
        c_workspace_size = ctypes.c_uint64(0)
        c_workspace_size_ptr = ctypes.c_void_p(ctypes.addressof(c_workspace_size))
        c_executor = ctypes.c_void_p(0)
        c_executor_ptr = ctypes.c_void_p(ctypes.addressof(c_executor))
        self._aclnn_api_call(func_name, None,
                             *args, c_workspace_size_ptr, c_executor_ptr)
        logging.debug(f"OpApi phase 1 [{func_name}] Success "
                      f"with workspace size [{c_workspace_size.value}] bytes.")
        return c_workspace_size.value, c_executor

    def acl_execute(self, api_name: str,
                    workspace_size: int, c_executor: ctypes.c_void_p,
                    stream: Optional[ctypes.c_void_p] = None):
        c_workspace_size = ctypes.c_uint64(workspace_size)
        c_workspace_ptr = None
        status = "OK"
        if workspace_size > 0:
            np_array = np.random.randint(0, 255,
                                         size=(int(workspace_size),),
                                         dtype=np.uint8)
            c_workspace_ptr = self._rts_interface.copy_nparray_to_hbm(np_array,
                                                                      fill_oob_flag=False)
        try:
            self._aclnn_api_call(api_name, None,
                                 c_workspace_ptr, c_workspace_size,
                                 c_executor, stream)
        except Exception as e:
            logging.error(f"{api_name} execute failed with error: {e}")
            status = "ACLNN_EXECUTE_FAILED"
        finally:
            # need to synchronize stream to clean up
            # no matter phase-2 is success or not.
            try:
                self._rts_interface.synchronize_with_stream(stream)
            except RuntimeError as e:
                sync_status = self._rts_interface.handle_stream_sync_exception(e.args[0])
                if status == "OK":
                    status = sync_status
            finally:
                if c_workspace_ptr is not None:
                    # TODO: check workspace oob
                    self._rts_interface.free(c_workspace_ptr)
        return status

    def free_all_memory(self):
        self._release_acl_memory()

    def _free_device_memory(self, acl_tensor: Union[ctypes.c_void_p, int]):
        """
        Free device memory used in aclTensor via aclTensor pointer.
        """
        if not isinstance(acl_tensor, ctypes.c_void_p):
            acl_tensor = ctypes.c_void_p(acl_tensor)
        device_mem_ptr = self.get_device_mem_addr(acl_tensor)
        if device_mem_ptr is not None:
            self._rts_interface.free(device_mem_ptr)
            del self._acl_tensor_to_device_mem[acl_tensor.value]

    def _free_acl_tensor(self, acl_tensor: Union[ctypes.c_void_p, int]):
        if not isinstance(acl_tensor, ctypes.c_void_p):
            acl_tensor = ctypes.c_void_p(acl_tensor)
        self._opbase_api_call("aclDestroyTensor", None, acl_tensor)
        self._free_device_memory(acl_tensor)

    def _free_acl_tensor_list(self, acl_tensor_list: Union[ctypes.c_void_p, int]):
        if not isinstance(acl_tensor_list, ctypes.c_void_p):
            acl_tensor_list = ctypes.c_void_p(acl_tensor_list)
        # NOTE: aclDestroyTensorList will remove elements in this list
        self._opbase_api_call("aclDestroyTensorList", None, acl_tensor_list)
        acl_tensors: list = list(self._acl_tensor_lists[acl_tensor_list.value])
        del self._acl_tensor_lists[acl_tensor_list.value]
        while acl_tensors:
            self._free_device_memory(acl_tensors.pop())

    def _free_acl_scalar(self, acl_scalar: Union[ctypes.c_void_p, int]):
        if not isinstance(acl_scalar, ctypes.c_void_p):
            acl_scalar = ctypes.c_void_p(acl_scalar)
        self._opbase_api_call("aclDestroyScalar", None, acl_scalar)

    def _free_acl_scalar_list(self, acl_scalar_list: Union[ctypes.c_void_p, int]):
        if not isinstance(acl_scalar_list, ctypes.c_void_p):
            acl_scalar_list = ctypes.c_void_p(acl_scalar_list)
        # NOTE: aclDestroyScalarList will remove elements in this list...
        self._opbase_api_call("aclDestroyScalarList", None, acl_scalar_list)

    def _free_acl_array(self, acl_array: Union[ctypes.c_void_p, int], typ: str):
        if not isinstance(acl_array, ctypes.c_void_p):
            acl_array = ctypes.c_void_p(acl_array)
        self._opbase_api_call(f"aclDestroy{typ}Array", None, acl_array)

    def _release_acl_memory(self):
        for tl_ptr in tuple(self._acl_tensor_lists.keys()):
            self._free_acl_tensor_list(tl_ptr)
        while self._acl_tensors:
            self._free_acl_tensor(self._acl_tensors.pop())
        while self._acl_scalar_lists:
            self._free_acl_scalar_list(self._acl_scalar_lists.pop())
        while self._acl_scalars:
            self._free_acl_scalar(self._acl_scalars.pop())
        for typ in self.AclArrayDict.keys():
            array_sets = getattr(self, f"_acl_{typ.lower()}_arrays")
            while array_sets:
                self._free_acl_array(array_sets.pop(), typ)

    def _acl_init(self):
        if not self._acl_inited:
            self._acl_api_call("aclInit", None, None)
            self._acl_inited = True

    def _acl_finalize(self):
        if self._acl_inited:
            self._acl_api_call("aclFinalize", None)
            self._acl_inited = False

    def _acl_set_device(self, device_id: int):
        self._acl_api_call("aclrtSetDevice", f"on device {device_id}",
                           ctypes.c_int32(device_id))
        self._device_id = device_id

    def _acl_reset_device(self):
        if self._device_id is None:
            return
        self._acl_api_call("aclrtResetDevice", None,
                           ctypes.c_int32(self._device_id))
        self._device_id = None

    def _acl_api_call(self, api_name: str, extra_log: Optional[str], *api_args):
        self._api_call("ACL", api_name, extra_log, *api_args)

    def _aclnn_api_call(self, api_name: str, extra_log: Optional[str], *api_args):
        self._api_call("ACLNN", api_name, extra_log, *api_args)

    def _opbase_api_call(self, api_name: str, extra_log: Optional[str], *api_args):
        self._api_call("OPBASE", api_name, extra_log, *api_args)

    def _opbase_api_call_with_ptr_return(self, api_name: str,
                                         extra_log: Optional[str],
                                         *api_args) -> ctypes.c_void_p:
        if extra_log is None:
            extra_log = ""
        api = self._get_opbase_dll(api_name)
        if not api:
            raise RuntimeError(f"ACL api [{api_name}] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_void_p
        rt_ptr = api(*api_args)
        if not rt_ptr:
            raise RuntimeError(f"Aclnn API {api_name} return nullptr. {extra_log}")
        logging.debug(f"Aclnn API Call {api_name}() Success, {extra_log} "
                      f"costs {round(time.time() - start_time, 3)} seconds")
        return ctypes.c_void_p(rt_ptr)

    def _api_call(self, kind: str, api_name: str, extra_log: Optional[str], *api_args):
        if extra_log is None:
            extra_log = ""
        if kind == "ACL":
            api = self._get_acl_api(api_name)
        elif kind == "OPBASE":
            api = self._get_opbase_dll(api_name)
        else:
            api = self._get_op_api(api_name)
        if not api:
            raise RuntimeError(f"ACL api [{api_name}] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        rt_error = api(*api_args)
        self.parse_error(rt_error, api_name,
                         f"{extra_log} costs {round(time.time() - start_time, 3)} seconds")

    def _on_exit(self):
        self.reset()
        self._acl_finalize()

    # Shared across all AclInterface instances. ctypes.CDLL loads each SO once per process, so sharing is safe.
    _SO_CACHE: dict = {}

    def _get_op_api(self, api_name):
        from .op_api_info_keeper import OpApiInfoKeeper
        lookup_name = api_name
        if lookup_name.endswith('GetWorkspaceSize'):
            lookup_name = lookup_name[:-16]
        info = OpApiInfoKeeper().info_of(lookup_name)
        if not info:
            logging.error(f"API [{api_name}] not found in any header directory")
            return None
        if not info.so_path:
            logging.error(f"API [{api_name}] has category=[{info.category}] but SO path is empty")
            return None
        dll = self._SO_CACHE.get(info.so_path)
        if dll is None:
            dll = ctypes.CDLL(info.so_path)
            self._SO_CACHE[info.so_path] = dll
        api = getattr(dll, api_name, None)
        if not api:
            logging.error(f"API [{api_name}] not found in SO [{info.so_path}]")
        return api

    def _get_acl_api(self, api_name):
        return getattr(self._acl_dll, api_name)

    def _get_opbase_dll(self, api_name):
        return getattr(self._opbase_dll, api_name)

