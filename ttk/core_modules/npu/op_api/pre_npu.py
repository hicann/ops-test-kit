#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Framework-owned ACLNN runner exposed only during a ``pre_npu`` hook."""

import ctypes
from typing import Any, Mapping, Optional, Sequence

import numpy

from ....utilities import numpy_to_torch_tensor, resolve_custom_numpy_dtypes
from ...aclnn import AclInterface, OpApiInfoKeeper


class PreNpuAclnnRunner:
    """Execute one auxiliary ACLNN API and copy named outputs to placeholders."""

    _C_BASE_TYPES = {
        "bool": ctypes.c_bool,
        "int8_t": ctypes.c_int8,
        "uint8_t": ctypes.c_uint8,
        "int": ctypes.c_int,
        "int32_t": ctypes.c_int32,
        "uint32_t": ctypes.c_uint32,
        "float": ctypes.c_float,
        "int64_t": ctypes.c_int64,
        "uint64_t": ctypes.c_uint64,
        "double": ctypes.c_double,
    }

    def __init__(self, device: AclInterface, stream=None):
        self._device = device
        self._stream = stream

    def __call__(
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
        info = OpApiInfoKeeper().info_of(api_name)
        if info is None:
            raise RuntimeError(f"pre-NPU ACLNN API is not installed: {api_name}")
        output_names = self._validate_outputs(api_name, info, tensors, output_names)

        tensor_formats = dict(tensor_formats or {})
        storage_shapes = dict(storage_shapes or {})
        scalar_values = dict(scalars or {})
        attributes = dict(attributes or {})
        tensor_ptrs = {}
        params = []
        try:
            for name, param in info.params.items():
                acl_type = param["type"]
                if acl_type == "aclTensor*":
                    if name not in tensors:
                        raise ValueError(
                            f"pre-NPU ACLNN {api_name} missing tensor parameter {name!r}"
                        )
                    value = tensors[name]
                    ptr = None
                    if value is not None:
                        ptr = self._device.create_acl_tensor(
                            value,
                            tensor_formats.get(name, "ND"),
                            storage_shapes.get(name),
                        )
                    tensor_ptrs[name] = ptr
                    params.append(ptr)
                elif acl_type == "aclTensorList*":
                    values = tensors.get(name)
                    if values is None:
                        tensor_ptrs[name] = None
                        params.append(None)
                        continue
                    ptrs = [
                        self._device.create_acl_tensor(value, tensor_formats.get(name, "ND"))
                        if value is not None else None
                        for value in values
                    ]
                    ptr = self._device.create_acl_tensor_list(ptrs)
                    tensor_ptrs[name] = ptr
                    params.append(ptr)
                elif acl_type in ("aclScalar*", "aclScalarList*"):
                    if name not in scalar_values:
                        raise ValueError(
                            f"pre-NPU ACLNN {api_name} missing scalar parameter {name!r}"
                        )
                    value = scalar_values[name]
                    if value is None:
                        params.append(None)
                    elif acl_type == "aclScalarList*":
                        ptrs = [self._device.create_acl_scalar(item) for item in value]
                        params.append(self._device.create_acl_scalar_list(ptrs))
                    else:
                        params.append(self._device.create_acl_scalar(value))
                else:
                    params.append(
                        self._attribute_value(
                            api_name, name, acl_type, param.get("default"), attributes
                        )
                    )

            workspace_size, executor = self._device.acl_get_workspace(api_name, params)
            status = self._device.acl_execute(
                api_name, workspace_size, executor, self._stream
            )
            if status != "OK":
                raise RuntimeError(
                    f"pre-NPU ACLNN {api_name} execution failed: {status}"
                )
            for output_name in output_names:
                ptr = tensor_ptrs.get(output_name)
                if ptr is None:
                    raise ValueError(
                        f"pre-NPU ACLNN output {output_name!r} is not an ACL tensor parameter"
                    )
                returned_shape = tuple(self._device.get_view_shape(ptr))
                target_shape = tuple(tensors[output_name].shape)
                if returned_shape != target_shape:
                    raise ValueError(
                        f"pre-NPU ACLNN output {output_name!r} shape {returned_shape} "
                        f"!= placeholder {target_shape}"
                    )
                self._copy_output(ptr, tensors[output_name], storage_shapes.get(output_name))
        finally:
            self._device.free_all_memory()

    @staticmethod
    def _validate_outputs(api_name, info, tensors, output_names):
        if isinstance(output_names, (str, bytes)):
            raise TypeError("pre-NPU ACLNN output_names must be a sequence of names")
        names = tuple(output_names or ())
        if not names:
            raise ValueError("pre-NPU ACLNN call requires at least one output name")
        if len(set(names)) != len(names):
            raise ValueError("pre-NPU ACLNN output_names must not contain duplicates")
        for name in names:
            if not isinstance(name, str) or not name:
                raise TypeError("pre-NPU ACLNN output names must be non-empty strings")
            param = info.params.get(name)
            if param is None:
                raise ValueError(
                    f"pre-NPU ACLNN {api_name} has no parameter named {name!r}"
                )
            if param["type"] != "aclTensor*":
                raise ValueError(
                    f"pre-NPU ACLNN output {name!r} must be an aclTensor* parameter, "
                    f"got {param['type']!r}"
                )
            if name not in tensors or tensors[name] is None:
                raise ValueError(
                    f"pre-NPU ACLNN output {name!r} needs a tensor placeholder"
                )
        return names

    def _attribute_value(self, api_name, name, acl_type, default, attributes):
        if name in attributes:
            value = attributes[name]
        elif default is not None:
            value = default
        else:
            raise ValueError(
                f"pre-NPU ACLNN {api_name} missing attribute parameter {name!r}"
            )
        if value is None:
            return None
        if "Array" in acl_type:
            array_type = acl_type[3:acl_type.index("Array")]
            return self._device.create_acl_array(value, array_type)
        if acl_type == "aclDataType":
            return ctypes.c_int32(value)
        if acl_type == "char*":
            return ctypes.c_char_p(str(value).encode("UTF-8"))
        c_type = self._C_BASE_TYPES.get(acl_type)
        if c_type is None:
            raise TypeError(
                f"pre-NPU ACLNN {api_name} does not support C type {acl_type!r} "
                f"for {name!r}"
            )
        return c_type(value)

    def _copy_output(self, acl_tensor, target, declared_storage_shape):
        dtype_name = self._dtype_name(target)
        dtype = resolve_custom_numpy_dtypes((dtype_name,))[0]
        storage_shape = self._storage_shape(target, declared_storage_shape)
        byte_size = int(numpy.prod(storage_shape, dtype=numpy.int64)) * numpy.dtype(dtype).itemsize
        payload = self._device.get_data_from_hbm(
            self._device.get_device_mem_addr(acl_tensor), byte_size
        )
        storage = numpy.frombuffer(payload, dtype=dtype).copy().reshape(storage_shape)

        if isinstance(target, numpy.ndarray):
            # The ACL tensor may describe a view into a larger storage. Rebuild
            # that same view over the returned storage before copying it back.
            base = target
            while isinstance(getattr(base, "base", None), numpy.ndarray):
                base = base.base
            base_address = base.__array_interface__["data"][0]
            view_address = target.__array_interface__["data"][0]
            offset = (view_address - base_address) // target.itemsize
            source = numpy.lib.stride_tricks.as_strided(
                storage.reshape(-1)[offset:],
                shape=target.shape,
                strides=target.strides,
            )
            target[...] = source
            return

        import torch

        source_storage = numpy_to_torch_tensor(storage)
        if target.dim() == 0:
            source = source_storage.reshape(())
        else:
            source = torch.as_strided(
                source_storage,
                tuple(target.shape),
                tuple(target.stride()),
                int(target.storage_offset()),
            )
        target.copy_(source.to(dtype=target.dtype, device=target.device))

    @staticmethod
    def _dtype_name(value):
        if isinstance(value, numpy.ndarray):
            return value.dtype.name
        return str(value.dtype).split(".")[-1]

    @staticmethod
    def _storage_shape(value, declared_storage_shape):
        if declared_storage_shape is not None:
            return tuple(int(item) for item in declared_storage_shape)
        if isinstance(value, numpy.ndarray):
            storage = value
            while isinstance(getattr(storage, "base", None), numpy.ndarray):
                storage = storage.base
            return tuple(storage.shape)
        if value.is_contiguous() and int(value.storage_offset()) == 0:
            return tuple(value.shape)
        raise ValueError(
            "non-contiguous pre-NPU ACLNN outputs require an explicit storage_shapes entry"
        )
