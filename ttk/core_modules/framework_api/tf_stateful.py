#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TF mutable-ref (stateful) op adaptation.

判定依据 — OpDef 的 is_ref 标志, 而非算子名或参数名:
- 参数名叫 ref 的输入不全是 mutable(部分 ref 实为 resource handle);
- mutable 输入也不全叫 ref(存在 var/accum/handle 等命名)。
唯一权威来源是 op_def_registry 中 input_arg.is_ref(TF 官方参数元数据)。

tf.raw_ops.* 的 is_ref 输入要求 TF1 ref-dtype 可变张量, TF2 无法构造(即使
在 tf.function 图内, resource 变量也会被 op_def_library 以 "be a mutable
tensor" 拒绝)。CSV 的 api_name 应直接写 TF2 可调用接口(tf.compat.v1 对应
的 snake_case 封装: 内部对 resource 变量分发 Resource 系算子, NPU 在线
支持见 npu_supported_ops.json, 并以 lazy-read 返回更新后的完整张量);
判定函数对 python snake_case 函数名做 CamelCase 归一化后查 OpDef。

tf.raw_ops.Resource* 系(如 ResourceSparseApplyAdadelta)无 snake_case 公开
封装, 只能直接调 raw op: 其 var/accum 等输入为 DT_RESOURCE(type=20),
并非 is_ref —— mutable 判定需同时覆盖两类; 且 DT_RESOURCE 位要求 resource
句柄, tf.Variable 在 op 边界会被自动解引用为值, 须显式传 .handle
(snake_case 封装能工作正因其内部自行取 handle 分发)。

变量放置 — 必须随默认设备落在 NPU, 不能用 context.device("/CPU:0") 之类
强制 CPU 放置: CPU 变量会让消费它的算子被 TF placer colocate 到 CPU,
npu_device 不触发 GE 编译, 算子静默跑 CPU(kernel 记录为空), 精度对比
退化成 CPU vs CPU 的假绿。NPU 放置的前提是 CANN 环境变量初始化可编译 ——
cann-9.2.0 中 op_host/lib/linux/x86_64/ 下 libpad_infershape.so 与
libophost_custom.so 缺 TbeLoadSoAndSaveToRegistry 导出接口, 会使变量
初始化编译失败(E40021), 需先移出这两个 so。
"""

import functools

# tensorflow dtypes.DT_RESOURCE 的枚举值(OpDef input_arg.type 序列化值)。
# type=0(DT_INVALID) 表示由 type_attr 描述, 显式 20 仅出现在 Resource* 系
# raw op 的 handle 输入上, 与 is_ref 互斥。
_DT_RESOURCE = 20


@functools.lru_cache(maxsize=None)
def _lookup_op_def(api_name):
    """按 API 名查 OpDef; python 函数名(scatter_div)归一化为算子名(ScatterDiv)。

    仅 TF 系 api_name 才 import tensorflow: torch 进程先加载 torch_npu 后再
    加载 TF 的 C 扩展会段错误(两大框架运行时冲突), 故非 TF 前缀直接返回。
    """
    if not isinstance(api_name, str) or not api_name.startswith(("tf.", "tensorflow.")):
        return None
    try:
        from tensorflow.python.framework import op_def_registry

        op_name = api_name.rsplit(".", 1)[-1]
        op_def = op_def_registry.get(op_name)
        if op_def is None and op_name[:1].islower():
            op_def = op_def_registry.get("".join(p.capitalize() for p in op_name.split("_")))
    except Exception:
        return None
    return op_def


def _is_mutable_input_arg(arg):
    """is_ref(TF1 mutable-ref)或 DT_RESOURCE(Resource* raw op 的 handle 位)。"""
    return arg.is_ref or arg.type == _DT_RESOURCE


def get_mutable_param_indexes(api_name):
    """返回该 TF 算子 OpDef 中可变输入参数(is_ref 或 DT_RESOURCE)的下标元组。

    下标与 api_info.tensors(张量参数列表)一一对位。无 OpDef 或解析失败
    返回空元组(普通 API 无 mutable 输入)。
    """
    op_def = _lookup_op_def(api_name)
    if op_def is None:
        return ()
    return tuple(i for i, arg in enumerate(op_def.input_arg) if _is_mutable_input_arg(arg))


def get_mutable_param_names(api_name):
    """返回该 TF 算子 mutable 输入的参数名列表(如 ['ref'] / ['var', 'accum'])。"""
    op_def = _lookup_op_def(api_name)
    if op_def is None:
        return []
    return [arg.name for arg in op_def.input_arg if _is_mutable_input_arg(arg)]


def get_resource_param_names(api_name):
    """返回该 TF 算子 OpDef 中 DT_RESOURCE 输入参数名(如 var/accum/handle)。

    仅 Resource* 系 raw op 存在此类参数; snake_case 封装与 torch API 返回空。
    """
    op_def = _lookup_op_def(api_name)
    if op_def is None:
        return []
    return [arg.name for arg in op_def.input_arg if arg.type == _DT_RESOURCE]


def get_resource_param_indexes(api_name):
    """返回该 TF 算子 OpDef 中 DT_RESOURCE 输入参数下标元组。"""
    op_def = _lookup_op_def(api_name)
    if op_def is None:
        return ()
    return tuple(i for i, arg in enumerate(op_def.input_arg) if arg.type == _DT_RESOURCE)


def convert_variables_to_handles(api_name, args=None, kwargs=None):
    """把 DT_RESOURCE 参数位的 tf.Variable 就地换成 .handle。

    Resource* raw op 的 handle 位要求 resource 张量; tf.Variable 在 op 边界
    会被自动解引用为值(dtype 不符直接报错), 必须显式传 .handle。args 按
    input_arg 下标对位(raw op 的张量参数居签名前列, 与 input_arg 顺序一致),
    kwargs 按参数名对位。非 Variable 值与非 resource 位不动。
    """
    indexes = get_resource_param_indexes(api_name) if args else ()
    names = set(get_resource_param_names(api_name)) if kwargs else set()
    if not indexes and not names:
        return
    import tensorflow as tf

    for i in indexes:
        if i < len(args) and isinstance(args[i], tf.Variable):
            args[i] = args[i].handle
    for name in names & set(kwargs):
        if isinstance(kwargs[name], tf.Variable):
            kwargs[name] = kwargs[name].handle


def collect_mutable_variables(api_name, args=None, kwargs=None):
    """收集 mutable 参数位(is_ref/DT_RESOURCE)的 tf.Variable, 按参数下标升序。

    无输出有状态算子的输出语义支撑: 结果体现在变量原地更新里, 调用后回读
    这些位取值(对齐 torch inplace 的"原地位即输出位")。args 按下标、kwargs
    按名对位(与 convert_variables_to_handles 同一规则); 非 Variable 位跳过。
    """
    op_def = _lookup_op_def(api_name)
    if op_def is None:
        return []
    import tensorflow as tf

    variables = []
    for i, arg in enumerate(op_def.input_arg):
        if not _is_mutable_input_arg(arg):
            continue
        v = args[i] if args and i < len(args) else (kwargs or {}).get(arg.name)
        if isinstance(v, tf.Variable):
            variables.append(v)
    return variables


def read_mutable_values(variables):
    """读取变量列表的当前值(原地更新后的结果), 返回 numpy 数组列表。"""
    return [v.read_value().numpy() for v in variables]


def is_ref_variable(value):
    """value 是否为 tf.Variable(to_device/clone 需 bypass 普通张量处理)。"""
    import tensorflow as tf

    return isinstance(value, tf.Variable)
