# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
"""免上传输入（zero-upload）：用"生成配方"代替"整包数据"过网络。

背景：三方腿把输入 npz 推给远端 xpu-server。跨公网跳板时这段上行是主要耗时
（实测 3~5 MB/s，一轮性能档 16.8 GB），且长时间占用链路会放大中断风险。

做法：输入本来就是 ``RandomData(dtype, shape, data_range).generate(distribution)``
在**播过种的 numpy 全局 RNG** 上生成的纯函数结果。于是客户端只发配方
（dtype/shape/data_range/distribution/seed）+ 每片叶子的 sha256，
服务端按同一配方重算并**逐片校验摘要**：

- 摘要一致 → 两端输入逐位相同，数据不必过网络；
- 摘要不一致（numpy/scipy 版本漂移、生成代码差异、非连续视图、
  Spec 的 customize_inputs 改过数据……）→ 服务端回 409，
  客户端自动回落成原来的整包上传。

即"能证明一致才免传"，不是"假定一致"。失效时只会退化成今天的行为，
绝不会出现两端比了不同数据却浑然不觉 —— 那是最危险的失效模式。
"""

import hashlib
from typing import Optional

import numpy

# 配方格式版本：两端不一致时直接不启用（而不是猜对方语义）
RECIPE_VERSION = 1


def digest_of(array) -> str:
    """输入叶子的内容指纹（sha256 十六进制）。

    自定义窄类型（bfloat16/float8_*）也支持：走 buffer 协议按原始字节哈希，
    不解释数值语义，因此 ±0.0 之类的表示差异同样会被识别出来。
    """
    contiguous = numpy.ascontiguousarray(array)
    try:
        # 自定义窄类型（bfloat16/float8_* 等）没有 buffer 协议导出，
        # 先按字节重解释；uint8 视图不复制数据。
        raw = contiguous.view(numpy.uint8)
    except (TypeError, ValueError):
        raw = numpy.frombuffer(contiguous.tobytes(), dtype=numpy.uint8)
    return hashlib.sha256(memoryview(raw)).hexdigest()


def seed_for(testcase_name: str, index: int) -> int:
    """(用例名, 叶子序号) → 确定性种子。

    用 md5 而不是内置 hash()：后者带 PYTHONHASHSEED 随机化，跨进程不稳定
    （跑批的 worker 是多进程，用 hash() 会让同一用例两次生成不同数据）。
    """
    raw = hashlib.md5(  # noqa: S324  # 非安全用途：确定性种子派生
        f"{testcase_name}#{index}".encode()
    ).digest()
    return int.from_bytes(raw[:4], "big")


def _dtype_name(dtype) -> str:
    """取 dtype 的**规范名**(如 "bfloat16"), 而不是 ``str(dtype)``。

    TTK 内部的自定义 dtype(bfloat16/fp8/… 来自 ml_dtypes)是**类对象**, ``str()`` 会得到
    ``"<class 'ml_dtypes.bfloat16'>"`` —— 服务端拿它去 ``RandomData`` 必然报
    "data type not understood", 于是回 409、客户端回落整包上传: 免传输对这些 dtype 全程失效,
    且只在日志里留一行 "falling back to full upload", 表现为"这个优化没什么收益"。
    ``numpy.dtype(cls).name`` 对类对象、dtype 实例、字符串三种输入都能给出同一个规范名,
    且实测与原写法生成的数据逐位一致。
    """
    try:
        return numpy.dtype(dtype).name
    except TypeError:
        return str(dtype)


def build_recipe(dtype, shape, data_range, distribution: Optional[str], seed: int) -> dict:
    """组装一条可 JSON 序列化的配方。

    ``data_range`` 传**已解析**的区间（RandomData 内部把 None 换成默认上下界后
    的那份），避免两端各自填默认值时产生歧义。
    """
    return {
        "v": RECIPE_VERSION,
        "dtype": _dtype_name(dtype),
        "shape": [int(x) for x in shape],
        "data_range": [_jsonable(x) for x in (data_range or [])],
        "distribution": distribution or "uniform",
        "seed": int(seed),
    }


def regenerate(recipe: dict):
    """按配方重算出输入叶子。必须与客户端生成路径逐位一致。"""
    if int(recipe.get("v", 0)) != RECIPE_VERSION:
        raise ValueError(f"recipe version mismatch: {recipe.get('v')} != {RECIPE_VERSION}")
    from ttk.utilities.data import RandomData

    numpy.random.seed(int(recipe["seed"]))
    random_data = RandomData(recipe["dtype"], recipe["shape"], recipe["data_range"])
    return random_data.generate(recipe.get("distribution") or "uniform")


def regenerate_and_verify(recipes: list, digests: list) -> list:
    """重算全部叶子并逐片核对摘要；任一片对不上就抛，由调用方回 409。"""
    if len(recipes) != len(digests):
        raise ValueError(f"recipe/digest count mismatch: {len(recipes)} vs {len(digests)}")
    leaves = []
    for index, (recipe, expected) in enumerate(zip(recipes, digests)):
        array = regenerate(recipe)
        actual = digest_of(array)
        if actual != expected:
            raise ValueError(
                f"leaf {index} digest mismatch: expected {expected[:16]}… got {actual[:16]}… "
                f"(dtype={recipe.get('dtype')} shape={recipe.get('shape')})"
            )
        leaves.append(array)
    return leaves


def _jsonable(value):
    """把 numpy 标量/inf/nan 转成 json 模块能原样往返的形式。"""
    if isinstance(value, numpy.generic):
        value = value.item()
    return value
