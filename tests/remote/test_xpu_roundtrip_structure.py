# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""XU round-trip 结构保真看护。嵌套组合 ((A,B), C, None, (D,E)) 两个方向。"""
import io

import numpy as np

from ttk.remote.dispatcher import (
    _build_input_schema,
    _load_npz_outputs,
    _serialize_to_file,
)
from ttk.remote.server.executor import _outputs_to_numpy


def _sample():
    A = np.array([1.0, 2.0])
    B = np.array([3.0, 4.0])
    C = np.array([5.0])
    D = np.array([6.0, 7.0])
    E = np.array([8.0, 9.0])
    return A, B, C, D, E


def test_input_direction_client_to_server(tmp_path):
    """方向 1: client 序列化 → server match_params_v1 恢复嵌套+None。"""
    from ttk.remote.server.execution_container import match_params_v1
    A, B, C, D, E = _sample()
    inputs = [(A, B), C, None, (D, E)]
    names = ["in0", "in1", "in2", "in3"]
    schema = _build_input_schema(inputs, names)
    path = _serialize_to_file(inputs, dir=str(tmp_path))
    npz = np.load(path)
    flat = [npz[f"a{i}"] for i in range(sum(
        len(e["indices"]) if "indices" in e else (1 if e.get("index") is not None else 0)
        for e in schema))]
    named = match_params_v1(schema, flat)
    assert isinstance(named["in0"], list) and len(named["in0"]) == 2
    assert np.array_equal(named["in0"][0], A) and np.array_equal(named["in0"][1], B)
    assert np.array_equal(named["in1"], C)
    assert named["in2"] is None
    assert isinstance(named["in3"], list) and len(named["in3"]) == 2
    assert np.array_equal(named["in3"][0], D) and np.array_equal(named["in3"][1], E)


def test_output_direction_server_to_client():
    """方向 2: server _outputs_to_numpy → npz → client _load_npz_outputs 嵌套保真（同 dtype）。"""
    A, B, C, D, E = _sample()
    raw_outputs = ((A, B), C, None, (D, E))
    schema, arrays = _outputs_to_numpy(raw_outputs, "numpy")
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": o for i, o in enumerate(arrays)})
    buf.seek(0)
    loaded = _load_npz_outputs(buf, schema)
    # loaded[0] = [A, B], loaded[1] = C, loaded[2] = None, loaded[3] = [D, E]
    assert isinstance(loaded[0], list) and len(loaded[0]) == 2
    assert np.array_equal(loaded[0][0], A) and np.array_equal(loaded[0][1], B)
    assert np.array_equal(loaded[1], C)
    assert loaded[2] is None
    assert isinstance(loaded[3], list) and len(loaded[3]) == 2
    assert np.array_equal(loaded[3][0], D) and np.array_equal(loaded[3][1], E)
