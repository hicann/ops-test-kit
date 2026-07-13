"""XU round-trip 结构保真看护。嵌套组合 ((A,B), C, None, (D,E)) 两个方向 + 多 dtype 嵌套 + bf16/f8 wire。"""
import io
import numpy as np
import pytest

from ttk.remote.dispatcher import (
    _build_input_schema, _serialize_to_file, _load_npz_outputs,
)
from ttk.remote.server.executor import _outputs_to_numpy, _to_numpy_pair


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


def test_output_mixed_dtype_nested():
    """方向 2b: 多 dtype 嵌套 + None 保真（numpy 可表达 dtype：fp32/fp64/int32）。
    组合 ((A_f32,B_f64), C_int32, None, (D_f32,E_f64))——嵌套 × dtype × None 三因素同时。"""
    A = np.array([1.0, 2.0], dtype=np.float32)
    B = np.array([3.0, 4.0], dtype=np.float64)
    C = np.array([5, 6, 7], dtype=np.int32)
    D = np.array([8.0, 9.0], dtype=np.float32)
    E = np.array([10.0], dtype=np.float64)
    raw_outputs = ((A, B), C, None, (D, E))
    schema, arrays = _outputs_to_numpy(raw_outputs, "numpy")
    assert len(schema) == 4                      # 顶层 4 slot
    assert schema[2] == {"index": None, "dtype": None}   # None slot
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": o for i, o in enumerate(arrays)})
    buf.seek(0)
    loaded = _load_npz_outputs(buf, schema)
    assert isinstance(loaded[0], list) and loaded[0][0].dtype == np.float32 and loaded[0][1].dtype == np.float64
    assert loaded[1].dtype == np.int32
    assert loaded[2] is None
    assert isinstance(loaded[3], list) and loaded[3][0].dtype == np.float32 and loaded[3][1].dtype == np.float64
    assert np.array_equal(loaded[0][0], A) and np.array_equal(loaded[1], C)
    assert np.array_equal(loaded[3][1], E)


def test_output_bfloat16():
    """方向 3: bfloat16 wire int16 + utilities reinterpret。"""
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    orig = np.array([1.5, -0.25, 3.0], dtype=ml_dtypes.bfloat16)
    schema, arrays = _outputs_to_numpy([orig], "numpy")
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": o for i, o in enumerate(arrays)})
    buf.seek(0)
    loaded = _load_npz_outputs(buf, schema)
    assert np.allclose(np.asarray(loaded[0], dtype=np.float32),
                       np.asarray(orig, dtype=np.float32), rtol=1e-2, atol=1e-2)


def test_output_float8():
    """方向 4: float8 wire uint8 + utilities reinterpret。"""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    orig = torch.tensor([1.0, 2.0, 0.5, 4.0], dtype=torch.float8_e5m2)
    schema, arrays = _outputs_to_numpy([orig], "torch")
    assert arrays[0].dtype == np.uint8
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": o for i, o in enumerate(arrays)})
    buf.seek(0)
    loaded = _load_npz_outputs(buf, schema)
    assert np.allclose(np.asarray(loaded[0], dtype=np.float32),
                       orig.to(torch.float32).numpy(), rtol=0.2, atol=0.2)
