"""executor.execute_request — direct in-process unit tests (DATA path)."""
import io
import json
import os
import subprocess
import threading
import traceback
import sys
import time

import http.client

import numpy as np
import pytest
import functools
import importlib.util


@functools.lru_cache(maxsize=1)
def _tf_probe_ok():
    """tensorflow 是否能安全 import（子进程隔离 CI 的 C 扩展 segfault，importorskip 捕获不到）。"""
    if importlib.util.find_spec("tensorflow") is None:
        return False
    try:
        return subprocess.run([sys.executable, "-c", "import tensorflow"],
                              capture_output=True, timeout=90).returncode == 0
    except Exception:
        return False


def _npz(path, **arrs):
    np.savez_compressed(str(path),
                        **{f"a{i}": v for i, v in enumerate(arrs.values())})
    return str(path)


def _kw(**over):
    base = dict(tenant_sync_dir="", exec_type="api", provider="numpy", api=None,
                spec_module=None, spec_class=None, mode=1,
                input_schema=[], attrs={}, tmp_in_path=None, input_count=0,
                device_id=0, use_device=False, output_dir=None)
    base.update(over)
    return base


class TestApiData:
    def test_numpy_add(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]), x2=np.array([3.0, 4.0]))
        env = executor.execute_request(**_kw(
            api="numpy.add", mode=DATA,
            input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
            tmp_in_path=inp, input_count=2, output_dir=str(tmp_path)))
        assert env["ok"] and env["http_status"] == 200 and env["output_count"] == 1
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"],
                                      np.array([4.0, 6.0]))

    def test_api_class_is_400(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        env = executor.execute_request(**_kw(api="numpy.ndarray", mode=DATA,
                                              output_dir=str(tmp_path)))
        assert env["ok"] is False and env["http_status"] == 400

    def test_api_missing_module_is_500(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        env = executor.execute_request(**_kw(
            api="definitely_not_a_module_xyz.fn", mode=DATA, output_dir=str(tmp_path)))
        assert env["ok"] is False and env["http_status"] == 500
        assert env["missing"] is None       # not a syncable 424


class TestSpecData:
    def _write(self, path, body):
        path.write_text(body)

    def test_mode_b_inputs_in_call(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        self._write(tmp_path / "addb.py",
                    "class AddImpl:\n"
                    "    def __init__(self, **kw): pass\n"
                    "    def __call__(self, x1, x2): return [x1 + x2]\n"
                    "class AddTestSpec:\n"
                    "    third_party = {'numpy': AddImpl}\n")
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]), x2=np.array([5.0]))
        env = executor.execute_request(**_kw(
            exec_type="spec", spec_module="addb", spec_class="AddTestSpec",
            mode=DATA, provider="numpy",
            input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
            tmp_in_path=inp, input_count=2, tenant_sync_dir=str(tmp_path),
            output_dir=str(tmp_path)))
        assert env["ok"]
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"], np.array([6.0]))

    def test_mode_a_inputs_in_init(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        self._write(tmp_path / "adda.py",
                    "class AddImpl:\n"
                    "    def __init__(self, x1, x2, **kw):\n"
                    "        self.x1, self.x2 = x1, x2\n"
                    "    def __call__(self):\n"
                    "        return [self.x1 + self.x2]\n"
                    "class AddTestSpec:\n"
                    "    third_party = {'numpy': AddImpl}\n")
        inp = _npz(tmp_path / "in.npz", x1=np.array([2.0]), x2=np.array([10.0]))
        env = executor.execute_request(**_kw(
            exec_type="spec", spec_module="adda", spec_class="AddTestSpec",
            mode=DATA, provider="numpy",
            input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
            tmp_in_path=inp, input_count=2, tenant_sync_dir=str(tmp_path),
            output_dir=str(tmp_path)))
        assert env["ok"]
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"], np.array([12.0]))

    def test_missing_spec_module_is_424(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        env = executor.execute_request(**_kw(
            exec_type="spec", spec_module="no_such_spec", spec_class="X",
            mode=DATA, provider="numpy", tenant_sync_dir=str(tmp_path),
            output_dir=str(tmp_path)))
        assert env["ok"] is False and env["http_status"] == 424
        assert env["missing"] == "no_such_spec"

    def test_unknown_param_is_400(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server import executor
        self._write(tmp_path / "bad.py",
                    "class BadImpl:\n"
                    "    def __call__(self, x1, bogus): return [x1]\n"
                    "class BadTestSpec:\n"
                    "    third_party = {'numpy': BadImpl}\n")
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]))
        env = executor.execute_request(**_kw(
            exec_type="spec", spec_module="bad", spec_class="BadTestSpec",
            mode=DATA, provider="numpy",
            input_schema=[{"name": "x1", "index": 0}],
            tmp_in_path=inp, input_count=1, tenant_sync_dir=str(tmp_path),
            output_dir=str(tmp_path)))
        assert env["ok"] is False and env["http_status"] == 400
        assert "bogus" in env["error"]

    def test_op_failure_error_is_sanitized_no_traceback(self, tmp_path):
        """Security (OWASP Improper Error Handling): a failing op's client error
        is TYPE+MESSAGE only — no traceback, no server FS path leaks over the
        wire / into xpu-metrics. Full traceback stays server-side
        (logging.exception). _client_error is the single control point."""
        from ttk.remote import DATA
        from ttk.remote.server import executor
        self._write(tmp_path / "boom.py",
                    "class BoomImpl:\n"
                    "    def __call__(self, x1): raise RuntimeError('op exploded')\n"
                    "class BoomSpec:\n"
                    "    third_party = {'numpy': BoomImpl}\n")
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]))
        env = executor.execute_request(**_kw(
            exec_type="spec", spec_module="boom", spec_class="BoomSpec",
            mode=DATA, provider="numpy",
            input_schema=[{"name": "x1", "index": 0}],
            tmp_in_path=inp, input_count=1, tenant_sync_dir=str(tmp_path),
            output_dir=str(tmp_path)))
        assert env["ok"] is False and env["http_status"] == 500
        assert env["error"] == "RuntimeError: op exploded"
        assert "Traceback" not in env["error"]
        assert ".py" not in env["error"]          # no server/spec file path leaked


class TestViaSubprocess:
    """_run_in_subprocess: execute_request in a real fresh child process."""

    def test_numpy_add_runs_in_child_process(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server.xpu_server import _run_in_subprocess
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]), x2=np.array([3.0, 4.0]))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        env = _run_in_subprocess(_kw(
            api="numpy.add", mode=DATA,
            input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
            tmp_in_path=inp, input_count=2, output_dir=str(out_dir)), deadline=60)
        assert env["ok"] and env["output_count"] == 1
        np.testing.assert_array_equal(np.load(env["output_path"])["a0"],
                                      np.array([4.0, 6.0]))

    def test_child_crash_returns_500(self, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.server.xpu_server import _run_in_subprocess
        (tmp_path / "crashfix.py").write_text(
            "import os\n"
            "class CrashImpl:\n"
            "    def __call__(self, x1): os._exit(139)\n"
            "class CrashTestSpec:\n"
            "    third_party = {'numpy': CrashImpl}\n")
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0]))
        env = _run_in_subprocess(_kw(
            exec_type="spec", spec_module="crashfix",
            spec_class="CrashTestSpec", mode=DATA, provider="numpy",
            input_schema=[{"name": "x1", "index": 0}],
            tmp_in_path=inp, input_count=1, tenant_sync_dir=str(tmp_path),
            output_dir=str(tmp_path)), deadline=60)
        assert env["ok"] is False and env["http_status"] == 500
        assert "139" in env["error"]


class TestPerfPath:
    """PERF timing: CPU NA-fallback + simulated CUDA timing (no GPU needed)."""

    def test_data_perf_cpu_marks_na(self, tmp_path):
        from ttk.remote import DATA, PERF
        from ttk.remote.server import executor
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]), x2=np.array([3.0, 4.0]))
        env = executor.execute_request(**_kw(
            api="numpy.add", mode=DATA | PERF,
            input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
            tmp_in_path=inp, input_count=2, output_dir=str(tmp_path)))
        assert env["ok"] and env["output_count"] == 1
        perf = env["perf"]
        assert perf["device_us"] == "NA" and perf["peak_memory_mb"] == "NA"

    def test_bfloat16_via_declared_dtype(self):
        # numpy has no native bfloat16; over the wire it's raw int16 bits. The
        # schema declares dtype='bfloat16', so the server reinterprets correctly
        # (no itemsize guessing). No ml_dtypes needed on the server side.
        import torch
        from ttk.remote.server import executor
        orig = torch.tensor([1.5, -2.25, 3.0], dtype=torch.bfloat16)
        wire = orig.view(torch.int16).numpy()          # the wire form (int16 bits)
        out = executor._to_vendor_tensor(wire, "torch", "cpu", "bfloat16")
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, orig)

    def test_non_bf16_does_not_get_bf16_view(self):
        # a plain int16 array must NOT be misinterpreted as bfloat16
        import torch
        from ttk.remote.server import executor
        wire = np.array([1, 2, 3], dtype=np.int16)
        out = executor._to_vendor_tensor(wire, "torch", "cpu", "int16")
        assert out.dtype == torch.int16
        assert torch.equal(out, torch.tensor([1, 2, 3], dtype=torch.int16))

    def test_tf_input_pinned_to_assigned_device(self, monkeypatch):
        """tf inputs pinned to the assigned device (multi-device correctness).
        Verifies tf.device(device_str) is entered for a device and skipped
        for cpu."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow not importable (crashes on import)")
        tf = pytest.importorskip("tensorflow")
        from ttk.remote.server import executor
        recorded = []

        class _SpyCtx:
            def __init__(self, d): self.d = d
            def __enter__(self): recorded.append(self.d); return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(tf, "device", lambda d: _SpyCtx(d))
        executor._to_vendor_tensor(np.array([1.0]), "tf", "/device:GPU:2")
        assert "/device:GPU:2" in recorded       # device_str -> pinned
        recorded.clear()
        executor._to_vendor_tensor(np.array([1.0]), "tf", "cpu")
        assert recorded == []                    # cpu -> no pin (TF default is cpu)

    def test_bfloat16_output_stored_as_int16_bits(self):
        # numpy can't hold bfloat16, so a bf16 output is shipped as int16 bits
        # with the semantic dtype declared -> client reinterprets.
        import torch
        from ttk.remote.server import executor
        orig = torch.tensor([1.5, -2.25, 3.0], dtype=torch.bfloat16)
        arr, dt = executor._to_numpy_pair(orig, "torch")
        assert dt == "bfloat16"
        assert arr.dtype == np.int16
        np.testing.assert_array_equal(arr, orig.view(torch.int16).numpy())

    def test_bfloat16_output_reinterpret_roundtrip(self):
        pytest.importorskip("ml_dtypes")   # client side reinterprets via ml_dtypes
        import torch
        from ttk.remote.server import executor
        from ttk.remote.dispatcher import _reinterpret_dtype
        orig = torch.tensor([1.5, -2.25, 3.0], dtype=torch.bfloat16)
        arr, dt = executor._to_numpy_pair(orig, "torch")
        back = _reinterpret_dtype(arr, dt)
        np.testing.assert_array_equal(
            back.astype(np.float32), orig.to(torch.float32).numpy())


class TestOutputSerialization:
    """_outputs_to_numpy nested schema + float8 wire (Task 1: XU server fix).

    Locks the new server-side serialization contract that cross_check (3rd-party
    comparison) relies on: top-level slots preserved (not flattened), None slots
    -> {index:null}, tensor-list slots -> {indices:[...]}, float8 -> uint8 bits.
    """

    def test_float8_output_stored_as_uint8_bits(self):
        # numpy has no native float8; shipped as raw uint8 bits with the float8
        # dtype name declared (bit-preserving, like bfloat16->int16).
        import torch
        from ttk.remote.server import executor
        orig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float8_e4m3fn)
        arr, dt = executor._to_numpy_pair(orig, "torch")
        assert dt == "float8_e4m3fn"
        assert arr.dtype == np.uint8
        np.testing.assert_array_equal(arr, orig.view(torch.uint8).numpy())

    def test_float8_e5m2_dtype_name_preserved(self):
        import torch
        from ttk.remote.server import executor
        orig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float8_e5m2)
        arr, dt = executor._to_numpy_pair(orig, "torch")
        assert dt == "float8_e5m2"
        assert arr.dtype == np.uint8

    def test_outputs_to_numpy_flat_tensors_schema(self):
        # N top-level tensor slots -> schema [{index:i, dtype}, ...]; arrays are
        # the leaf arrays in npz order.
        import torch
        from ttk.remote.server import executor
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0, 5.0])
        schema, arrays = executor._outputs_to_numpy([a, b], "torch")
        assert schema == [{"index": 0, "dtype": "float32"},
                          {"index": 1, "dtype": "float32"}]
        assert len(arrays) == 2
        np.testing.assert_array_equal(arrays[0], [1.0, 2.0])

    def test_outputs_to_numpy_none_slot_index_null(self):
        # A None top-level slot (op output placeholder) -> {index:null, dtype:null}
        # and contributes NO leaf array.
        import torch
        from ttk.remote.server import executor
        a = torch.tensor([1.0, 2.0])
        schema, arrays = executor._outputs_to_numpy([a, None], "torch")
        assert schema == [{"index": 0, "dtype": "float32"},
                          {"index": None, "dtype": None}]
        assert len(arrays) == 1   # None slot contributes no leaf

    def test_outputs_to_numpy_tensor_list_slot_indices(self):
        # A tensor-list slot -> {indices:[leaf indices], dtype}; leaves appended
        # to the flat arrays list.
        import torch
        from ttk.remote.server import executor
        single = torch.tensor([1.0])
        lst = [torch.tensor([2.0, 3.0]), torch.tensor([4.0])]
        schema, arrays = executor._outputs_to_numpy([single, lst], "torch")
        assert schema == [{"index": 0, "dtype": "float32"},
                          {"indices": [1, 2], "dtype": "float32"}]
        assert len(arrays) == 3   # 1 (single) + 2 (list leaves)

    def test_outputs_to_numpy_single_output_wrapped(self):
        # A bare non-list/tuple output is wrapped into a single-slot schema.
        from ttk.remote.server import executor
        schema, arrays = executor._outputs_to_numpy(
            np.array([1.0, 2.0]), "numpy")
        assert schema == [{"index": 0, "dtype": "float64"}]
        assert len(arrays) == 1

    def test_ok_carries_schema_kwarg(self):
        # _ok now carries schema in the envelope (X-Output-Schema wire source).
        from ttk.remote.server import executor
        schema = [{"index": 0, "dtype": "float32"}]
        env = executor._ok("/p", 1, [[2]], dtypes=None, perf=None,
                           api="torch.add", schema=schema)
        assert env["ok"] is True
        assert env["schema"] == schema

    def test_ok_schema_defaults_none(self):
        from ttk.remote.server import executor
        env = executor._ok("/p", 1, [[2]], dtypes=None, perf=None)
        assert env["schema"] is None

    def test_handle_run_envelope_carries_schema(self, tmp_path):
        # End-to-end through execute_request: a 2-output op produces an envelope
        # whose schema has 2 top-level slots and whose npz has 2 leaf arrays.
        from ttk.remote import DATA
        from ttk.remote.server import executor
        inp = _npz(tmp_path / "in.npz", x1=np.array([1.0, 2.0]),
                   x2=np.array([3.0, 4.0]))
        env = executor.execute_request(**_kw(
            api="numpy.add", mode=DATA,
            input_schema=[{"name": "x1", "index": 0}, {"name": "x2", "index": 1}],
            tmp_in_path=inp, input_count=2, output_dir=str(tmp_path)))
        assert env["ok"]
        assert env["schema"] == [{"index": 0, "dtype": "float64"}]
        assert env["output_count"] == 1   # len(schema) = top-level slots
        npz = np.load(env["output_path"])
        assert sorted(npz.files) == ["a0"]


class TestDeviceHelpers:
    """_device_available + _device_time (spec §4.3.2 / §4.3.4)."""

    def test_device_available_torch(self, monkeypatch):
        """torch: getattr(torch, torch_lib).is_available() drives availability."""
        import torch
        from unittest.mock import MagicMock
        from ttk.remote.server import executor
        fake = MagicMock()
        fake.is_available.return_value = True
        monkeypatch.setattr(torch, "cuda", fake)
        assert executor._device_available("torch", {"torch_lib": "cuda"}) is True

    def test_device_available_tf_missing_type(self):
        """tf without tf_device_type -> False (graceful degrade, not KeyError)."""
        from ttk.remote.server import executor
        assert executor._device_available("tf", {}) is False

    def test_device_available_tf_with_type(self, monkeypatch):
        """tf with tf_device_type -> list_physical_devices(type) drives it."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TF not installed")
        from ttk.remote.server import executor
        monkeypatch.setattr(
            tf.config, "list_physical_devices", lambda t: [object()])
        assert executor._device_available("tf", {"tf_device_type": "GPU"}) is True

    def test_device_time_zero_not_falsy(self):
        """Candidate1=0.0 must NOT be skipped (is not None, not truthiness).

        Regression guard: a naive ``or`` fallback would treat 0.0 as missing
        and wrongly return candidate2. 0.0 is a valid self_device_time_total.
        """
        from types import SimpleNamespace
        from ttk.remote.server import executor
        evt = SimpleNamespace(self_device_time_total=0.0,
                              self_cuda_time_total=999.0)
        assert executor._device_time(evt, device="cuda") == 0.0

    def test_device_time_fallback_to_candidate2(self):
        """Candidate1 None -> candidate2 self_{device}_time_total."""
        from types import SimpleNamespace
        from ttk.remote.server import executor
        evt = SimpleNamespace(self_device_time_total=None,
                              self_mlu_time_total=42.0)
        assert executor._device_time(evt, device="mlu") == 42.0

    def test_device_time_missing_both_defaults_zero(self):
        """Both candidates absent -> 0.0 (then filtered to NA upstream)."""
        from types import SimpleNamespace
        from ttk.remote.server import executor
        evt = SimpleNamespace()
        assert executor._device_time(evt, device="cuda") == 0.0

    def test_profiler_missing_activities_raises(self, monkeypatch):
        """Missing torch_profiler/activities -> RuntimeError (server config error
        -> 500), NOT an NA degrade. Config errors must surface, not be masked."""
        from ttk.remote.server import executor
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)
        profile = {"torch_lib": "cuda", "torch_profiler": {}}  # no activities
        with pytest.raises(RuntimeError, match="activities"):
            executor._run_perf(lambda **kw: None, {}, {}, "torch", 0, True,
                               profile=profile, runtime=3)

    def test_profiler_unknown_activity_raises(self, monkeypatch):
        """Unknown ProfilerActivity name -> RuntimeError (-> 500)."""
        import torch
        from ttk.remote.server import executor
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)
        # patch ProfilerActivity so only CUDA exists; BOGUS is unknown
        from types import SimpleNamespace
        monkeypatch.setattr(torch.profiler, "ProfilerActivity",
                            SimpleNamespace(CUDA="cuda"))
        profile = {"torch_lib": "cuda",
                   "torch_profiler": {"activities": ["BOGUS"]}}
        with pytest.raises(RuntimeError, match="unknown ProfilerActivity"):
            executor._run_perf(lambda **kw: None, {}, {}, "torch", 0, True,
                               profile=profile, runtime=3)


class TestPerfMeasurement:
    """v2 perf measurement: profiler-based device_us (μs, 3 sig figs)."""
    import torch  # module-level import: monkeypatch patches the GLOBAL torch

    # profiles consumed by _run_perf (now profile-driven, spec §4.3.2/§4.3.4).
    # torch_profiler.activities must match the ProfilerActivity attrs the test
    # patches onto torch.profiler.ProfilerActivity (CUDA here).
    _TORCH_PROFILE = {"torch_lib": "cuda",
                      "torch_profiler": {"activities": ["CUDA"]}}
    _TF_PROFILE = {"torch_lib": "cuda", "tf_device_type": "GPU"}

    # --- 11 core tests ---

    def test_torch_device_us_via_profiler(self, monkeypatch):
        """torch.profiler key_averages -> device_us = total/runtime."""
        import torch
        from types import SimpleNamespace
        from ttk.remote.server import executor

        # torch 2.7+ uses self_device_time_total (older: self_cuda_time_total)
        fake_avg = SimpleNamespace(self_device_time_total=450.0,
                                   self_cuda_time_total=450.0)
        fake_prof = SimpleNamespace(key_averages=lambda: [fake_avg],
                                    step=lambda: None)

        class FakeProfileCtx:
            def __enter__(self): return fake_prof
            def __exit__(self, *a): return False

        # patch GLOBAL torch.profiler (NOT executor.torch — no module-level import)
        monkeypatch.setattr(torch.profiler, "profile", lambda **kw: FakeProfileCtx())
        monkeypatch.setattr(torch.profiler, "ProfilerActivity",
                            SimpleNamespace(CUDA="cuda"))
        monkeypatch.setattr(torch.profiler, "schedule", lambda **kw: None)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)
        assert perf["device_us"] == 150.0  # 450/3

    def test_torch_profiler_failure_na(self, monkeypatch):
        """Profiler raises -> device_us=NA."""
        import torch
        from ttk.remote.server import executor

        def boom(**kw): raise AssertionError("No CUDA")
        monkeypatch.setattr(torch.profiler, "profile", boom)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)
        assert perf["device_us"] == "NA"

    def test_op_failure_propagates_not_masked_as_na(self, monkeypatch):
        """Op execution failure inside the profiler pass must PROPAGATE (-> FAIL),
        not turn into PASS + NA/NA. Uses a fake profiler ctx so the op is reached
        without real CUDA.
        """
        import torch
        from types import SimpleNamespace
        from ttk.remote.server import executor

        fake_prof = SimpleNamespace(step=lambda: None, key_averages=lambda: [])

        class Ctx:
            def __enter__(self): return fake_prof
            def __exit__(self, *a): return False
        monkeypatch.setattr(torch.profiler, "profile", lambda **kw: Ctx())
        monkeypatch.setattr(torch.profiler, "ProfilerActivity",
                            SimpleNamespace(CUDA="cuda"))
        monkeypatch.setattr(torch.profiler, "schedule", lambda **kw: None)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        def boom(**kw): raise RuntimeError("op exploded")
        with pytest.raises(RuntimeError, match="op exploded"):
            executor._run_perf(boom, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)

    def test_tf_op_failure_propagates_not_masked_as_na(self, monkeypatch):
        """TF op execution failure must PROPAGATE (-> FAIL) — symmetric with the
        torch pass. The TF pass has a different shape (warmup outside the
        start/stop window), so the contract is locked independently."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TF not installed")
        from ttk.remote.server import executor

        prof_mod = type("M", (), {})()
        prof_mod.start = lambda d: None
        prof_mod.stop = lambda: None
        monkeypatch.setattr(tf.profiler, "experimental", prof_mod)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        def boom(**kw): raise RuntimeError("op exploded")
        with pytest.raises(RuntimeError, match="op exploded"):
            executor._run_perf(boom, {}, {}, "tf", 0, True, profile=self._TF_PROFILE, runtime=3)

    def test_profiler_start_failure_is_logged(self, monkeypatch, caplog):
        """Profiler-infra failure -> device_us=NA AND a server-side log line
        (profiler mechanics only)."""
        import torch
        import logging as _logging
        from ttk.remote.server import executor

        def boom(**kw): raise AssertionError("no CUDA")
        monkeypatch.setattr(torch.profiler, "profile", boom)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        with caplog.at_level(_logging.ERROR):
            outputs, perf = executor._run_perf(
                noop, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)
        assert perf["device_us"] == "NA"
        assert any("torch.profiler start failed" in r.message
                   for r in caplog.records)

    def test_profiler_readout_failure_keeps_outputs(self, monkeypatch):
        """Profiler readout (key_averages) failure -> device_us=NA but the op's
        OUTPUTS survive. In DATA|PERF mode the op result must not be lost to a
        flaky profiler (the 'don't over-fail' half of the error policy)."""
        import torch
        from types import SimpleNamespace
        from ttk.remote.server import executor

        marker = object()
        fake_prof = SimpleNamespace(
            step=lambda: None,
            key_averages=lambda: (_ for _ in ()).throw(RuntimeError("readout boom")))

        class Ctx:
            def __enter__(self): return fake_prof
            def __exit__(self, *a): return False
        monkeypatch.setattr(torch.profiler, "profile", lambda **kw: Ctx())
        monkeypatch.setattr(torch.profiler, "ProfilerActivity",
                            SimpleNamespace(CUDA="cuda"))
        monkeypatch.setattr(torch.profiler, "schedule", lambda **kw: None)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        op = lambda **kw: marker
        outputs, perf = executor._run_perf(op, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)
        assert perf["device_us"] == "NA"
        assert outputs is marker

    def test_tf_device_us_via_xplane(self, monkeypatch, tmp_path):
        """TF xplane.pb parse -> device_us = sum(duration_ps/1e6)/runtime."""
        from ttk.remote.server import executor

        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        try:
            from tensorflow.core.profiler.protobuf import xplane_pb2
            import tensorflow as tf
        except ImportError:
            pytest.skip("TF not installed")

        xs = xplane_pb2.XSpace()
        plane = xs.planes.add()
        plane.name = "/device:GPU:0"
        line = plane.lines.add()
        for _ in range(3):
            ev = line.events.add()
            ev.duration_ps = 13_000_000_000  # 13ms = 13000us each
        host_plane = xs.planes.add()
        host_plane.name = "/host:CPU"  # should be excluded
        host_line = host_plane.lines.add()
        host_ev = host_line.events.add()
        host_ev.duration_ps = 999999999

        pb_path = tmp_path / "plugins" / "profile" / "test"
        pb_path.mkdir(parents=True)
        pb_file = pb_path / "host.xplane.pb"
        pb_file.write_bytes(xs.SerializeToString())

        # patch GLOBAL tf.profiler.experimental + tempfile.mkdtemp
        prof_mod = type("M", (), {})()
        prof_mod.start = lambda d: None
        prof_mod.stop = lambda: None
        monkeypatch.setattr(tf.profiler, "experimental", prof_mod)
        import tempfile as _tempfile
        monkeypatch.setattr(_tempfile, "mkdtemp", lambda **kw: str(tmp_path))
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "tf", 0, True, profile=self._TF_PROFILE, runtime=3)
        expected = float(f"{39000.0 / 3:.3g}")
        assert perf["device_us"] == expected
        assert perf["device_us"] != "NA"

    def test_tf_xplane_not_found_na(self, monkeypatch, tmp_path):
        """Empty logdir -> NA."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        import tensorflow as tf
        from ttk.remote.server import executor

        prof_mod = type("M", (), {})()
        prof_mod.start = lambda d: None
        prof_mod.stop = lambda: None
        monkeypatch.setattr(tf.profiler, "experimental", prof_mod)
        import tempfile as _tempfile
        monkeypatch.setattr(_tempfile, "mkdtemp", lambda **kw: str(tmp_path))
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "tf", 0, True, profile=self._TF_PROFILE, runtime=3)
        assert perf["device_us"] == "NA"

    def test_cpu_mode_device_us_na(self):
        """use_device=False -> device_us=NA."""
        from ttk.remote.server import executor
        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "torch", 0, False, runtime=3)
        assert perf["device_us"] == "NA"

    def test_sig_figs_3(self):
        """3 significant figures via %.3g."""
        assert float(f"{13.4567:.3g}") == 13.5
        assert float(f"{0.01234:.3g}") == 0.0123
        assert float(f"{999.6:.3g}") == 1000.0

    def test_timing_peak_split(self, monkeypatch):
        """Profiler pass: no empty_cache. Peak pass: empty_cache not called
        (production never calls empty_cache in either pass)."""
        import torch
        from types import SimpleNamespace
        from ttk.remote.server import executor

        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache",
                            lambda *a, **kw: calls.append("empty_cache"))
        fake_prof = SimpleNamespace(
            key_averages=lambda: [SimpleNamespace(self_device_time_total=1.0)],
            step=lambda: None)

        class Ctx:
            def __enter__(self): return fake_prof
            def __exit__(self, *a): return False
        monkeypatch.setattr(torch.profiler, "profile", lambda **kw: Ctx())
        monkeypatch.setattr(torch.profiler, "ProfilerActivity",
                            SimpleNamespace(CUDA="cuda"))
        monkeypatch.setattr(torch.profiler, "schedule", lambda **kw: None)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        executor._run_perf(noop, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)
        # empty_cache is not invoked in the profiler pass nor the peak pass
        assert calls == []

    # --- 5 supplemental guards (mock variants) ---

    def test_tf_corrupt_xplane_na(self, monkeypatch, tmp_path):
        """TF corrupt .xplane.pb (random bytes) -> NA."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        import tensorflow as tf
        from ttk.remote.server import executor

        pb_path = tmp_path / "junk.xplane.pb"
        pb_path.write_bytes(b"\x00\x01\x02not a valid proto\xff\xfe")

        prof_mod = type("M", (), {})()
        prof_mod.start = lambda d: None
        prof_mod.stop = lambda: None
        monkeypatch.setattr(tf.profiler, "experimental", prof_mod)
        import tempfile as _tempfile
        monkeypatch.setattr(_tempfile, "mkdtemp", lambda **kw: str(tmp_path))
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "tf", 0, True, profile=self._TF_PROFILE, runtime=3)
        assert perf["device_us"] == "NA"

    def test_tf_start_stop_raise_na(self, monkeypatch, tmp_path):
        """TF start/stop raise -> device_us=NA."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        import tensorflow as tf
        from ttk.remote.server import executor

        prof_mod = type("M", (), {})()
        prof_mod.start = lambda d: (_ for _ in ()).throw(RuntimeError("boom"))
        prof_mod.stop = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        monkeypatch.setattr(tf.profiler, "experimental", prof_mod)
        import tempfile as _tempfile
        monkeypatch.setattr(_tempfile, "mkdtemp", lambda **kw: str(tmp_path))
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "tf", 0, True, profile=self._TF_PROFILE, runtime=3)
        assert perf["device_us"] == "NA"

    def test_peak_pass_raise_keeps_device_us(self, monkeypatch):
        import torch
        from types import SimpleNamespace
        from ttk.remote.server import executor

        fake_prof = SimpleNamespace(
            key_averages=lambda: [SimpleNamespace(self_device_time_total=300.0,
                                                 self_cuda_time_total=300.0)],
            step=lambda: None)

        class Ctx:
            def __enter__(self): return fake_prof
            def __exit__(self, *a): return False
        monkeypatch.setattr(torch.profiler, "profile", lambda **kw: Ctx())
        monkeypatch.setattr(torch.profiler, "ProfilerActivity",
                            SimpleNamespace(CUDA="cuda"))
        monkeypatch.setattr(torch.profiler, "schedule", lambda **kw: None)
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)
        # peak pass: reset_peak raises
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("peak boom")))

        noop = lambda **kw: None
        outputs, perf = executor._run_perf(
            noop, {}, {}, "torch", 0, True, profile=self._TORCH_PROFILE, runtime=3)
        assert perf["device_us"] == 100.0  # 300/3 unaffected
        assert perf["peak_memory_mb"] == "NA"

    def test_tf_logdir_rmtree_on_stop_raise(self, monkeypatch, tmp_path):
        """TF logdir rmtree runs in finally even if stop() raises."""
        if not _tf_probe_ok():
            pytest.skip("tensorflow crashes on import")
        import tensorflow as tf
        from ttk.remote.server import executor
        import shutil

        prof_mod = type("M", (), {})()
        prof_mod.start = lambda d: None
        prof_mod.stop = lambda: (_ for _ in ()).throw(RuntimeError("stop boom"))
        monkeypatch.setattr(tf.profiler, "experimental", prof_mod)

        rmtree_calls = []
        real_mkdtemp = __import__("tempfile").mkdtemp
        created = real_mkdtemp(prefix="tfptest_")

        import tempfile as _tempfile
        monkeypatch.setattr(_tempfile, "mkdtemp", lambda **kw: created)
        monkeypatch.setattr(shutil, "rmtree",
                            lambda p, **kw: rmtree_calls.append(str(p)))
        monkeypatch.setattr(executor, "_device_available", lambda p, prof: True)

        noop = lambda **kw: None
        executor._run_perf(noop, {}, {}, "tf", 0, True, profile=self._TF_PROFILE, runtime=3)
        assert any(created == c for c in rmtree_calls), \
            f"rmtree not called with {created}; calls={rmtree_calls}"


class TestRuntimeClamp:
    """Server-side X-Runtime clamp (spec §8 guard). Header is clamped to [1,100];
    non-numeric falls back to 3. Tests _clamp_runtime (extracted from _handle_run)."""

    def test_over_high_clamps_to_100(self):
        from ttk.remote.server.xpu_server import _clamp_runtime
        assert _clamp_runtime("200") == 100

    def test_zero_clamps_to_1(self):
        from ttk.remote.server.xpu_server import _clamp_runtime
        assert _clamp_runtime("0") == 1

    def test_non_numeric_falls_back_to_default_3(self):
        from ttk.remote.server.xpu_server import _clamp_runtime
        assert _clamp_runtime("abc") == 3

    def test_none_falls_back_to_default_3(self):
        # _get_header may return None in some paths; TypeError branch covers it
        from ttk.remote.server.xpu_server import _clamp_runtime
        assert _clamp_runtime(None) == 3

    def test_in_range_passes_through(self):
        from ttk.remote.server.xpu_server import _clamp_runtime
        assert _clamp_runtime("50") == 50


def _post_run(port, body, headers, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", "/v1/run", body=body, headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def _sync_module(port, tenant, rel_path, content):
    import base64
    import hashlib
    body = json.dumps({"files": {
        rel_path: {"content": base64.b64encode(content.encode()).decode(),
                   "hash": hashlib.sha256(content.encode()).hexdigest()}}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/sync", body=body,
              headers={"Content-Type": "application/json", "X-Tenant-ID": tenant})
    r = c.getresponse()
    r.read()
    c.close()
    return r.status


class TestServerRunViaSubprocess:
    """End-to-end: real non-dry-run server runs /v1/run in a child process."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory):
        sync = tmp_path_factory.mktemp("sync")
        tmp = tmp_path_factory.mktemp("tmp")
        
        # Create a config file for the server
        import yaml
        config_file = tmp_path_factory.mktemp("config") / "xpu_server.yaml"
        config_data = {
            "storage": {
                "sync_dir": str(sync),
                "tmp_dir": str(tmp),
            },
            "server": {
                "run_deadline_s": 60,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        proc = subprocess.Popen(
            [sys.executable, "-m", "server.xpu_server",
             "--port", "19150", "--devices", "cpu", "--config", str(config_file)],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(40):
            try:
                c = http.client.HTTPConnection("127.0.0.1", 19150, timeout=1)
                c.request("GET", "/v1/heartbeat")
                r = c.getresponse(); c.close()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, OSError):
                pass
            time.sleep(0.5)
        yield proc
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_run_numpy_add(self, server):
        from ttk.remote import DATA
        a0 = np.array([1.0, 2.0])
        a1 = np.array([3.0, 4.0])
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=a0, a1=a1)
        c = http.client.HTTPConnection("127.0.0.1", 19150, timeout=60)
        c.request("POST", "/v1/run", body=buf.getvalue(), headers={
            "X-Execution-Type": "api", "X-Provider": "numpy", "X-API": "numpy.add",
            "X-Mode": str(DATA),
            "X-Input-Schema": json.dumps([{"name": "x1", "index": 0},
                                          {"name": "x2", "index": 1}]),
            "X-Input-Count": "2", "X-Attrs": "{}", "X-Tenant-ID": "e2e1",
            "Content-Type": "application/octet-stream"})
        r = c.getresponse()
        data = r.read()
        c.close()
        assert r.status == 200
        np.testing.assert_array_equal(np.load(io.BytesIO(data))["a0"], a0 + a1)

    def test_crash_then_server_keeps_serving(self, server):
        from ttk.remote import DATA
        _sync_module(19150, "cr", "crashfix.py",
                     "import os\n"
                     "class CrashImpl:\n"
                     "    def __call__(self, x1): os._exit(139)\n"
                     "class CrashTestSpec:\n"
                     "    third_party = {'numpy': CrashImpl}\n")
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([1.0]))
        h = {"X-Execution-Type": "spec", "X-Provider": "numpy",
             "X-Spec-Module": "crashfix", "X-Spec-Class": "CrashTestSpec",
             "X-Mode": str(DATA),
             "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}]),
             "X-Input-Count": "1", "X-Attrs": "{}", "X-Tenant-ID": "cr",
             "Content-Type": "application/octet-stream"}
        s1, _ = _post_run(19150, buf.getvalue(), h)
        assert s1 == 500                       # child crashed

        addbuf = io.BytesIO()
        np.savez_compressed(addbuf, a0=np.array([1.0]), a1=np.array([2.0]))
        s2, b2 = _post_run(19150, addbuf.getvalue(), {
            "X-Execution-Type": "api", "X-Provider": "numpy", "X-API": "numpy.add",
            "X-Mode": str(DATA),
            "X-Input-Schema": json.dumps([{"name": "x1", "index": 0},
                                          {"name": "x2", "index": 1}]),
            "X-Input-Count": "2", "X-Attrs": "{}", "X-Tenant-ID": "cr2",
            "Content-Type": "application/octet-stream"})
        assert s2 == 200                        # server survived, fresh child
        np.testing.assert_array_equal(np.load(io.BytesIO(b2))["a0"], np.array([3.0]))

    def test_424_then_sync_then_success(self, server):
        from ttk.remote import DATA
        add_src = ("class AddImpl:\n"
                   "    def __call__(self, x1, x2): return [x1 + x2]\n"
                   "class AddTestSpec:\n"
                   "    third_party = {'numpy': AddImpl}\n")
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([2.0]), a1=np.array([3.0]))
        h = {"X-Execution-Type": "spec", "X-Provider": "numpy",
             "X-Spec-Module": "addfix", "X-Spec-Class": "AddTestSpec",
             "X-Mode": str(DATA),
             "X-Input-Schema": json.dumps([{"name": "x1", "index": 0},
                                          {"name": "x2", "index": 1}]),
             "X-Input-Count": "2", "X-Attrs": "{}", "X-Tenant-ID": "resync",
             "Content-Type": "application/octet-stream"}
        s1, b1 = _post_run(19150, buf.getvalue(), h)
        assert s1 == 424
        assert json.loads(b1).get("missing") in ("addfix", "addfix.py")

        assert _sync_module(19150, "resync", "addfix.py", add_src) == 200
        s3, b3 = _post_run(19150, buf.getvalue(), h)
        assert s3 == 200                        # fresh child reads the synced file
        np.testing.assert_array_equal(np.load(io.BytesIO(b3))["a0"], np.array([5.0]))

    def test_cross_tenant_isolation(self, server):
        # Two tenants sync the SAME module name with DIFFERENT impls; concurrent
        # requests must each get their own impl (fresh sys.modules per child).
        from ttk.remote import DATA
        _sync_module(19150, "ta", "sharedfix.py",
                     "class AddImpl:\n"
                     "    def __call__(self, x1, x2): return [x1 + x2]\n"
                     "class AddTestSpec:\n"
                     "    third_party = {'numpy': AddImpl}\n")
        _sync_module(19150, "tb", "sharedfix.py",
                     "class AddImpl:\n"
                     "    def __call__(self, x1, x2): return [x1 * x2]\n"
                     "class AddTestSpec:\n"
                     "    third_party = {'numpy': AddImpl}\n")
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([2.0]), a1=np.array([3.0]))

        def hdr(t):
            return {"X-Execution-Type": "spec", "X-Provider": "numpy",
                    "X-Spec-Module": "sharedfix", "X-Spec-Class": "AddTestSpec",
                    "X-Mode": str(DATA),
                    "X-Input-Schema": json.dumps([{"name": "x1", "index": 0},
                                                  {"name": "x2", "index": 1}]),
                    "X-Input-Count": "2", "X-Attrs": "{}", "X-Tenant-ID": t,
                    "Content-Type": "application/octet-stream"}

        results = []
        errors = []

        def fire(t):
            try:
                results.append((t, _post_run(19150, buf.getvalue(), hdr(t))))
            except Exception:
                errors.append((t, traceback.format_exc()))

        threads = []
        for _ in range(8):
            threads.append(threading.Thread(target=fire, args=("ta",)))
            threads.append(threading.Thread(target=fire, args=("tb",)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=70)

        assert not errors, f"fire raised: {errors}"
        assert len(results) == 16, f"results={len(results)} errors={errors}"
        for tenant, (status, data) in results:
            assert status == 200
            out = np.load(io.BytesIO(data))["a0"]
            expected = np.array([5.0]) if tenant == "ta" else np.array([6.0])
            np.testing.assert_array_equal(out, expected)

    def test_perf_mode_returns_x_perf_header(self, server):
        from ttk.remote import DATA, PERF
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([1.0]), a1=np.array([2.0]))
        c = http.client.HTTPConnection("127.0.0.1", 19150, timeout=60)
        c.request("POST", "/v1/run", body=buf.getvalue(), headers={
            "X-Execution-Type": "api", "X-Provider": "numpy", "X-API": "numpy.add",
            "X-Mode": str(DATA | PERF),
            "X-Input-Schema": json.dumps([{"name": "x1", "index": 0},
                                          {"name": "x2", "index": 1}]),
            "X-Input-Count": "2", "X-Attrs": "{}", "X-Tenant-ID": "perf1",
            "Content-Type": "application/octet-stream"})
        r = c.getresponse()
        perf_h = r.getheader("X-Perf")
        r.read()
        c.close()
        assert r.status == 200
        assert perf_h is not None
        perf = json.loads(perf_h)


class TestBackpressure:
    """MAX_CONCURRENT gate: overflow -> 503; control plane bypasses it."""

    @pytest.fixture(scope="class")
    def gate_server(self, tmp_path_factory):
        sync = tmp_path_factory.mktemp("sync_g")
        tmp = tmp_path_factory.mktemp("tmp_g")
        
        # Create a config file for the server
        import yaml
        config_file = tmp_path_factory.mktemp("config") / "xpu_server.yaml"
        config_data = {
            "storage": {
                "sync_dir": str(sync),
                "tmp_dir": str(tmp),
            },
            "server": {
                "max_concurrent": 1,
                "gate_wait_s": 0.3,
                "run_deadline_s": 60,
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        proc = subprocess.Popen(
            [sys.executable, "-m", "server.xpu_server",
             "--port", "19151", "--devices", "cpu", "--config", str(config_file)],
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(40):
            try:
                c = http.client.HTTPConnection("127.0.0.1", 19151, timeout=1)
                c.request("GET", "/v1/heartbeat")
                r = c.getresponse(); c.close()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, OSError):
                pass
            time.sleep(0.5)
        yield proc
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_gate_full_returns_503_and_control_plane_ok(self, gate_server):
        from ttk.remote import DATA
        slow = ("import time\n"
                "class SlowImpl:\n"
                "    def __call__(self, x1):\n"
                "        time.sleep(2.0)\n"
                "        return [x1]\n"
                "class SlowTestSpec:\n"
                "    third_party = {'numpy': SlowImpl}\n")
        _sync_module(19151, "g", "slowfix.py", slow)
        buf = io.BytesIO()
        np.savez_compressed(buf, a0=np.array([1.0]))
        h = {"X-Execution-Type": "spec", "X-Provider": "numpy",
             "X-Spec-Module": "slowfix", "X-Spec-Class": "SlowTestSpec",
             "X-Mode": str(DATA),
             "X-Input-Schema": json.dumps([{"name": "x1", "index": 0}]),
             "X-Input-Count": "1", "X-Attrs": "{}", "X-Tenant-ID": "g",
             "Content-Type": "application/octet-stream"}

        holder = {}

        def fire():
            holder["s"] = _post_run(19151, buf.getvalue(), h)[0]

        t = threading.Thread(target=fire)
        t.start()
        time.sleep(0.8)                       # let the holder acquire the 1 slot
        s2, _ = _post_run(19151, buf.getvalue(), h)
        assert s2 == 503                        # gate full, 0.3s wait elapses

        c = http.client.HTTPConnection("127.0.0.1", 19151, timeout=2)   # control plane
        c.request("GET", "/v1/heartbeat")
        r = c.getresponse(); r.read(); c.close()
        assert r.status == 200

        t.join(timeout=15)
        assert holder.get("s") == 200
