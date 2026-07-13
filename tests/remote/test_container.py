"""Container sandbox backend tests (no real docker — patches subprocess.run)."""
import json
import os
import subprocess
import sys
import time

import pytest


def _kwargs(tmp_path, with_tmp_in=True):
    req = tmp_path / "req"
    req.mkdir()
    return {
        "tenant_sync_dir": str(tmp_path / "sync"),
        "output_dir": str(req),
        "tmp_in_path": str(req / "body.npz") if with_tmp_in else None,
        "exec_type": "api", "provider": "torch", "profile": {"torch_lib": "cuda"},
        "api": "torch.add", "spec_module": None, "spec_class": None,
        "op_name": "add", "op_type": "Add",
        "mode": 1, "input_schema": [], "attrs": {"axis": -1},
        "input_count": 2, "device_id": 0, "use_device": False,
    }


class _Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _Runner:
    """Fake subprocess.run: records the docker cmd + reads the kwargs.json written."""
    def __init__(self, envelope, captured_paths):
        self.envelope = envelope
        self.captured_paths = captured_paths
        self.cmd = None
        self.ckwargs = None

    def __call__(self, cmd, **kw):
        self.cmd = cmd
        with open(self.captured_paths[-1]) as f:
            self.ckwargs = json.load(f)
        return _Completed(stdout=json.dumps(self.envelope))


def _patch_mkstemp(container, monkeypatch, holder):
    real = container.tempfile.mkstemp

    def cap(*a, **k):
        fd, p = real(*a, **k)
        holder.append(p)
        return fd, p
    monkeypatch.setattr(container.tempfile, "mkstemp", cap)


class TestRunInContainerCmd:
    def test_cpu_cmd_flags_and_path_rewrite(self, tmp_path, monkeypatch):
        from ttk.remote.server import container
        kwargs = _kwargs(tmp_path)
        cap = []
        _patch_mkstemp(container, monkeypatch, cap)
        runner = _Runner({"ok": True, "http_status": 200, "output_path": "/work/out.npz"}, cap)
        monkeypatch.setattr(container.subprocess, "run", runner)
        env = container._run_in_container(
            kwargs, deadline=time.monotonic() + 60, image="img", use_device=False)

        cmd = runner.cmd
        assert f"{kwargs['output_dir']}:/work" in cmd
        assert f"{kwargs['tenant_sync_dir']}:/sync:ro" in cmd
        assert "--tmpfs" in cmd and "/tmp" in cmd
        assert "HOME=/tmp" in cmd
        assert "--read-only" in cmd
        assert "--cap-drop=ALL" in cmd
        assert "--gpus" not in cmd
        assert "--user" in cmd
        ck = runner.ckwargs
        assert ck["output_dir"] == "/work"
        assert ck["tenant_sync_dir"] == "/sync"
        assert ck["tmp_in_path"] == "/work/body.npz"
        assert ck["api"] == "torch.add"            # non-path field untouched
        assert ck["op_name"] == "add"              # new non-path kwargs survive the allowlist rewrite
        assert ck["op_type"] == "Add"
        assert ck["attrs"] == {"axis": -1}
        assert env["output_path"] == os.path.join(kwargs["output_dir"], "out.npz")

    def test_gpu_cmd_uses_gpus_not_capdrop(self, tmp_path, monkeypatch):
        from ttk.remote.server import container
        kwargs = _kwargs(tmp_path)
        cap = []
        _patch_mkstemp(container, monkeypatch, cap)
        runner = _Runner({"ok": True, "output_path": "/work/out.npz"}, cap)
        monkeypatch.setattr(container.subprocess, "run", runner)
        container._run_in_container(kwargs, deadline=time.monotonic() + 60,
                                    image="img", use_device=True,
                                    docker_args=["--gpus", "all"])
        cmd = runner.cmd
        assert "--gpus" in cmd and "all" in cmd
        assert "--security-opt" in cmd and "no-new-privileges" in cmd
        assert "--cap-drop=ALL" not in cmd

    def test_docker_args_in_cmd_not_in_ckwargs(self, tmp_path, monkeypatch):
        import time
        from ttk.remote.server import container
        kwargs = _kwargs(tmp_path)
        cap = []
        _patch_mkstemp(container, monkeypatch, cap)
        runner = _Runner({"ok": True, "output_path": "/work/out.npz"}, cap)
        monkeypatch.setattr(container.subprocess, "run", runner)
        container._run_in_container(
            kwargs, deadline=time.monotonic() + 60, image="img",
            use_device=True, docker_args=["--gpus", "device=0"])
        assert "--gpus" in runner.cmd and "device=0" in runner.cmd
        assert "docker_args" not in runner.ckwargs

    def test_none_tmp_in_and_none_output_noop(self, tmp_path, monkeypatch):
        from ttk.remote.server import container
        kwargs = _kwargs(tmp_path, with_tmp_in=False)
        cap = []
        _patch_mkstemp(container, monkeypatch, cap)
        runner = _Runner({"ok": True, "output_path": None}, cap)
        monkeypatch.setattr(container.subprocess, "run", runner)
        env = container._run_in_container(kwargs, deadline=time.monotonic() + 60, image="img")
        assert runner.ckwargs.get("tmp_in_path") is None
        assert env["output_path"] is None   # PERF-mode None: slice-map no-op

    def test_docker_not_found_returns_500(self, tmp_path, monkeypatch):
        from ttk.remote.server import container
        kwargs = _kwargs(tmp_path)

        def boom(*a, **k):
            raise FileNotFoundError
        monkeypatch.setattr(container.subprocess, "run", boom)
        env = container._run_in_container(
            kwargs, deadline=time.monotonic() + 60, image="img")
        assert env["ok"] is False
        assert env["http_status"] == 500
        assert "docker" in env["error"].lower()


class TestExecutorMainExitCode:
    def test_exit0_when_envelope_printed(self, tmp_path):
        """executor_main exits 0 whenever a valid envelope is printed (incl
        ok:False / 424); exit 1 only on infra failure. This preserves the 424
        signal the client's sync→retry loop depends on (container.py treats
        nonzero exit as 500, which would swallow 424)."""
        kwargs = {"exec_type": "api", "provider": "numpy", "api": "numpy.add",
                  "mode": 1, "use_device": False, "input_schema": [],
                  "attrs": {}, "input_count": 0, "device_id": 0,
                  "output_dir": str(tmp_path), "tenant_sync_dir": "",
                  "tmp_in_path": None, "profile": {"torch_lib": "cuda"},
                  "spec_module": None, "spec_class": None}
        kp = str(tmp_path / "kwargs.json")
        with open(kp, "w") as f:
            json.dump(kwargs, f)
        r = subprocess.run(
            [sys.executable, "ttk/remote/server/executor_main.py", kp],
            capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, f"exit {r.returncode}, stderr={r.stderr[:300]}"
        assert "ok" in json.loads(r.stdout)
