"""Docker sandbox backend for xpu-server.

Deployment constraint: stdlib + subprocess only. No ttk.* imports.
"""

import json
import logging
import os
import subprocess
import tempfile
import time


def _run_in_container(
    kwargs: dict,
    deadline: float,
    image: str,
    memory: str = "8g",
    network: str = "none",
    use_device: bool = False,
    docker_args=None,
) -> dict:
    """Run execute_request in a Docker container (per-Container isolation).

    Bind-mounts req_dir (rw, tmp_in + out) + tenant_sync_dir (ro, specs) at
    fixed container paths, rewrites the 3 path fields in kwargs to those paths,
    runs executor_main.py, maps envelope.output_path back to host.
    """
    host_req_dir = kwargs["output_dir"]
    host_sync_dir = kwargs["tenant_sync_dir"]
    _api = kwargs.get("api") or kwargs.get("spec_class")

    # Shallow copy; rebind ONLY the 3 path fields to container paths (allowlist,
    # not a loop — avoids mangling api/attrs string values).
    ckwargs = dict(kwargs)
    ckwargs["output_dir"] = "/work"
    ckwargs["tenant_sync_dir"] = "/sync"
    if ckwargs.get("tmp_in_path"):
        ckwargs["tmp_in_path"] = "/work/" + os.path.basename(kwargs["tmp_in_path"])

    fd, kwargs_path = tempfile.mkstemp(suffix=".json", prefix="xpu_kwargs_")
    timeout_s = max(1, int(deadline - time.monotonic()))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ckwargs, f)

        if os.getuid() == 0:
            logging.warning(
                "_run_in_container: server running as root — container will be "
                "root, C3 symlink-escape protection inactive"
            )
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{host_req_dir}:/work",  # rw: tmp_in + out.npz
            "-v",
            f"{host_sync_dir}:/sync:ro",  # ro: specs
            "--tmpfs",
            "/tmp",  # writable tmp (--read-only root)
            "-e",
            "HOME=/tmp",  # torch ~/.cache → tmpfs
            "--read-only",
            "--network",
            network,
            "--memory",
            memory,
            "--user",
            f"{os.getuid()}:{os.getgid()}",  # non-root: blocks symlink escape
            "-v",
            f"{kwargs_path}:/kwargs.json:ro",
        ]
        if use_device:
            cmd.extend(docker_args or [])
            cmd.extend(["--security-opt", "no-new-privileges"])
        else:
            cmd.append("--cap-drop=ALL")
        cmd.extend([image, "python", "/executor/executor_main.py", "/kwargs.json"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)

        if result.returncode == 0 and result.stdout.strip():
            try:
                envelope = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {
                    "ok": False,
                    "http_status": 500,
                    "error": f"invalid envelope JSON: {result.stdout[:200]}",
                    "api": _api,
                }
            # Map /work/<x> back to host_req_dir/<x> (slice, not replace)
            op = envelope.get("output_path")
            if op and op.startswith("/work/"):
                envelope["output_path"] = host_req_dir + op[len("/work") :]
            return envelope
        return {
            "ok": False,
            "http_status": 500,
            "error": "container execution failed" if result.stderr else f"exit code {result.returncode}",
            "api": _api,
        }

    except subprocess.TimeoutExpired:
        return {"ok": False, "http_status": 500, "error": f"container timed out after {timeout_s}s", "api": _api}
    except FileNotFoundError:
        return {"ok": False, "http_status": 500, "error": "docker not found", "api": _api}
    except Exception as e:
        return {"ok": False, "http_status": 500, "error": str(e), "api": _api}
    finally:
        if os.path.exists(kwargs_path):
            os.unlink(kwargs_path)
