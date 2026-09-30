# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
"""免上传（zero-upload）：配方可复算性、免传判定、以及端到端回落。

核心不变量只有一条：**远端按配方重算出的字节，必须与客户端手上那份完全相同**，
且这件事要能被证明（摘要），不成立时必须回落整包上传而不是继续算。
"""

import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from ttk.remote import dispatcher
from ttk.remote.input_recipe import build_recipe, digest_of, regenerate, seed_for
from ttk.utilities.data import RandomData

# 覆盖易碎路径：自定义窄类型、原生整型(独立 RNG)、全值域(指数造数)、
# 常量区间(low==high)、inf 区间、normal(走 scipy truncnorm)
RECIPE_CASES = [
    ("float32", (64, 32), [-1.0, 1.0], "uniform"),
    ("float16", (1000,), [-65504.0, 65504.0], "uniform"),
    ("bfloat16", (128, 8), [-3.0, 3.0], "uniform"),
    ("int32", (256,), [-100, 100], "uniform"),
    ("int64", (33,), [-5, 5], "uniform"),
    ("int8", (64,), [-128, 127], "uniform"),
    ("bool", (50,), [0, 1], "uniform"),
    ("float32", (512,), [None, None], "uniform"),
    ("float32", (77,), [0.0, 0.0], "uniform"),
    ("float32", (40,), [float("-inf"), float("inf")], "uniform"),
    ("float16", (300,), [-1.0, 1.0], "normal"),
]


@pytest.mark.parametrize(("dtype", "shape", "data_range", "distribution"), RECIPE_CASES)
def test_recipe_regenerates_bit_identical(dtype, shape, data_range, distribution):
    """同一配方两次生成必须逐位相同，且配方要经得起 JSON 往返（它走 HTTP 头）。"""
    seed = seed_for("case_probe", 0)
    np.random.seed(seed)
    random_data = RandomData(dtype, shape, data_range)
    original = random_data.generate(distribution)

    recipe = json.loads(json.dumps(build_recipe(dtype, shape, random_data.data_range, distribution, seed)))
    assert digest_of(regenerate(recipe)) == digest_of(original)


def test_digest_distinguishes_different_data():
    """负控：判据必须能失败，否则"摘要一致"毫无意义。"""
    np.random.seed(seed_for("probe", 0))
    a = RandomData("float32", (64, 32), [-1.0, 1.0]).generate("uniform")
    np.random.seed(seed_for("probe", 9))
    b = RandomData("float32", (64, 32), [-1.0, 1.0]).generate("uniform")
    assert digest_of(a) != digest_of(b)


def test_seed_is_process_stable():
    """种子必须跨进程稳定：内置 hash() 带 PYTHONHASHSEED 随机化，会让跑批不可复现。"""
    code = "from ttk.remote.input_recipe import seed_for; print(seed_for('caseX', 3))"
    env = dict(os.environ, PYTHONHASHSEED="0")
    first = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    env["PYTHONHASHSEED"] = "12345"
    second = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    assert first.strip() == second.strip()


def test_plan_requires_every_leaf_to_be_provable():
    """只要有一片叶子无法用配方证明，就整体放弃免传——宁可多传，不可比错。"""
    np.random.seed(seed_for("plan", 0))
    random_data = RandomData("float32", (8, 4), [-1.0, 1.0])
    known = random_data.generate("uniform")
    recipes = {digest_of(known): build_recipe("float32", (8, 4), random_data.data_range, "uniform", 0)}
    stranger = np.ones((8, 4), dtype=np.float32)  # 比如被 input 插件原地改过的那种

    assert dispatcher._plan_zero_upload([known], recipes) is not None
    assert dispatcher._plan_zero_upload([known, stranger], recipes) is None
    assert dispatcher._plan_zero_upload([known], None) is None


# --------------------------------------------------------------------------
# 端到端：需要 torch，起一个 standalone server（与生产部署同一入口）
# --------------------------------------------------------------------------
try:
    has_torch = (
        subprocess.run([sys.executable, "-c", "import torch"], capture_output=True, timeout=90, check=False).returncode
        == 0
    )
except Exception:
    has_torch = False

needs_torch = pytest.mark.skipif(not has_torch, reason="torch not installed")


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(scope="module")
def zero_upload_server():
    import http.client

    port = _free_port()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, env.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ttk.remote.server.xpu_server", "--port", str(port), "--devices", "cpu"],
        env=env,
        cwd=repo_root,
        # 服务端输出丢弃, 不能用 PIPE: 没人读时管道缓冲区(64 KB)写满会让服务端
        # **永久阻塞**(框架预载一次就能吐 100 KB+), 表现为请求超时而非报错。
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ready = False
    for _ in range(40):
        time.sleep(0.5)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            conn.request("GET", "/v1/heartbeat")
            ready = conn.getresponse().status == 200
        except OSError:
            ready = False
        if ready:
            break
    if not ready:
        proc.kill()
        pytest.skip("xpu_server did not become ready")
    yield port
    proc.kill()


def _echo(port, array, recipes, tmp_path):
    """让远端 clone 这份输入并回传，顺带统计发生过几次 npz 序列化。"""
    calls = []
    original = dispatcher._serialize_to_file

    def counting(inputs, dir=None):
        calls.append(1)
        return original(inputs, dir=dir)

    dispatcher._serialize_to_file = counting
    try:
        outputs = dispatcher.dispatch_to_remote(
            op_name="clone",
            inputs=[array],
            input_names=["input"],
            provider="torch",
            endpoint_host="127.0.0.1",
            endpoint_port=port,
            tenant_id="zero_upload_test",
            mode="data",
            api="torch.clone",
            execution_type="api",
            input_recipes=recipes,
            tmp_root=str(tmp_path),
        )
    finally:
        dispatcher._serialize_to_file = original
    return outputs[0], len(calls)


@needs_torch
def test_zero_upload_end_to_end(zero_upload_server, tmp_path, monkeypatch):
    """回显验证：免传时远端实际算用的输入与本地逐字节相同；配方坏掉则回落。"""
    from ttk.config.loader import load_config

    config = tmp_path / "ttk.conf.yaml"
    config.write_text(f"remote:\n  endpoints:\n    - host: 127.0.0.1\n      port: {zero_upload_server}\n")
    load_config(str(config))

    seed = seed_for("zero_upload_test", 0)
    np.random.seed(seed)
    random_data = RandomData("float32", (128, 64), [-1.0, 1.0])
    local = random_data.generate("uniform")
    recipes = {digest_of(local): build_recipe("float32", (128, 64), random_data.data_range, "uniform", seed)}

    echoed, serializations = _echo(zero_upload_server, local, recipes, tmp_path)
    assert serializations == 0, "配方可用时不应再序列化输入"
    assert digest_of(echoed) == digest_of(local), "远端算用的输入必须与本地逐字节相同"

    # 配方被改坏 -> 服务端摘要对不上 -> 409 -> 客户端整包重发，结果依旧正确
    broken = {k: dict(v, seed=v["seed"] ^ 0xBEEF) for k, v in recipes.items()}
    echoed_after_fallback, serializations = _echo(zero_upload_server, local, broken, tmp_path)
    assert serializations == 1, "摘要不符时必须回落成整包上传"
    assert digest_of(echoed_after_fallback) == digest_of(local)
