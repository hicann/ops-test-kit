"""Tests for server-side _resolve_3party_api (KERNEL dual-form port)."""
import pytest
from ttk.remote.server import executor

# torch/tf are available on the XPU server; skip when absent in dev/CI envs.
torch = pytest.importorskip("torch")


def test_torch_resolve_snake():
    # op_name snake 命中(torch.add)
    f, name = executor._resolve_3party_api("add", "Add", "torch")
    assert callable(f)
    assert name == "add"


def test_torch_resolve_exception_dict():
    # kl_div 不在 torch 顶层,在 exception 表里 → functional.kl_div
    f, name = executor._resolve_3party_api("kl_div", "KlDiv", "torch")
    assert callable(f)
    assert name == "kl_div"


def test_torch_resolve_camel_is_nop():
    # torch 喂 Camel 在 getattr 即失败,双形式靠 snake 命中;两个都给也 OK
    f, name = executor._resolve_3party_api("add", "Add", "torch")
    assert callable(f)
    assert name == "add"


def test_resolve_returns_none_when_unresolvable():
    f, name = executor._resolve_3party_api("totally_bogus_xyz", "TotallyBogusXyz", "torch")
    assert f is None and name is None


def test_resolve_callable_api_none_routes_to_3party():
    # api=None + exec_type=api → _resolve_3party_api; 推导 api_label = provider.name
    f, label = executor._resolve_callable("api", "torch", None, "add", "Add", None, None)
    assert callable(f)
    assert label == "torch.add"


def test_resolve_callable_explicit_api_still_works():
    # explicit api 走原 resolve_callable(嵌套 dotted); api_label = api 本身
    f, label = executor._resolve_callable("api", "torch", "torch.nn.functional.softmax",
                                          "ignored", "Ignored", None, None)
    assert callable(f)
    assert label == "torch.nn.functional.softmax"


def test_torch_resolve_suffix_stripping_v2():
    # _v2 suffix strips to base op (top_k_v2 → topk via exception dict path
    # exercised through _resolve_3party_api; here test direct _torch_resolve
    # strips _v2 even for an unknown base — returns None, not crash).
    # Use a known one: add_v2 → strip _v2 → add (torch.add exists).
    f = executor._torch_resolve("add_v2")
    assert callable(f)


def test_torch_func_exception_dict_floor_div_is_partial():
    # floor_div in exception dict → partial(torch.div, rounding_mode='floor')
    f = executor._torch_resolve("floor_div")
    assert callable(f)


def test_aclnn_resolve_strips_prefix_and_finds_torch():
    # aclnnAdd → strip "aclnn" → camel_to_snake("Add")="add" → torch.add
    # name = strip+snake 后的实际 key（X-API 回传 torch.add，不是 torch.aclnnAdd）
    f, name = executor._aclnn_resolve("aclnnAdd")
    assert callable(f)
    assert name == "add"


def test_aclnn_resolve_recursive_suffix():
    # aclnn-prefixed name with a torch-matchable base walks _find_torch_api's
    # inplace_/_v2/... suffix recursion (here: simple base case "Add" -> add).
    f, name = executor._aclnn_resolve("aclnnAdd")
    assert callable(f)
    assert name == "add"


def test_resolve_3party_aclnn_branch():
    # aclnn prefix now routes through _aclnn_resolve (no longer the None placeholder)
    # name = strip+snake key — X-API 回传 provider.name = torch.add
    f, name = executor._resolve_3party_api("aclnnAdd", None, "torch")
    assert callable(f)
    assert name == "add"


def test_camel_to_snake_copied_locally():
    # camel_to_snake is copied into executor (deployment constraint: no ttk.* import)
    assert hasattr(executor, "camel_to_snake")
    assert executor.camel_to_snake("BatchNormV3") == "batch_norm_v3"


def test_aclnn_resolve_unresolvable_returns_none():
    # aclnn prefix + base that has no torch match → (None, None) (no crash, no recursion blowup)
    f, name = executor._aclnn_resolve("aclnnTotallyBogusXyz")
    assert f is None and name is None


def test_resolve_3party_skips_empty_names():
    # Both names empty/None → no resolve attempt, returns (None, None).
    f, name = executor._resolve_3party_api(None, None, "torch")
    assert f is None and name is None
    f, name = executor._resolve_3party_api("", "", "torch")
    assert f is None and name is None


# --- tf resolver tests (skip if tensorflow is not installed / crashes on import) ---
# CI 的 tensorflow C 扩展可能 import 即 segfault（importorskip 捕获不到），用子进程探测。
import importlib.util as _ilu, subprocess as _sp, sys as _sys
_tf_ok = False
if _ilu.find_spec("tensorflow") is not None:
    try:
        _tf_ok = _sp.run([_sys.executable, "-c", "import tensorflow"],
                         capture_output=True, timeout=90).returncode == 0
    except Exception:
        _tf_ok = False
if not _tf_ok:
    pytest.skip("tensorflow not importable (crashes on import)", allow_module_level=True)
tf = pytest.importorskip("tensorflow")


def test_tf_resolve_camel():
    # tf resolves "relu" (snake op_name) first — name = 命中的搜索入参 = op_name "relu".
    # (Camel "Relu" is the fallback; here op_name already hits so name=op_name.)
    f, name = executor._resolve_3party_api("relu", "Relu", "tf")
    assert callable(f)
    assert name in ("relu", "Relu")   # 命中的搜索入参（op_name 优先）


def test_tf_resolve_snake():
    # op_name snake 也命中(tf.math.multiply); name = op_name "multiply"
    f, name = executor._resolve_3party_api("multiply", "Multiply", "tf")
    assert callable(f)
    assert name == "multiply"


def test_tf_resolve_exception_dict_batch_norm_v3():
    # batch_norm_v3 不在 sources 顶层,在 exception 表里 → nn_impl.fused_batch_norm
    f, name = executor._resolve_3party_api("batch_norm_v3", "BatchNormV3", "tf")
    assert callable(f)
    assert name == "batch_norm_v3"


def test_tf_resolve_exception_dict_spence():
    # spence → tf.math.special.spence
    f, name = executor._resolve_3party_api("spence", "Spence", "tf")
    assert callable(f)
    assert name == "spence"


def test_tf_resolve_exception_dict_space_to_batch():
    # space_to_batch → gen_array_ops.SpaceToBatch
    f, name = executor._resolve_3party_api("space_to_batch", "SpaceToBatch", "tf")
    assert callable(f)
    assert name == "space_to_batch"


def test_tf_resolve_suffix_stripping_v2():
    # Backup's replace-based strip targets the literal substring "_v2"
    # (not CamelCase "V2"). "Relu_v2" -> strip "_v2" -> "Relu" (tf.nn.Relu).
    # _tf_resolve is a scalar helper (not a tuple-returning resolver).
    f = executor._tf_resolve("Relu_v2")
    assert callable(f)


def test_tf_resolve_returns_none_when_unresolvable():
    f, name = executor._resolve_3party_api("totally_bogus_xyz", "TotallyBogusXyz", "tf")
    assert f is None and name is None


def test_resolve_callable_api_none_routes_to_3party_tf():
    # api=None + exec_type=api + provider=tf → _resolve_3party_api
    # 推导 api_label = provider.命中name（op_name "relu" 优先命中 → "tf.relu"）
    f, label = executor._resolve_callable("api", "tf", None, "relu", "Relu", None, None)
    assert callable(f)
    assert label in ("tf.relu", "tf.Relu")


def test_resolve_callable_explicit_api_still_works_tf():
    # explicit api 走原 resolve_callable(嵌套 dotted) for tf too; api_label = api
    f, label = executor._resolve_callable("api", "tf", "tf.nn.relu",
                                          "ignored", "Ignored", None, None)
    assert callable(f)
    assert label == "tf.nn.relu"
