import pytest
import torch
from collections import OrderedDict
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.core_modules.npu.op_api.golden_generation import GoldenGenerator
from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfo


def _mock_switches(golden_mode="Promote"):
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"; sw.short_soc_version = "Ascend910B"
    sw.golden_mode = golden_mode; sw.plugin_path = None; sw.overflow_mode = 0
    return sw


def _make_op_api_info(tensor_names, scalar_names=()):
    params = OrderedDict()
    for n in tensor_names:
        params[n] = {"type": "aclTensor*"}
    for n in scalar_names:
        params[n] = {"type": "aclScalar*"}
    return OpApiInfo(params=params)


def _make_testcase(api_name, tensors, tensor_dtypes):
    case = TestcaseAclnn()
    case.testcase_name = f"test_{api_name}_promote"
    case.api_name = api_name
    case.tensors = list(tensors)
    case.tensor_dtypes = tensor_dtypes
    case.scalars = None
    case.attributes = {}
    case.output_tensor_indexes = ()
    case._pure_output_indexes = ()
    case.manual_golden_binaries = None
    return case


class _RecordDtype:
    """class-form golden:无自定义 __init__(走 cls() 守卫);__call__ 记录收到的 dtype。"""
    received = {}
    def __call__(self, x):
        type(self).received["dtype"] = str(x.dtype)
        return [x]


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        monkeypatch.delenv(k, raising=False)


@patch('ttk.core_modules.npu.op_api.golden_generation.OpApiInfoKeeper')
@patch('ttk.core_modules.npu.op_api.golden_generation.get_global_storage')
@patch('ttk.core_modules.npu.op_api.golden_generation.get_plugin_function')
class TestAclnnPromote:
    def test_promote_float16_to_float32_and_restore(self, mock_get_plugin, mock_sw, mock_op_info):
        """Promote: float16 输入 → golden(class __call__)收 float32;退出后 ctx 还原 float16。"""
        _RecordDtype.received = {}
        mock_get_plugin.return_value = _RecordDtype          # class-form → _invoke_class
        mock_sw.return_value = _mock_switches("Promote")
        mock_op_info.return_value.info_of.return_value = _make_op_api_info(["x"])
        inp = torch.tensor([1.0, 2.0], dtype=torch.float16)
        case = _make_testcase("aclnnRec", [inp], ("float16",))
        GoldenGenerator(case)._generate_golden()
        assert _RecordDtype.received.get("dtype") == "torch.float32"   # 提升后喂 golden
        assert str(case.tensors[0].dtype) == "torch.float16"           # 退出 promote context 后还原

    def test_noop_when_no_low_precision(self, mock_get_plugin, mock_sw, mock_op_info):
        """无低精度 dtype(float64 不在 DTYPE_PROMOTE_MAP)→ 不提升,golden 收原 dtype。"""
        _RecordDtype.received = {}
        mock_get_plugin.return_value = _RecordDtype
        mock_sw.return_value = _mock_switches("Promote")
        mock_op_info.return_value.info_of.return_value = _make_op_api_info(["x"])
        case = _make_testcase("aclnnRec", [torch.tensor([1.0], dtype=torch.float64)], ("float64",))
        GoldenGenerator(case)._generate_golden()
        assert _RecordDtype.received.get("dtype") == "torch.float64"   # 未提升(float64 不在 map)


def test_scenario2_guard_raises(monkeypatch):
    """source 非原生 + target 原生(scenario 2)→ _promote_dtype 直接抛 NotImplementedError。
    直接驱动 _promote_dtype 生成器(绕过 _generate_golden 的 except 吞噬)。"""
    import ttk.core_modules.npu.op_api.golden_generation as mod
    monkeypatch.setattr(mod, "is_torch_native_dtype",
                        lambda d: d not in ("float16", "bfloat16"),
                        raising=False)  # float16 判非原生,float32 原生;raising=False 兼容 RED 态(未 import)
    case = _make_testcase("aclnnRec", [torch.tensor([1.0], dtype=torch.float16)], ("float16",))
    with patch('ttk.core_modules.npu.op_api.golden_generation.get_global_storage') as mock_sw:
        mock_sw.return_value = _mock_switches("Promote")
        gen = GoldenGenerator(case)
        it = gen._promote_dtype()
        with pytest.raises(NotImplementedError, match="native boundary"):
            next(it)   # guard fires before first yield → next() triggers it
