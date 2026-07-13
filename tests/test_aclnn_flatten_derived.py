import torch
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn


def test_flatten_tensors_derives_from_tensors():
    """flatten_tensors 是 deep_flatten(tensors) 的派生缓存,不是独立赋值的 plain attr。"""
    case = TestcaseAclnn()
    t1, t2, t3 = torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])
    case.tensors = ((t1, t2), t3)   # 嵌套:TensorList (t1,t2) + 单 t3
    flat = case.flatten_tensors
    assert list(flat) == [t1, t2, t3]   # deep_flatten 展平,顺序保持
    # 缓存生效:再读同一对象
    assert case.flatten_tensors is flat


def test_flatten_scalars_derives_from_scalars():
    case = TestcaseAclnn()
    s1, s2 = torch.tensor(5), torch.tensor(6)
    case.scalars = (s1, s2)
    assert list(case.flatten_scalars) == [s1, s2]


def test_flatten_none_when_tensors_none():
    case = TestcaseAclnn()
    assert case.flatten_tensors is None
    assert case.flatten_scalars is None


def test_invalidate_flat_cache_clears_and_rederives():
    """invalidate_flat_cache("tensors") 清 _flat_tensors,下次读 flatten_tensors 重派生。"""
    case = TestcaseAclnn()
    t1 = torch.tensor([1.0])
    case.tensors = (t1,)
    first = case.flatten_tensors          # 派生并缓存
    assert first is case.flatten_tensors  # 缓存命中
    # 改 tensors,不 invalidate → 仍读旧缓存(派生语义:缓存优先)
    t2 = torch.tensor([2.0])
    case.tensors = (t2,)
    assert case.flatten_tensors is first  # 还是旧缓存(_flat_tensors 未清)
    # invalidate 后重派生
    case.invalidate_flat_cache("tensors")
    assert list(case.flatten_tensors) == [t2]
    # 不存在的 field 安全跳过(hasattr 守卫)
    case.invalidate_flat_cache("nonexistent_field")   # 不抛
