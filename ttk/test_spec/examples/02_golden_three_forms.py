"""golden in three forms: string / function / class

kernel 流程 golden 收到 numpy.ndarray，需内部转 torch 计算后转回 numpy。
ACLNN/E2E 流程 golden 直接收到 torch.Tensor，无需转换。
"""

import torch


class AbsStrSpec:
    """Form 1: string — framework auto-maps. Simplest, for ops with direct API."""
    golden = "torch.abs"


class AbsFuncSpec:
    """Form 2: function — params before * are inputs, after * are attrs."""
    def golden(x, **kwargs):
        return [torch.abs(torch.from_numpy(x)).numpy()]


class _AbsGolden:
    """golden 形式3：类 — 需要状态管理时。__init__ 可选 + __call__ 必须。"""
    def __init__(self, **kwargs):
        self.low_precision = kwargs.get("low_precision", False)

    def __call__(self, x, **kwargs):
        result = torch.abs(torch.from_numpy(x))
        if self.low_precision:
            result = result.to(torch.float16)
        return [result.numpy()]


class AbsClassSpec:
    """Form 3: golden 指向一个 golden 类（状态管理时用）。"""
    golden = _AbsGolden


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "abs_str": "AbsStrSpec",
    "abs_func": "AbsFuncSpec",
    "abs_class": "AbsClassSpec",
}
