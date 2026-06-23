"""golden in three forms: string / function / class"""

import numpy


class AbsStrSpec:
    """Form 1: string — framework auto-maps. Simplest, for ops with direct numpy API."""
    golden = "numpy.abs"


class AbsFuncSpec:
    """Form 2: function — params before * are inputs, after * are attrs."""
    def golden(x, **kwargs):
        return [numpy.abs(x)]


class _AbsGolden:
    """golden 形式3：类 — 需要状态管理时。__init__ 可选 + __call__ 必须。"""
    def __init__(self, **kwargs):
        self.low_precision = kwargs.get("low_precision", False)

    def __call__(self, x, **kwargs):
        result = numpy.abs(x)
        if self.low_precision:
            result = result.astype(numpy.float16)
        return [result]


class AbsClassSpec:
    """Form 3: golden 指向一个 golden 类（状态管理时用）。"""
    golden = _AbsGolden


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "abs_str": AbsStrSpec,
    "abs_func": AbsFuncSpec,
    "abs_class": AbsClassSpec,
}
