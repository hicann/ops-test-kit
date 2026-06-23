"""Advanced: small-op composition, golden reusing bench, customize_inputs"""

import numpy


class SoftmaxAdvancedSpec:
    """Small-op composition + golden reusing bench"""

    def golden(x, *, axis=-1, **kwargs):
        """golden reusing bench logic — calls ThirdPartyImpl on CPU tensor"""
        import torch
        cpu_tensor = torch.from_numpy(x)
        result = SoftmaxAdvancedSpec._BenchImpl()(cpu_tensor, axis=axis)
        return [r.numpy() for r in result]

    class _BenchImpl:
        def __init__(self, *, axis=-1, **kwargs):
            self.axis = axis

        def __call__(self, x, *, axis=None, **kwargs):
            import torch
            dim = axis if axis is not None else self.axis
            return [torch.nn.functional.softmax(x, dim=dim)]

    third_party = {
        "torch": "torch.nn.functional.softmax",
        "npu_decompose": _BenchImpl,
    }

    tolerance = {
        "float32": {"standard": "IsClose", "rtol": 1e-4, "atol": 1e-4},
        "float16": {"standard": "IsClose", "rtol": 1e-3, "atol": 1e-3},
    }


class HistogramInputSpec:
    """customize_inputs — custom input generation"""
    def golden(x, min_val, max_val, **kwargs):
        return [numpy.histogram(x, bins=10, range=(min_val[0], max_val[0]))[0]]

    def customize_inputs(x, min_val, max_val, **kwargs):
        """Parameter names match operator definition, returns modified input arrays"""
        min_data = min_val[0]
        max_data = max_val[0]
        if min_data == max_data:
            min_val[0] = numpy.min(x)
            max_val[0] = numpy.max(x)
        return (x, min_val, max_val)


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "softmax_advanced": SoftmaxAdvancedSpec,
    "histogram_input": HistogramInputSpec,
}
