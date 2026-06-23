"""third_party in three forms: string / dict / class"""

import numpy


class SoftmaxSimpleSpec:
    """Single API string — torch auto-mapped"""
    def golden(x, *, axis=-1, **kwargs):
        x = x.astype("float32")
        max_x = numpy.amax(x, axis=axis, keepdims=True)
        exp_x = numpy.exp(x - max_x)
        return [exp_x / numpy.sum(exp_x, axis=axis, keepdims=True)]

    third_party = "torch.nn.functional.softmax"


class SoftmaxMultiVendorSpec:
    """Dict multi-vendor — supports torch / tf / flash_attn / npu_decompose"""
    def golden(x, *, axis=-1, **kwargs):
        x = x.astype("float32")
        exp_x = numpy.exp(x)
        return [exp_x / numpy.sum(exp_x, axis=axis, keepdims=True)]

    third_party = {
        "torch": "torch.nn.functional.softmax",
        "tf": "tf.raw_ops.Softmax",
    }


class SoftmaxComposeSpec:
    """Class form — small-op composition"""
    def golden(x, *, axis=-1, **kwargs):
        exp_x = numpy.exp(x)
        return [exp_x / numpy.sum(exp_x, axis=axis, keepdims=True)]

    class NpuDecomposeImpl:
        def __init__(self, *, axis=-1, **kwargs):
            self.axis = axis

        def __call__(self, x, **kwargs):
            import torch
            exp_x = torch.exp(x)
            sum_exp = torch.sum(exp_x, dim=self.axis, keepdim=True)
            return [exp_x / sum_exp]

    third_party = {
        "torch": "torch.nn.functional.softmax",
        "npu_decompose": NpuDecomposeImpl,
    }


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "softmax_simple": SoftmaxSimpleSpec,
    "softmax_multi_vendor": SoftmaxMultiVendorSpec,
    "softmax_compose": SoftmaxComposeSpec,
}
