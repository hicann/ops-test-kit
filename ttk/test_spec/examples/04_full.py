"""Full example: golden + third_party + tolerance + compare"""

import numpy


class LayerNormFullSpec:
    """LayerNorm — demonstrates all optional attributes"""

    # -- golden — function form --
    def golden(x, gamma, beta, *, epsilon=1e-5, **kwargs):
        mean = numpy.mean(x, axis=-1, keepdims=True)
        var = numpy.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / numpy.sqrt(var + epsilon)
        return [x_norm * gamma + beta]

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(self, *, epsilon=1e-5, **kwargs):
            self.eps = epsilon

        def __call__(self, x, gamma, beta, **kwargs):
            import torch
            return [torch.nn.functional.layer_norm(
                x, [x.shape[-1]], gamma, beta, self.eps
            )]

    third_party = {"torch": ThirdPartyImpl, "tf": "tf.raw_ops.LayerNorm"}

    # -- tolerance — per-dtype precision standard --
    tolerance = {
        "float32": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 5.0,
            "rmse_rtol": 2.0,
            "small_value": 1e-6,
            "small_value_atol": 1e-9,
        },
        "float16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0,
            "max_re_rtol": 10.0,
            "rmse_rtol": 2.0,
            "small_value": 0.001,
            "small_value_atol": 1e-5,
        },
    }

    # -- compare — custom comparison (optional, default cosine_similarity) --
    def compare(*outputs, **kwargs):
        """Custom comparison: returns dict (single output) or list[dict] (multi output).
        Each dict: pass(bool, required), precision(str|float, required),
        error_info(str, optional), metrics(dict, optional), diff_indices(list, optional).
        float precision is percentage (99.98, not 0.9998)."""
        npu_out, golden_out = outputs[0], outputs[1]
        cos_sim = numpy.dot(npu_out.flatten(), golden_out.flatten()) / (
            numpy.linalg.norm(npu_out.flatten()) * numpy.linalg.norm(golden_out.flatten()))
        return {
            "pass": cos_sim > kwargs.get("threshold", 0.99),
            "precision": cos_sim * 100,
            "metrics": {"cosine_similarity": cos_sim},
        }


# Explicit registration: class name uses *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "layer_norm_full": LayerNormFullSpec,
}
