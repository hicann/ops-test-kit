# ttk/core_modules/comparison/stat_rel_err.py
import numpy as np

from .registry import ComparisonBase, EachCompareResult, register_comparison, FAIL_REASONS

# 算法常量（不是配置——配置由 resolve_tolerance 解析后经 params 传入）
MARE_RATIO = 10
FLOOR = 1e-7


@register_comparison("stat_rel_err")
class StatRelErrComparison(ComparisonBase):
    STANDARD_NAME = "stat_rel_err"

    def compare_impl(self) -> EachCompareResult:
        # 空数组已在 compare() 入口 _check_empty 统一短路，这里只处理非空
        # threshold 是 resolve_tolerance 解析好的最终值（经 params → options 传入），不查表
        actual, golden = self.output, self.golden
        T = np.promote_types(np.dtype(np.float32),
                             np.promote_types(actual.dtype, golden.dtype))
        a = actual.astype(T)
        g = golden.astype(T)
        th = self.tol_options["threshold"]

        with np.errstate(invalid="ignore", divide="ignore"):
            rel_err = np.abs(a - g) / (np.abs(g) + FLOOR)
            a_nan, g_nan = np.isnan(a), np.isnan(g)
            a_inf, g_inf = np.isinf(a), np.isinf(g)
            same_nan = a_nan & g_nan                              # 都 NaN（NaN 无符号）
            same_inf = a_inf & g_inf & (np.sign(a) == np.sign(g))  # 都 Inf 且同号
            mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
            mismatch_idx = np.where(mismatch)[0]
            mask = np.isfinite(a) & np.isfinite(g)
            n_finite = int(mask.sum())

            if mismatch_idx.size:                  # NaN/Inf 不一致：污染，不算 mere/mare
                return EachCompareResult("FAIL", diff_index=mismatch_idx, is_pass=False,
                                         standard="stat_rel_err",
                                         metrics=_metrics(th, None, None, False,
                                                          FAIL_REASONS["nan_inf_mismatch"]))
            if n_finite == 0:                       # 全非有限且一致
                return EachCompareResult("PASS", is_pass=True,
                                         standard="stat_rel_err", metrics=_metrics(th, None, None, True))
            mere = float(np.mean(rel_err[mask]))
            mare = float(np.max(rel_err[mask]))
            passed = mere < th and mare < MARE_RATIO * th
            if passed:
                diff_idx = None
                reason = None
            else:                                   # 数值 FAIL：finite 全部按 rel_err 降序(worst first)
                finite_idx = np.where(mask)[0]      #   完整列表不截断；log(_log_diff_output)控制打印数
                diff_idx = finite_idx[np.argsort(-rel_err[finite_idx])]
                exceeded = []
                if mere >= th:
                    exceeded.append(f"mere({mere:.2e}>={th})")
                if mare >= MARE_RATIO * th:
                    exceeded.append(f"mare({mare:.2e}>={MARE_RATIO * th})")
                reason = FAIL_REASONS["threshold_exceeded"].format(metrics=", ".join(exceeded))

        return EachCompareResult("PASS" if passed else "FAIL",
                                 diff_index=diff_idx, is_pass=passed,
                                 standard="stat_rel_err", metrics=_metrics(th, mere, mare, passed, reason))


def _metrics(th, mere, mare, passed, reason=None):
    # 全 Python 字面量，保 CSV eval 往返；mere/mare=None（非 NaN）当 mismatch 或全 match
    m = {"standard": "stat_rel_err", "mere": mere, "mare": mare,
         "threshold": th, "pass": bool(passed)}
    if reason:
        m["reason"] = reason
    return m
