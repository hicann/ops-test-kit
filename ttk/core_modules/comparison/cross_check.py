# ttk/core_modules/comparison/cross_check.py
import numpy as np
from .registry import ComparisonBase, EachCompareResult, register_comparison, FAIL_REASONS


def safe_div(num, den, err):
    """三比值的除法:分母一律夹小值域阈值 err。

    精度标准:正常值域的 mare/mere/rmse 三比值,分母按 dtype 取的 small_value
    (见 resolve.py)夹底。err 既防除零,也保证竞品误差落在噪声地板以下时不把比值
    放大——固定常数会让判定随输出量级漂移。
    """
    if np.isnan(den) or np.isinf(den):
        return float("inf")
    if num == 0 and den == 0:
        return 1.0   # both perfect match -> ratio=1 (consistent)
    return float(num / max(den, err))


@register_comparison(["cross_check"])
class CrossCheckComparison(ComparisonBase):
    STANDARD_NAME = "cross_check"

    def compare_impl(self) -> EachCompareResult:
        t, g, b = self.output, self.golden, self.third_party
        if b is None:
            return EachCompareResult("GOLDEN_FAILURE", is_pass=False, standard="cross_check",
                                     metrics={"standard": "cross_check", "pass": False,
                                              "level": self.tol_options.get("level"),
                                              "reason": FAIL_REASONS["third_party_unavailable"]})

        T = np.promote_types(np.float32, np.promote_types(t.dtype, np.promote_types(g.dtype, b.dtype)))
        t, g, b = t.astype(T), g.astype(T), b.astype(T)

        sv = {"small_value": self.tol_options["small_value"],
              "small_value_atol": self.tol_options["small_value_atol"]}
        mare_limit = self.tol_options["mare_ratio"]
        mere_limit = self.tol_options["mere_ratio"]
        rmse_limit = self.tol_options["rmse_ratio"]

        special = np.isnan(t) | np.isinf(t) | np.isnan(g) | np.isinf(g) | np.isnan(b) | np.isinf(b)
        finite = ~special
        large = finite & (np.abs(g) >= sv["small_value"])
        small = finite & (np.abs(g) < sv["small_value"])

        # Path A: special positions
        match_g = (t == g) | (np.isnan(t) & np.isnan(g))
        match_b = (t == b) | (np.isnan(t) & np.isnan(b))
        special_ok = np.all((match_g | match_b)[special]) if special.any() else True

        with np.errstate(invalid="ignore", divide="ignore"):
            # Path B-1: large positions — mare/mere/rmse ratio
            if large.any():
                rel_npu = np.abs(t[large] - g[large]) / (np.abs(g[large]) + 1e-7)
                rel_party = np.abs(b[large] - g[large]) / (np.abs(g[large]) + 1e-7)
                mare_ratio = safe_div(rel_npu.max(), rel_party.max(), sv["small_value"])
                mere_ratio = safe_div(rel_npu.mean(), rel_party.mean(), sv["small_value"])
                rmse_npu = np.sqrt(np.mean((t[large] - g[large]) ** 2))
                rmse_party = np.sqrt(np.mean((b[large] - g[large]) ** 2))
                rmse_ratio = safe_div(rmse_npu, rmse_party, sv["small_value"])
                exceeded = []
                if mare_ratio > mare_limit:
                    exceeded.append(f"mare({mare_ratio:.2f}>{mare_limit})")
                if mere_ratio > mere_limit:
                    exceeded.append(f"mere({mere_ratio:.2f}>{mere_limit})")
                if rmse_ratio > rmse_limit:
                    exceeded.append(f"rmse({rmse_ratio:.2f}>{rmse_limit})")
                ratio_ok = not exceeded
                diff_idx = self._top_diff(t, g, large)
            else:
                exceeded = []
                mare_ratio = mere_ratio = rmse_ratio = None   # 无大值域：未算 ratio（None 表 N/A，非 0.0 误导）
                ratio_ok = True
                diff_idx = []

            # Path B-2: small positions — ErrorCount ratio（guard 与 large 对称，metrics 字段总定义）
            if small.any():
                err_target = int(np.sum(small & (np.abs(t - g) > sv["small_value_atol"])))
                err_third = int(np.sum(small & (np.abs(b - g) > sv["small_value_atol"])))
                small_ratio = err_target / max(err_third, 1)
                small_ok = small_ratio <= 2.0
                diff_idx += self._top_diff(t, g, small)
            else:
                err_target = err_third = 0
                small_ratio = 0.0
                small_ok = True

        passed = ratio_ok and special_ok and small_ok
        if not passed:
            if not special_ok:
                reason = FAIL_REASONS["nan_inf_mismatch"]
            elif not small_ok:
                reason = FAIL_REASONS["small_value_exceeded"].format(ratio=f"{small_ratio:.2f}>2.0")
            else:
                reason = FAIL_REASONS["ratio_exceeded"].format(exceeded=", ".join(exceeded))
        else:
            reason = None

        metrics = {"standard": "cross_check", "level": self.tol_options.get("level"),
                   "config": {"mare": mare_limit, "mere": mere_limit, "rmse": rmse_limit,
                              "small_value": sv["small_value"], "small_value_atol": sv["small_value_atol"]},
                   "result": {"mare": float(mare_ratio) if mare_ratio is not None else None,
                              "mere": float(mere_ratio) if mere_ratio is not None else None,
                              "rmse": float(rmse_ratio) if rmse_ratio is not None else None,
                              "small_err_cnt_target": err_target, "small_err_cnt_third": err_third,
                              "small_err_ratio": float(small_ratio)},
                   "pass": passed}
        if reason:
            metrics["reason"] = reason
        _diff_arr = np.array(diff_idx, dtype=int) if diff_idx else None
        return EachCompareResult("PASS" if passed else "FAIL", is_pass=passed,
                                 diff_index=_diff_arr if not passed else None,
                                 standard="cross_check", metrics=metrics)

    def _top_diff(self, t, g, mask):
        """top MAX_DIFF_OUTPUT |t-g|>0 positions (worst-first) under mask. Returns list."""
        _diff = np.abs(t[mask] - g[mask])
        _li = np.where(mask)[0]
        _mis = _diff > 0
        if not _mis.any():
            return []
        _md, _ml = _diff[_mis], _li[_mis]
        _k = min(self.MAX_DIFF_OUTPUT, _md.size)
        _top = np.argpartition(-_md, _k - 1)[:_k]
        result = _ml[_top[np.argsort(-_md[_top])]].tolist()
        del _md, _ml, _top, _diff, _li, _mis
        return result

    def _log_diff_output(self, diff_index):
        """Three-party diff (t/g/b + |t-g| + |b-g|), worst-first.
        large positions use %.6e, small positions use %.10e (small values need
        more digits). [L]/[S] tag distinguishes large/small in each row."""
        log = ""
        if diff_index is None or len(diff_index) == 0:
            return log
        t, g, b = self.output, self.golden, self.third_party
        small_value = self.tol_options.get("small_value", 1e-7)
        n = len(diff_index)
        log += "Output %d Compare Difference length %d (worst-first)\n" % (self.output_idx, n)
        for idx in range(n):
            i = int(diff_index[idx])
            is_small = abs(float(g[i])) < small_value
            fmt = "%.10e" if is_small else "%.6e"
            tag = "S" if is_small else "L"
            log += ("Index: %03d RealIndex: %06d [%s] t=%s g=%s b=%s |t-g|=%s |b-g|=%s\n"
                    % (idx, i, tag, fmt % float(t[i]), fmt % float(g[i]), fmt % float(b[i]),
                       fmt % float(abs(t[i] - g[i])), fmt % float(abs(b[i] - g[i]))))
        return log

