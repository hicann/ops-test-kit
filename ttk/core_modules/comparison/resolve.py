# ttk/core_modules/comparison/resolve.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResolvedStandard:
    token: str
    params: dict = field(default_factory=dict)


# 2.1 stat_rel_err 阈值表（resolve_tolerance 是唯一解析点——比对类不查表）
THRESHOLDS = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
    "float8_e4m3fn": 2**-3,
    "float8_e5m2": 2**-2,
}
DEFAULT_THRESHOLD = 2**-13

# cross_check level 预设（level 矩阵）
LEVEL_PRESETS = {
    "L0": {"mare_ratio": 10.0, "mere_ratio": 2.0, "rmse_ratio": 2.0},
    "L1": {"mare_ratio": 5.0,  "mere_ratio": 1.5, "rmse_ratio": 1.5},
    "L2": {"mare_ratio": 2.0,  "mere_ratio": 1.2, "rmse_ratio": 1.2},
}

# small-value thresholds (key 必须跟 _dtype_str 输出一致；cross_check 只支持 fp16/bf16/fp32)
SMALL_VALUE = {
    "float16":   {"small_value": 2**-11, "small_value_atol": 2**-16},
    "bfloat16":  {"small_value": 2**-8,  "small_value_atol": 2**-16},
    "float32":   {"small_value": 2**-14, "small_value_atol": 2**-30},
}

_REQUANT_DTYPES = {"float8_e5m2", "float8_e4m3fn", "hifloat8"}
_BIN_DTYPES = {"float4_e2m1", "float4_e1m2", "float8_e8m0"}


def _dtype_str(dtype) -> str:
    # torch.float16 -> "float16"
    return str(dtype).split(".")[-1]


def _is_int_or_bool(s: str) -> bool:
    return s == "bool" or s.startswith("int") or s.startswith("uint")


def _is_complex(s: str) -> bool:
    return "complex" in s


def _float_choice(tolerance: Optional[dict], dtype_str: str,
                  compare_method: Optional[str]) -> str:
    """普通浮点族的三级优先级：CLI > Spec.tolerance > stat_rel_err。"""
    if compare_method:                       # a) CLI 显式指定
        return compare_method.lower()
    if tolerance and dtype_str in tolerance:  # b) Spec 配了该 dtype
        std = tolerance[dtype_str].get("standard")
        if std:
            return std.lower()
    return "stat_rel_err"                     # c) 默认社区标准


def _resolve_params(standard, tolerance, dtype_str) -> dict:
    cfg = (tolerance or {}).get(dtype_str, {})
    extra = {k: v for k, v in cfg.items() if k != "standard"}

    if standard == "stat_rel_err":
        th = extra.pop("threshold", None) or THRESHOLDS.get(dtype_str, DEFAULT_THRESHOLD)
        return {"threshold": th}

    elif standard == "cross_check":
        if dtype_str not in SMALL_VALUE:
            raise ValueError(f"[{dtype_str}] cross_check unsupported dtype; supported: {sorted(SMALL_VALUE)}")
        level = extra.get("level")
        if level is not None:
            if level not in LEVEL_PRESETS:
                raise ValueError(f"[{dtype_str}] cross_check unknown level: {level!r}; expected {sorted(LEVEL_PRESETS)}")
            limits = dict(LEVEL_PRESETS[level])
        else:
            level = "L1"
            limits = dict(LEVEL_PRESETS[level])
        for k in ("mare_ratio", "mere_ratio", "rmse_ratio"):
            if k in extra:
                limits[k] = extra[k]
        missing = [k for k in ("mare_ratio", "mere_ratio", "rmse_ratio") if k not in limits]
        if missing:
            raise ValueError(f"[{dtype_str}] cross_check missing ratios: {missing}; specify level or all 3 ratios")
        resolved_level = level or "L1"
        sv_default = SMALL_VALUE[dtype_str]
        return {"level": resolved_level, **limits,
                "small_value": extra.get("small_value", sv_default["small_value"]),
                "small_value_atol": extra.get("small_value_atol", sv_default["small_value_atol"])}

    else:
        return {}


def resolve_tolerance(tolerance, precision_tolerances, absolute_precision,
                      output_dtypes, compare_method):
    standards = []
    for idx, dtype in enumerate(output_dtypes):
        s = _dtype_str(dtype)
        if dtype is None:
            token = "binary_equal"
        elif _is_int_or_bool(s):
            token = "binary_equal"
        elif _is_complex(s):
            choice = _float_choice(tolerance, s, compare_method)
            token = "binary_equal" if choice in ("bin", "binary") else "isclose"
        elif s in _REQUANT_DTYPES:
            choice = _float_choice(tolerance, s, compare_method)
            token = choice if choice in ("bin", "binary", "requant") else "requant"
        elif s in _BIN_DTYPES:
            token = "bin"
        else:
            token = _float_choice(tolerance, s, compare_method)
        params = _resolve_params(token, tolerance, s)
        # legacy 注入（per-output 标量，命名空间隔离）
        params["legacy"] = {
            "rtol": precision_tolerances[idx][0] if precision_tolerances and idx < len(precision_tolerances) else None,
            "ptol": precision_tolerances[idx][1] if precision_tolerances and idx < len(precision_tolerances) else None,
            "atol": absolute_precision[idx] if isinstance(absolute_precision, (tuple, list)) and idx < len(absolute_precision) else (absolute_precision if not isinstance(absolute_precision, (tuple, list)) else None),            
        }
        standards.append(ResolvedStandard(token, params))
    return standards
