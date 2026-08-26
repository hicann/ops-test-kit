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
    # torch.float16 -> "float16"；ml_dtypes.bfloat16 类对象 / numpy dtype 也需归一。
    # 直接 str().split('.')[-1] 对类对象会得到 "bfloat16'>"（残留 "'>"），
    # 使 THRESHOLDS 查表 miss 而回落 DEFAULT(2**-13, fp32 级)，令 bf16 阈值被误判为过严。
    # 优先用 numpy.dtype(...).name 做鲁棒解析，失败再回退旧逻辑。
    try:
        import numpy as _np
        return _np.dtype(dtype).name
    except Exception:
        return str(dtype).split(".")[-1].rstrip("'>\" ")


def _is_int_or_bool(s: str) -> bool:
    return s == "bool" or s.startswith("int") or s.startswith("uint")


def _is_complex(s: str) -> bool:
    return "complex" in s


# 量化输出的目标 dtype：标准限定 int4 / int8。
# 不含 int32/int64——那些通常承载索引、计数、掩码等精确整数语义，差 1 就是错。
_QUANT_OUT_DTYPES = {"int4", "int8"}

# --compare 中对【整型输出】有意义的取值：可覆盖 Spec 的 quant 声明。
# 其余取值均为浮点判据，对量化整型无意义，不参与覆盖。
_INT_APPLICABLE_TOKENS = {"bin", "binary", "binary_equal", "requant", "quant"}


def _is_float_dtype(s: str) -> bool:
    return ("float" in s) or (s in ("half", "double", "bf16", "bfloat16"))


def _spec_standard(tolerance, dtype_str):
    """取 Spec.tolerance 为该 dtype 显式声明的标准名；未声明返回 None。"""
    if not tolerance or dtype_str not in tolerance:
        return None
    std = (tolerance[dtype_str] or {}).get("standard")
    return std.lower() if isinstance(std, str) else None


def _check_quant_applicable(dtype_str, input_dtypes):
    """声明 quant 时校验前提，不匹配直接报错——**不静默纠正**。

    【为何要校验】quant 的判据是绝对误差 <= 1，只对"浮点输入 + int4/int8 输出"的
    量化场景成立。若用在索引/计数类输出上，±1 的容忍会放过真缺陷，而且悄无声息。
    【为何报错而非降级】误用是配置错误，应当让作者当场看见并改正；静默换成别的判据
    会让 Spec 声明与实际判定不一致，是更隐蔽的坑。
    【为何要求至少一个浮点输入】量化必然存在 float→int 的转换，被量化的数据必为
    浮点。但量化算子天然带整型量化参数（如 zero_points / antiquant offset），要求
    "全部输入浮点"会误伤这类合法算子。改为"至少一个浮点输入"——既保留对纯整型算子
    （索引/计数/掩码，无任何浮点输入）的拦截，又不拒绝带整型量化参数的量化算子。
    """
    if dtype_str not in _QUANT_OUT_DTYPES:
        raise ValueError(
            f"Spec.tolerance 为 [{dtype_str}] 声明了 standard='quant'，但 quant 仅适用于 "
            f"{sorted(_QUANT_OUT_DTYPES)} 输出（绝对误差<=1 是量化语义；索引/计数类输出差 1 即错）。")
    if input_dtypes is not None:
        ins = [_dtype_str(d) for d in input_dtypes if d is not None]
        if ins and not any(_is_float_dtype(i) for i in ins):
            raise ValueError(
                f"Spec.tolerance 为 [{dtype_str}] 声明了 standard='quant'，但输入 dtype {ins} "
                f"全为整型。quant 的前提是【浮点型输入 + int4/int8 输出】的量化场景；"
                f"纯整型算子（索引/计数/掩码）应使用 binary_equal。")


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
                      output_dtypes, compare_method, input_dtypes=None):
    """input_dtypes 可选：仅供 quant 的护栏校验"输入是否全为浮点"，不参与判据选择。
    不传时跳过该校验，Spec 声明的 quant 依然生效——判据由算子作者的声明决定，
    拿不到输入 dtype 只是少了一道校验，不应反过来否定声明。
    未声明 quant 的存量算子不受影响（整数仍走 binary_equal）。"""
    standards = []
    for idx, dtype in enumerate(output_dtypes):
        s = _dtype_str(dtype)
        spec_std = _spec_standard(tolerance, s)
        if dtype is None:
            token = "binary_equal"
        elif spec_std == "quant":
            # Spec 显式声明 quant：算子自己知道输出是量化值，判据为绝对误差 <= 1。
            # 该分支排在整数短路之前——原先整数短路排在读 tolerance 之前，等于"算子说了不算"。
            # 不做 dtype 自动推断：dtype 组合只是必要条件，真正知道语义的是算子作者；
            # 推断错会把该严的判松（漏报真缺陷且无人察觉），代价远大于漏声明导致的误报。
            _check_quant_applicable(s, input_dtypes)
            # 优先级仍守 CLI > Spec.tolerance > 默认，但只认【整数适用】的 CLI 取值：
            # --compare 的其余取值（close/cosine/stat_rel_err/cross_check）是浮点判据，
            # 对量化整型输出无意义，不应连带覆盖掉算子的 quant 声明。
            cli = (compare_method or "").lower()
            token = cli if cli in _INT_APPLICABLE_TOKENS else "quant"
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
