# ttk/test_spec/validator.py

from typing import Dict

from . import InvalidSpecError

# Spec.tolerance 只收 2.1 四标准；框架增强/别名走 CLI --compare
_SPEC_TOLERANCE_STANDARDS = {"stat_rel_err", "binary_equal", "cross_check", "quant"}   # 2.1 官方
_FRAMEWORK_TOKENS = {"isclose", "close", "cosine", "bin", "binary", "requant"}          # CLI-only

# Valid types for each attribute: tuple of types
_CHECK_RULES: Dict[str, tuple] = {
    "golden": (str, type),
    "third_party": (str, dict, type),
    "compare": (),
    "pre_compare": (),
    "customize_inputs": (),
    "pre_npu": (),
    "tolerance": (dict,),
    "torch_graph": (type,),
    "describe": (),
}

# These attributes additionally accept callable
_CALLABLE_OK = {
    "golden", "third_party", "compare", "pre_compare", "customize_inputs",
    "pre_npu", "describe",
}

# callable 属性里允许 class 的(golden/third_party 文档有 class 形式);
# 其余 callable 属性(compare/pre_compare/customize_inputs/describe)只接受 function。
_CLASS_OK = {"golden", "third_party"}


def validate(spec_cls: type) -> None:
    """Shallow type check. Raises InvalidSpecError listing all mismatches (fail-fast).

    Args:
        spec_cls: spec class

    Raises:
        InvalidSpecError: if any declared attribute has a disallowed type.
    """
    errors = []

    for attr_name in _CHECK_RULES:
        if not hasattr(spec_cls, attr_name):
            continue

        value = getattr(spec_cls, attr_name)
        if value is None:
            continue

        # torch_graph 深语义:必须是 torch.nn.Module 子类(延迟 import torch)
        if attr_name == "torch_graph":
            try:
                import torch.nn as nn
                ok = isinstance(value, type) and issubclass(value, nn.Module)
            except ImportError:
                ok = isinstance(value, type)  # torch 未装:fallback 只查是类
            if not ok:
                errors.append(
                    f"{spec_cls.__name__}.torch_graph must be a torch.nn.Module subclass, "
                    f"got {type(value).__name__}"
                )
            continue

        # tolerance 深校验：只收 2.1 四标准；框架 token 指路 --compare；CamelCase → generic unknown
        if attr_name == "tolerance":
            _validate_tolerance(spec_cls.__name__, value, errors)
            continue

        # isinstance check
        type_ok = False
        if _CHECK_RULES[attr_name] and isinstance(value, _CHECK_RULES[attr_name]):
            type_ok = True
        elif attr_name in _CALLABLE_OK and callable(value):
            # golden/third_party 允许 class;其余 callable 属性只接受 function
            if isinstance(value, type) and attr_name not in _CLASS_OK:
                type_ok = False
            else:
                type_ok = True

        if not type_ok:
            errors.append(
                f"{spec_cls.__name__}.{attr_name} type mismatch: "
                f"got {type(value).__name__}, expected one of {_expected_types_str(attr_name)}"
            )

    if errors:
        raise InvalidSpecError("; ".join(errors))


def _expected_types_str(attr_name: str) -> str:
    types = list(_CHECK_RULES.get(attr_name, ()))
    parts = [t.__name__ for t in types if isinstance(t, type)]
    if attr_name in _CALLABLE_OK:
        parts.append("callable")
    return " | ".join(parts)


def _validate_tolerance(spec_name, tolerance, errors):
    if not isinstance(tolerance, dict):
        errors.append(f"{spec_name}.tolerance must be dict, got {type(tolerance).__name__}")
        return
    for dtype, cfg in tolerance.items():
        if not isinstance(cfg, dict):
            errors.append(f"{spec_name}.tolerance[{dtype!r}] must be dict, got {type(cfg).__name__}")
            continue
        std = cfg.get("standard")
        if std is None:
            continue
        if not isinstance(std, str):
            errors.append(f"{spec_name}.tolerance[{dtype!r}].standard must be str, got {type(std).__name__}")
        elif std in _SPEC_TOLERANCE_STANDARDS:
            pass  # 2.1 合法（cross_check/quant 即便 P1 未实现也放过，runtime 报）
        elif std in _FRAMEWORK_TOKENS:
            errors.append(f"{spec_name}.tolerance[{dtype!r}].standard {std!r} is a framework enhancement/alias "
                          f"(CLI-only via --compare); Spec.tolerance only accepts 2.1 standards "
                          f"{sorted(_SPEC_TOLERANCE_STANDARDS)}")
        else:
            errors.append(f"{spec_name}.tolerance[{dtype!r}].standard unknown: {std!r}; "
                          f"expected one of {sorted(_SPEC_TOLERANCE_STANDARDS)}")
