# ttk/test_spec/validator.py

from . import InvalidSpecError

# Valid types for each attribute: tuple of types
_CHECK_RULES: dict[str, tuple] = {
    "golden": (str, type),
    "third_party": (str, dict, type),
    "compare": (),
    "pre_compare": (),
    "customize_inputs": (),
    "tolerance": (dict,),
    "torch_graph": (type,),
    "describe": (),
}

# These attributes additionally accept callable
_CALLABLE_OK = {"golden", "third_party", "compare", "pre_compare", "customize_inputs", "describe"}

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
