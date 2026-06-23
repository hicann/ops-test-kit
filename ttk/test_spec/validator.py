# ttk/test_spec/validator.py
import warnings

from . import InvalidSpecWarning

# Valid types for each attribute: tuple of types
_CHECK_RULES: dict[str, tuple] = {
    "golden": (str, type),
    "third_party": (str, dict, type),
    "compare": (dict,),
    "pre_compare": (),
    "customize_inputs": (),
    "tolerance": (dict,),
    "torch_graph": (type,),
    "describe": (),
}

# These attributes additionally accept callable
_CALLABLE_OK = {"golden", "third_party", "compare", "pre_compare", "customize_inputs", "describe"}


def validate(spec_cls: type) -> list[str]:
    """Shallow type check. Returns list of warning messages, does not block.

    Args:
        spec_cls: spec class

    Returns:
        list[str]: warning messages
    """
    warnings_list = []

    for attr_name in _CHECK_RULES:
        if not hasattr(spec_cls, attr_name):
            continue

        value = getattr(spec_cls, attr_name)
        if value is None:
            continue

        # isinstance check
        type_ok = False
        if _CHECK_RULES[attr_name] and isinstance(value, _CHECK_RULES[attr_name]):
            type_ok = True
        elif attr_name in _CALLABLE_OK and callable(value):
            type_ok = True

        if not type_ok:
            msg = (
                f"[TestSpec] {spec_cls.__name__}.{attr_name} "
                f"type mismatch: got {type(value).__name__}, "
                f"expected one of {_expected_types_str(attr_name)}"
            )
            warnings.warn(msg, InvalidSpecWarning, stacklevel=2)
            warnings_list.append(msg)

    return warnings_list


def _expected_types_str(attr_name: str) -> str:
    types = list(_CHECK_RULES.get(attr_name, ()))
    if attr_name in _CALLABLE_OK:
        types.append("callable")
    return " | ".join(t.__name__ for t in types if isinstance(t, type))
