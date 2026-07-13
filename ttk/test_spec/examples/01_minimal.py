"""Simplest example: only a golden function, everything else defaults."""

# __spec__ dict registration (optional, naming convention also works)
# 值为类名字符串 —— loader 用 AST 静态扫描建索引（不 exec 文件），故可写在顶部
__spec__ = {"add": "AddTestSpec"}

import numpy


class AddTestSpec:
    """Add operator — minimal spec"""
    def golden(x, y, **kwargs):
        return [numpy.add(x, y)]
