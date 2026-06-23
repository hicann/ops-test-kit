"""Simplest example: only a golden function, everything else defaults."""

import numpy


class AddTestSpec:
    """Add operator — minimal spec"""
    def golden(x, y, **kwargs):
        return [numpy.add(x, y)]


# __spec__ dict registration (optional, naming convention also works)
__spec__ = {"add": AddTestSpec}
