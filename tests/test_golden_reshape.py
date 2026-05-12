"""
Tests for __golden_flatten in output_generation.py:
- Flat ndarray output
- Nested (TensorList) output
- numpy.generic scalar conversion
- Mixed scalar + ndarray
- Single non-sequence output
"""

import numpy as np
import pytest

from ttk.core_modules.npu.op import output_generation as _mod

_golden_reshape = getattr(_mod, '__golden_flatten')


class TestGoldenReshapeFlat:

    def test_flat_ndarrays(self):
        a = np.ones((3,), dtype="float16")
        b = np.ones((4,), dtype="float32")
        result = _golden_reshape((a, b))
        assert len(result) == 2
        assert result[0] is a
        assert result[1] is b

    def test_single_ndarray(self):
        a = np.ones((3,), dtype="float16")
        result = _golden_reshape(a)
        assert len(result) == 1
        assert result[0] is a

    def test_scalar_converted(self):
        s = np.float32(1.5)
        result = _golden_reshape((s,))
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (1,)
        assert result[0][0] == 1.5

    def test_mixed_scalar_and_array(self):
        a = np.ones((3,), dtype="float16")
        s = np.int32(42)
        result = _golden_reshape((a, s))
        assert len(result) == 2
        assert result[0] is a
        assert isinstance(result[1], np.ndarray)
        assert result[1][0] == 42


class TestGoldenReshapeNested:

    def test_nested_tensorlist(self):
        a1 = np.ones((3,), dtype="float16")
        a2 = np.ones((4,), dtype="float16")
        a3 = np.ones((5,), dtype="float32")
        result = _golden_reshape(((a1, a2), a3))
        assert len(result) == 3
        assert result[0] is a1
        assert result[1] is a2
        assert result[2] is a3

    def test_deeply_nested(self):
        a1 = np.ones((2,), dtype="float16")
        a2 = np.ones((3,), dtype="float16")
        a3 = np.ones((4,), dtype="float32")
        a4 = np.ones((5,), dtype="float32")
        result = _golden_reshape(((a1, a2), (a3, a4)))
        assert len(result) == 4
        assert result == [a1, a2, a3, a4]

    def test_nested_with_scalar(self):
        a1 = np.ones((3,), dtype="float16")
        s = np.float32(2.0)
        a2 = np.ones((4,), dtype="float32")
        result = _golden_reshape(((a1, s), a2))
        assert len(result) == 3
        assert result[0] is a1
        assert isinstance(result[1], np.ndarray)
        assert result[1][0] == 2.0
        assert result[2] is a2

    def test_single_tensorlist_group(self):
        a1 = np.ones((3,), dtype="float16")
        a2 = np.ones((4,), dtype="float16")
        result = _golden_reshape(((a1, a2),))
        assert len(result) == 2
        assert result[0] is a1
        assert result[1] is a2

    def test_multiple_tensorlist_groups(self):
        a1 = np.ones((2,), dtype="float16")
        a2 = np.ones((3,), dtype="float16")
        a3 = np.ones((4,), dtype="float32")
        a4 = np.ones((5,), dtype="float32")
        a5 = np.ones((6,), dtype="int32")
        result = _golden_reshape(((a1, a2), (a3, a4), a5))
        assert len(result) == 5
        assert result == [a1, a2, a3, a4, a5]


class TestGoldenReshapeEdgeCases:

    def test_empty_tuple(self):
        result = _golden_reshape(())
        assert result == []

    def test_list_input(self):
        a = np.ones((3,), dtype="float16")
        result = _golden_reshape([a])
        assert len(result) == 1

    def test_non_sequence_scalar_wrapped(self):
        s = np.float32(1.0)
        result = _golden_reshape(s)
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
