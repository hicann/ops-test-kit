from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.comparison.batch_consistency import compare_batch_consistency
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e


def batch_metadata(first_slice, second_slice):
    return {
        "batch_axis": ((0,),),
        "batch_slice_info": (((first_slice, second_slice),),),
        "batch_seed": (((74123, 74123),),),
    }


@pytest.mark.parametrize("case_class", (TestcaseE2e, TestcaseAclnn))
def test_batch_group_id_ignores_logical_relation_offset(case_class):
    first = case_class()
    second = case_class()
    for case, metadata in (
        (first, batch_metadata((0, 1, 1), (1, 2, 1))),
        (second, batch_metadata((0, 1, 1), (2, 3, 1))),
    ):
        case.batch_axis = metadata["batch_axis"]
        case.batch_slice_info = metadata["batch_slice_info"]
        case.batch_seed = metadata["batch_seed"]
        case._generate_batch_consistency_id()

    assert first.batch_consistency_id == second.batch_consistency_id


def batch_case(name, slice_info):
    return SimpleNamespace(
        testcase_name=name,
        batch_consistency_id="relation",
        batch_axis=((0,),),
        batch_slice_info=((slice_info,),),
    )


def canonical_batch_case(name, slices):
    return SimpleNamespace(
        testcase_name=name,
        batch_consistency_id="relation",
        batch_axis=((0,),),
        batch_slice_info=((tuple(slices),),),
    )


def test_batch_compare_slices_ndarray_output_before_hashing():
    first = batch_case("first", (0, 1, 1))
    second = batch_case("second", (1, 2, 1))
    first_result = SimpleNamespace(
        output_bytes=(np.array([[7], [1]], dtype=np.int32),)
    )
    second_result = SimpleNamespace(
        output_bytes=(np.array([[9], [7]], dtype=np.int32),)
    )

    results = compare_batch_consistency([
        (first, first_result),
        (second, second_result),
    ])

    assert results[0]["supported"] is True
    assert results[0]["pass"] is True
    assert all(member["status"] == "ok" for member in results[0]["members"])


def test_batch_compare_handles_all_canonical_relation_slices():
    first = canonical_batch_case("first", ((0, 1, 1), (1, 2, 1)))
    second = canonical_batch_case("second", ((0, 1, 1), (2, 3, 1)))
    first_output = np.array([[7], [7]], dtype=np.int32)
    second_output = np.array([[7], [9], [7]], dtype=np.int32)

    results = compare_batch_consistency([
        (first, SimpleNamespace(output_bytes=(first_output,))),
        (second, SimpleNamespace(output_bytes=(second_output,))),
    ])

    assert results[0]["supported"] is True
    assert results[0]["pass"] is True


def test_batch_compare_detects_difference_in_any_canonical_relation_slice():
    first = canonical_batch_case("first", ((0, 1, 1), (1, 2, 1)))
    second = canonical_batch_case("second", ((0, 1, 1), (2, 3, 1)))
    first_output = np.array([[7], [7]], dtype=np.int32)
    second_output = np.array([[7], [9], [8]], dtype=np.int32)

    results = compare_batch_consistency([
        (first, SimpleNamespace(output_bytes=(first_output,))),
        (second, SimpleNamespace(output_bytes=(second_output,))),
    ])

    assert results[0]["supported"] is True
    assert results[0]["pass"] is False


def test_batch_compare_rejects_matching_cross_case_patterns_with_intra_case_difference():
    first = canonical_batch_case("first", ((0, 1, 1), (1, 2, 1)))
    second = canonical_batch_case("second", ((0, 1, 1), (2, 3, 1)))
    first_output = np.array([[7], [8]], dtype=np.int32)
    second_output = np.array([[7], [9], [8]], dtype=np.int32)

    results = compare_batch_consistency([
        (first, SimpleNamespace(output_bytes=(first_output,))),
        (second, SimpleNamespace(output_bytes=(second_output,))),
    ])

    assert results[0]["supported"] is True
    assert results[0]["pass"] is False
    assert all(len(member["sample_md5s"]) == 2 for member in results[0]["members"])


@pytest.mark.parametrize("case_class", (TestcaseE2e, TestcaseAclnn))
@pytest.mark.parametrize("invalid_slice", ((-1, 0, 1), (1, 0, -1)))
def test_batch_group_id_keeps_legacy_nonnegative_forward_slice_contract(
    case_class, invalid_slice
):
    case = case_class()
    metadata = batch_metadata(invalid_slice, (0, 1, 1))
    case.batch_axis = metadata["batch_axis"]
    case.batch_slice_info = metadata["batch_slice_info"]
    case.batch_seed = metadata["batch_seed"]

    case._generate_batch_consistency_id()

    assert "74123_0_0_" in str(case.batch_consistency_id)


def test_batch_compare_marks_missing_worker_output_unsupported():
    first = batch_case("first", (0, 1, 1))
    second = batch_case("second", (1, 2, 1))

    results = compare_batch_consistency([
        (first, SimpleNamespace()),
        (second, SimpleNamespace()),
    ])

    assert results[0]["supported"] is False
    assert results[0]["pass"] is False
    assert results[0]["reason"] == "NO_OUTPUT"
    assert [member["md5"] for member in results[0]["members"]] == [
        "NO_OUTPUT",
        "NO_OUTPUT",
    ]


def test_batch_compare_rejects_untyped_raw_bytes():
    first = batch_case("first", (0, 1, 1))
    second = batch_case("second", (1, 2, 1))

    results = compare_batch_consistency([
        (first, SimpleNamespace(output_bytes=(b"first",))),
        (second, SimpleNamespace(output_bytes=(b"second",))),
    ])

    assert results[0]["supported"] is False
    assert results[0]["pass"] is False
    assert results[0]["reason"] == "UNSLICEABLE_OUTPUT"


def test_batch_compare_rejects_invalid_slice_instead_of_hashing_full_output():
    first = batch_case("first", (0, 3, 1))
    second = batch_case("second", (0, 3, 1))
    output = np.array([[7], [1]], dtype=np.int32)

    results = compare_batch_consistency([
        (first, SimpleNamespace(output_bytes=(output,))),
        (second, SimpleNamespace(output_bytes=(output,))),
    ])

    assert results[0]["supported"] is False
    assert results[0]["pass"] is False
    assert results[0]["reason"] == "INVALID_SLICE"
    assert all(member["status"] == "unsupported" for member in results[0]["members"])


def test_batch_compare_marks_missing_slice_metadata_unsupported():
    first = batch_case("first", (0, 1, 1))
    second = batch_case("second", (1, 2, 1))
    first.batch_slice_info = None
    output = np.array([[7], [7]], dtype=np.int32)

    results = compare_batch_consistency([
        (first, SimpleNamespace(output_bytes=(output,))),
        (second, SimpleNamespace(output_bytes=(output,))),
    ])

    assert results[0]["supported"] is False
    assert results[0]["pass"] is False
    assert results[0]["reason"] == "MISSING_SLICE"
