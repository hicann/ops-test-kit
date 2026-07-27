import logging
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from ttk.core_modules.framework_api.input_generation import generate_inputs
from ttk.core_modules.manual_data import (
    ManualDataError,
    ManualDataStore,
    load_manual_data_case,
    manual_data_store,
    prepare_manual_data_store,
    register_manual_data_directory_provider,
    replay_manual_data_store,
    unregister_manual_data_directory_provider,
)
from ttk.core_modules.npu.op_api.input_generation import InputGenerator
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
from ttk.utilities import resolve_custom_numpy_dtypes


def _e2e_case(name="case/with spaces"):
    case = TestcaseE2e()
    case.testcase_name = name
    case.api_name = "torch.add"
    case.tensor_view_shapes = ((2, 2), (2, 2))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((3, 3), (2, 2))
    case.tensor_view_offsets = (1, 0)
    case.tensor_view_strides = ((3, 1), (2, 1))
    case.output_tensor_indexes = (1,)
    case.inplace_input_indexes = ()
    case.attributes = {"alpha": 1}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.golden_api = ""
    case._tensor_list_dist = (0, 0)
    case._pure_output_indexes = [1]
    return case


def _aclnn_case():
    case = TestcaseAclnn()
    case.testcase_name = "aclnn_case"
    case.api_name = "aclnnAdd"
    case.tensor_view_shapes = ((2,), (2,))
    case.tensor_dtypes = ("float32", "float32")
    case.tensor_formats = ("ND", "ND")
    case.tensor_storage_shapes = ((2,), (2,))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((1,), (1,))
    case.output_tensor_indexes = (1,)
    case.output_inplace_indexes = ()
    case.inplace_input_indexes = ()
    case.attributes = {}
    case.input_data_ranges = ((-1, 1), (-1, 1))
    case.scalar_dtypes = ("float32",)
    case.scalar_data_ranges = ((0, 1),)
    case._tensor_list_dist = (0, 0)
    case._scalar_list_dist = (0,)
    case._pure_output_indexes = [1]
    return case


def _kernel_case(name="kernel_case"):
    return SimpleNamespace(
        testcase_name=name,
        flat_input_shapes=((2,), None),
        flat_input_dtypes=("float32", "float32"),
        flat_output_shapes=((2,),),
        flat_output_dtypes=("float32",),
    )


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_round_trip_uses_only_typed_data_files(tmp_path, file_format):
    case = _aclnn_case()
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalars = [np.array(0.25, dtype=np.float32)]
    goldens = [np.array([3.0, 4.0], dtype=np.float32)]
    store = ManualDataStore(tmp_path)

    case_dir = store.write_case(
        case, "aclnn", inputs, goldens, scalars=scalars,
        file_format=file_format,
    )
    loaded = store.load_case(case, "aclnn")
    loaded_goldens = loaded.load_goldens(references=goldens)

    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(loaded.inputs[1], inputs[1])
    np.testing.assert_array_equal(loaded.scalars[0], scalars[0])
    np.testing.assert_array_equal(loaded_goldens[0], goldens[0])
    assert {path.name for path in case_dir.iterdir()} == {
        f"input_0_float32.{file_format}",
        f"input_1_float32.{file_format}",
        f"scalar_0_float32.{file_format}",
        (
            "golden_0_float32__shape_2.bin"
            if file_format == "bin"
            else f"golden_0_float32.{file_format}"
        ),
    }


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_complete_dataset_remains_loadable_after_directory_move(tmp_path, file_format):
    case = _aclnn_case()
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalars = [np.array(0.25, dtype=np.float32)]
    goldens = [np.array([3.0, 4.0], dtype=np.float32)]
    source = tmp_path / "source"
    destination = tmp_path / "destination"

    ManualDataStore(source).write_case(
        case, "aclnn", inputs, goldens, scalars=scalars, file_format=file_format
    )
    shutil.copytree(source, destination)
    loaded = ManualDataStore(destination).load_case(case, "aclnn")
    loaded_goldens = loaded.load_goldens(references=goldens)

    assert loaded.case_dir.parent == destination.resolve()
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    np.testing.assert_array_equal(loaded.scalars[0], scalars[0])
    np.testing.assert_array_equal(loaded_goldens[0], goldens[0])


def test_npy_round_trip_restores_custom_bfloat16_dtype(tmp_path):
    case = _e2e_case("bfloat16_npy")
    case.tensor_dtypes = ("bfloat16", "bfloat16")
    dtype = resolve_custom_numpy_dtypes(("bfloat16",))[0]
    inputs = [np.arange(9, dtype=np.float32).astype(dtype).reshape(3, 3),
              np.zeros((2, 2), dtype=dtype)]
    golden = [np.ones((2, 2), dtype=dtype)]
    store = ManualDataStore(tmp_path)

    store.write_case(case, "e2e", inputs, golden, file_format="npy")
    loaded = store.load_case(case, "e2e")
    loaded_golden = loaded.load_goldens(references=golden)

    assert loaded.inputs[0].dtype.name == "bfloat16"
    assert loaded_golden[0].dtype.name == "bfloat16"
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
@pytest.mark.parametrize(
    "logical_dtype",
    ["complex32", "uint1", "int4", "float8_e4m3fn", "float8_e5m2"],
)
def test_special_physical_storage_round_trip(tmp_path, file_format, logical_dtype):
    case = _e2e_case(f"{logical_dtype}_{file_format}")
    case.tensor_dtypes = (logical_dtype, "float32")
    if logical_dtype == "complex32":
        storage = np.arange(18, dtype=np.float16).reshape(3, 3, 2)
    elif logical_dtype == "uint1":
        storage = np.array([0x55, 0x01], dtype=np.uint8)
    else:
        dtype = resolve_custom_numpy_dtypes((logical_dtype,))[0]
        storage = np.arange(-4, 5, dtype=np.int8).astype(dtype).reshape(3, 3)
    store = ManualDataStore(tmp_path)

    store.write_case(
        case,
        "e2e",
        [storage, np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
        file_format=file_format,
    )
    loaded = store.load_case(case, "e2e")

    assert loaded.inputs[0].dtype.name == storage.dtype.name
    assert loaded.inputs[0].shape == storage.shape
    if logical_dtype == "int4" and file_format == "bin":
        np.testing.assert_array_equal(loaded.inputs[0], storage)
    else:
        assert loaded.inputs[0].tobytes() == storage.tobytes()


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_kernel_round_trip_uses_kernel_csv_shapes(tmp_path, file_format):
    case = _kernel_case(f"kernel_{file_format}")
    inputs = [np.array([1.0, 2.0], np.float32), None]
    goldens = [np.array([3.0, 4.0], np.float32)]
    store = ManualDataStore(tmp_path)

    case_dir = store.write_case(
        case, "kernel", inputs, goldens, file_format=file_format
    )
    loaded = store.load_case(case, "kernel")
    loaded_goldens = loaded.load_goldens(
        shapes=case.flat_output_shapes,
        dtypes=case.flat_output_dtypes,
    )

    assert loaded.file_format == file_format
    assert (case_dir / f"input_1_none.{file_format}").stat().st_size == 0
    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    assert loaded.inputs[1] is None
    np.testing.assert_array_equal(loaded_goldens[0], goldens[0])


def test_e2e_noncontiguous_backing_storage_and_none_slot_round_trip(tmp_path):
    case = _e2e_case()
    first = np.arange(9, dtype=np.float32).reshape(3, 3)
    second = np.ones((2, 2), dtype=np.float32)
    golden = [np.ones((2, 2), dtype=np.float32), None]
    store = ManualDataStore(tmp_path)

    store.write_case(case, "e2e", [first, second], golden)
    loaded = store.load_case(case, "e2e")
    loaded_golden = loaded.load_goldens(references=golden)

    np.testing.assert_array_equal(loaded.inputs[0], first)
    assert loaded_golden[1] is None
    assert (loaded.case_dir / "golden_1_none.bin").stat().st_size == 0


def test_tensor_list_grouping_is_rebuilt_from_csv_structure(tmp_path):
    case = _e2e_case("tensor_list")
    case.tensor_view_shapes = (((2,), (3,)), (2,))
    case.tensor_dtypes = (("float32", "int32"), "float32")
    case.tensor_formats = (("ND", "ND"), "ND")
    case.tensor_storage_shapes = (((2,), (3,)), (2,))
    case.tensor_view_offsets = ((0, 0), 0)
    case.tensor_view_strides = (((1,), (1,)), (1,))
    case.output_tensor_indexes = (1,)
    case._tensor_list_dist = (2, 0)
    case._pure_output_indexes = [2]
    inputs = [
        np.array([1.0, 2.0], np.float32),
        np.array([3, 4, 5], np.int32),
        np.zeros(2, np.float32),
    ]
    store = ManualDataStore(tmp_path)
    store.write_case(case, "e2e", inputs, [np.ones(2, np.float32)])

    loaded = store.load_case(case, "e2e")
    switches = SimpleNamespace(plugin_path=("must-not-be-scanned",))
    backend = SimpleNamespace(alias=lambda: "npu")
    generate_inputs(case, switches, backend, object(), stored_inputs=loaded.inputs)

    assert isinstance(case.tensors[0], list)
    assert len(case.tensors[0]) == 2
    assert tuple(case.tensors[0][0].shape) == (2,)
    assert tuple(case.tensors[0][1].shape) == (3,)
    assert tuple(case.tensors[1].shape) == (2,)


def test_changed_csv_dtype_is_rejected_by_filename(tmp_path):
    case = _e2e_case("contract_case")
    store = ManualDataStore(tmp_path)
    store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    case.tensor_dtypes = ("float16", "float32")

    case.invalidate_flat_cache("tensor_dtypes")
    with pytest.raises(ManualDataError, match="filename dtype"):
        store.load_case(case, "e2e")


def test_precision_fields_can_change_without_invalidating_prepared_data(tmp_path):
    case = _e2e_case("precision_case")
    case.precision_tolerances = ((0.001, 0.001),)
    case.absolute_precision = (1e-5,)
    store = ManualDataStore(tmp_path)
    store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )

    case.precision_tolerances = ((0.005, 0.005),)
    case.absolute_precision = (1e-4,)

    assert store.load_case(case, "e2e").case_dir.name == "precision_case"


def test_corrupt_file_is_rejected_before_loading(tmp_path):
    case = _e2e_case("hash_case")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    with (case_dir / "input_0_float32.bin").open("ab") as stream:
        stream.write(b"corrupt")

    with pytest.raises(ManualDataError, match="byte size"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize("file_format", ["npy", "pt"])
def test_self_describing_format_rejects_same_size_wrong_shape(tmp_path, file_format):
    case = _e2e_case(f"wrong_{file_format}_shape")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
        file_format=file_format,
    )
    path = case_dir / f"input_0_float32.{file_format}"
    wrong_shape = np.zeros((1, 9), np.float32)
    if file_format == "npy":
        np.save(path, wrong_shape)
    else:
        import torch
        torch.save(torch.from_numpy(wrong_shape), path)

    with pytest.raises(ManualDataError, match="stored shape"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize("file_format", ["npy", "pt"])
def test_self_describing_format_rejects_same_size_wrong_dtype(tmp_path, file_format):
    case = _e2e_case(f"wrong_{file_format}_dtype")
    case.tensor_dtypes = ("bfloat16", "bfloat16")
    dtype = resolve_custom_numpy_dtypes(("bfloat16",))[0]
    inputs = [
        np.zeros((3, 3), dtype=dtype),
        np.zeros((2, 2), dtype=dtype),
    ]
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        inputs,
        [np.zeros((2, 2), dtype=dtype)],
        file_format=file_format,
    )
    path = case_dir / f"input_0_bfloat16.{file_format}"
    wrong_dtype = np.zeros((3, 3), np.int16)
    if file_format == "npy":
        np.save(path, wrong_dtype)
    else:
        import torch
        torch.save(torch.from_numpy(wrong_dtype), path)

    with pytest.raises(ManualDataError, match="dtype .* != filename dtype"):
        store.load_case(case, "e2e")


def test_pt_raw_byte_payload_keeps_shape_inside_data_file(tmp_path):
    import torch

    case = _e2e_case("raw_byte_pt_shape")
    case.tensor_dtypes = ("float128", "float32")
    inputs = [
        np.arange(9, dtype=np.float128).reshape(3, 3),
        np.zeros((2, 2), np.float32),
    ]
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        inputs,
        [np.zeros((2, 2), np.float32)],
        file_format="pt",
    )
    path = case_dir / "input_0_float128.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert set(payload) == {"ttk_raw_bytes", "ttk_shape"}
    assert tuple(payload["ttk_shape"]) == (3, 3)

    payload["ttk_shape"] = (1, 9)
    torch.save(payload, path)
    with pytest.raises(ManualDataError, match="stored shape"):
        store.load_case(case, "e2e")


def test_pt_legacy_torch_load_warns_about_pickle_fallback(tmp_path, monkeypatch, caplog):
    import torch

    case = _e2e_case("legacy_pt_load")
    store = ManualDataStore(tmp_path)
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    store.write_case(case, "e2e", inputs, [np.zeros((2, 2), np.float32)], file_format="pt")
    original_load = torch.load

    def legacy_load(*args, **kwargs):
        if "weights_only" in kwargs:
            raise TypeError("weights_only is unsupported")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", legacy_load)
    with caplog.at_level(logging.WARNING):
        loaded = store.load_case(case, "e2e")

    np.testing.assert_array_equal(loaded.inputs[0], inputs[0])
    assert "falling back to pickle-based loading" in caplog.text


def test_pt_load_type_error_does_not_fall_back_to_pickle(tmp_path, monkeypatch, caplog):
    import torch

    case = _e2e_case("pt_load_type_error")
    store = ManualDataStore(tmp_path)
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    store.write_case(case, "e2e", inputs, [np.zeros((2, 2), np.float32)], file_format="pt")

    def broken_load(path, map_location, weights_only):
        raise TypeError("payload decoder failed")

    monkeypatch.setattr(torch, "load", broken_load)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ManualDataError, match="payload decoder failed"):
            store.load_case(case, "e2e")

    assert "falling back to pickle-based loading" not in caplog.text


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_none_slots_are_explicit_empty_data_files(tmp_path, file_format):
    case = _e2e_case("none_input")
    case.tensor_view_shapes = (None, (2, 2))
    case.tensor_storage_shapes = (None, (2, 2))
    case.tensor_view_offsets = (None, 0)
    case.tensor_view_strides = (None, (2, 1))
    case._flat_tensor_view_shapes = None
    case._flat_tensor_storage_shapes = None
    case._flat_tensor_view_offsets = None
    case._flat_tensor_view_strides = None
    case_dir = ManualDataStore(tmp_path).write_case(
        case,
        "e2e",
        [None, np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format=file_format,
    )

    loaded = ManualDataStore(tmp_path).load_case(case, "e2e")

    assert loaded.inputs[0] is None
    assert (case_dir / f"input_0_none.{file_format}").stat().st_size == 0


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_missing_none_marker_is_rejected(tmp_path, file_format):
    case = _e2e_case("missing_none_marker")
    case.tensor_view_shapes = (None, (2, 2))
    case.tensor_storage_shapes = (None, (2, 2))
    case.tensor_view_offsets = (None, 0)
    case.tensor_view_strides = (None, (2, 1))
    case._flat_tensor_view_shapes = None
    case._flat_tensor_storage_shapes = None
    case._flat_tensor_view_offsets = None
    case._flat_tensor_view_strides = None
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [None, np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format=file_format,
    )
    (case_dir / f"input_0_none.{file_format}").unlink()

    with pytest.raises(ManualDataError, match="slot count|contiguous"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_zero_element_tensor_is_not_confused_with_none(tmp_path, file_format):
    case = _e2e_case("zero_element_tensor")
    case.tensor_view_shapes = ((0, 3), (2, 2))
    case.tensor_storage_shapes = ((0, 3), (2, 2))
    case.tensor_view_offsets = (0, 0)
    case.tensor_view_strides = ((3, 1), (2, 1))
    case._flat_tensor_view_shapes = None
    case._flat_tensor_storage_shapes = None
    case._flat_tensor_view_offsets = None
    case._flat_tensor_view_strides = None
    empty = np.empty((0, 3), np.float32)
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [empty, np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format=file_format,
    )

    loaded = store.load_case(case, "e2e")

    assert loaded.inputs[0].shape == (0, 3)
    assert loaded.inputs[0].dtype == np.dtype("float32")
    assert (case_dir / f"input_0_float32.{file_format}").is_file()


def test_unknown_sidecar_is_rejected(tmp_path):
    case = _e2e_case("no_sidecars")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    (case_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ManualDataError, match="unexpected file"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize(
    "formats, expected_format, expected_value",
    [
        (("pt", "npy"), "npy", 2.0),
        (("pt", "npy", "bin"), "bin", 3.0),
    ],
)
def test_mixed_data_formats_use_whole_dataset_priority(
        tmp_path, formats, expected_format, expected_value):
    case = _e2e_case("mixed_formats")
    values = {"pt": 1.0, "npy": 2.0, "bin": 3.0}
    target_store = ManualDataStore(tmp_path / "mixed")
    target_dir = target_store.case_dir(case.testcase_name)
    target_dir.mkdir(parents=True)
    for file_format in formats:
        source_store = ManualDataStore(tmp_path / f"source-{file_format}")
        source_dir = source_store.write_case(
            case,
            "e2e",
            [
                np.full((3, 3), values[file_format], np.float32),
                np.zeros((2, 2), np.float32),
            ],
            [np.full((2, 2), values[file_format], np.float32)],
            file_format=file_format,
        )
        for path in source_dir.iterdir():
            shutil.copy2(path, target_dir / path.name)

    loaded = target_store.load_case(case, "e2e")
    loaded_golden = loaded.load_goldens(
        references=[np.zeros((2, 2), np.float32)]
    )

    assert loaded.file_format == expected_format
    np.testing.assert_array_equal(
        loaded.inputs[0], np.full((3, 3), expected_value, np.float32)
    )
    np.testing.assert_array_equal(
        loaded_golden[0], np.full((2, 2), expected_value, np.float32)
    )


def test_incomplete_high_priority_format_does_not_fall_back(tmp_path):
    case = _e2e_case("incomplete_priority")
    target_store = ManualDataStore(tmp_path / "mixed")
    bin_dir = ManualDataStore(tmp_path / "bin").write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
        file_format="bin",
    )
    npy_dir = ManualDataStore(tmp_path / "npy").write_case(
        case,
        "e2e",
        [np.ones((3, 3), np.float32), np.ones((2, 2), np.float32)],
        [np.ones((2, 2), np.float32)],
        file_format="npy",
    )
    target_dir = target_store.case_dir(case.testcase_name)
    target_dir.mkdir(parents=True)
    for source_dir in (bin_dir, npy_dir):
        for path in source_dir.iterdir():
            shutil.copy2(path, target_dir / path.name)
    (target_dir / "golden_0_float32__shape_2x2.bin").unlink()

    loaded = target_store.load_case(case, "e2e")
    assert loaded.file_format == "bin"
    with pytest.raises(ManualDataError, match="golden slot count"):
        loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])


def test_long_case_name_maps_stably_within_directory_limit(tmp_path):
    name = "kernel command / " + "very-long-case-" * 20
    case = _e2e_case(name)
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )

    assert len(case_dir.name) <= 120
    assert store.case_dir(name) == case_dir
    assert store.load_case(case, "e2e").case_dir == case_dir
    assert store.case_dir(name + "changed") != case_dir


@pytest.mark.parametrize(
    "filename", ["input_1_float32.bin", "golden_0_float32__shape_2x2.bin"]
)
def test_missing_data_slot_is_rejected(tmp_path, filename):
    case = _e2e_case("missing_slot")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    (case_dir / filename).unlink()

    with pytest.raises(ManualDataError, match="slot count|contiguous"):
        loaded = store.load_case(case, "e2e")
        loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])


def test_extra_data_slot_is_rejected(tmp_path):
    case = _e2e_case("extra_slot")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    (case_dir / "input_2_float32.bin").write_bytes(b"")

    with pytest.raises(ManualDataError, match="slot count"):
        store.load_case(case, "e2e")


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_nonempty_none_marker_is_rejected(tmp_path, file_format):
    case = _e2e_case("none_golden")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [None],
        file_format=file_format,
    )
    (case_dir / f"golden_0_none.{file_format}").write_bytes(b"not-empty")

    loaded = store.load_case(case, "e2e")
    with pytest.raises(ManualDataError, match="must be empty"):
        loaded.load_goldens(references=[None])


def test_bin_golden_filename_shape_matches_device_output(tmp_path):
    case = _e2e_case("deferred_bin_shape")
    store = ManualDataStore(tmp_path)
    golden = np.arange(4, dtype=np.float32).reshape(2, 2)
    case_dir = store.write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [golden],
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])

    assert restored[0].shape == (2, 2)
    assert (case_dir / "golden_0_float32__shape_2x2.bin").is_file()
    np.testing.assert_array_equal(restored[0], golden)


def test_bin_golden_rejects_same_numel_wrong_device_shape(tmp_path):
    case = _e2e_case("same_numel_wrong_shape")
    store = ManualDataStore(tmp_path)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.arange(6, dtype=np.float32).reshape(2, 3)],
    )

    loaded = store.load_case(case, "e2e")
    with pytest.raises(ManualDataError, match="saved shape .*device output shape"):
        loaded.load_goldens(references=[np.zeros((3, 2), np.float32)])


@pytest.mark.parametrize(
    "shape, token",
    [
        ((), "scalar"),
        ((0,), "0"),
        ((2, 0, 3), "2x0x3"),
    ],
)
def test_bin_golden_shape_filename_handles_scalar_and_zero_dimensions(
        tmp_path, shape, token):
    case = _e2e_case(f"shape_{token}")
    store = ManualDataStore(tmp_path)
    golden = np.zeros(shape, dtype=np.float32)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [golden],
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.zeros(shape, np.float32)])

    assert (case_dir / f"golden_0_float32__shape_{token}.bin").is_file()
    expected_storage_shape = (1,) if shape == () else shape
    assert restored[0].shape == expected_storage_shape


@pytest.mark.parametrize("file_format", ["bin", "npy", "pt"])
def test_scalar_golden_round_trip_preserves_shape(tmp_path, file_format):
    case = _e2e_case(f"scalar_{file_format}")
    store = ManualDataStore(tmp_path)
    golden = np.array(1.0, dtype=np.float32)
    store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [golden],
        file_format=file_format,
    )

    loaded = store.load_case(case, "e2e")
    restored = loaded.load_goldens(references=[np.array(0.0, np.float32)])

    assert restored[0].shape == (1,)
    np.testing.assert_array_equal(restored[0], golden)


def test_bin_golden_without_shape_suffix_is_rejected(tmp_path):
    case = _e2e_case("bin_golden_without_shape")
    store = ManualDataStore(tmp_path)
    case_dir = store.write_case(
        case,
        "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.arange(4, dtype=np.float32).reshape(2, 2)],
    )
    current = case_dir / "golden_0_float32__shape_2x2.bin"
    current.rename(case_dir / "golden_0_float32.bin")

    with pytest.raises(ManualDataError, match="must include a shape suffix"):
        store.load_case(case, "e2e")


def test_golden_sentinel_never_publishes_case_directory(tmp_path):
    case = _e2e_case("failed_prepare")
    store = ManualDataStore(tmp_path)

    with pytest.raises(ManualDataError, match="sentinel"):
        store.write_case(
            case, "e2e",
            [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
            ["GOLDEN_FAILURE"],
        )

    assert not store.case_dir(case.testcase_name).exists()


def test_failed_reprepare_invalidates_previous_case(tmp_path):
    case = _e2e_case("failed_reprepare")
    store = ManualDataStore(tmp_path)
    inputs = [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    old_golden = np.ones((2, 2), np.float32)
    store.write_case(case, "e2e", inputs, [old_golden])

    with pytest.raises(ManualDataError, match="sentinel"):
        store.write_case(case, "e2e", inputs, ["GOLDEN_FAILURE"])

    assert not store.case_dir(case.testcase_name).exists()


def test_invalidate_case_unlinks_directory_symlink_without_touching_target(tmp_path):
    case = _e2e_case("symlinked_case")
    store = ManualDataStore(tmp_path / "prepared")
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    case_dir = store.case_dir(case.testcase_name)
    case_dir.parent.mkdir(parents=True)
    case_dir.symlink_to(external, target_is_directory=True)

    store.invalidate_case(case.testcase_name)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not case_dir.is_symlink()


def test_prepare_and_replay_helpers_share_store_policy(tmp_path):
    prepare_case = _e2e_case("prepare_helper")
    prepare_switches = SimpleNamespace(
        manual_data_mode="prepare", manual_data_dirs=(str(tmp_path),)
    )

    prepare_store = prepare_manual_data_store(prepare_case, "e2e", prepare_switches)

    assert isinstance(prepare_store, ManualDataStore)
    assert not prepare_store.case_dir(prepare_case.testcase_name).exists()

    replay_case = _e2e_case("replay_helper")
    inputs = [np.ones((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    ManualDataStore(tmp_path).write_case(
        replay_case, "e2e", inputs, [np.zeros((2, 2), np.float32)]
    )
    replay_switches = SimpleNamespace(
        manual_data_mode="replay",
        manual_data_dirs=(str(tmp_path),),
        golden_mode="Enable",
        validate_only=False,
        force_cpu=False,
    )
    loaded_status = []

    manual_case = load_manual_data_case(
        replay_case, "e2e", replay_switches, before_load=lambda: loaded_status.append(True)
    )

    assert loaded_status == [True]
    np.testing.assert_array_equal(manual_case.inputs[0], inputs[0])


def test_registered_provider_is_an_extension_point_for_future_csv_sources(tmp_path):
    case = _e2e_case("provider_case")
    ManualDataStore(tmp_path).write_case(
        case, "e2e",
        [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)],
        [np.zeros((2, 2), np.float32)],
    )
    switches = SimpleNamespace(manual_data_dirs=())

    def provider(testcase, case_type, current_switches):
        assert testcase is case
        assert case_type == "e2e"
        assert current_switches is switches
        return tmp_path

    register_manual_data_directory_provider(provider)
    try:
        loaded = manual_data_store(case, "e2e", switches).load_case(case, "e2e")
    finally:
        unregister_manual_data_directory_provider(provider)

    assert loaded.case_dir == ManualDataStore(tmp_path).case_dir(case.testcase_name)


def test_case_provider_takes_priority_over_cli_batch_directories(tmp_path):
    case = _e2e_case("provider_priority")
    inputs = [np.zeros((3, 3), np.float32), np.zeros((2, 2), np.float32)]
    cli_root = tmp_path / "cli"
    provider_root = tmp_path / "provider"
    ManualDataStore(cli_root).write_case(
        case, "e2e", inputs, [np.full((2, 2), 1, np.float32)]
    )
    ManualDataStore(provider_root).write_case(
        case, "e2e", inputs, [np.full((2, 2), 2, np.float32)]
    )
    switches = SimpleNamespace(manual_data_dirs=(cli_root,))

    def provider(*_):
        return provider_root

    register_manual_data_directory_provider(provider)
    try:
        loaded = manual_data_store(case, "e2e", switches).load_case(case, "e2e")
    finally:
        unregister_manual_data_directory_provider(provider)

    assert loaded.case_dir.parent == provider_root.resolve()
    np.testing.assert_array_equal(
        loaded.load_goldens(references=[np.zeros((2, 2), np.float32)])[0],
        np.full((2, 2), 2, np.float32),
    )


def test_provider_replay_obeys_device_stage_constraints(tmp_path):
    case = _e2e_case("provider_constraints")
    switches = SimpleNamespace(
        manual_data_dirs=(), golden_mode="Enable", validate_only=False, force_cpu=True
    )

    def provider(*_):
        return tmp_path

    register_manual_data_directory_provider(provider)
    try:
        with pytest.raises(ManualDataError, match="cannot use --cpu"):
            replay_manual_data_store(case, "e2e", switches)
    finally:
        unregister_manual_data_directory_provider(provider)


def test_e2e_restore_rebuilds_view_without_running_input_plugin():
    case = _e2e_case("restore_e2e")
    storage = np.arange(9, dtype=np.float32).reshape(3, 3)
    output = np.zeros((2, 2), dtype=np.float32)
    switches = SimpleNamespace(plugin_path=("must-not-be-scanned",))
    backend = SimpleNamespace(alias=lambda: "npu")

    raw = generate_inputs(
        case, switches, backend, object(), stored_inputs=[storage, output]
    )

    assert tuple(case.tensors[0].stride()) == (3, 1)
    assert float(case.tensors[0][0, 0]) == float(storage.ravel()[1])
    assert np.shares_memory(raw[0], storage)


def test_aclnn_restore_rebuilds_tensor_and_scalar_without_plugins(monkeypatch):
    case = _aclnn_case()
    switches = SimpleNamespace(plugin_path=("must-not-be-scanned",))
    monkeypatch.setattr(
        "ttk.core_modules.npu.op_api.input_generation.get_global_storage",
        lambda: switches,
    )
    inputs = [np.arange(2, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    scalar = np.array(0.5, dtype=np.float32)

    InputGenerator(case).gen(stored_inputs=inputs, stored_scalars=[scalar])

    assert tuple(case.tensors[0].stride()) == (1,)
    assert float(case.scalars[0]) == pytest.approx(0.5)
