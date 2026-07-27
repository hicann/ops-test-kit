#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Restorable input and CPU-golden data shared by E2E, ACLNN, and Kernel.

Directory providers are process-global extensions. Callers that register one must
unregister it in ``finally`` (or an equivalent teardown) so it cannot affect an
unrelated case in the same process.
"""

import hashlib
import inspect
import logging
import math
import os
import pathlib
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy

from ttk.utilities import dump_to_file, load_numpy_data, resolve_custom_numpy_dtypes
from ttk.utilities.dtypes import torch_to_numpy_tensor

SUPPORTED_MANUAL_DATA_FORMATS = ("bin", "npy", "pt")
_SAFE_CASE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_DATA_FILE_NAME = re.compile(
    r"^(input|scalar|golden)_(0|[1-9][0-9]*)_([A-Za-z0-9][A-Za-z0-9_.-]*?)"
    r"(?:__shape_(scalar|[0-9]+(?:x[0-9]+)*))?\.(bin|pt|npy)$"
)
_NONE_DTYPE = "none"
_UNKNOWN_SHAPE = object()
_DIRECTORY_PROVIDERS: List[Callable] = []
_DIRECTORY_PROVIDERS_LOCK = threading.RLock()
_CUSTOM_NUMPY_DTYPES = {
    "bfloat16", "int4", "float8_e5m2", "float8_e4m3fn", "float8_e8m0",
    "float4_e2m1", "float4_e1m2", "hifloat8", "hifloat4",
}


class ManualDataError(RuntimeError):
    """Prepared data is missing, corrupt, or incompatible with the CSV case."""


@dataclass(frozen=True)
class _ManualDataFile:
    role: str
    index: int
    dtype: str
    file_format: str
    path: pathlib.Path
    saved_shape: Optional[Tuple[int, ...]] = None


@dataclass
class ManualDataCase:
    inputs: List[Any]
    scalars: List[Any]
    case_dir: pathlib.Path
    file_format: str
    _golden_files: Tuple[_ManualDataFile, ...]

    def load_goldens(self, references=None, shapes=None, dtypes=None) -> List[Any]:
        """Load saved goldens after device output shapes are available.

        Raw ``bin`` payloads do not carry shape metadata, so non-None bin Golden
        filenames encode it. E2E passes device outputs as references, while ACLNN
        passes returned output shapes and the CSV output dtypes.
        """
        if references is not None and shapes is not None:
            raise ManualDataError("golden references and explicit shapes are mutually exclusive")

        expected_shapes = []
        storage_shapes = []
        if references is not None:
            for index, value in enumerate(references):
                if value is None:
                    expected_shapes.append(None)
                    storage_shapes.append(None)
                elif isinstance(value, str):
                    expected_shapes.append(_UNKNOWN_SHAPE)
                    storage_shapes.append(_UNKNOWN_SHAPE)
                else:
                    label = f"output[{index}]"
                    logical_shape = _value_shape(value, label)
                    storage = _as_numpy(value, label)
                    storage_shape = tuple(storage.shape)
                    expected_shape = (
                        logical_shape if not logical_shape and storage.size == 1
                        else storage_shape
                    )
                    expected_shapes.append(expected_shape)
                    storage_shapes.append(storage_shape)
        elif shapes is not None:
            shape_values = list(shapes)
            dtype_values = list(dtypes or ())
            if dtypes is not None and len(dtype_values) != len(shape_values):
                raise ManualDataError(
                    f"golden dtype count {len(dtype_values)} != output shape count {len(shape_values)}"
                )
            for index, shape in enumerate(shape_values):
                if shape is None:
                    expected_shapes.append(None)
                    storage_shapes.append(None)
                    continue
                logical_shape = tuple(int(dimension) for dimension in shape)
                if dtypes is None:
                    storage_shape = _saved_array_shape(shape)
                else:
                    storage_shape = _physical_array_spec(dtype_values[index], shape)[1]
                expected_shape = (
                    logical_shape if not logical_shape and math.prod(storage_shape) == 1
                    else storage_shape
                )
                expected_shapes.append(expected_shape)
                storage_shapes.append(storage_shape)
        else:
            expected_shapes = [_UNKNOWN_SHAPE] * len(self._golden_files)
            storage_shapes = [_UNKNOWN_SHAPE] * len(self._golden_files)

        if len(expected_shapes) != len(self._golden_files):
            raise ManualDataError(
                f"golden slot count {len(self._golden_files)} != device output count "
                f"{len(expected_shapes)}"
            )

        values = []
        for entry, expected_shape, storage_shape in zip(
                self._golden_files, expected_shapes, storage_shapes):
            if expected_shape is None:
                values.append(_load_none(entry, "golden"))
                continue
            if entry.dtype == _NONE_DTYPE:
                raise ManualDataError(
                    f"golden[{entry.index}] is saved as None but the device returned an output"
                )
            if entry.saved_shape is not None:
                if (expected_shape is not _UNKNOWN_SHAPE and
                        tuple(expected_shape) != entry.saved_shape):
                    raise ManualDataError(
                        f"golden[{entry.index}] saved shape {entry.saved_shape} "
                        f"!= device output shape {tuple(expected_shape)}"
                    )
            if storage_shape is _UNKNOWN_SHAPE:
                storage_shape = entry.saved_shape
            values.append(_load_array(entry, storage_shape))
        return values


def register_manual_data_directory_provider(provider: Callable) -> Callable:
    """Register a per-case manual-data directory provider.

    A provider receives ``(testcase, case_type, switches)`` and returns one path
    or an iterable of paths. Provider paths take priority over CLI directories.
    Registrations are process-global; unregister dynamically registered providers
    in ``finally`` or test teardown.
    """
    if not callable(provider):
        raise TypeError("manual data directory provider must be callable")
    with _DIRECTORY_PROVIDERS_LOCK:
        if provider not in _DIRECTORY_PROVIDERS:
            _DIRECTORY_PROVIDERS.append(provider)
    return provider


def unregister_manual_data_directory_provider(provider: Callable):
    """Remove a process-global directory provider registered by the caller."""
    with _DIRECTORY_PROVIDERS_LOCK:
        if provider in _DIRECTORY_PROVIDERS:
            _DIRECTORY_PROVIDERS.remove(provider)


def resolve_manual_data_directories(testcase, case_type: str, switches,
                                    include_providers: bool = True,
                                    required: bool = True) -> Tuple[pathlib.Path, ...]:
    """Resolve per-case provider paths before CLI batch-search directories."""
    values: List[Any] = []
    if include_providers:
        with _DIRECTORY_PROVIDERS_LOCK:
            providers = tuple(_DIRECTORY_PROVIDERS)
        for provider in providers:
            provided = provider(testcase, case_type, switches)
            if provided is None:
                continue
            if isinstance(provided, (str, os.PathLike)):
                values.append(provided)
            else:
                values.extend(provided)
    values.extend(getattr(switches, "manual_data_dirs", ()) or ())

    result = []
    seen = set()
    for value in values:
        path = pathlib.Path(value).expanduser().resolve()
        if path not in seen:
            result.append(path)
            seen.add(path)
    if not result and required:
        raise ManualDataError(
            "no manual data directory was resolved; pass --manual-data-dirs "
            "or register a data-source provider"
        )
    return tuple(result)


def manual_data_store(testcase, case_type: str, switches,
                      include_providers: bool = True) -> "ManualDataStore":
    return ManualDataStore(resolve_manual_data_directories(
        testcase, case_type, switches, include_providers=include_providers
    ))


def replay_manual_data_store(testcase, case_type: str, switches) -> Optional["ManualDataStore"]:
    """Return a replay store selected by a per-case provider or the CLI."""
    directories = resolve_manual_data_directories(
        testcase, case_type, switches, include_providers=True, required=False
    )
    if directories:
        if getattr(switches, "golden_mode", "Enable") != "Enable":
            raise ManualDataError("manual-data replay requires --golden-mode Enable")
        if getattr(switches, "validate_only", False):
            raise ManualDataError("manual-data replay cannot be combined with --validate")
        if case_type == "e2e" and getattr(switches, "force_cpu", False):
            raise ManualDataError("manual-data replay is the device stage and cannot use --cpu")
        return ManualDataStore(directories)
    if getattr(switches, "manual_data_mode", None) == "replay":
        raise ManualDataError("--manual-data-dirs did not resolve any usable directory")
    return None


def prepare_manual_data_store(testcase, case_type: str, switches) -> Optional["ManualDataStore"]:
    """Create and invalidate the one output store selected for prepare mode."""
    if getattr(switches, "manual_data_mode", None) != "prepare":
        return None
    store = manual_data_store(testcase, case_type, switches, include_providers=False)
    store.invalidate_case(testcase.testcase_name)
    return store


def load_manual_data_case(testcase, case_type: str, switches,
                          before_load: Optional[Callable[[], None]] = None) -> Optional[ManualDataCase]:
    """Load a replay case through the shared provider and CLI directory policy."""
    if getattr(switches, "manual_data_mode", None) == "prepare":
        return None
    store = replay_manual_data_store(testcase, case_type, switches)
    if store is None:
        return None
    if before_load is not None:
        before_load()
    return store.load_case(testcase, case_type)


def snapshot_manual_values(values: Sequence[Any], label: str) -> List[Any]:
    """Copy values before golden callbacks can mutate their backing storage."""
    snapshots = []
    for index, value in enumerate(values or ()):
        if value is None:
            snapshots.append(None)
        else:
            snapshots.append(_as_numpy(value, f"{label}[{index}]").copy())
    return snapshots


def _case_directory_name(testcase_name: str) -> str:
    safe = _SAFE_CASE_NAME.sub("_", testcase_name).strip("._") or "case"
    if safe == testcase_name and len(safe) <= 120:
        return safe
    digest = hashlib.sha256(testcase_name.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:96]}-{digest}"


def _encode_shape(shape: Sequence[int]) -> str:
    shape = tuple(int(dimension) for dimension in shape)
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def _decode_shape(token: Optional[str]) -> Optional[Tuple[int, ...]]:
    if token is None:
        return None
    if token == "scalar":
        return ()
    return tuple(int(dimension) for dimension in token.split("x"))


def _as_numpy(value, label: str) -> numpy.ndarray:
    if isinstance(value, str):
        raise ManualDataError(f"cannot save {label} sentinel {value!r}")
    if isinstance(value, numpy.ndarray):
        array = value
    elif isinstance(value, numpy.generic):
        array = numpy.asarray(value)
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        array = torch_to_numpy_tensor(value.detach().cpu())
    else:
        try:
            array = numpy.asarray(value)
        except Exception as exc:
            raise ManualDataError(f"unsupported {label} value type: {type(value)!r}") from exc
    if array.dtype.hasobject or array.dtype.fields:
        raise ManualDataError(f"cannot save {label} with dtype {array.dtype}")
    return numpy.ascontiguousarray(array)


def _value_shape(value, label: str) -> Tuple[int, ...]:
    """Return logical shape before contiguous conversion promotes scalars to (1,)."""
    if isinstance(value, numpy.ndarray):
        return tuple(value.shape)
    if hasattr(value, "shape"):
        return tuple(int(dimension) for dimension in value.shape)
    try:
        return tuple(numpy.asarray(value).shape)
    except Exception as exc:
        raise ManualDataError(f"cannot determine {label} shape: {type(value)!r}") from exc


def _saved_array_shape(shape) -> Tuple[int, ...]:
    result = tuple(int(dimension) for dimension in shape)
    return result if result else (1,)


def _physical_array_spec(logical_dtype, logical_shape) -> Tuple[str, Tuple[int, ...]]:
    """Return the dtype/shape of the numpy backing array saved by TTK."""
    dtype_name = getattr(logical_dtype, "name", str(logical_dtype))
    shape = tuple(int(dimension) for dimension in logical_shape)
    if dtype_name == "complex32":
        return "float16", shape + (2,)
    if dtype_name in ("uint1", "int1"):
        element_count = math.prod(shape) if shape else 1
        return "uint8", ((element_count + 7) // 8,)
    try:
        resolved = resolve_custom_numpy_dtypes((dtype_name,))[0]
        return numpy.dtype(resolved).name, _saved_array_shape(shape)
    except Exception as exc:
        raise ManualDataError(f"cannot resolve manual-data dtype {dtype_name!r}") from exc


def _input_specs(testcase, case_type: str) -> List[Optional[Tuple[str, Tuple[int, ...]]]]:
    if case_type == "kernel":
        shapes = list(getattr(testcase, "flat_input_shapes", ()) or ())
        dtypes = list(getattr(testcase, "flat_input_dtypes", ()) or ())
    else:
        shapes = list(getattr(testcase, "flat_tensor_view_shapes", ()) or ())
        dtypes = list(getattr(testcase, "flat_tensor_dtypes", ()) or ())
    if len(dtypes) != len(shapes):
        raise ManualDataError(f"CSV tensor dtype count {len(dtypes)} != tensor slot count {len(shapes)}")
    result = []
    for index, shape in enumerate(shapes):
        if shape is None:
            result.append(None)
        elif case_type == "kernel":
            result.append(_physical_array_spec(dtypes[index], shape))
        else:
            result.append(_physical_array_spec(dtypes[index], testcase.flat_storage_shape(index)))
    return result


def _scalar_specs(testcase, case_type: str) -> List[Optional[Tuple[str, Tuple[int, ...]]]]:
    if case_type != "aclnn":
        return []
    result = []
    for dtype in (getattr(testcase, "flat_scalar_dtypes", ()) or ()):
        if dtype is None:
            result.append(None)
        else:
            result.append(_physical_array_spec(dtype, ()))
    return result


def _expected_golden_count(testcase, case_type: str) -> Optional[int]:
    if case_type in ("aclnn", "kernel"):
        return len(getattr(testcase, "flat_output_dtypes", ()) or ())
    return None


def _resolve_file_dtype(dtype_name: str):
    try:
        return resolve_custom_numpy_dtypes((dtype_name,))[0]
    except Exception as exc:
        raise ManualDataError(f"cannot resolve dtype {dtype_name!r} from manual-data filename") from exc


def _expected_bin_size(dtype_name: str, shape: Tuple[int, ...]) -> int:
    count = math.prod(shape)
    if "int4" in dtype_name or "float4" in dtype_name:
        return (count + 1) // 2
    try:
        return count * numpy.dtype(_resolve_file_dtype(dtype_name)).itemsize
    except Exception as exc:
        if isinstance(exc, ManualDataError):
            raise
        raise ManualDataError(f"cannot determine byte size for dtype {dtype_name!r}") from exc


def _load_pt(path: pathlib.Path, dtype, shape: Optional[Tuple[int, ...]]) -> numpy.ndarray:
    try:
        import torch

        try:
            supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
        except (TypeError, ValueError):
            supports_weights_only = True
        if supports_weights_only:
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
        else:
            logging.warning(
                "torch.load on this Torch version does not support weights_only=True; "
                "falling back to pickle-based loading for %s. Only load trusted manual-data files.",
                path,
            )
            payload = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise ManualDataError(f"cannot load pt file {path}: {exc}") from exc

    if isinstance(payload, dict) and set(payload) == {"ttk_raw_bytes", "ttk_shape"}:
        raw_tensor = payload["ttk_raw_bytes"]
        if not hasattr(raw_tensor, "detach"):
            raise ManualDataError(f"invalid raw-byte pt payload: {path}")
        try:
            stored_shape = tuple(int(dimension) for dimension in payload["ttk_shape"])
        except Exception as exc:
            raise ManualDataError(f"invalid raw-byte pt shape: {path}") from exc
        if shape is not None and stored_shape != tuple(shape):
            raise ManualDataError(
                f"{path} stored shape {stored_shape} != expected {tuple(shape)}"
            )
        raw = raw_tensor.detach().cpu().contiguous().numpy().tobytes()
        try:
            array = numpy.frombuffer(raw, dtype=dtype).copy().reshape(stored_shape)
        except ValueError as exc:
            raise ManualDataError(
                f"{path} raw bytes cannot be reshaped to stored shape {stored_shape}"
            ) from exc
    elif hasattr(payload, "detach"):
        array = torch_to_numpy_tensor(payload.detach().cpu())
        if shape is not None and tuple(array.shape) != tuple(shape):
            raise ManualDataError(
                f"{path} stored shape {tuple(array.shape)} != expected {tuple(shape)}"
            )
    else:
        raise ManualDataError(f"invalid pt payload: {path}")
    return array


def _load_npy(path: pathlib.Path, dtype, shape: Optional[Tuple[int, ...]]) -> numpy.ndarray:
    array = numpy.load(str(path), allow_pickle=False)
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ManualDataError(
            f"{path} stored shape {tuple(array.shape)} != expected {tuple(shape)}"
        )

    expected_dtype = numpy.dtype(dtype)
    if array.dtype.name == expected_dtype.name:
        return array
    if expected_dtype.name not in _CUSTOM_NUMPY_DTYPES:
        raise ManualDataError(
            f"{path} stored dtype {array.dtype.name!r} != filename dtype {expected_dtype.name!r}"
        )
    if array.dtype.kind != "V":
        raise ManualDataError(
            f"{path} stored dtype {array.dtype.name!r} != filename dtype {expected_dtype.name!r}"
        )
    if array.dtype.itemsize != expected_dtype.itemsize:
        raise ManualDataError(
            f"{path} itemsize {array.dtype.itemsize} != filename dtype itemsize "
            f"{expected_dtype.itemsize}"
        )
    return array.view(dtype)


def _load_array(entry: _ManualDataFile, shape: Optional[Tuple[int, ...]]) -> numpy.ndarray:
    if entry.dtype == _NONE_DTYPE:
        raise ManualDataError(f"{entry.role}[{entry.index}] is None, not an array")
    dtype = _resolve_file_dtype(entry.dtype)
    try:
        if entry.file_format == "bin":
            if shape is not None:
                expected_size = _expected_bin_size(entry.dtype, shape)
                if entry.path.stat().st_size != expected_size:
                    raise ManualDataError(
                        f"{entry.path} byte size {entry.path.stat().st_size} != expected {expected_size}"
                    )
                array = load_numpy_data(str(entry.path), dtype, shape)
            else:
                array = numpy.fromfile(str(entry.path), dtype=dtype)
        elif entry.file_format == "npy":
            array = _load_npy(entry.path, dtype, shape)
        elif entry.file_format == "pt":
            array = _load_pt(entry.path, dtype, shape)
        else:
            raise ManualDataError(f"unsupported manual data format: {entry.file_format!r}")
    except ManualDataError:
        raise
    except Exception as exc:
        raise ManualDataError(f"cannot load {entry.path}: {exc}") from exc

    if not array.flags.c_contiguous:
        array = numpy.ascontiguousarray(array)
    expected_dtype = numpy.dtype(dtype).name
    if array.dtype.name != expected_dtype:
        raise ManualDataError(
            f"{entry.path} dtype {array.dtype.name!r} != filename dtype {expected_dtype!r}"
        )
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ManualDataError(f"{entry.path} shape {tuple(array.shape)} != expected {tuple(shape)}")
    return array


def _load_none(entry: _ManualDataFile, role: str):
    if entry.dtype != _NONE_DTYPE:
        raise ManualDataError(f"{role}[{entry.index}] should be None but file dtype is {entry.dtype!r}")
    if entry.path.stat().st_size != 0:
        raise ManualDataError(f"None marker must be empty: {entry.path}")
    return None


class ManualDataStore:
    """Write one complete per-case dataset and replay it by testcase name."""

    def __init__(self, roots: Iterable[Any]):
        if isinstance(roots, (str, os.PathLike)):
            roots = (roots,)
        self.roots = tuple(pathlib.Path(root).expanduser().resolve() for root in roots)
        if not self.roots:
            raise ManualDataError("at least one manual data directory is required")

    def case_dir(self, testcase_name: str, root_index: int = 0) -> pathlib.Path:
        return self.roots[root_index] / _case_directory_name(testcase_name)

    def invalidate_case(self, testcase_name: str):
        """Remove an older same-name case before a new prepare attempt."""
        if len(self.roots) != 1:
            raise ManualDataError("manual data preparation requires exactly one output directory")
        self._remove_existing(self.case_dir(testcase_name))

    def write_case(self, testcase, case_type: str, inputs, goldens, scalars=(),
                   file_format: str = "bin") -> pathlib.Path:
        self._validate_case_type(case_type)
        if file_format not in SUPPORTED_MANUAL_DATA_FORMATS:
            raise ManualDataError(
                f"manual data format {file_format!r} is not supported; "
                f"choose {', '.join(SUPPORTED_MANUAL_DATA_FORMATS)}"
            )
        if len(self.roots) != 1:
            raise ManualDataError("manual data preparation requires exactly one output directory")

        input_values = list(inputs or ())
        scalar_values = list(scalars or ())
        golden_values = list(goldens or ())
        input_specs = _input_specs(testcase, case_type)
        scalar_specs = _scalar_specs(testcase, case_type)
        self._validate_value_count("input", input_values, len(input_specs))
        self._validate_value_count("scalar", scalar_values, len(scalar_specs))
        expected_goldens = _expected_golden_count(testcase, case_type)
        if expected_goldens is not None:
            self._validate_value_count("golden", golden_values, expected_goldens)

        root = self.roots[0]
        root.mkdir(parents=True, exist_ok=True)
        target = self.case_dir(testcase.testcase_name)
        self._remove_existing(target)
        temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(root)))
        try:
            self._write_values(temporary, input_values, "input", file_format, input_specs)
            self._write_values(temporary, scalar_values, "scalar", file_format, scalar_specs)
            self._write_values(temporary, golden_values, "golden", file_format)
            self._scan_case(temporary)
            os.replace(str(temporary), str(target))
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise
        logging.info("[%s] prepared restorable input/golden data at %s",
                     testcase.testcase_name, target)
        return target

    def load_case(self, testcase, case_type: str) -> ManualDataCase:
        self._validate_case_type(case_type)
        case_dir = self._find_case(testcase.testcase_name)
        entries, file_format = self._scan_case(case_dir)
        input_specs = _input_specs(testcase, case_type)
        input_files = self._ordered_files(entries, "input", len(input_specs))
        scalar_specs = _scalar_specs(testcase, case_type)
        scalar_files = self._ordered_files(entries, "scalar", len(scalar_specs))

        golden_count = _expected_golden_count(testcase, case_type)
        golden_files = self._ordered_files(entries, "golden", golden_count)
        inputs = self._load_expected_values(input_files, input_specs, "input")
        scalars = self._load_expected_values(scalar_files, scalar_specs, "scalar")
        logging.info("[%s] loaded prepared input/scalar data from %s",
                     testcase.testcase_name, case_dir)
        return ManualDataCase(
            inputs, scalars, case_dir, file_format, tuple(golden_files)
        )

    @staticmethod
    def _validate_case_type(case_type: str):
        if case_type not in ("e2e", "aclnn", "kernel"):
            raise ManualDataError(f"unsupported manual data case type: {case_type}")

    @staticmethod
    def _validate_value_count(role: str, values: Sequence[Any], expected: int):
        if len(values) != expected:
            raise ManualDataError(f"{role} slot count {len(values)} != CSV {expected}")

    @staticmethod
    def _remove_existing(path: pathlib.Path):
        # A symlink must be unlinked before directory handling can follow it.
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(str(path))

    def _write_values(self, case_dir: pathlib.Path, values: Sequence[Any], role: str,
                      file_format: str,
                      specs: Optional[Sequence[Optional[Tuple[str, Tuple[int, ...]]]]] = None):
        for index, value in enumerate(values):
            expected = specs[index] if specs is not None else None
            if value is None:
                if specs is not None and expected is not None:
                    raise ManualDataError(f"{role}[{index}] is None but CSV declares a tensor")
                path = case_dir / f"{role}_{index}_{_NONE_DTYPE}.{file_format}"
                path.touch()
                continue
            if specs is not None and expected is None:
                raise ManualDataError(f"{role}[{index}] has data but CSV declares None")

            label = f"{role}[{index}]"
            logical_shape = _value_shape(value, label)
            array = _as_numpy(value, label)
            if expected is not None:
                expected_dtype, expected_shape = expected
                if array.dtype.name != expected_dtype:
                    raise ManualDataError(
                        f"{role}[{index}] dtype {array.dtype.name!r} != CSV storage dtype "
                        f"{expected_dtype!r}"
                    )
                if tuple(array.shape) != expected_shape:
                    raise ManualDataError(
                        f"{role}[{index}] shape {tuple(array.shape)} != CSV storage shape "
                        f"{expected_shape}"
                    )

            dtype_name = array.dtype.name
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", dtype_name):
                raise ManualDataError(f"dtype {dtype_name!r} cannot be encoded in a data filename")
            stem = f"{role}_{index}_{dtype_name}"
            saved_shape = None
            if role == "golden" and file_format == "bin":
                saved_shape = (
                    logical_shape if not logical_shape and array.size == 1
                    else tuple(array.shape)
                )
                stem += f"__shape_{_encode_shape(saved_shape)}"
            path = case_dir / f"{stem}.{file_format}"
            self._write_array(path, array, stem, file_format)
            restored = _load_array(
                _ManualDataFile(
                    role, index, dtype_name, file_format, path, saved_shape
                ),
                tuple(array.shape),
            )
            is_packed_4bit = file_format == "bin" and (
                "int4" in dtype_name or "float4" in dtype_name
            )
            if is_packed_4bit:
                verified = numpy.array_equal(restored, array)
            else:
                verified = restored.tobytes(order="C") == array.tobytes(order="C")
            if not verified:
                raise ManualDataError(f"{path} did not pass write/read verification")

    @staticmethod
    def _write_array(path: pathlib.Path, array: numpy.ndarray,
                     stem: str, file_format: str):
        if file_format == "bin":
            dump_to_file(array, str(path.parent), stem, file_format=file_format)
            return
        if file_format == "npy":
            payload = array
            if array.dtype.name in _CUSTOM_NUMPY_DTYPES:
                payload = array.view(numpy.dtype(f"V{array.dtype.itemsize}"))
            dump_to_file(payload, str(path.parent), stem, file_format=file_format)
            return

        try:
            import torch

            from ttk.utilities.dtypes import numpy_to_torch_tensor

            try:
                payload = numpy_to_torch_tensor(array)
            except Exception:
                raw = numpy.frombuffer(array.tobytes(order="C"), dtype=numpy.uint8).copy()
                payload = {
                    "ttk_raw_bytes": torch.from_numpy(raw),
                    "ttk_shape": tuple(array.shape),
                }
            torch.save(payload, str(path))
        except Exception as exc:
            raise ManualDataError(f"cannot write pt file {path}: {exc}") from exc

    def _find_case(self, testcase_name: str) -> pathlib.Path:
        checked = []
        for root in self.roots:
            case_dir = root / _case_directory_name(testcase_name)
            checked.append(str(case_dir))
            if case_dir.exists():
                if case_dir.is_symlink() or not case_dir.is_dir():
                    raise ManualDataError(f"prepared testcase path is not a regular directory: {case_dir}")
                return case_dir
        raise ManualDataError(
            f"prepared testcase {testcase_name!r} was not found; checked: " + ", ".join(checked)
        )

    @staticmethod
    def _scan_case(case_dir: pathlib.Path):
        entries_by_format = {
            file_format: {} for file_format in SUPPORTED_MANUAL_DATA_FORMATS
        }
        try:
            paths = sorted(case_dir.iterdir(), key=lambda item: item.name)
        except Exception as exc:
            raise ManualDataError(f"cannot list prepared testcase directory {case_dir}: {exc}") from exc
        if not paths:
            raise ManualDataError(f"prepared testcase directory is empty: {case_dir}")

        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ManualDataError(f"manual-data case contains a non-regular file: {path}")
            match = _DATA_FILE_NAME.fullmatch(path.name)
            if match is None:
                raise ManualDataError(f"unexpected file in manual-data case: {path.name}")
            role, index_text, dtype_name, shape_token, current_format = match.groups()
            if (role == "golden" and current_format == "bin" and
                    dtype_name != _NONE_DTYPE and shape_token is None):
                raise ManualDataError(
                    f"non-None bin Golden filename must include a shape suffix: {path.name}"
                )
            if shape_token is not None and (
                    role != "golden" or current_format != "bin" or dtype_name == _NONE_DTYPE):
                raise ManualDataError(
                    f"shape suffix is only valid for non-None bin Golden files: {path.name}"
                )
            key = (role, int(index_text))
            format_entries = entries_by_format[current_format]
            if key in format_entries:
                raise ManualDataError(
                    f"duplicate {current_format} manual-data slot {role}[{index_text}]"
                )
            if dtype_name != _NONE_DTYPE:
                _resolve_file_dtype(dtype_name)
            format_entries[key] = _ManualDataFile(
                role,
                int(index_text),
                dtype_name,
                current_format,
                path,
                _decode_shape(shape_token),
            )

        available_formats = [
            file_format for file_format in SUPPORTED_MANUAL_DATA_FORMATS
            if entries_by_format[file_format]
        ]
        selected_format = available_formats[0]
        if len(available_formats) > 1:
            logging.info(
                "manual-data case %s contains formats %s; selected %s by priority",
                case_dir,
                available_formats,
                selected_format,
            )
        return entries_by_format[selected_format], selected_format

    @staticmethod
    def _ordered_files(entries, role: str, expected_count: Optional[int]):
        files = sorted(
            (entry for (entry_role, _), entry in entries.items() if entry_role == role),
            key=lambda entry: entry.index,
        )
        indexes = [entry.index for entry in files]
        if indexes != list(range(len(files))):
            raise ManualDataError(f"{role} file indexes must be contiguous from zero, got {indexes}")
        if expected_count is not None and len(files) != expected_count:
            raise ManualDataError(f"{role} slot count {len(files)} != CSV {expected_count}")
        return files

    @staticmethod
    def _load_expected_values(files, specs, role: str) -> List[Any]:
        values = []
        for entry, expected in zip(files, specs):
            if expected is None:
                values.append(_load_none(entry, role))
                continue
            expected_dtype, expected_shape = expected
            if entry.dtype == _NONE_DTYPE:
                raise ManualDataError(f"{role}[{entry.index}] is None but CSV declares a tensor")
            if entry.dtype != expected_dtype:
                raise ManualDataError(
                    f"{role}[{entry.index}] filename dtype {entry.dtype!r} != CSV storage dtype "
                    f"{expected_dtype!r}"
                )
            values.append(_load_array(entry, expected_shape))
        return values
