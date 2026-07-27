# Manual-Data Prepare and Replay

Two-stage execution separates input and CPU-golden generation from target-device
execution. It is intended for workflows where preparation and device execution run
on different hosts.

```text
prepare: CSV -> generate/customize inputs -> CPU golden -> save
replay:  CSV -> restore inputs/goldens -> NPU API or kernel -> compare
```

The feature supports `ttk e2e`, `ttk aclnn`, and `ttk kernel`. Both stages require
the same CSV and match data by `testcase_name`; TTK does not create cases by scanning
the data directory.

## 1. Select A Stage

| Mode | Options | Input | Golden | Device execution | Compare |
| --- | --- | --- | --- | --- | --- |
| Direct | No two-stage options | Generate | Generate | Run | Run |
| Prepare | `--no-prof --dump in,golden` | Generate and save | Generate and save | Skip | Skip |
| Replay | `--manual-data-dirs DIR...` | Restore | Restore | Run | Run |

Prepare currently saves input and golden together. Input-only and golden-only stages
are not supported. `--dump-format` accepts `bin`, `npy`, or `pt` and defaults to `bin`.

### Kernel `--no-prof` And `--co`

Standalone Kernel `--no-prof` keeps its original dry-run behavior: TTK generates
input, golden, and workspace data but disables dynamic, const, and binary execution.
Only the exact `--no-prof --dump in,golden` pair selects manual-data prepare.

`--co/--compile-only` returns after compilation or tiling and before input/golden
generation, so it cannot produce or consume a two-stage dataset. Kernel prepare still
runs the selected mode's compilation or binary lookup and tiling, then returns before
the device lock and target-kernel execution.

## 2. Quick Start

Prepare and replay must use compatible CSV and assets files.

### 2.1 E2E

```bash
# Prepare
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# Replay
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

E2E prepare may use `--cpu` to force the CPU backend. Replay is the device stage and
rejects `--cpu`.

### 2.2 ACLNN

```bash
# Prepare
python3 -m ttk aclnn \
  -i /path/to/aclnn_cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  --plat Ascend950 \
  -o /path/to/prepare_result.csv

# Replay
python3 -m ttk aclnn \
  -i /path/to/aclnn_cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

ACLNN prepare does not call the target API, query device count, or compile warmup
helper kernels. CSV and API metadata still require a CANN/OPP environment. Pass
`--plat` when the prepare host cannot detect the target SoC.

### 2.3 Kernel

Use the same dynamic, const, or release-binary selection in both stages. This example
uses a release binary:

```bash
# Prepare: compile/lookup and tile, then save without running the target kernel
python3 -m ttk kernel \
  -i /path/to/kernel_cases.csv \
  --plugin /path/to/kernel_assets \
  --plat Ascend910_9362 \
  -d=false -c=false -b=release \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# Replay: restore data, run the release kernel, and compare
python3 -m ttk kernel \
  -i /path/to/kernel_cases.csv \
  --plugin /path/to/kernel_assets \
  --plat Ascend910_9362 \
  -d=false -c=false -b=release \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

When TTK has no built-in input/golden implementation, the assets must register
`__input__["kernel"]` and `__golden__["kernel"]` for the CSV `op_name`. An E2E-only
TestSpec does not provide raw-kernel callbacks.

## 3. Data Directories

### 3.1 Prepare Output

An explicit `--manual-data-dirs DIR` is the prepare output. Without it, TTK uses:

1. `<plugin>/manual_data` for one directory plugin;
2. `<plugin-parent>/manual_data` for one `.py` plugin;
3. `<current-working-directory>/manual_data` without a plugin.

Prepare accepts one output root. Multiple plugin paths require an explicit output.

### 3.2 Replay Search Roots

```bash
python3 -m ttk e2e -i cases.csv \
  --manual-data-dirs /data/current /data/archive
```

TTK searches roots in argument order for each case. The first matching case directory
is authoritative. If it is corrupt, replay fails instead of searching a later copy or
falling back to random generation.

### 3.3 Case Directory Names

A `testcase_name` containing only `[A-Za-z0-9_.-]` and at most 120 characters is used
directly. Other names are sanitized and receive the first 12 hexadecimal characters
of the complete name's SHA-256. Prepare and replay map the same complete name to the
same directory; this is not fuzzy matching and the original long name cannot be
recovered from the directory alone.

Re-preparing a case invalidates its old directory first. New files are written and
verified in a hidden temporary directory and published only after the whole dataset
passes validation.

## 4. File Protocol

A case directory may contain:

```text
<manual-data-dir>/<case-directory>/
├── input_0_bfloat16.bin
├── input_1_int32.bin
├── input_2_none.bin
├── scalar_0_float32.bin
└── golden_0_bfloat16__shape_2x8x128.bin
```

The ordinary filename for inputs, scalars, `None` markers, and npy/pt goldens is:

```text
<input|scalar|golden>_<zero-based-flat-index>_<numpy-dtype|none>.<bin|npy|pt>
```

A non-None raw-bin golden also encodes its prepare-time output shape:

```text
golden_<index>_<dtype>__shape_<shape-token>.bin
```

Dimensions are joined with `x`: `(2, 0, 3)` becomes `2x0x3`, and scalar `()` becomes
`scalar`. Replay compares this shape with the device output or CSV output shape before
loading values. It therefore detects a wrong output shape even when the element count
is unchanged. A non-None bin Golden without the shape suffix is rejected; re-run
prepare to regenerate the case directory.

### 4.1 Slots And `None`

- Input, scalar, and golden indexes are independently contiguous from zero.
- Optional `None` uses a zero-byte `*_none.<format>` marker and must not be omitted.
- `*_none.npy` and `*_none.pt` are TTK markers, not native NumPy/PyTorch files.
- A zero-element tensor keeps its real dtype name, such as `input_0_float32.bin`.
- Inputs store final backing storage. The current CSV rebuilds views, strides, offsets,
  TensorList/ScalarList grouping, and pure-output roles.

### 4.2 Formats

| Format | Storage | Shape source |
| --- | --- | --- |
| `bin` | Contiguous raw bytes | CSV for input/scalar; filename checked against device/CSV shape for golden |
| `npy` | NumPy array | Embedded shape; custom dtypes use fixed-width `voidN` and the filename restores the logical dtype |
| `pt` | CPU Torch data | Embedded shape; unsupported Torch dtypes store raw bytes and shape in the same pt file |

Prepare writes one format. If a delivered directory contains multiple complete copies,
replay selects one whole dataset by `bin > npy > pt`; it never mixes input from one
format with golden from another. A present but incomplete or corrupt high-priority copy
fails instead of falling back.

JSON, logs, checksums, subdirectories, symlinks, and other sidecars are rejected. Each
non-None file is immediately read back and checked for dtype, shape, and complete bytes.

## 5. Replay Behavior

After a dataset matches, replay skips random input generation, attribute data fills,
the input plugin, and CPU golden generation. It still performs CSV/ParamPlan parsing,
API or kernel resolution, required compilation/tiling, wrappers, device execution,
comparison, and result output. Continue to pass the required plugin and `PYTHONPATH`.

Prepare snapshots final tensor backing storage and ACLNN scalars before calling the
golden callback. An in-place golden callback therefore cannot alter the input later
consumed by replay.

### Comparison

Two-stage execution does not replace comparison behavior:

- E2E pre-compare, custom compare, and built-in compare keep their precedence.
- ACLNN/Kernel tolerances and CSV precision fields use their normal path.
- Pass `--compare close` when CSV `rtol/ptol/atol` must select close comparison.
- Replay may change precision criteria without re-running prepare.

A custom compare must not depend on process-global state written by an input/golden
plugin during prepare. Replay on another process or host cannot restore that state.

## 6. Provider Extension

`register_manual_data_directory_provider(provider)` selects per-case data roots. A
provider receives `(testcase, case_type, switches)` and returns one path, ordered paths,
or `None`. Replay searches provider paths before CLI batch roots.

This supports a future/custom CSV field for exceptional cases while
`--manual-data-dirs` handles the batch. No dedicated CSV header is built in; extension
code may read `testcase.original_dict`. Prepare always writes to the CLI/default root,
and providers cannot bypass replay's device-stage constraints.

ACLNN `manual_tensor_binaries/manual_golden_binaries` are separate historical fields
and do not automatically register this provider or join the two-stage protocol.

## 7. Option Constraints

- `-i/--input` and `-o/--output` each accept one CSV.
- `--plugin` may contain comma-separated search paths in one argument.
- `--manual-data-dirs` accepts at most one prepare root and multiple replay roots.
- Prepare rejects output/full dump, `print`, `--dump-on-fail`,
  `--golden-mode Disable`, and `--validate`; E2E also rejects graph options, and
  Kernel rejects `--compile-only`.
- Replay rejects `--no-prof`, `--validate`, `--golden-mode Disable`, E2E `--cpu`, and
  Kernel `--compile-only`.
- `--seed` affects prepare-time random input only. Replay reads files and does not use
  the seed as a lookup key.
- Use the same `-t/--testcase` in both stages to deliver selected cases only.

## 8. Delivery Acceptance

At minimum, verify:

| Test | Acceptance point |
| --- | --- |
| Direct baseline | Same CSV, plugin, and compare configuration can run directly |
| Three formats | Separate `bin`, `npy`, and `pt` prepare/replay runs |
| Directory move | Replay after copying the whole root to another absolute path |
| Shape | A same-numel wrong device shape is rejected by new bin, npy, and pt data |
| Compare | Custom compare and CSV close tolerances remain active during replay |
| Priority | Whole-dataset `bin > npy > pt`; corrupt high priority does not fall back |
| Failure protection | Missing case/slot/None marker and dtype/shape/byte errors fail clearly |

Successful prepare reports `MANUAL_DATA_PREPARED/PASS`. After moving data, compare the
direct and replay case sets, execution status, precision status, and custom-compare
result.

## 9. Known Limits

Without a manifest or sidecar, TTK cannot automatically validate:

- whether a same-name case still uses the same API, attributes, wrapper, or plugin;
- an input-bin shape change that preserves dtype and total element count;
- same-size replacement or bit corruption, because there is no checksum;
- accidental sharing of a compatible same-name directory across E2E/ACLNN/Kernel;
- seed, generation environment, or protocol-version history.

Re-run prepare after changing API, attributes, input shape/view, wrapper, or assets.
Use different manual-data roots for different commands and data versions, and do not
rename files inside a case.

Common errors:

| Message | Check |
| --- | --- |
| `prepared testcase ... was not found` | Complete CSV `testcase_name` and search roots |
| `slot count ... != CSV` | Missing files, contiguous indexes, and CSV changes |
| `filename dtype ... != CSV storage dtype` | Tensor/scalar dtype in both stages |
| `saved shape ... != device output shape` | Device output regression or stale CSV/assets |
| `byte size ... != expected` | Truncated files or changed dtype/shape |
| `unexpected file in manual-data case` | Remove logs, JSON, subdirectories, and sidecars |
