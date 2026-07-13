# Result Analysis

[toc]

---

# Test Results

Per-case precision status and overall pass rate are printed to the terminal during execution. Use `-o` to save results to CSV:

```shell
python3 -m ttk kernel -i cases.csv -o results.csv
```

# Device Info

```shell
python3 -m ttk info
```

Shows hardware info for all Ascend NPU devices (chip model, temperature, utilization).

# Case Preview

```shell
python3 -m ttk list -i cases.csv
python3 -m ttk list -i cases.csv --op add
```

# Precision Comparison Methods

Use `--compare` to select the comparison method:

| Method | Value | Description | Default Tolerance |
|--------|-------|-------------|-------------------|
| Statistical relative error (community standard) | `stat_rel_err` (default) | Per-dtype statistical relative error with community thresholds | Per-dtype threshold table |
| Numeric approximation | `close` | Uses `np.isclose()`/`torch.isclose()` | fp16/bf16: rtol=0.001; fp32: rtol=0.0001; atol=1e-8 |
| Cosine similarity | `cosine` | Vector cosine similarity | rtol=0.01 |
| Binary exact | `binary` | Bit-exact comparison | No tolerance |
| Requantization | `requant` | For float8 types (e5m2/e4m3fn/hifloat8) | Auto-adapted |
| Three-party cross-check | `cross_check` | output/golden/third_party ratio; needs `third_party` (fp16/bf16/fp32) | mare/mere/rmse ratio + level presets |

> Default: `--compare` unset → per-output routing via `Spec.tolerance` (needs `--plugin`), else `stat_rel_err`.

## Auto-Switch Rules

| Data Type | Auto-switches to |
|-----------|-----------------|
| float8\_e5m2, float8\_e4m3fn, hifloat8 | `requant` |
| float4, int4 | `binary` |

# Precision Debugging

## Dump Data

```shell
python3 -m ttk kernel -i cases.csv --dump full
python3 -m ttk kernel -i cases.csv --dump in,golden --dump-format npy
python3 -m ttk kernel -i cases.csv --dump full --dump-format pt
```

| Format | Description |
|--------|-------------|
| `bin` (default) | Raw binary data |
| `npy` | NumPy array file |
| `pt` | PyTorch tensor file |
| `print` | Print to terminal |

## Auto-Dump on Failure

```shell
python3 -m ttk kernel -i cases.csv --dump-on-fail
```

## Per-Case Debugging

Combine `-t` with `--dump-on-fail` / `--single-log` for the most detailed debug output on a single case:

```shell
python3 -m ttk kernel -i cases.csv -t add_01 --dump-on-fail --single-log
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy
```

## Reproducible Results

```shell
python3 -m ttk kernel -i cases.csv --seed 42
```

# Input Data Distribution

| Distribution | Value | Description |
|-------------|-------|-------------|
| Uniform | `uniform` (default) | Uniform sampling within `input_data_ranges` |
| Normal | `normal` | Normal distribution sampling within range |

# Golden Mode

| Mode | Description |
|------|-------------|
| `Enable` (default) | Generate golden and compare |
| `Disable` | Skip golden generation (compile + execute only) |
| `Promote` | Use promoted precision for golden computation |
