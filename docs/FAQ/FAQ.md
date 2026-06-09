# FAQ

[toc]

---

# Self-Check Before Debugging

1. Is TTK up to date? Run `python3 -m ttk -v`
2. Check command syntax: `python3 -m ttk kernel --help`
3. Is CANN installed and environment sourced?
4. Check operator name spelling in CSV
5. If using custom plugins, verify registration names match CSV operator names

# Environment Issues

## "ASCEND_HOME_PATH not found"

Install CANN and source environment:

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## CANN Sub-Package Environment

```shell
source /usr/local/Ascend/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/latest
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/latest
```

Common errors without sourcing: `ImportError: libhccl.so`, `AttributeError: 'NoneType' object has no attribute 'acl_init'`

# CSV Writing Issues

## How to write dtype strings?

Supported: `float16`/`fp16`, `float32`/`fp32`, `float64`/`fp64`, `bfloat16`/`bf16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`, `complex64`, `complex128`.

## Fields with special characters must be double-quoted

```csv
"((128, 1024), (1, 1024))"     # Correct
((128, 1024), (1, 1024))       # Wrong - commas parsed as CSV delimiters
```

## Output shape depends on input tensor values, cannot be inferred from input shapes (e.g. NonZero)

Set `output_shapes` to a placeholder shape (non-empty), and mark the unknown output index in `output_shape_unknown_indexes`:

```csv
output_shapes,"((8, 1716),)"
output_shape_unknown_indexes,"(0,)"
```

The framework will determine the actual output shape at runtime.

## TensorList operators

Express TensorList via nesting in `input_shapes`. Group tensors in the same inner tuple:

```csv
input_shapes,"(((3,3),(3,2),(3,4)),(3,5))"
```

## Preview CSV test cases

```shell
python3 -m ttk list -i cases.csv
python3 -m ttk list -i cases.csv --op add
```

# Execution Issues

## Timeout or OOM with large operators

- `--proc-timeout=300` (per-case timeout)
- `-l 10` (10GB HBM limit)
- `--reserve-hbm=512` (reserve HBM)

## Reproduce test results

```shell
python3 -m ttk kernel -i cases.csv --seed 42
```

## Quick environment check

```shell
python3 -m ttk info
python3 -m ttk kernel -i examples/case_store/kernel/add.csv
```

# Precision Issues

## How to debug precision failures?

```shell
python3 -m ttk kernel -i cases.csv --dump-on-fail
python3 -m ttk kernel -i cases.csv --dump in,golden --dump-format npy
python3 -m ttk kernel -i cases.csv -t add_01 --single-log --dump full
```

## Comparison method recommendations

| Scenario | Method | Flag |
|----------|--------|------|
| General float | Numeric approximation | `--compare close` (default) |
| Large vector trends | Cosine similarity | `--compare cosine` |
| Integer / exact match | Binary exact | `--compare binary` |
| float8 types | Requantization | `--compare requant` (auto) |

## Adjust tolerance in CSV

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

# Plugin Issues

## Custom golden not taking effect

1. Verify `--plugin` path is correct
2. Check `@register_golden(["op_name"])` matches CSV `op_name`
3. Check plugin file for syntax errors

## Duplicate operator name registration

Later-loaded plugins override earlier ones (with a warning). Ensure unique registration names.
