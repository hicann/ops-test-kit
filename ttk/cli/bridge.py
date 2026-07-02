import pathlib
import logging


def args_to_switches(args):
    from ttk.utilities.classes import SWITCHES, DumpConfig, DumpLevel, SoCSimtCfg

    sw = SWITCHES()
    sw.logging_to_file = True

    sw.input_files = [args.input]
    sw.output_file_name = args.output

    if hasattr(args, 'testcase') and args.testcase:
        sw.selected_testcases = args.testcase.split(",")
    if hasattr(args, 'testcase_index') and args.testcase_index:
        sw.selected_testcase_indexes = _parse_indexes(args.testcase_index)
    if hasattr(args, 'testcase_count') and args.testcase_count is not None:
        sw.selected_testcase_count = args.testcase_count

    if hasattr(args, 'priority') and args.priority:
        sw.priorities = _parse_priorities(args.priority)
    if hasattr(args, 'operator') and args.operator:
        sw.selected_operators = tuple(args.operator.split(","))
    if hasattr(args, 'exclude_operator') and args.exclude_operator:
        sw.excluded_operators = tuple(args.exclude_operator.split(","))

    if hasattr(args, 'random_seed') and args.random_seed is not None:
        sw.random_seed = args.random_seed
    if hasattr(args, 'input_dist'):
        sw.input_distribution = args.input_dist
    if hasattr(args, 'compare'):
        sw.compare_method = args.compare
    if hasattr(args, 'golden_mode'):
        sw.golden_mode = args.golden_mode

    if hasattr(args, 'dump') and args.dump is not None:
        _apply_dump_config(sw.dump_config, args.dump)
    if hasattr(args, 'dump_format'):
        sw.dump_config.file_format = args.dump_format
    if hasattr(args, 'dump_on_fail') and args.dump_on_fail:
        sw.dump_config.dump_on_fail = True

    if hasattr(args, 'plugin') and args.plugin:
        sw.plugin_path = tuple(
            pathlib.Path(p.strip()).resolve()
            for p in args.plugin.split(',')
            if p.strip()
        )

    if hasattr(args, 'rerun') and args.rerun:
        sw.rerun_targets = args.rerun.lower().split(",")

    if hasattr(args, 'title') and args.title:
        sw.custom_columns = args.title.split(",")
    if hasattr(args, 'csv_preserve') and args.csv_preserve:
        sw.preserve_original_csv = True
    if hasattr(args, 'single_log') and args.single_log:
        sw.single_testcase_log_mode = True
    if hasattr(args, 'summary_print') and not args.summary_print:
        sw.summary_print = False
    if hasattr(args, 'proc_no_reuse') and args.proc_no_reuse:
        sw.proc_no_reuse = True
    if hasattr(args, 'no_memory_check') and args.no_memory_check:
        sw.no_memory_check = True
    if hasattr(args, 'task_prof') and not args.task_prof:
        sw.TASK_PROFILING = False
    if hasattr(args, 'progress_output') and args.progress_output:
        sw.progress_output = args.progress_output
    if hasattr(args, 'device') and args.device is not None:
        sw.device_count = args.device
    if hasattr(args, 'device_blacklist') and args.device_blacklist:
        sw.device_blacklist = tuple(int(x) for x in args.device_blacklist.split(","))
    if hasattr(args, 'device_whitelist') and args.device_whitelist:
        sw.device_whitelist = tuple(int(x) for x in args.device_whitelist.split(","))
    if hasattr(args, 'process_count') and args.process_count is not None:
        sw.process_per_device = args.process_count
    if hasattr(args, 'platform') and args.platform:
        sw.dev_plat = args.platform
    if hasattr(args, 'proc_timeout') and args.proc_timeout:
        sw.proc_timeout = args.proc_timeout
    if hasattr(args, 'validate_only') and args.validate_only:
        sw.validate_only = True
    if hasattr(args, 'warmup') and not args.warmup:
        sw.warmup = False
    if hasattr(args, 'run') and args.run is not None:
        sw.run_time = args.run
    if hasattr(args, 'npu_timeout') and args.npu_timeout:
        sw.run_timeout = args.npu_timeout

    return sw


def apply_kernel_args(sw, args):
    if hasattr(args, 'dynamic') and args.dynamic is not None:
        val = args.dynamic
        if isinstance(val, str):
            val = val.lower() not in ('false', '0', 'no', 'off')
        sw.dyn_switches.enabled = val
    if hasattr(args, 'const') and args.const is not None:
        val = args.const
        if isinstance(val, str):
            val = val.lower() not in ('false', '0', 'no', 'off')
        sw.cst_switches.enabled = val
    if hasattr(args, 'binary') and args.binary is not None:
        if args.binary == "release":
            sw.bin_switches.enabled = True
            sw.bin_switches.realtime = "release"
        elif args.binary is not True:
            val = args.binary
            if isinstance(val, str):
                val = val.lower() not in ('false', '0', 'no', 'off')
            sw.bin_switches.enabled = val
    if hasattr(args, 'compile_only') and args.compile_only:
        sw.compile_only = True
    # --compile-opts key=value (append, direct passthrough)
    if hasattr(args, 'compile_opts') and args.compile_opts:
        for pair in args.compile_opts:
            if '=' not in pair:
                raise ValueError(f"--compile-opts requires KEY=VALUE format, got: {pair}")
            key, value = pair.split('=', 1)
            sw.compile_options[key] = value
    if hasattr(args, 'impl_mode') and args.impl_mode:
        sw.op_impl_mode = args.impl_mode
    if hasattr(args, 'limit') and args.limit is not None:
        sw.DAVINCI_HBM_SIZE_LIMIT = args.limit
    if hasattr(args, 'reuse_hbm') and args.reuse_hbm:
        sw.reuse_hbm = True
    if hasattr(args, 'reserve_hbm') and args.reserve_hbm is not None:
        sw.reserve_hbm = args.reserve_hbm
    if hasattr(args, 'no_prof') and args.no_prof:
        sw.dyn_switches.prof = False
        sw.cst_switches.prof = False
        sw.bin_switches.prof = False
    if hasattr(args, 'tiling_run') and args.tiling_run is not None:
        sw.tiling_run_time = args.tiling_run
    if hasattr(args, 'cce') and args.cce is not None:
        _apply_cce(sw, args.cce)
    if hasattr(args, 'clear_atomic') and args.clear_atomic:
        _apply_clear_atomic(sw, args.clear_atomic)
    if hasattr(args, 'clear_ub') and args.clear_ub is not None:
        import numpy
        sw.force_clear_ub = _parse_clean_val("UB", args.clear_ub)
    if hasattr(args, 'clear_l1') and args.clear_l1 is not None:
        import numpy
        sw.force_clear_l1 = _parse_clean_val("L1", args.clear_l1)
    if hasattr(args, 'force_block_dim') and args.force_block_dim is not None:
        bd = args.force_block_dim
        if isinstance(bd, int):
            sw.force_block_dim = [bd] * 3
        else:
            sw.force_block_dim = list(bd)
    if hasattr(args, 'simt_ub') and args.simt_ub is not None:
        v = args.simt_ub
        if isinstance(v, int):
            sw.force_simt_ub_size = [v] * 3
        else:
            sw.force_simt_ub_size = list(v[:3])
    if hasattr(args, 'simt_stack_dcu') and args.simt_stack_dcu is not None:
        sw.simt_cfg.dcu_stack = args.simt_stack_dcu
    if hasattr(args, 'simt_stack_dvg') and args.simt_stack_dvg is not None:
        sw.simt_cfg.dvg_stack = args.simt_stack_dvg


def apply_aclnn_args(sw, args):
    pass


def apply_e2e_args(sw, args):
    if hasattr(args, 'dynamic') and args.dynamic is not None:
        val = args.dynamic
        if isinstance(val, str):
            val = val.lower() not in ('false', '0', 'no', 'off')
        sw.dyn_switches.enabled = val
    if hasattr(args, 'const') and args.const is not None:
        val = args.const
        if isinstance(val, str):
            val = val.lower() not in ('false', '0', 'no', 'off')
        sw.cst_switches.enabled = val
    if hasattr(args, 'backend') and args.backend:
        sw.backend_name = args.backend.lower()
    if hasattr(args, 'fullgraph'):
        sw.fullgraph = args.fullgraph


def run_with_switches(sw):
    import ttk
    from ttk.utilities import set_global_storage
    from ttk.core_modules.tbe_logging import default_logging_config

    set_global_storage(sw)
    default_logging_config(file_handler=sw.logging_to_file)

    from ttk.utilities import set_process_name, set_thread_name
    set_process_name()
    set_thread_name()

    logging.info(f"Command: ttk {sw.test_mode} -i {sw.input_files[0] if sw.input_files else ''}")

    if sw.test_mode == "framework-api":
        from ttk.core_modules.framework_api.instance import FrameworkApiInstance
        ins = FrameworkApiInstance()
    else:
        from ttk.core_modules.npu.instance_refactor import NpuInstance
        ins = NpuInstance()

    ins.profile()


def _parse_indexes(spec):
    selected = []
    for part in spec.split(","):
        if '-' in part:
            lo, hi = part.split('-', 1)
            selected.extend(range(int(lo), int(hi) + 1))
        else:
            selected.append(int(part))
    return tuple(selected)


def _parse_priorities(spec):
    priorities = []
    for p in spec.split(","):
        if '-' in p:
            lo, hi = p.split('-', 1)
            priorities.append((int(lo) if lo else 0, int(hi) if hi else float('inf')))
        else:
            priorities.append((int(p), int(p)))
    return tuple(priorities)


def _apply_dump_config(dump_config, value):
    if value == "full":
        dump_config.enable_all()
        return
    for m in value.lower().split(","):
        if m == "in":
            dump_config.enable_input()
        elif m == "out":
            dump_config.enable_output()
        elif m == "golden":
            dump_config.enable_golden()
        elif m == "full":
            dump_config.enable_all()


def _apply_cce(sw, value):
    if value is True or value is None:
        sw.dyn_switches.realtime = False
        sw.cst_switches.realtime = False
        if sw.bin_switches.realtime != "release":
            sw.bin_switches.realtime = False
    else:
        for mode in value.lower().split(','):
            if mode in ('d', 'dyn', 'dynamic'):
                sw.dyn_switches.realtime = False
            elif mode in ('c', 'cst', 'const'):
                sw.cst_switches.realtime = False
            elif mode in ('b', 'bin', 'binary'):
                if sw.bin_switches.realtime != "release":
                    sw.bin_switches.realtime = False


def _apply_clear_atomic(sw, value):
    if value is True:
        sw.force_clear_atomic = [True] * 3
    else:
        for mode in value.lower().split(","):
            if mode in ("d", "dyn"):
                sw.force_clear_atomic[0] = True
            elif mode in ("c", "cst"):
                sw.force_clear_atomic[1] = True
            elif mode in ("b", "bin"):
                sw.force_clear_atomic[2] = True


def _parse_clean_val(name, value):
    import numpy
    if not value:
        return numpy.int32(0)
    val = value.strip().lower()
    if "int" not in val and "float" not in val:
        if val in ("inf", "nan", "-inf"):
            val = f'float32("{val}")'
        else:
            val = f"float32({val})" if "." in val else f"int32({val})"
    val = f"numpy.{val}"
    try:
        return eval(val)
    except Exception as e:
        raise ValueError(f"Cannot parse {name} clean value [{val}]: {e}")
