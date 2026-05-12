import argparse


def add_common_args(parser):
    parser.add_argument("-i", "--input", required=True, help="CSV test case file")
    parser.add_argument("-o", "--output", help="Output CSV file")
    parser.add_argument("-t", "--testcase", help="Specify testcase name(s), comma-separated")
    parser.add_argument("--ti", "--testcase-index", dest="testcase_index",
                        help="Specify testcase indexes, e.g. --ti=1,3,6 or --ti=1-5")
    parser.add_argument("--tc", "--testcase-count", type=int, dest="testcase_count",
                        help="Randomly pick N test cases")
    parser.add_argument("--priority", help="Filter by priority, e.g. --priority=1,3 or --priority=1-3")
    parser.add_argument("--op", "--operator", dest="operator", help="Filter by operator name(s)")
    parser.add_argument("--no-op", "--exclude-operator", dest="exclude_operator",
                        help="Exclude operator name(s)")
    parser.add_argument("--seed", "--random-seed", type=int, dest="random_seed", help="Random seed")
    parser.add_argument("--input-dist", dest="input_dist", choices=["uniform", "normal"],
                        default="uniform", help="Input data distribution (default: uniform)")
    parser.add_argument("--compare", default="close",
                        choices=["close", "cosine", "binary", "requant"],
                        help="Comparison method (default: close)")
    parser.add_argument("--golden-mode", dest="golden_mode", default="Enable",
                        choices=["Enable", "Disable", "Promote"],
                        help="Golden generation mode (default: Enable)")
    parser.add_argument("--dump", nargs="?", const="full",
                        help="Dump data: full, in, out, golden (comma-separated)")
    parser.add_argument("--dump-format", dest="dump_format", default="bin",
                        choices=["bin", "npy", "pt", "print"],
                        help="Dump file format (default: bin)")
    parser.add_argument("--dump-on-fail", dest="dump_on_fail", action="store_true",
                        help="Dump all data when precision check fails")
    parser.add_argument("--plugin", help="External plugin path for customized golden/inputs")
    parser.add_argument("--rerun", help="Rerun failed cases, e.g. --rerun=precision_status")
    parser.add_argument("--title", "--titles", dest="title",
                        help="Custom output columns, e.g. --title=testcase_name,dyn_perf_us")
    parser.add_argument("--csv-preserve", dest="csv_preserve", action="store_true",
                        help="Preserve original CSV headers")
    parser.add_argument("--single-log", dest="single_log", action="store_true",
                        help="Each test case gets its own log file")
    parser.add_argument("--print", dest="summary_print", default=True,
                        type=lambda x: x.lower() != "false",
                        help="Print summary info periodically (default: true)")
    parser.add_argument("--proc-no-reuse", dest="proc_no_reuse", action="store_true",
                        help="Create new process for each case")
    parser.add_argument("--no-memory-check", dest="no_memory_check", action="store_true",
                        help="Skip host memory check")
    parser.add_argument("--task-prof", dest="task_prof", default=True,
                        type=lambda x: x.lower() != "false",
                        help="Profile E2E duration (default: true)")
    parser.add_argument("--po", "--progress-output", dest="progress_output",
                        help="Progress output file")
