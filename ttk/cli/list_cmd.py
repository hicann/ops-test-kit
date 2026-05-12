def register_list_command(subparsers):
    parser = subparsers.add_parser("list", help="List test cases from CSV file")
    parser.add_argument("-i", "--input", required=True, help="CSV test case file")
    parser.add_argument("--op", "--operator", dest="operator", help="Filter by operator name")
    parser.set_defaults(handler=_handle_list)


def _handle_list(args):
    from ttk.core_modules.testcase_manager import UniversalTestcaseFactory
    with open(args.input) as f:
        factory = UniversalTestcaseFactory(f, skip_validate=True)
    cases = factory.testcases
    if args.operator:
        op_filter = args.operator.split(",")
        cases = [c for c in cases if getattr(c, 'op_name', getattr(c, 'api_name', None)) in op_filter]
    for case in cases:
        print(case.testcase_name)
    print(f"\nTotal: {len(cases)} case(s)")
