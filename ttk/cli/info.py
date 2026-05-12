def register_info_command(subparsers):
    parser = subparsers.add_parser("info", help="Show device/environment information")
    parser.set_defaults(handler=_handle_info)


def _handle_info(args):
    try:
        from ttk.core_modules.dsmi import DSMIInterface
        dsmi = DSMIInterface()
        count = dsmi.get_device_count()
        if count == 0:
            print("No Ascend devices found.")
            return
        print(f"Found {count} device(s):\n")
        print(f"{'Device':>8}  {'Chip Info':30s}  {'Temp':>6}  {'Util':>5}")
        print("-" * 60)
        for i in range(count):
            try:
                info = dsmi.get_chip_info(i)
            except Exception:
                info = "N/A"
            try:
                temp = dsmi.get_device_temperature(i)
                temp = f"{temp}C"
            except Exception:
                temp = "N/A"
            try:
                util = dsmi.get_device_utilization(i)
                util = f"{util}%"
            except Exception:
                util = "N/A"
            print(f"{i:>8}  {str(info):30s}  {temp:>6}  {util:>5}")
    except ImportError:
        print("DSMI module not available. Is CANN installed?")
    except Exception as e:
        print(f"Error querying devices: {e}")
