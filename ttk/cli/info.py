def register_info_command(subparsers):
    parser = subparsers.add_parser("info", help="Show device/environment information")
    parser.set_defaults(handler=_handle_info)


def _handle_info(args):
    import logging

    _prev_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    try:
        from ttk.core_modules.dsmi import DSMIInterface

        dsmi = DSMIInterface()
        count = dsmi.get_device_count()
        if count == 0:
            print("No Ascend devices found.")
            return
        print(f"Found {count} device(s):\n")
        print(f"  {'ID':>3}  {'Platform':<16s} {'Ver':<4s}  {'Temp':>5}  {'Util':>5}")
        print("-" * 45)
        for i in range(count):
            try:
                chip = dsmi.get_chip_info(i)
                platform = chip.get_complete_platform()
                ver = chip.get_ver()
            except Exception:
                platform = "N/A"
                ver = ""
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
            print(f"  {i:>3}  {platform:<16s} {ver:<4s}  {temp:>5}  {util:>5}")
    except ImportError:
        print("DSMI module not available. Is CANN installed?")
    except Exception as e:
        print(f"Error querying devices: {e}")
    finally:
        logging.getLogger().setLevel(_prev_level)
