#!/usr/bin/env python3
"""Docker entrypoint: receive kwargs as JSON via argv[1], run executor, print envelope.

Reads a JSON file path from argv[1] containing the execute_request kwargs dict.
Runs execute_request, prints the envelope JSON to stdout. Exit code 0 on success.

The script's directory (/executor/) is on sys.path by default, so the executor
package is importable directly.
"""

import json
import sys

# Flat-layout assumption: Dockerfile COPYs the server dir to /executor/, so this
# script's dir (sys.path[0]) contains executor.py as a top-level module.
from executor import execute_request


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: executor_main.py <kwargs.json>"}))
        sys.exit(1)

    with open(sys.argv[1]) as f:
        kwargs = json.load(f)

    envelope = execute_request(**kwargs)
    print(json.dumps(envelope))
    # Exit 0 whenever a valid envelope was printed (incl. ok:False / 424 / 400) —
    # the http_status field carries the result. Exit 1 only on infra failure
    # (handled above). Otherwise container.py treats 424 as exit!=0 → 500 and
    # the client's 424→sync→retry loop never fires.
    sys.exit(0)


if __name__ == "__main__":
    main()
