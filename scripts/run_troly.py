#!/usr/bin/env python3
"""
Super-simple launcher for the Gradio app in troly_dontu/troly.py.

Assumes you have an `all_laws.txt` in the project root. If the
absolute path expected by troly.py is missing, this redirects it
to ./all_laws.txt and then launches the app.
"""
import builtins
import sys
import os


ABS_ALL_LAWS = "/home/ubuntu/various_tools/troly_dontu/all_laws.txt"
FALLBACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "all_laws.txt"))

# Ensure project root is importable when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


orig_open = builtins.open  # type: ignore[attr-defined]


def open_wrapper(file, *args, **kwargs):  # type: ignore[override]
    try:
        target = os.fspath(file)
    except TypeError:
        target = file
    if target == ABS_ALL_LAWS and os.path.isfile(FALLBACK):
        return orig_open(FALLBACK, *args, **kwargs)
    return orig_open(file, *args, **kwargs)


# Patch only for import-time read
builtins.open = open_wrapper  # type: ignore[assignment]
from troly_dontu import troly  # noqa: E402
builtins.open = orig_open  # type: ignore[assignment]


if __name__ == "__main__":
    # Launch with Gradio defaults (localhost:7860)
    troly.demo.launch()
