#!/usr/bin/env python3
"""
Helper script to launch the Gradio app defined in troly_dontu/troly.py.

This script safely imports the module even if the original absolute
path for `all_laws.txt` does not exist by temporarily monkey‑patching
`open()` to redirect that path to a local fallback (if provided).

Usage examples:
  - python scripts/run_troly_demo.py --host 0.0.0.0 --port 7860
  - python scripts/run_troly_demo.py --dry-run   # verify import only

Optional env vars:
  - OPENAI_API_KEY: required for actual retrieval and LLM calls.
  - TROLY_ALL_LAWS_FALLBACK: path to a local all_laws.txt file.
"""
import argparse
import builtins
import contextlib
import os
import sys
from types import ModuleType
from typing import Callable, Optional


ABS_ALL_LAWS = "all_laws.txt"


@contextlib.contextmanager
def patch_open_for_all_laws(fallback_path: Optional[str]):
    if not fallback_path:
        yield
        return

    orig_open: Callable = builtins.open  # type: ignore[attr-defined]

    def open_wrapper(file, *args, **kwargs):  # type: ignore[override]
        try:
            target = os.fspath(file)
        except TypeError:
            target = file
        if target == ABS_ALL_LAWS and os.path.isfile(fallback_path):
            return orig_open(fallback_path, *args, **kwargs)
        return orig_open(file, *args, **kwargs)

    builtins.open = open_wrapper  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.open = orig_open  # type: ignore[assignment]


def import_troly_with_fallback(fallback_path: Optional[str]) -> ModuleType:
    # Ensure project root on sys.path (script lives in scripts/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    with patch_open_for_all_laws(fallback_path):
        # Import here so the patched open() affects module import-time reads
        from troly_dontu import troly  # type: ignore
    return troly


def main():
    parser = argparse.ArgumentParser(description="Run troly Gradio app")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio public sharing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only import module and verify the Gradio app object",
    )
    parser.add_argument(
        "--fallback-all-laws",
        default=os.getenv("TROLY_ALL_LAWS_FALLBACK"),
        help=(
            "Local path to use as a fallback for all_laws.txt when "
            "the absolute path in troly.py is unavailable."
        ),
    )
    args = parser.parse_args()

    troly = import_troly_with_fallback(args.fallback_all_laws)

    # Validate the object exists
    demo = getattr(troly, "demo", None)
    if demo is None:
        print(
            "Error: troly.demo not found. Ensure troly_dontu/troly.py defines a Gradio Blocks named 'demo'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dry_run:
        print("troly.demo loaded successfully (dry run).")
        return

    # Launch the app
    print(
        f"Launching troly.demo on http://{args.host}:{args.port} (share={args.share})"
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
