"""
Entry point for running alfars lab as a module.

Usage:
    python -m alfars
    uv run -m alfars
    uv run -m alfars --no-browser
    uv run -m alfars --backend rust --no-browser --no-wait
"""
import sys
import argparse
from alfars.lab import main

if __name__ == "__main__":
    # Fix argv[0] for argparse
    if sys.argv:
        sys.argv[0] = "alfars"

    parser = argparse.ArgumentParser(description="Start alfars lab environment")
    parser.add_argument(
        "--backend",
        choices=["rust", "python", "auto"],
        default="auto",
        help="Backend type (default: auto)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Don't wait for processes"
    )

    args = parser.parse_args()

    main(
        backend_type=args.backend,
        open_browser=not args.no_browser,
        wait=not args.no_wait,
    )
