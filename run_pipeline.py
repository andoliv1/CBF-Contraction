"""
Run the vanishing CBF-QP example and generate plots.

Andreas Oliveira, Mustafa Bozdag
03/2026
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_name: str) -> None:
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed running {script_name} (exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run linear or nonlinear pipeline")
    parser.add_argument(
        "--mode",
        choices=["linear", "nonlinear"],
        required=True,
        help="Pipeline mode to run",
    )
    args = parser.parse_args()

    if args.mode == "linear":
        run_step("linear/example.py")
        run_step("linear/plots.py")
    else:
        run_step("nonlinear/example_nonlinear.py")
        run_step("nonlinear/plots_nonlinear.py")

if __name__ == "__main__":
    main()
