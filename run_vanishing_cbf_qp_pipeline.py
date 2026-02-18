"""
Run the vanishing CBF-QP example and generate plots.
"""

import subprocess
import sys
from pathlib import Path


def run_step(script_name: str) -> None:
    script_path = Path(__file__).with_name(script_name)
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Failed running {script_name} (exit {result.returncode})")


def main() -> None:
    run_step("example_vanishing_cbf_qp.py")
    run_step("plot_vanishing_cbf_qp.py")


if __name__ == "__main__":
    main()
