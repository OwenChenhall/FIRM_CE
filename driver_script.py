# Command line argument runner for multiple subsequent scenarios for FIRM_CE
# Copyright (c) 2025 Owen Chenhall
# Licensed under the MIT Licence
# Correspondence: owen.chenhall@gmail.com

import sys
import subprocess
from pathlib import Path

def run_script(script_path, args=None, stdin_input=None):
    args = args or []
    script_path = Path(script_path).resolve()
    project_dir = script_path.parent

    cmd = [sys.executable, script_path.name, *args]
    print(f"\n=== Running: {' '.join(cmd)} in {project_dir} ===")

    result = subprocess.run(
        cmd,
        cwd=project_dir,
        input=stdin_input,
        text=True
    )
    print(f"=== Finished {script_path} with code {result.returncode} ===\n")
    return result.returncode

if __name__ == "__main__":
    run_script("Optimisation.py", ["-i", "1","-p", "5", "-n", "NSW", "-steps", "1"],stdin_input="n\n") # A simple test for function

    # run_script("Optimisation.py", ["-i", "1000","-p", "20", "-n", "NSW", "-steps", "1"],stdin_input="n\n")
    # run_script("Optimisation.py", ["-i", "1000","-p", "20", "-n", "NSW", "-steps", "2"],stdin_input="n\n")
    # run_script("Optimisation.py", ["-i", "2000","-p", "20", "-n", "NSW", "-steps", "4"],stdin_input="n\n")
    # run_script("Optimisation.py", ["-i", "2000","-p", "40", "-n", "NSW", "-steps", "8"],stdin_input="n\n")

