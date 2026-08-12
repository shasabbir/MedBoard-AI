"""Run the complete local MedBoard release gate."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    commands = [
        [sys.executable, "-m", "flake8", "medboard", "tests", "scripts", "app.py"],
        [sys.executable, "-m", "mypy", "medboard", "scripts"],
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=medboard",
            "--cov-report=term-missing",
        ],
        [sys.executable, "-m", "pip", "check"],
    ]
    for command in commands:
        print(f"\n==> {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    if _evaluation_determinism_check():
        return 1
    return _streamlit_smoke_test()


def _evaluation_determinism_check() -> int:
    command = [
        sys.executable,
        "-m",
        "medboard.evaluation",
        "--output",
        "evaluation/results",
    ]
    print(f"\n==> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    paths = [
        ROOT / "evaluation/results/evaluation_results.json",
        ROOT / "evaluation/results/evaluation_results.md",
    ]
    first = [_file_digest(path) for path in paths]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if first != [_file_digest(path) for path in paths]:
        print("Evaluation determinism check failed.")
        return 1
    print("Evaluation determinism: identical SHA-256 outputs")
    return 0


def _file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _streamlit_smoke_test() -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        "8521",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"\n==> {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(40):
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8521", timeout=1
                ) as response:
                    body = response.read().decode("utf-8")
                    if response.status == 200 and "Streamlit" in body:
                        print("Streamlit smoke test: HTTP 200")
                        return 0
            except OSError:
                time.sleep(0.25)
        stdout, stderr = process.communicate(timeout=2)
        print((stdout + stderr).decode("utf-8", errors="replace"))
        return 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
