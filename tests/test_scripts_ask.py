import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ask_with_no_args_prints_usage_and_exits_1():
    result = subprocess.run(
        [sys.executable, "scripts/ask.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Usage:" in result.stdout
