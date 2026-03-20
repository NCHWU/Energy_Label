import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "energy_label.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "energy_label" in result.stdout.lower() or "Energy" in result.stdout


def test_cli_evaluate_subcommand_help():
    result = subprocess.run(
        [sys.executable, "-m", "energy_label.cli", "evaluate", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--tasks" in result.stdout


def test_cli_score_subcommand_help():
    result = subprocess.run(
        [sys.executable, "-m", "energy_label.cli", "score", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--input" in result.stdout
