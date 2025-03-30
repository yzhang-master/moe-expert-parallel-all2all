import os
import socket
import subprocess
import sys
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_smoke(capacity=None):
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(free_port()), WORLD_SIZE="2")
    command = [sys.executable, str(root / "scripts" / "smoke_distributed.py"), "--check"]
    if capacity is not None:
        command += ["--capacity-factor", str(capacity)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rank=0" in result.stdout and "rank=1" in result.stdout
    return result.stdout


def test_two_rank_all_to_all_forward_backward_and_balance():
    run_smoke()


def test_two_rank_capacity_dropping_matches_dense():
    output = run_smoke(0.5)
    assert "dropped=0" not in output
