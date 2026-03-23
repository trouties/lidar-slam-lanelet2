"""Git metadata utilities."""

from __future__ import annotations

import subprocess


def get_git_sha(short: bool = False) -> str:
    """Return the current git HEAD SHA.

    Args:
        short: If True, return the 7-char abbreviated SHA.

    Returns:
        SHA string, or ``'unknown'`` if git is unavailable.
    """
    cmd = ["git", "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
