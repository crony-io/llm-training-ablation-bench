"""Dual logger: prints to console and writes to a .log file in the same batch.

Usage:
    from logger import log, init_log

    init_log(Path("test/results/bench_20260325.log"))
    log("hello")       # prints to console AND appends to the log file
    log(f"step {i}")   # same as print() but dual-output
"""

from __future__ import annotations

from pathlib import Path

_log_file: Path | None = None
_log_handle = None


def init_log(path: Path) -> Path:
    """Open the log file for writing. Returns the path for display."""
    global _log_file, _log_handle
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = path
    _log_handle = open(path, "w", encoding="utf-8")  # noqa: SIM115
    return path


def log(msg: str = "") -> None:
    """Print to console and append to the log file in one call."""
    print(msg)
    if _log_handle is not None:
        _log_handle.write(msg + "\n")
        _log_handle.flush()


def close_log() -> None:
    """Flush and close the log file."""
    global _log_handle
    if _log_handle is not None:
        _log_handle.flush()
        _log_handle.close()
        _log_handle = None
