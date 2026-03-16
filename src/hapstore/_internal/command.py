"""Subprocess command execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import subprocess
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Result of an executed command."""

    cmd: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    cwd: str | None

    @property
    def cmd_display(self) -> str:
        return " ".join(shlex.quote(part) for part in self.cmd)


class CommandExecutionError(RuntimeError):
    """Raised when a command fails with check=True."""

    def __init__(self, result: CommandResult):
        message = (
            "Command failed\n"
            f"CMD: {result.cmd_display}\n"
            f"RETURN_CODE: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        super().__init__(message)
        self.result = result


def run_cmd(
    cmd: Sequence[str],
    check: bool = True,
    capture_output: bool = True,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
) -> CommandResult:
    """Run a subprocess command and return a structured result."""
    if not cmd:
        raise ValueError("cmd must be non-empty")

    cmd_tuple = tuple(str(part) for part in cmd)
    run_env = os.environ.copy()
    if env:
        run_env.update({str(k): str(v) for k, v in env.items()})

    cwd_str = str(cwd) if cwd is not None else None
    start = time.perf_counter()

    if log_path is not None and not capture_output:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write("CMD: " + " ".join(shlex.quote(c) for c in cmd_tuple) + "\n")
            log_file.flush()
            proc = subprocess.run(
                list(cmd_tuple),
                cwd=cwd_str,
                env=run_env,
                timeout=timeout,
                check=False,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.perf_counter() - start
        result = CommandResult(
            cmd=cmd_tuple,
            returncode=proc.returncode,
            stdout="",
            stderr="",
            elapsed_seconds=elapsed,
            cwd=cwd_str,
        )
    else:
        proc = subprocess.run(
            list(cmd_tuple),
            cwd=cwd_str,
            env=run_env,
            timeout=timeout,
            check=False,
            text=True,
            capture_output=capture_output,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout if proc.stdout is not None else ""
        stderr = proc.stderr if proc.stderr is not None else ""

        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write("CMD: " + " ".join(shlex.quote(c) for c in cmd_tuple) + "\n")
                if stdout:
                    log_file.write(stdout)
                    if not stdout.endswith("\n"):
                        log_file.write("\n")
                if stderr:
                    log_file.write(stderr)
                    if not stderr.endswith("\n"):
                        log_file.write("\n")

        result = CommandResult(
            cmd=cmd_tuple,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
            cwd=cwd_str,
        )

    if check and result.returncode != 0:
        raise CommandExecutionError(result)
    return result
