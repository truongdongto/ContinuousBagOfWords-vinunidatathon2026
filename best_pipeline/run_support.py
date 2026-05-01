"""Shared helpers: confirmation gate and subprocess runs."""
import subprocess
import sys

from config import DATA_ROOT


def should_execute(prompt: str, *, dry_run: bool, yes: bool) -> bool:
    """If dry_run: print intent and continue (caller prints per-script commands). If yes: True. Else ask."""
    if dry_run:
        print(f"[dry-run] {prompt}")
        return True
    if yes:
        print(f"[--yes] {prompt}")
        return True
    return input(f"{prompt}\nExecute? [y/N]: ").strip().lower() == "y"


def run_python_script(script_basename: str, *, dry_run: bool) -> int:
    """Run `python script_basename` with cwd=DATA_ROOT."""
    cmd = [sys.executable, script_basename]
    cwd = str(DATA_ROOT)
    print(f"\n>>> cd {cwd}\n>>> {' '.join(cmd)}\n")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=cwd)


def check_files(required: tuple[str, ...]) -> tuple[list[str], list[str]]:
    missing = []
    ok = []
    for rel in required:
        p = DATA_ROOT / rel
        if p.exists():
            ok.append(rel)
        else:
            missing.append(rel)
    return ok, missing
