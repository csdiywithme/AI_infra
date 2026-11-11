#!/usr/bin/env python3
"""Reproduce the format-gate boundary with an exact, separately fetched script.

No network, GPU, kernel, profiler or model call is performed. The only upstream
program executed is the inspected Bash gate at the pinned Git blob below.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

PINNED_BLOB = "ba6ab5e7362923651b95521264f7711ce0d68f9f"
PINNED_COMMIT = "21f10bbe3c2dc854e533b62fcd306186f88c76c6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_script", type=Path)
    args = parser.parse_args()
    gate = args.gate_script.resolve(strict=True)
    source = gate.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    if blob != PINNED_BLOB:
        raise SystemExit(f"Refusing different gate source: {blob}; expected {PINNED_BLOB}")
    root = Path(tempfile.mkdtemp(prefix="kernel-agent-gate-audit-"))
    version = root / "fake-run" / "v0"
    version.mkdir(parents=True)
    for name in ("correctness.md", "ncu_summary.md", "ncu_details.md"):
        (version / name).write_text("FAILED; no GPU measurement was performed.\n")
    (version / "kernel.py").write_text("# placeholder; not executable kernel code\n")
    (version / "kbs_evidence.md").write_text(
        "# KBS Evidence\n## NCU Facts\n## Queries\n## Selected Evidence\n"
        "## Rejected / Limits\n## Decision Link\n"
    )
    (version / "hypothesis.txt").write_text("\n".join([
        "Version:", "Current evidence:", "Decision link:", "Single change:",
        "Rationale:", "Expected metric movement:", "Risk:", "Evidence used:",
        "Decision rule:"
    ]) + "\n")
    completed = subprocess.run(
        ["bash", str(gate), str(version), "--verbose"],
        capture_output=True, text=True, timeout=30,
    )
    (root / "gate-audit.log").write_text(completed.stdout + completed.stderr)
    state_file = root / "fake-run" / "current_iteration.txt"
    result = {
        "audit": "format gate accepts explicit failed/no-measurement artifacts",
        "source_repo": "minho-fan/kernel-opt-agent",
        "source_commit": PINNED_COMMIT,
        "source_git_blob_sha": blob,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "audit_protocol_date": "2026-11-04",
        "exit_code": completed.returncode,
        "state": state_file.read_text().strip() if state_file.exists() else None,
        "gpu_execution": False,
        "scope": "only gate script executed; no candidate code, profiler, API or GPU job",
    }
    (root / "gate-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"artifacts": str(root), **result}, ensure_ascii=False, indent=2))
    return 0 if completed.returncode == 0 and result["state"] == "v1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
