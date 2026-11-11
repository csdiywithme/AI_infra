"""Sealed, single-allocation B300 evaluator. Importing does not submit a job.

Run through `modal run --detach ... --allow-cloud`; no GPU job is started by
merely importing this file. Results are retrievable using the saved call ID.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import modal

app = modal.App("codex-b300-gemm-workflow")
image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "cuda-nsight-compute-13-1")
)

MODES = {"check", "bench", "profile"}
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 96 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe artifact path: {name!r}")
    return p


def canonical_json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def verify_payload(payload):
    if not isinstance(payload, dict) or not all(isinstance(v, bytes) for v in payload.values()):
        raise ValueError("Payload must map relative paths to bytes")
    if sum(map(len, payload.values())) > MAX_PAYLOAD_BYTES:
        raise ValueError("Source bundle exceeds 16 MiB")
    if "manifest.json" not in payload:
        raise ValueError("Missing manifest.json")
    manifest = json.loads(payload["manifest.json"])
    if not isinstance(manifest, dict) or "manifest.json" in manifest:
        raise ValueError("Manifest must map source paths to SHA256, excluding itself")
    if set(payload) != set(manifest) | {"manifest.json"}:
        raise ValueError("Payload differs from the sealed manifest")
    for name, digest in manifest.items():
        safe_relative(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid manifest digest")
        if sha(payload[name]) != digest:
            raise ValueError("Source hash mismatch: " + name)
    required = {"contract.json", "evaluator.cu", "modal_worker.py", "src/cuda/common.h",
                "src/cuda/m4_gemm/baseline.cu", "src/cuda/m4_gemm/candidate.cu"}
    if not required <= set(manifest):
        raise ValueError("Incomplete source bundle")
    return manifest


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"Invalid {name}")
    return value


def contract_header(contract):
    if contract.get("schema") != "b300-gemm-contract.v1":
        raise ValueError("Unsupported task contract")
    if (contract.get("gpu"), contract.get("arch"), contract.get("stages")) != ("B300", "sm_103a", 2):
        raise ValueError("This adapter requires B300, sm_103a and STAGES=2")
    shapes = contract.get("shapes")
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= 8:
        raise ValueError("Expected 1..8 shapes")
    entries = []
    acceptance = 0
    seen = set()
    for s in shapes:
        if not isinstance(s, dict):
            raise ValueError("Shape must be an object")
        m, n, k = [integer(s.get(key), key, 1, 16384) for key in ("m", "n", "k")]
        if m % 128 or n % 256 or k % 64 or k > 8192:
            raise ValueError("Shape outside frozen baseline supported multiples/K bound")
        if (m, n, k) in seen:
            raise ValueError("Duplicate shape")
        seen.add((m, n, k))
        role = s.get("role")
        if role not in {"validation", "acceptance"}:
            raise ValueError("Unknown shape role")
        acceptance += role == "acceptance"
        entries.append(f"{{{m},{n},{k},{str(role == 'acceptance').lower()}}}")
    if not acceptance:
        raise ValueError("Need at least one acceptance shape")
    seeds = contract.get("seeds")
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid seeds")
    for seed in seeds:
        integer(seed, "seed", 0, 2**31 - 1)
    num = contract.get("numeric", {})
    for key in ("atol", "rtol"):
        v = num.get(key)
        if type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= .1:
            raise ValueError("Invalid numeric tolerance")
    if type(num.get("integer_exact")) is not bool:
        raise ValueError("integer_exact must be bool")
    timing = contract.get("timing", {})
    blocks = integer(timing.get("blocks"), "blocks", 2, 48)
    if blocks % 2:
        raise ValueError("Balanced ABBA/BAAB needs an even block count")
    warmup = integer(timing.get("warmup"), "warmup", 1, 1000)
    iterations = integer(timing.get("iterations"), "iterations", 1, 1000)
    order_seed = integer(timing.get("order_seed"), "order_seed", 0, 2**31 - 1)
    return ("#pragma once\nstruct ContractShape { int m,n,k; bool acceptance; };\n"
            f"constexpr ContractShape kShapes[] = {{{','.join(entries)}}};\n"
            f"constexpr int kShapeCount={len(shapes)};\n"
            f"constexpr unsigned kSeeds[] = {{{','.join(map(str,seeds))}}};\n"
            f"constexpr int kSeedCount={len(seeds)};\n"
            f"constexpr double kAtol={float(num['atol'])!r}, kRtol={float(num['rtol'])!r};\n"
            f"constexpr bool kIntegerExact={str(num['integer_exact']).lower()};\n"
            f"constexpr int kBlocks={blocks}, kWarmup={warmup}, kIterations={iterations};\n"
            f"constexpr unsigned kOrderSeed={order_seed};\n")


def save_result(output, result):
    """Materialize only safe artifact names; remote-result never contains bytes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in result.get("files", {}).items():
        safe_relative(name)
        if not isinstance(data, bytes):
            raise ValueError("Non-byte artifact")
        atomic_write(output / "artifacts" / name, data)
    metadata = {key: value for key, value in result.items() if key != "files"}
    atomic_write(output / "remote-result.json", canonical_json(metadata))


@app.function(image=image, gpu="B300", cpu=4, memory=16384, timeout=600,
              retries=0, max_containers=1, scaledown_window=2)
def run(payload: dict[str, bytes], mode: str = "check"):
    """Build and evaluate both versions in one invocation/allocation."""
    root = Path(tempfile.mkdtemp(prefix="gemm-task-"))
    artifacts = root / "artifacts"
    artifacts.mkdir()
    result = {"status": "failed", "mode": mode, "started_at": now(), "command_records": []}
    deadline = time.monotonic() + 540

    def command(argv, log, limit=180):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise RuntimeError("Worker time budget exhausted")
        path = artifacts / log
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        record = {"argv": list(map(str, argv)), "log": log, "started_at": now()}
        result["command_records"].append(record)
        env = os.environ.copy()
        env["NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT"] = "1"
        with path.open("wb") as stream:
            process = subprocess.Popen(argv, cwd=root, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                process.wait(timeout=min(limit, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                record["timed_out"] = True
        record.update(returncode=process.returncode, seconds=time.monotonic() - start)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed ({process.returncode}): {argv[0]}; see {log}")
        return record

    try:
        if mode not in MODES:
            raise ValueError("Unknown mode")
        manifest = verify_payload(payload)
        result["manifest_sha256"] = sha(payload["manifest.json"])
        executed_worker_sha256 = sha(Path(__file__).read_bytes())
        result["executed_worker_sha256"] = executed_worker_sha256
        if executed_worker_sha256 != manifest["modal_worker.py"]:
            raise RuntimeError("Executed worker source differs from sealed modal_worker.py")
        for name, data in payload.items():
            safe_relative(name)
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        contract = json.loads(payload["contract.json"])
        header = contract_header(contract)
        (root / "contract.h").write_text(header)
        command(["nvidia-smi"], "logs/nvidia-smi.txt", 20)
        command(["nvcc", "--version"], "logs/nvcc-version.txt", 20)
        binary = artifacts / "evaluator"
        # tcgen05 inline PTX needs the architecture-specific virtual target too.
        # The -arch shorthand can also emit generic compute_103 PTX in CUDA 13.1.
        flags = ["nvcc", "-O3", "-lineinfo", "-std=c++17",
                 "-gencode=arch=compute_103a,code=sm_103a",
                 "--expt-relaxed-constexpr", "--expt-extended-lambda",
                 "evaluator.cu", "-lcuda", "-lcublas", "-o", str(binary)]
        command(flags, "logs/build.log", 180)
        environment = {
            "manifest_sha256": result["manifest_sha256"], "contract_sha256": sha(payload["contract.json"]),
            "binary_sha256": sha(binary.read_bytes()), "source_sha256": manifest,
            "generated_contract_header_sha256": sha(header.encode()), "compile_argv": flags,
            "mode": mode, "reference": "cublasGemmEx BF16 inputs / FP32 output / COMPUTE_32F",
            "executed_worker_sha256": executed_worker_sha256,
            "timer": "CUDA events; per-slot repeated launches; no CUDA Graph; hot repeated buffers",
        }
        atomic_write(artifacts / "environment.json", canonical_json(environment))
        command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem",
                 "--format=csv,noheader,nounits"], "logs/gpu-telemetry-before.csv", 20)
        # Profile mode has its own full correctness run; its event times are never a benchmark.
        command([str(binary), "check" if mode == "profile" else mode], "logs/evaluator.jsonl", 300)
        if mode == "profile":
            command(["ncu", "--version"], "logs/ncu-version.txt", 20)
            report = artifacts / "profile/candidate.ncu-rep"
            report.parent.mkdir()
            argv = ["ncu", "--section", "SpeedOfLight", "--section", "LaunchStats",
                    "--kernel-name-base", "demangled", "--rename-kernels", "off",
                    "--kernel-name", "regex:.*candidate::gemm_pipeline.*",
                    "--launch-skip", str(contract["timing"]["warmup"]), "--launch-count", "1",
                    "--cache-control", "none", "--clock-control", "none", "--import-source", "yes",
                    "--export", str(report), str(binary), "profile-target"]
            command(argv, "logs/profile-target.jsonl", 240)
            if not report.is_file() or report.stat().st_size == 0:
                raise RuntimeError("No NCU report was generated")
            command(["ncu", "--import", str(report), "--page", "raw", "--csv"], "profile/metrics.csv", 30)
        command(["nvidia-smi"], "logs/nvidia-smi-after.txt", 20)
        command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem",
                 "--format=csv,noheader,nounits"], "logs/gpu-telemetry-after.csv", 20)
        result["status"] = "complete"
    except BaseException as exc:
        import traceback
        result["error"] = repr(exc)
        atomic_write(artifacts / "logs/worker-error.txt", traceback.format_exc().encode())
    finally:
        result["finished_at"] = now()
        atomic_write(artifacts / "worker-status.json", canonical_json(result))
        # The executable is identified by hash; avoid returning a large redundant binary.
        files = {}
        total = 0
        omitted = []
        for path in sorted(artifacts.rglob("*")):
            if not path.is_file() or path == artifacts / "evaluator":
                continue
            size = path.stat().st_size
            name = path.relative_to(artifacts).as_posix()
            if size > MAX_FILE_BYTES or total + size > MAX_OUTPUT_BYTES:
                omitted.append({"path": name, "size": size})
                continue
            files[name] = path.read_bytes()
            total += size
        if omitted:
            result.update(status="failed", error="Artifact limit exceeded", omitted_artifacts=omitted)
        result["files"] = files
    return result


@app.local_entrypoint()
def main(bundle: str, output: str, mode: str = "check", allow_cloud: bool = False):
    if not allow_cloud:
        raise RuntimeError("Explicit --allow-cloud is required to upload sources and start paid GPU execution")
    if mode not in MODES:
        raise ValueError("Unknown mode")
    source = Path(bundle).resolve(strict=True)
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    payload = {name: (source / str(safe_relative(name))).read_bytes() for name in manifest}
    payload["manifest.json"] = manifest_bytes
    verify_payload(payload)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    invocation = out / "invocation.json"
    # Reserve the attempt before spawn. Unknown submission is not silently retried.
    with invocation.open("x") as stream:
        json.dump({"state": "submitting", "mode": mode, "created_at": now(),
                   "bundle_manifest_sha256": sha(manifest_bytes)}, stream, indent=2)
        stream.write("\n")
    call = run.spawn(payload, mode)
    record = {"state": "submitted", "call_id": call.object_id, "mode": mode,
              "bundle_manifest_sha256": sha(manifest_bytes), "created_at": now()}
    atomic_write(invocation, canonical_json(record))
    print(json.dumps(record), flush=True)
    try:
        result = call.get(timeout=30)
    except TimeoutError:
        print("PENDING: use saved call_id to poll; do not resubmit", flush=True)
        return
    save_result(out, result)
    if result.get("status") != "complete":
        raise RuntimeError("Remote evaluation failed; inspect remote-result.json and artifact logs")
