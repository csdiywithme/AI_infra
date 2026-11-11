#!/usr/bin/env python3
"""Local supervisor. No cloud work without the explicit submit --allow-cloud flag."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = Path.home() / 'csdiy' / 'wmhpc-training-camp-x-lcpu-ai-infra-seminars'
BASELINE_REL = Path('assignment02/cuda/m4_gemm/04l_wide_streamed.cu')
COMMON_REL = Path('assignment02/cuda/common.h')
BASELINE_SHA = '456e9499a9389a9bb5f5f0ae3205e90e4456c79654e586afdf9ffada4912c0d7'
COMMON_SHA = '80017b942556fdea270cb4b3790d0f22c05043a675c471ed1a17ef82e9c86619'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_snapshot(path):
    path = path.resolve(strict=True)
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts}
    if actual != set(manifest) | {'manifest.json'}:
        raise ValueError('Snapshot file set changed')
    for name, expected in manifest.items():
        file = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts or file.is_symlink():
            raise ValueError('Invalid snapshot path')
        if sha(file) != expected:
            raise ValueError('Snapshot changed: ' + name)
    return manifest, sha(manifest_path)


def doctor(args):
    try:
        version = importlib.metadata.version('modal')
    except importlib.metadata.PackageNotFoundError:
        version = None
    print(json.dumps({'python': sys.executable, 'modal_sdk': version,
                      'local_nvcc': shutil.which('nvcc'),
                      'baseline_available': (args.project / BASELINE_REL).exists(),
                      'remote_probe': 'NOT_RUN',
                      'note': 'macOS needs no local CUDA; Modal SDK is needed only for submit/poll/cancel'}, indent=2))


def prepare(args):
    work = args.work.resolve()
    if work.exists():
        raise ValueError('Workspace exists; use it or choose a new --work path')
    baseline, common = args.project / BASELINE_REL, args.project / COMMON_REL
    if sha(baseline) != BASELINE_SHA or sha(common) != COMMON_SHA:
        raise ValueError('Existing baseline/dependency differs from audited source; review and revise task first')
    (work / 'fixed').mkdir(parents=True)
    (work / 'candidate').mkdir()
    shutil.copyfile(baseline, work / 'fixed/baseline.cu')
    shutil.copyfile(common, work / 'fixed/common.h')
    shutil.copyfile(baseline, work / 'candidate/candidate.cu')
    shutil.copyfile(ROOT / 'contract.json', work / 'fixed/contract.json')
    dump(work / 'origin.json', {'project': str(args.project), 'baseline': str(baseline),
                               'baseline_sha256': BASELINE_SHA, 'common_sha256': COMMON_SHA,
                               'candidate_initially_equals_baseline': True})
    print(work)


def seal(args):
    work = args.work.resolve(strict=True)
    if sha(work / 'fixed/baseline.cu') != BASELINE_SHA or sha(work / 'fixed/common.h') != COMMON_SHA:
        raise ValueError('Frozen baseline changed')
    contract = json.loads((work / 'fixed/contract.json').read_text())
    if contract != json.loads((ROOT / 'contract.json').read_text()):
        raise ValueError('Task contract changed; create a reviewed task revision')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = work / 'snapshots' / run_id
    files = {
        'src/cuda/m4_gemm/baseline.cu': work / 'fixed/baseline.cu',
        'src/cuda/m4_gemm/candidate.cu': work / 'candidate/candidate.cu',
        'src/cuda/common.h': work / 'fixed/common.h',
        'contract.json': work / 'fixed/contract.json',
        'evaluator.cu': ROOT / 'evaluator.cu',
        'modal_worker.py': ROOT / 'modal_worker.py',
        'policy.py': ROOT / 'policy.py',
    }
    for name, source in files.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    manifest = {name: sha(out / name) for name in sorted(files)}
    dump(out / 'manifest.json', manifest)
    verify_snapshot(out)
    print(out)


def submit(args):
    if not args.allow_cloud:
        raise ValueError('submit requires --allow-cloud: uploads this snapshot and starts billable B300 execution')
    _, digest = verify_snapshot(args.snapshot)
    attempt = args.output.resolve()
    if attempt.exists():
        raise ValueError('Attempt exists; poll/cancel it instead of resubmitting')
    # Verify the local runtime before reserving the attempt.
    importlib.metadata.version('modal')
    attempt.mkdir(parents=True)
    dump(attempt / 'dispatch.json', {'snapshot': str(args.snapshot.resolve()),
                                   'bundle_manifest_sha256': digest, 'mode': args.mode,
                                   'state': 'DISPATCHING', 'profile': args.profile})
    command = [sys.executable, '-m', 'modal', 'run', '--detach',
               str(args.snapshot.resolve() / 'modal_worker.py'),
               '--bundle', str(args.snapshot.resolve()), '--output', str(attempt),
               '--mode', args.mode, '--allow-cloud']
    env = os.environ.copy()
    env['MODAL_PROFILE'] = args.profile
    # A lost submission response is UNKNOWN, never an automatic retry.
    with (attempt / 'dispatch.log').open('w') as log:
        p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    info = json.loads((attempt / 'invocation.json').read_text()) if (attempt / 'invocation.json').exists() else {}
    valid_call = (bool(info.get('call_id')) and info.get('bundle_manifest_sha256') == digest
                  and info.get('mode') == args.mode)
    state = 'RESULT_AVAILABLE' if (attempt / 'remote-result.json').exists() else (
        'CALL_RECORDED' if valid_call else 'SUBMISSION_UNKNOWN')
    dump(attempt / 'dispatch-status.json', {'state': state, 'returncode': p.returncode,
                                          'log': 'dispatch.log'})
    print(json.dumps({'attempt': str(attempt), 'state': state,
                      'next': 'gate if result exists; otherwise poll. UNKNOWN: inspect Modal app before any new submission'}, indent=2))
    return p.returncode


def store_result(attempt, result):
    files = result.pop('files', {})
    for name, content in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not isinstance(content, bytes):
            raise ValueError('Invalid remote artifact')
        dest = attempt / 'artifacts' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    dump(attempt / 'remote-result.json', result)


def poll_or_cancel(args):
    attempt = args.attempt.resolve(strict=True)
    dispatch = json.loads((attempt / 'dispatch.json').read_text())
    os.environ['MODAL_PROFILE'] = dispatch['profile']
    import modal
    info = json.loads((attempt / 'invocation.json').read_text())
    if (not info.get('call_id') or info.get('bundle_manifest_sha256') != dispatch['bundle_manifest_sha256']
            or info.get('mode') != dispatch['mode']):
        raise ValueError('Submission identity is unknown; inspect Modal app, do not automatically resubmit')
    call = modal.FunctionCall.from_id(info['call_id'])
    if args.command == 'cancel':
        call.cancel()
        dump(attempt / 'cancel-request.json', {'call_id': info['call_id'], 'state': 'CANCEL_REQUESTED'})
        print('Cancellation requested; this does not claim that remote cleanup has completed')
        return
    if (attempt / 'remote-result.json').exists():
        print('Result already saved; run gate')
        return
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        print('PENDING_OR_RUNNING: no new GPU submission was made')
        return
    store_result(attempt, result)
    print('Result saved: ' + str(attempt / 'remote-result.json'))


def gate(args):
    attempt = args.attempt.resolve(strict=True)
    dispatch = json.loads((attempt / 'dispatch.json').read_text())
    snap = Path(dispatch['snapshot'])
    manifest, digest = verify_snapshot(snap)
    result = json.loads((attempt / 'remote-result.json').read_text())
    if dispatch.get('mode') != 'bench' or result.get('mode') != 'bench':
        raise ValueError('Only a bench attempt can produce a performance verdict')
    # Require matching request, snapshot, runtime and post-build identity.
    if dispatch['bundle_manifest_sha256'] != digest or result.get('manifest_sha256') != digest:
        raise ValueError('Result/source snapshot identity mismatch')
    # A build failure legitimately has no post-build environment or timing log.
    # Report the original failure before asking for evidence it could not create.
    if result.get('status') != 'complete':
        verdict = {
            'verdict': 'INVALID_EVIDENCE',
            'reasons': ['Remote evaluation did not complete: ' + str(result.get('error', result.get('status')))],
            'remote_status': result.get('status'),
            'bundle_manifest_sha256': digest,
            'candidate_sha256': manifest['src/cuda/m4_gemm/candidate.cu'],
            'baseline_sha256': manifest['src/cuda/m4_gemm/baseline.cu'],
            'gate_supervisor_sha256': sha(Path(__file__)),
            'baseline_replaced': False,
        }
        dump(attempt / 'verdict.json', verdict)
        print(json.dumps(verdict, indent=2))
        return 2
    environment = json.loads((attempt / 'artifacts/environment.json').read_text())
    if environment.get('mode') != 'bench':
        raise ValueError('Build/runtime mode is not bench')
    if environment.get('manifest_sha256') != digest:
        raise ValueError('Build environment belongs to a different snapshot')
    if environment.get('contract_sha256') != manifest['contract.json']:
        raise ValueError('Runtime contract identity mismatch')
    if environment.get('executed_worker_sha256') != manifest['modal_worker.py']:
        raise ValueError('Executed worker differs from sealed worker')
    if environment.get('source_sha256') != manifest:
        raise ValueError('Runtime source inventory differs from sealed snapshot')
    if not environment.get('binary_sha256'):
        raise ValueError('Missing compiled binary identity')
    rows = [json.loads(line) for line in (attempt / 'artifacts/logs/evaluator.jsonl').read_text().splitlines()
            if line.strip().startswith('{')]
    policy = module(snap / 'policy.py', 'sealed_gate_policy')
    verdict = policy.evaluate(json.loads((snap / 'contract.json').read_text()), rows, result)
    verdict.update(bundle_manifest_sha256=digest, candidate_sha256=manifest['src/cuda/m4_gemm/candidate.cu'],
                   baseline_sha256=manifest['src/cuda/m4_gemm/baseline.cu'],
                   binary_sha256=environment['binary_sha256'], baseline_replaced=False)
    dump(attempt / 'verdict.json', verdict)
    print(json.dumps(verdict, indent=2))
    return 2 if verdict['verdict'] == 'INVALID_EVIDENCE' else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('doctor', 'prepare'):
        q = sub.add_parser(cmd)
        q.add_argument('--project', type=Path, default=PROJECT)
        if cmd == 'prepare':
            q.add_argument('--work', type=Path, default=ROOT / 'work')
    q = sub.add_parser('seal'); q.add_argument('--work', type=Path, default=ROOT / 'work')
    q = sub.add_parser('submit')
    q.add_argument('--snapshot', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--mode', choices=('check', 'bench', 'profile'), default='check')
    profile = os.environ.get('MODAL_PROFILE')
    q.add_argument('--profile', default=profile, required=not profile)
    q.add_argument('--allow-cloud', action='store_true')
    for cmd in ('poll', 'cancel', 'gate'):
        q = sub.add_parser(cmd); q.add_argument('--attempt', type=Path, required=True)
    args = p.parse_args()
    funcs = {'doctor': doctor, 'prepare': prepare, 'seal': seal, 'submit': submit,
             'poll': poll_or_cancel, 'cancel': poll_or_cancel, 'gate': gate}
    try:
        return funcs[args.command](args) or 0
    except Exception as exc:
        if args.command == 'gate' and args.attempt.is_dir():
            dump(args.attempt / 'verdict.json', {'verdict': 'INVALID_EVIDENCE',
                                                'reasons': [type(exc).__name__ + ': ' + str(exc)],
                                                'baseline_replaced': False})
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
