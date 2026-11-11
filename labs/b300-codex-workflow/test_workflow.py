"""CPU-only acceptance counterexamples. All timing data here are synthetic."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import argparse
import contextlib
import io
import shutil

import policy
import workflow


ROOT = Path(__file__).resolve().parent


def fixture(reduction=.10):
    c = json.loads((ROOT / 'contract.json').read_text())
    c['acceptance']['bootstrap_resamples'] = 100
    rows = [{'event': 'environment', 'gpu_name': 'NVIDIA B300', 'cc_major': 10, 'cc_minor': 3}]
    for s in c['shapes']:
        sid = policy.shape_id(s)
        for phase in ('before', 'after'):
            for impl in ('baseline', 'candidate'):
                for seed in c['seeds']:
                    for data in ('integer_exact', 'random_bf16', 'structured'):
                        rows.append(dict(event='validation', phase=phase, shape_id=sid, impl=impl,
                                         seed=seed, data=data, elements=s['m'] * s['n'], bad=0,
                                         nonfinite=0, reference_nonfinite=0, passed=True))
        if s['role'] != 'acceptance':
            continue
        for phase in ('timing_before', 'timing_after'):
            rows.append(dict(event='input_integrity', shape_id=sid, phase=phase, passed=True,
                             bad_a=0, bad_b=0))
        for block in range(12):
            pattern = 'ABBA' if block % 2 == 0 else 'BAAB'
            for slot, letter in enumerate(pattern):
                rows.append(dict(event='timing', shape_id=sid, block=block, pattern=pattern,
                                 slot=slot, impl='baseline' if letter == 'A' else 'candidate',
                                 latency_ms=1.0 if letter == 'A' else 1.0 - reduction, iterations=100))
            rows.append(dict(event='timing_validation', shape_id=sid, block=block,
                             impl='baseline' if pattern[-1] == 'A' else 'candidate',
                             elements=s['m'] * s['n'], passed=True, bad=0, nonfinite=0))
    rows.append({'event': 'complete', 'mode': 'bench'})
    return c, rows, {'status': 'complete', 'mode': 'bench'}


class AcceptanceTests(unittest.TestCase):
    def verdict(self, mutate=None, reduction=.10):
        c, rows, remote = fixture(reduction)
        if mutate:
            mutate(rows, remote)
        return policy.evaluate(c, rows, remote)['verdict']

    def test_valid_scoped_gain(self):
        self.assertEqual(self.verdict(), 'ACCEPT_SCOPED')

    def test_no_gain_keeps_baseline(self):
        self.assertEqual(self.verdict(reduction=0), 'KEEP_BASELINE')

    def test_missing_correctness_cannot_pass(self):
        self.assertEqual(self.verdict(lambda rows, _: rows.pop(1)), 'INVALID_EVIDENCE')

    def test_failed_check_cannot_be_masked(self):
        self.assertEqual(self.verdict(lambda rows, _: rows[1].update(passed=False)), 'INVALID_EVIDENCE')

    def test_nan_timing_rejected(self):
        def mutate(rows, _):
            next(r for r in rows if r['event'] == 'timing')['latency_ms'] = float('nan')
        self.assertEqual(self.verdict(mutate), 'INVALID_EVIDENCE')

    def test_missing_or_duplicate_slot_rejected(self):
        def mutate(rows, _):
            rows.append(copy.deepcopy(next(r for r in rows if r['event'] == 'timing')))
        self.assertEqual(self.verdict(mutate), 'INVALID_EVIDENCE')

    def test_live_output_or_input_mutation_rejected(self):
        for event in ('input_integrity', 'timing_validation'):
            def mutate(rows, _, event=event):
                next(r for r in rows if r['event'] == event)['passed'] = False
            self.assertEqual(self.verdict(mutate), 'INVALID_EVIDENCE')

    def test_profiler_is_not_performance_evidence(self):
        self.assertEqual(self.verdict(lambda _, remote: remote.update(mode='profile')), 'INVALID_EVIDENCE')

    def test_missing_completion_rejected(self):
        self.assertEqual(self.verdict(lambda rows, _: rows.pop()), 'INVALID_EVIDENCE')


class SnapshotTests(unittest.TestCase):
    def test_build_failure_reports_remote_error_without_post_build_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snap = root / 'snapshot'
            for name in ('src/cuda/m4_gemm/baseline.cu', 'src/cuda/m4_gemm/candidate.cu'):
                path = snap / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic test input\n')
            manifest = {str(p.relative_to(snap)): workflow.sha(p) for p in snap.rglob('*') if p.is_file()}
            workflow.dump(snap / 'manifest.json', manifest)
            digest = workflow.sha(snap / 'manifest.json')
            attempt = root / 'attempt'
            workflow.dump(attempt / 'dispatch.json', {'snapshot': str(snap),
                          'bundle_manifest_sha256': digest, 'mode': 'bench'})
            workflow.dump(attempt / 'remote-result.json', {'status': 'failed', 'mode': 'bench',
                          'manifest_sha256': digest, 'error': 'nvcc failed; see logs/build.log'})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(workflow.gate(argparse.Namespace(attempt=attempt)), 2)
            verdict = json.loads((attempt / 'verdict.json').read_text())
            self.assertEqual(verdict['verdict'], 'INVALID_EVIDENCE')
            self.assertIn('nvcc failed', verdict['reasons'][0])
            self.assertFalse(verdict['baseline_replaced'])

    def test_full_local_gate_binds_snapshot_and_result(self):
        # Synthetic evidence validates plumbing only, not CUDA or real performance.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snap = root / 'snapshot'
            snap.mkdir()
            c, rows, remote = fixture()
            workflow.dump(snap / 'contract.json', c)
            shutil.copyfile(ROOT / 'policy.py', snap / 'policy.py')
            for name in ('modal_worker.py', 'src/cuda/m4_gemm/baseline.cu', 'src/cuda/m4_gemm/candidate.cu'):
                path = snap / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic test input\n')
            manifest = {str(p.relative_to(snap)): workflow.sha(p) for p in snap.rglob('*') if p.is_file()}
            workflow.dump(snap / 'manifest.json', manifest)
            digest = workflow.sha(snap / 'manifest.json')
            attempt = root / 'attempt'
            workflow.dump(attempt / 'dispatch.json', {'snapshot': str(snap),
                          'bundle_manifest_sha256': digest, 'mode': 'bench'})
            remote['manifest_sha256'] = digest
            workflow.dump(attempt / 'remote-result.json', remote)
            workflow.dump(attempt / 'artifacts/environment.json', {
                'mode': 'bench', 'manifest_sha256': digest,
                'contract_sha256': manifest['contract.json'],
                'executed_worker_sha256': manifest['modal_worker.py'],
                'source_sha256': manifest, 'binary_sha256': 'a' * 64})
            log = attempt / 'artifacts/logs/evaluator.jsonl'
            log.parent.mkdir()
            log.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(workflow.gate(argparse.Namespace(attempt=attempt)), 0)
            result = json.loads((attempt / 'verdict.json').read_text())
            self.assertEqual(result['verdict'], 'ACCEPT_SCOPED')
            remote['manifest_sha256'] = 'b' * 64
            workflow.dump(attempt / 'remote-result.json', remote)
            with self.assertRaises(ValueError):
                workflow.gate(argparse.Namespace(attempt=attempt))

    def test_source_mutation_and_extra_files_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'candidate.cu').write_bytes(b'original')
            workflow.dump(root / 'manifest.json', {'candidate.cu': workflow.sha(root / 'candidate.cu')})
            workflow.verify_snapshot(root)
            (root / 'candidate.cu').write_bytes(b'modified')
            with self.assertRaises(ValueError):
                workflow.verify_snapshot(root)
            (root / 'candidate.cu').write_bytes(b'original')
            (root / 'undeclared.h').write_bytes(b'header')
            with self.assertRaises(ValueError):
                workflow.verify_snapshot(root)

    def test_real_source_layout_and_worker_header(self):
        tree = ast.parse((ROOT / 'modal_worker.py').read_text())
        allowed = {'integer', 'contract_header'}
        code = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name in allowed], type_ignores=[])
        import math
        ns = {'math': math}
        exec(compile(code, 'selected_contract_helpers', 'exec'), ns)
        c = json.loads((ROOT / 'contract.json').read_text())
        h = ns['contract_header'](c)
        self.assertIn('kShapeCount=3', h)
        self.assertIn('{4096,4096,4096,true}', h)
        c['shapes'][0]['n'] = 769
        with self.assertRaises(ValueError):
            ns['contract_header'](c)


if __name__ == '__main__':
    unittest.main()
