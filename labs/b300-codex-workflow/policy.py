"""CPU-only, fail-closed acceptance of sealed B300 evaluator evidence."""
import itertools
import math
import random
import statistics


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def shape_id(shape):
    return f"{shape['m']}x{shape['n']}x{shape['k']}"


def geomean(values):
    return math.exp(statistics.fmean(math.log(x) for x in values))


def evaluate(contract, rows, remote):
    """No model-written summary, file-presence check, or success-only metric."""
    try:
        require(remote.get('status') == 'complete', 'Remote invocation did not complete')
        require(remote.get('mode') == 'bench', 'Only unprofiled bench evidence is admissible')
        completed = [r for r in rows if r.get('event') == 'complete']
        require(len(completed) == 1 and completed[0].get('mode') == 'bench',
                'Missing/duplicate bench completion marker')
        require(not any(r.get('event') == 'error' for r in rows), 'Evaluator error event')
        environments = [r for r in rows if r.get('event') == 'environment']
        require(len(environments) == 1, 'Missing/duplicate environment record')
        env = environments[0]
        require('B300' in env.get('gpu_name', ''), 'Wrong GPU')
        require((env.get('cc_major'), env.get('cc_minor')) == (10, 3), 'Wrong compute capability')
        shapes = {shape_id(s): s for s in contract['shapes']}
        expected = set(itertools.product(
            ('before', 'after'), shapes, ('baseline', 'candidate'),
            contract['seeds'], ('integer_exact', 'random_bf16', 'structured')))
        checks = [r for r in rows if r.get('event') == 'validation']
        seen = set()
        for r in checks:
            key = (r.get('phase'), r.get('shape_id'), r.get('impl'), r.get('seed'), r.get('data'))
            require(key in expected and key not in seen, 'Unknown/duplicate correctness case')
            seen.add(key)
            s = shapes[r['shape_id']]
            require(r.get('elements') == s['m'] * s['n'], 'Partial output validation')
            require(r.get('passed') is True and all(r.get(k) == 0 for k in
                    ('bad', 'nonfinite', 'reference_nonfinite')), 'Correctness check failed')
        require(seen == expected, 'Incomplete before/after correctness matrix')
        timed_shapes = {shape_id(s) for s in contract['shapes'] if s['role'] == 'acceptance'}
        require(bool(timed_shapes), 'No acceptance shapes')
        timing = contract['timing']
        slots = {}
        for r in rows:
            if r.get('event') != 'timing':
                continue
            sid, block, pos = r.get('shape_id'), r.get('block'), r.get('slot')
            require(sid in timed_shapes, 'Timing for unknown shape')
            require(type(block) is int and 0 <= block < timing['blocks'], 'Invalid block index')
            require(type(pos) is int and 0 <= pos < 4, 'Invalid slot position')
            pattern = r.get('pattern')
            require(pattern in ('ABBA', 'BAAB'), 'Invalid order pattern')
            require(r.get('impl') == ('baseline' if pattern[pos] == 'A' else 'candidate'),
                    'Slot implementation/order mismatch')
            require(r.get('iterations') == timing['iterations'], 'Iteration count changed')
            latency = r.get('latency_ms')
            require(type(latency) in (int, float) and math.isfinite(latency) and latency > 0,
                    'Invalid/nonfinite latency')
            key = sid, block, pos
            require(key not in slots, 'Duplicate timing slot')
            slots[key] = r
        require(len(slots) == len(timed_shapes) * timing['blocks'] * 4, 'Incomplete timing schedule')
        live = [r for r in rows if r.get('event') == 'timing_validation']
        live_seen = set()
        for r in live:
            key = r.get('shape_id'), r.get('block')
            require(key[0] in timed_shapes and type(key[1]) is int and
                    0 <= key[1] < timing['blocks'] and key not in live_seen,
                    'Invalid/duplicate live-output check')
            live_seen.add(key)
            last = slots[key[0], key[1], 3]
            s = shapes[key[0]]
            require(r.get('impl') == last['impl'] and r.get('elements') == s['m'] * s['n'] and
                    r.get('passed') is True and r.get('bad') == 0 and r.get('nonfinite') == 0,
                    'Repeated-call output validation failed')
        require(len(live_seen) == len(timed_shapes) * timing['blocks'], 'Missing live-output checks')
        integrity = [r for r in rows if r.get('event') == 'input_integrity']
        integrity_keys = {(r.get('shape_id'), r.get('phase')) for r in integrity}
        require(len(integrity) == 2 * len(timed_shapes) and
                integrity_keys == set(itertools.product(timed_shapes, ('timing_before', 'timing_after'))) and
                all(r.get('passed') is True and r.get('bad_a') == 0 and r.get('bad_b') == 0
                    for r in integrity), 'Input mutation or missing integrity checks')
        by_shape = {}
        accept = contract['acceptance']
        for sid in sorted(timed_shapes):
            ratios, patterns = [], {'ABBA': [], 'BAAB': []}
            for block in range(timing['blocks']):
                block_rows = [slots[sid, block, pos] for pos in range(4)]
                require(len({r['pattern'] for r in block_rows}) == 1, 'Mixed block pattern')
                base = statistics.fmean(r['latency_ms'] for r in block_rows if r['impl'] == 'baseline')
                cand = statistics.fmean(r['latency_ms'] for r in block_rows if r['impl'] == 'candidate')
                ratio = base / cand
                ratios.append(ratio)
                patterns[block_rows[0]['pattern']].append(ratio)
            require(all(len(v) == timing['blocks'] // 2 for v in patterns.values()),
                    'ABBA/BAAB blocks not balanced')
            rng = random.Random(accept['bootstrap_seed'])
            boot = sorted(geomean(rng.choices(ratios, k=len(ratios)))
                          for _ in range(accept['bootstrap_resamples']))
            require(len(boot) >= 100, 'Insufficient bootstrap resamples')
            interval = [boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]]
            by_shape[sid] = {'ratio': geomean(ratios), 'block_ratios': ratios,
                             'descriptive_block_bootstrap_95pct': interval,
                             'pattern_ratios': {p: geomean(v) for p, v in patterns.items()}}
        overall = geomean(v['ratio'] for v in by_shape.values())
        reduction = 1 - 1 / overall
        reasons = []
        if reduction < accept['minimum_latency_reduction']:
            reasons.append('Overall latency reduction below predeclared threshold')
        for sid, value in by_shape.items():
            if value['ratio'] < accept['minimum_shape_ratio']:
                reasons.append(sid + ': shape regression')
            if accept['require_each_shape_ci_lower_gt_one'] and value['descriptive_block_bootstrap_95pct'][0] <= 1:
                reasons.append(sid + ': benefit unresolved within paired blocks')
            if accept['require_both_order_patterns_gt_one'] and min(value['pattern_ratios'].values()) <= 1:
                reasons.append(sid + ': improvement not consistent across order patterns')
        return {'verdict': 'KEEP_BASELINE' if reasons else 'ACCEPT_SCOPED',
                'reasons': reasons, 'overall_ratio_baseline_over_candidate': overall,
                'overall_latency_reduction': reduction, 'by_shape': by_shape,
                'scope': 'Only declared shapes, dtype, launch ABI and this GPU session; not independent multi-session replication'}
    except (EvidenceError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        return {'verdict': 'INVALID_EVIDENCE', 'reasons': [str(exc)]}
