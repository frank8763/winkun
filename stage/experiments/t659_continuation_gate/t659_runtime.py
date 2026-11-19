"""No learning: matched native continuation measurements and independent audit."""
from pathlib import Path
import fcntl
import gzip
import json
import os
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t659_continuation_gate'
DOC = ROOT / 'stage/docs/day_20261119/t659'
sys.path.insert(0, str(ROOT / 'stage/experiments/t645_iterative_recovery_relay'))
import control as Resource
sys.path.insert(0, str(ROOT / 'stage/experiments/t654_verified_deviation_recovery'))
import t654_runtime as Prior
import t659_native as Native

R = Prior.R
read, save, sha, digest, np, torch = Prior.read, Prior.save, Prior.sha, Prior.digest, Prior.np, Prior.torch


def plan():
    return read(HERE / 'protocol.json')


def verify(p, *args):
    assert p['authorization_approved'] and time.time() < p['absolute_all_stop']
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    assert R.used(OUT) <= p['max_bytes']
    assert R.used(ROOT / 'stage/runs') - R.used(OUT) + p['max_bytes'] < p['global_bytes']
    assert shutil.disk_usage(ROOT).free >= 40 * 2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    Resource.resources()
    helper = read(DOC/'AFFINITY_GUARD.json')
    assert helper['state'] == 'ACTIVE_RESOURCE_ONLY' and Path('/proc', str(helper['pid'])).exists()
    for f, h in p['inputs'].items():
        assert sha(f) == h, f


def setup(p):
    Prior.OUT, Prior.DOC, Prior.HERE, Prior.verify = OUT, DOC, HERE, verify
    Prior.setup(p)
    R.N.worker, R.N.inspect, Prior.O.N.inspect = Native.worker, Native.inspect, Native.inspect


def safe(row):
    return bool(row['complete'] and not row['administrative_censor'] and row['success'] and not row['collision']
                and not ('fullfield_start_pose' in row['request']['job'] and row.get('field_boundary', False)))


def panel(p, rows):
    fresh = [r for r in rows if not r.get('exact')]
    answer = []
    for query in p['queries']:
        rr = {r['request']['job']['continuation_kind']: r for r in fresh
              if r['request']['job']['query_key'] == query['query_key']}
        assert set(rr) == {'reference', 'candidate_once', 'candidate_full'}
        complete = all(r['complete'] and not r['administrative_censor'] for r in rr.values())
        b, one, whole = [int(safe(rr[k])) for k in ['reference', 'candidate_once', 'candidate_full']]
        answer.append(dict(query_key=query['query_key'], id=query['id'], prefix_first=query['prefix_first'],
                           original_start=query['prefix_first'] == 0, source_category=query['source_category'],
                           complete=complete, reference=b if complete else None,
                           first_action_then_reference=one if complete else None,
                           full_candidate=whole if complete else None,
                           local_old_continuation_difference=one-b if complete else None,
                           complete_policy_difference=whole-b if complete else None,
                           suffix_replacement_effect=whole-one if complete else None,
                           first_action_proxy_mismatch=bool(complete and one != whole),
                           rows={k: v['path'] for k, v in rr.items()}))
    return dict(queries=answer,
                proxy_mismatch_queries=[v['query_key'] for v in answer if v['first_action_proxy_mismatch']],
                all_complete=all(v['complete'] for v in answer),
                predictor_trained=False, actor_updated=False, alpha_gradient_validated=False,
                independent_generalization_or_nonforgetting_proven=False,
                inference='Per-query replacement effects only; not a policy-improvement theorem test')


def main():
    p = plan()
    verify(p)
    setup(p)
    R.E.L.install_stop_handlers()
    lock = (ROOT / 'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'science_started.json').exists()
    save(OUT / 'science_started.json', dict(pid=os.getpid(), epoch=time.time()))
    rt = R.Runtime(p)
    error, result = None, None
    try:
        rt.pool = R.N.Pool(dict(p, out=str(OUT), native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        for kind, model in [('reference', p['reference']), ('candidate_once', p['reference']), ('candidate_full', p['candidate'])]:
            jobs = [dict(q, continuation_kind=kind) for q in p['queries']]
            vv = rt.request(jobs, model, kind, True)
            rt.request(jobs, model, 'exact_' + kind, False, [r for a, r in vv])
        result = panel(p, rt.rows)
        save(OUT / 'PANEL.json', result)
    except BaseException:
        error = traceback.format_exc()
        save(OUT / 'failure.json', dict(error=error, no_restart=True, raw_preserved=True))
    finally:
        if rt.pool:
            rt.pool.close()
        rt.commands += 3
        save(OUT / 'summary.json', dict(state='FAILED' if error else 'COMPLETE', pid=os.getpid(), error=error,
             physics=sum(rt.counts), commands=rt.commands, elapsed_s=time.time()-rt.start,
             result=result, new_Actor=0, new_Critic=0, new_optimizer=0, full407=None, retained=p['safe_retained']))
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert error is None, error


def audit():
    p = plan()
    verify(p)
    setup(p)
    s = read(OUT / 'summary.json')
    assert s['state'] == 'COMPLETE' and not Path('/proc', str(s['pid'])).exists()
    for generation in read(OUT / 'worker_generations.json'):
        assert all(not Path('/proc', str(pid)).exists() for pid in [generation['parent'], *generation['workers']])
    rows = [json.loads(x) for x in gzip.decompress((OUT / 'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by = {r['path']: r for r in rows}
    assert len(rows) == len(by) == 6 * len(p['queries'])
    for row in rows:
        assert sha(row['path']) == row['sha256']
        a = R.E.N.S.arrays(row['path'])
        if row['full']:
            Native.inspect(p, a, row)
        else:
            assert all(digest(v) == row['array_hashes'][key] for key, v in a.items())
        if row.get('exact'):
            ref = by[row['reference_path']]
            assert row['array_hashes'] == ref['array_hashes'] and row['integration_sha256'] == ref['integration_sha256']
    assert panel(p, rows) == read(OUT / 'PANEL.json') == s['result']
    assert sum(r['physics'] for r in rows) == s['physics'] <= p['max_physics']
    assert s['commands'] == len(rows) + 6 == p['max_commands']
    for model in [p['reference'], p['candidate'], p['safe_retained']]:
        assert sha(model['path']) == model['sha256']
    save(OUT / 'audit.json', dict(state='PASS', all_full_causal_recurrences_and_forced_actions=True,
         every_fresh_exact_replay_verified=True, original_start_and_prefix_separated=True,
         frozen_models_unchanged=True, new_native=0, new_optimizer=0))


if __name__ == '__main__':
    {'main': main, 'audit': audit}[sys.argv[1]]()
