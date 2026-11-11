"""Register only after the previous writer, audits and raw backup finish."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy
import gzip
import io
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t479_runtime as T
L, P, E, M, R, torch, np = T.L, T.P, T.E, T.M, T.R, T.torch, T.np
read, save, sha, digest = T.read, T.save, T.sha, T.digest
OUT, DOC = T.OUT, ROOT / 'stage/docs/review_20261111'


def prepare():
    import t479_cache as K
    assert not OUT.exists()
    previous = ROOT / 'stage/runs/t478_wide8_pooled_credit'
    K._idle()
    report = read(DOC / 'T478_FINAL_REPORT.json')
    assert report['state'] == 'COMPLETE_AUDITED_LOCAL'
    assert report['remote'] == K.COMPLETE
    assert read(previous / 'learn/audit.json')['state'] == 'PASS'
    assert read(ROOT / 'stage/docs/long_distance_20261101/release_receipt_T478.json')['state'] == K.COMPLETE
    # No copying a rejected proposal: only the previous actual retained model.
    base = report['selected']
    # This preregistered pilot concerns precisely the documented 5-left/3-right
    # coverage gap. A different retained initial state requires a fresh design.
    assert base['wide8']['success'] == 8 and base['wide6']['success'] == 53 and base['old275']['success'] == 274
    assert len(base['success_ids']) == 335
    stop = datetime(2026, 11, 11, 15, 0, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    assert time.time() + 8400 < stop, 'Full new bounded window does not fit'
    old = read(previous / 'protocol.json')
    p = copy.deepcopy(old)
    capture_ids = [x for x in base['success_ids'] if x.startswith('wide117_8m_')]
    assert sum('_LEFT_' in x for x in capture_ids) == 5 and len(capture_ids) == 8
    import t478_storage as S
    historic = ROOT / 'stage/runs/t477_wide8_recovery/longcredit/cases.jsonl'
    rr = [json.loads(x) for x in S.text(historic).splitlines()]
    cached = {r['job_id']: r for r in rr if r['role'] == 'FINAL_ALL383'}
    assert all(cached[x]['success'] and cached[x]['loaded_model_sha256'] == base['model']['sha256'] for x in capture_ids)
    first = [r for r in T.rows(previous / 'learn') if r['role'] == 'TRAIN_0' and r['success'] and not r['collision']]
    assert len(first) == 3 and sum('_LEFT_' in r['job_id'] for r in first) == 1
    memory = old['own_reference_rows'] + first
    assert len(memory) == len({r['sha256'] for r in memory}) == 5
    jobs = {j['id']: j for j in old['wide8_jobs']}
    p.update(id='T479_MATCHED_BALANCED_SUCCESS_PPO_VS_SUCCESS_SIL',
             out=str(OUT), initial=base['model'], initial_result=base,
             initial_success_ids=base['success_ids'], capture_success_ids=capture_ids,
             capture_jobs=[jobs[x] for x in capture_ids],
             capture_cached={x: cached[x] for x in capture_ids}, sil_rows=memory,
             seed=20261111479, epochs=16, keep_weight=1.,
             sil_actor_weight=.1, sil_value_weight=.01,
             max_bytes=30 * 2**20, full_bytes=2**20, compact_bytes=8192,
             model_bytes=1792 * 1024, max_physics=24000000, max_commands=1000,
             main_seconds=5400, learning_seconds=1200, audit_seconds=900,
             root_max_physics=24000000, root_max_commands=1000,
             root_max_main_seconds=5400, all_seconds=7800,
             absolute_all_stop=stop, absolute_native_stop=stop-2700,
             absolute_learning_stop=stop-7200,
             user_authorization='Nov11 nextstep: independent finite matched pilot, not a restart or enlargement of T478. All by15:00Beijing; bounded admission may refuse launch.',
             mechanism='A: pooled recurrentPPO + complete8ownsuccess LEFT/RIGHT equal-weight function retention. B: identical plus offpolicy success-filtered SIL actor.1/value.01. Both same fresh4/matureinitial/epochs; common KL earlystop. Standard SIL adaptation not original.',
             acceptance='Each arm full383. All335 current successes retained AND eight count strictly grows. Rejected entire Actor/Value/bothAdam rollback. No further rounds. Choose mosteight successes among admitted, tie A. No holdout selection.',
             reference_semantics='Deterministic8capture historical repeat for preservation, not onpolicy RL. Four fresh stochastic trajectories shared by initial-identical arms. Five old stochastic own GOAL episodes explicitly offpolicy SIL only, equal side/episode weights.',
             decision='Only one matched paired proposal, no seed/epoch/weight grid and no retry. Fresh16/reload78 not allocated, therefore no stability claim.',
             own_reference_rows=[],
             full_capture_max_files=12, compact_eval_max_files=766,
             metadata_reserved_bytes=8 * 2**20)
    files = set(old['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC / 'T479_PLAN.md'), str(DOC / 'T478_FINAL_REPORT.json'),
              str(previous / 'protocol.json'), str(previous / 'learn/audit.json'),
              str(ROOT / 'stage/docs/long_distance_20261101/release_receipt_T478.json'),
              str(ROOT / 'stage/experiments/t468_offline_recovery_night/t468_cache.py')}
    files |= {r['path'] for r in memory}
    p['protected_files'] = dict(old['protected_files'])
    p['protected_files'].update({base['model']['path']: base['model']['sha256']})
    files |= set(p['protected_files'])
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['constructor_reads'] = sorted(files)
    needed = max(1, M.used() + p['max_bytes'] + 2**20 - 2**30)
    cache = K.plan(files, needed)
    for group in cache['groups']:
        for key in ('receipt', 'manifest'):
            f = str(ROOT / group[key])
            p['inputs'][f] = sha(f)
    p['cache_total_reserved_bytes'] = p['max_bytes']
    p['cache_hard_stop_epoch'] = time.time() + 900
    p['remote_cache_path'] = str(HERE / 'cache_plan.json')
    (HERE / 'cache_plan.json').write_text(json.dumps(cache, ensure_ascii=False, separators=(',', ':')) + '\n')
    p['remote_cache_sha256'] = sha(HERE / 'cache_plan.json')
    worst = (2 * sum(j['case']['step_cap'] for j in p['capture_jobs'])
             + 2 * sum(j['case']['step_cap'] for j in p['training_jobs'])
             + 2 * sum(j['case']['step_cap'] for j in p['wide8_jobs']+p['six_jobs']+p['old275_jobs']))
    storage = 12*p['full_bytes'] + 766*p['compact_bytes'] + 2*p['model_bytes'] + p['metadata_reserved_bytes']
    assert worst <= p['max_physics'] and storage <= p['max_bytes']
    assert 24+766+8+6 <= p['max_commands']
    OUT.mkdir()
    save(HERE / 'protocol.json', p)
    save(OUT / 'protocol.json', p)
    save(DOC / 'T479_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED',
         max_physics=p['max_physics'], worst_physics=worst, max_commands=1000,
         worst_commands=804, max_live_Actor_Adam=32, max_live_Value_Adam=32,
         corresponding_shadow_Adam=64, independent_historical_audit_Adam=64,
         preflight_discarded_Adam=16, full12_compact766_models2_metadata_bound=storage,
         own_reserved_bytes=p['max_bytes'], global_limit=2**30,
         cache_required_net_release=needed, cache_verified_restore_markers=True,
         all_stop_epoch=stop, absolute_learning_stop=p['absolute_learning_stop'],
         absolute_native_stop=p['absolute_native_stop'], fresh16=None, reload78=None))
    print('T479_REGISTERED_NOT_LAUNCHED', flush=True)


def cache_protocol():
    p = read(OUT / 'protocol.json')
    assert sha(p['remote_cache_path']) == p['remote_cache_sha256']
    return dict(p, remote_cache=read(p['remote_cache_path']))


def cache():
    import t479_cache as K
    K.evict(cache_protocol())


def cache_recovery():
    import t479_cache as K
    K.evict(cache_protocol(), recovery=True)


def cache_audit():
    import t479_cache as K
    K.cacheaudit(cache_protocol())


def preflight():
    p = read(OUT / 'protocol.json')
    T.verify(p, True)
    assert read(OUT / 'cache_audit.json')['state'] == 'PASS_REFERENCE_RECEIPT_MARKERS_AND_PROTECTED_INPUTS'
    assert not (OUT / 'preflight.json').exists()
    # Engineering fixture only: eight distinct ids and full recurrent histories
    # exercise padding/weights. These are NOT actual eight-success capture.
    seedrows = p['sil_rows'][:2] + p['sil_rows'][2:]
    fixtures = []
    for i, name in enumerate(p['capture_success_ids']):
        source = next(r for r in seedrows if L.side(r['job_id']) == L.side(name))
        fixtures.append(dict(source, job_id=name))
    refs = L.references(p, fixtures)
    objects = {a: L.make(p) for a in ('A', 'B')}
    shadows = {a: L.make(p) for a in ('A', 'B')}
    memory = L.success_memory(p, next(objects['A'][0].parameters()).device)
    assert float(L.keep_loss(objects['A'][0], refs, 0)[0].detach()) < 1e-24
    arrays = [E.N.S.arrays(r['path']) for r in p['preflight_rows']]
    # Same initial policy as historical fixture is required by exact density.
    aa = L.fit_pair(objects, arrays, p, refs, memory, lambda: None, lambda *_: None, steps=2)
    bb = L.fit_pair(shadows, arrays, p, refs, memory, lambda: None, lambda *_: None, steps=2)
    assert aa == bb
    counts = aa['matched_epochs'] * 8
    assert counts <= 16
    serialized = []
    for arm in objects:
        assert digest(P.state(*objects[arm])) == digest(P.state(*shadows[arm]))
        buf = io.BytesIO()
        state = P.state(*objects[arm])
        torch.save(dict(joint_state=state, policy_hash=objects[arm][0].phash(),
                        joint_hash=digest(state), architecture='plain120_128_64GRU_2', single_actor=True), buf)
        blob = gzip.compress(buf.getvalue(), mtime=0)
        assert len(blob) <= p['model_bytes']
        assert digest(torch.load(io.BytesIO(gzip.decompress(blob)), map_location='cpu', weights_only=False)['joint_state']) == digest(state)
        serialized.append(len(blob))
        for obj in (objects[arm], shadows[arm]):
            P.load(obj, P.checkpoint(p['initial']['path'])['joint_state'])
            assert digest(P.state(*obj)) == p['initial']['joint_hash']
    # SIL detach: its actor loss must not create Value gradients.
    q, v, ao, vo = objects['B']
    ao.zero_grad(set_to_none=True); vo.zero_grad(set_to_none=True)
    al, vl, positive = L.sil_losses(q, v, memory, p)
    al.backward()
    assert all(z.grad is None for z in v.parameters())
    assert any(z.grad is not None and z.grad.abs().sum() > 0 for z in q.parameters())
    ao.zero_grad(set_to_none=True)
    folder = OUT / 'preflight_native'
    folder.mkdir()
    counts_shared = R.M.mp.get_context('spawn').RawArray('q', 3)
    pool = T.T.Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts_shared, lambda: None)
    try:
        reply = pool.request([dict(model=p['initial']['path'], job=j, seed=None, cancel_probe=True) for j in p['capture_jobs'][:2]])
        assert all(r['state'] == 'CANCEL_PASS' for r in reply) and sum(counts_shared) == 0
    finally:
        pool.close()
    save(OUT / 'preflight.json', dict(state='PASS', native=0, commands=6,
         discarded_historical_fixture_Adam=counts, maximum_discarded_Adam=16,
         fixture_eight_ids_NOT_physical_success_verification=True,
         actual_eight_capture_still_required=True, matched_fit=aa,
         serialized_bytes=serialized, Actor_Value_both_Adam_shadow_rollback_exact=True,
         SIL_actor_detached_from_Value=True, SIL_positive_fraction=positive))
    print('T479_PREFLIGHT_PASS_NOT_LAUNCHED', flush=True)


def launch():
    p = read(OUT / 'protocol.json')
    T.verify(p, True)
    assert read(OUT / 'preflight.json')['state'] == 'PASS'
    assert not (OUT / 'launch.json').exists()
    src = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    assert now + 7800 < p['absolute_all_stop']
    with (OUT / 'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block',
            '--who=Stage-T479', '--why=Finite matched success preservation and SIL',
            'timeout', '--signal=TERM', '--kill-after=2', '7800', 'taskset', '-c', '0',
            str(ROOT / 'run_stage.sh'), str(HERE / 'driver.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT / 'launch.json', dict(pid=c.pid, epoch=now, all_stop=now+7800,
         source_remote=src['remote_commit'], protocol_sha256=sha(OUT / 'protocol.json'),
         offline_no_LLM_network_install=True, independent_new_budget_not_T478_extension=True))
    print('T479_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT / f'error_control_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
