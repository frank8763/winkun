"""One native writer: capture eight retained successes, then matched A/B."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t489_learning as L
sys.path.insert(0, str(ROOT / 'stage/experiments/t478_wide8_pooled_credit'))
import t478_runtime as T

B, P, E, M, R, torch, np = T.B, T.P, T.E, T.M, T.R, T.torch, T.np
read, save, sha, digest = T.read, T.save, T.sha, T.digest
OUT = ROOT / 'stage/runs/t489_left_budget_recovery'
ACTIVE = OUT


def verify(p, full=False):
    root = read(OUT / 'protocol.json')
    assert sha(HERE / 'protocol.json') == sha(OUT / 'protocol.json')
    assert M.used(OUT) <= root['max_bytes']
    assert M.used() - M.used(OUT) + root['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40 * 2**30
    for f, h in (root['inputs'] if full else root['protected_files']).items():
        assert sha(f) == h, f


def setup(folder):
    global ACTIVE
    ACTIVE = Path(folder)
    B.OUT, B.ACTIVE, B.verify = OUT, ACTIVE, verify
    M.OUT, M.verify = ACTIVE, verify
    M.setup_runtime()


def rows(folder, name='cases.jsonl'):
    f = Path(folder) / name
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def stopped(folder):
    ids = [read(folder / 'summary.json')['pid']]
    for g in read(folder / 'worker_generations.json'):
        ids += [g['parent']] + g['workers']
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    return ids


def assess(values,model,p):
    rr=[r for a,r in values]
    assert len(rr)==54 and len({r['job_id'] for r in rr})==54
    counts=R.M.tally(rr);success=set(T.safe(rr))
    return dict(model=model,wide8=counts,success_ids=sorted(success),lost_ids=sorted(set(p['initial_success_ids'])-success),
        accepted=counts['success']>p['initial_result']['wide8']['success'] and counts['contact']<=p['initial_result']['wide8']['contact'],
        old275=None,wide6=None,eight_only_research_NOT_joint_retention=True)

def event(which):
    def emit(arm, stage, k):
        with (ACTIVE / 'optimizer_calls.jsonl').open('a') as f:
            f.write(json.dumps([which, arm, stage, k]) + '\n')
    return emit


def main(mode):
    folder = OUT / mode
    p = read(folder / 'protocol.json')
    setup(folder)
    verify(p, True)
    E.N.offline()
    E.L.install_stop_handlers()
    rt = B.Runtime(p)
    state, error, result = 'FAILED', None, {}
    objects = shadows = None
    try:
        rt.pool = T.Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        if mode == 'capture':
            seeds = [[p['seed'], i] for i in range(4)]
            train=rt.request(p['training_jobs'],p['initial'],'FRESH_EIGHT_ONLY',seeds=seeds,full=True)
            rt.request(p['training_jobs'],p['initial'],'EXACT_FRESH_EIGHT',seeds=seeds,full=True,references=[r for a,r in train])
            result=dict(samples=[r for a,r in train],diagnostics=p['prior_diagnostic_rows'],no_old_or_six_native=True,diagnostics_reused_NOT_new_independent=True)
        else:
            assert mode=='pair' and read(OUT/'capture/audit.json')['state']=='PASS'
            cap=read(OUT/'capture/summary.json')['result'];arrays=[E.N.S.arrays(r['path']) for r in cap['samples']]
            objects={'B':L.make(p)};shadows={'B':L.make(p)}
            update=L.fit(objects['B'],arrays,p,lambda:rt.guard(True),event('live'))
            duplicate=L.fit(shadows['B'],arrays,p,lambda:rt.guard(True),event('shadow'))
            assert update==duplicate and digest(P.state(*objects['B']))==digest(P.state(*shadows['B']))
            proposal=P.export(objects['B'],folder/'proposal_B.pt.gz',p['model_bytes'])
            vv=rt.request(p['wide8_jobs'],proposal,'FULL54_EIGHT_ONLY');candidate=assess(vv,proposal,p)
            if not candidate['accepted']:
                for obj in [objects['B'],shadows['B']]:P.load(obj,P.checkpoint(p['initial']['path'])['joint_state'])
            result=dict(update=update,candidates={'B':candidate},chosen_arm='B' if candidate['accepted'] else None,
                selected=candidate if candidate['accepted'] else p['initial_result'],
                joint_hash_after_rollback={'B':digest(P.state(*objects['B']))},old275=None,wide6=None)
        state = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        rollback = {}
        if objects is not None:
            before = P.checkpoint(p['initial']['path'])['joint_state']
            for label, group in [('live', objects), ('shadow', shadows or {})]:
                for arm, obj in group.items():
                    P.load(obj, before)
                    rollback[label + arm] = digest(P.state(*obj))
                    assert rollback[label + arm] == p['initial']['joint_hash']
        save(folder / 'failure.json', dict(error=error, raw_preserved=True, no_restart=True,
             interrupted_entire_Actor_Value_both_Adam_rollback=rollback))
    finally:
        if rt.pool:
            rt.pool.close()
            rt.commands += 2
        save(folder / 'summary.json', dict(state=state, pid=os.getpid(), mode=mode,
             error=error, physics=sum(rt.counts), commands=rt.commands,
             elapsed_s=time.time() - rt.start, result=result,
             fresh16=None, reload78=None, eight_stability_gate=False))
        fcntl.flock(rt.lock, fcntl.LOCK_UN)
        rt.lock.close()
    assert state == 'COMPLETE', error


def audit(mode):
    folder = OUT / mode
    p = read(folder / 'protocol.json')
    setup(folder)
    verify(p, True)
    s = read(folder / 'summary.json')
    assert s['state'] == 'COMPLETE' and not (folder / 'audit.json').exists()
    stopped(folder)
    rr, replays = rows(folder), rows(folder, 'replays.jsonl')
    bypath = {r['path']: r for r in rr}
    for r in rr:
        assert sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert all(digest(v) == r['array_hashes'][k] for k, v in a.items())
        assert int(a['native_counts'].sum()) == r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert a['end_flags'][-1].tolist() == [int(r['collision']), int(r['success']), int(r['timeout'])]
        if r['full']:
            M.audit_full_history(p, r, a)
    for r in replays:
        ref = bypath[r['reference_path']]
        assert r['exact'] and r['array_digest'] == ref['array_digest']
        assert r['array_hashes'] == ref['array_hashes'] and r['integration_sha256'] == ref['integration_sha256']
    discarded = 0
    if mode=='capture':
        assert s['result']['samples']==rr and len(rr)==len(replays)==4
        obj=L.make(p)
        with torch.no_grad():L.C.batch(obj,[E.N.S.arrays(r['path']) for r in rr],p)
    else:
        cap=read(OUT/'capture/summary.json')['result'];obj=L.make(p)
        rebuilt=L.fit(obj,[E.N.S.arrays(r['path']) for r in cap['samples']],p,lambda:None,lambda *_:None)
        assert rebuilt==s['result']['update']
        c=s['result']['candidates']['B'];assert digest(P.state(*obj))==c['model']['joint_hash']
        assert sha(c['model']['path'])==c['model']['sha256']
        assert assess([(None,r) for r in rr],c['model'],p)==c
        if not c['accepted']:P.load(obj,P.checkpoint(p['initial']['path'])['joint_state'])
        assert digest(P.state(*obj))==s['result']['joint_hash_after_rollback']['B']
        assert s['result']['chosen_arm']==('B' if c['accepted'] else None)
        discarded=2*rebuilt['epochs']
    assert s['physics'] == sum(r['physics'] for r in rr + replays)
    assert s['commands'] == len(rr) + len(replays) + 4
    save(folder / 'audit.json', dict(state='PASS', pid=os.getpid(), native=0,
         historical_discarded_Actor_and_Value_Adam=discarded,
         physics=s['physics'], commands=s['commands'],
         full_training_history_density_hidden_reward_and_joint_Adam_exact=True,
         compact_evaluation_NOT_full120_reconstructed=True,
         eight_success_count_and_contact_acceptance_rollback_verified=True))


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]](*sys.argv[2:])
    except BaseException:
        if OUT.exists():
            save(OUT / f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
