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
import t488_learning as L
sys.path.insert(0, str(ROOT / 'stage/experiments/t478_wide8_pooled_credit'))
import t478_runtime as T

B, P, E, M, R, torch, np = T.B, T.P, T.E, T.M, T.R, T.torch, T.np
read, save, sha, digest = T.read, T.save, T.sha, T.digest
OUT = ROOT / 'stage/runs/t488_left_failure_focus'
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


def diagnose(a,r):
    poses=a['poses'];nextposes=a['next_poses'];goal=a['task_goal_xy'];dt=a['native_counts'].astype(float)*.002
    distances=np.linalg.norm(nextposes[:,:2]-goal,axis=1)
    tail=np.cumsum(dt)>=np.sum(dt)-15
    vec=goal-nextposes[:,:2];error=np.arctan2(np.sin(np.arctan2(vec[:,1],vec[:,0])-nextposes[:,2]),np.cos(np.arctan2(vec[:,1],vec[:,0])-nextposes[:,2]))
    travel=np.linalg.norm(nextposes[:,:2]-poses[:,:2],axis=1)
    return dict(id=r['job_id'],success=r['success'],collision=r['collision'],timeout=r['timeout'],elapsed_s=float(dt.sum()),
        remaining_m=float(distances[-1]),min_recorded_distance_m=float(distances.min()),path_length_m=float(travel.sum()),
        last15_mean_speed_mps=float(travel[tail].sum()/dt[tail].sum()),last15_progress_m=float(distances[np.flatnonzero(tail)[0]]-distances[-1]),
        last15_abs_heading_error_median_deg=float(np.median(np.abs(error[tail]))*180/np.pi),
        last15_mean_drive=float(np.mean(a['actions'][tail,0])),last15_mean_abs_steer=float(np.mean(np.abs(a['actions'][tail,1]))),
        last15_steer_saturation_fraction=float(np.mean(np.abs(a['actions'][tail,1])>=.99)),
        native_original_full_digest_reconstructed=True,endpoint_metrics_not_continuous_native_minimum=True)

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
            diag=rt.request(p['diagnostic_jobs'],p['initial'],'FROZEN_DIAGNOSTIC',full=True)
            for a,r in diag:
                previous=p['diagnostic_original_rows'][r['job_id']]
                assert r['array_digest']==previous['array_digest'] and r['integration_sha256']==previous['integration_sha256']
            rt.request(p['diagnostic_jobs'],p['initial'],'EXACT_DIAGNOSTIC',full=True,references=[r for a,r in diag])
            train=rt.request(p['training_jobs'],p['initial'],'FRESH_EIGHT_ONLY',seeds=seeds,full=True)
            rt.request(p['training_jobs'],p['initial'],'EXACT_FRESH_EIGHT',seeds=seeds,full=True,references=[r for a,r in train])
            result=dict(samples=[r for a,r in train],diagnostics=[r for a,r in diag],diagnostic_metrics=[diagnose(a,r) for a,r in diag],no_old_or_six_native=True)
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
        assert s['result']['diagnostics']+s['result']['samples']==rr and len(rr)==len(replays)==8
        assert s['result']['diagnostic_metrics']==[diagnose(E.N.S.arrays(r['path']),r) for r in rr[:4]]
        for r in rr[:4]:
            prev=p['diagnostic_original_rows'][r['job_id']]
            assert r['array_digest']==prev['array_digest'] and r['integration_sha256']==prev['integration_sha256']
        obj=L.make(p)
        with torch.no_grad():L.C.batch(obj,[E.N.S.arrays(r['path']) for r in s['result']['samples']],p)
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
