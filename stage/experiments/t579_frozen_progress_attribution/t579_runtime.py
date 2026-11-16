"""Fixed six tasks, three immutable actors; full causal traces and exact replays."""
from pathlib import Path
import fcntl
import os
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t579_frozen_progress_attribution'
DOC = ROOT/'stage/docs/day_20261116/t579'
sys.path.insert(0, str(ROOT/'stage/experiments/t577_offline_recovery_night'))
import t577_runtime as R
L, E, F, G, Base, torch, np = R.L, R.E, R.F, R.G, R.Base, R.torch, R.np
read, save, sha, digest = R.read, R.save, R.sha, R.digest


def used(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p, full=False):
    root=read(OUT/'protocol.json')
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop']
    assert used(OUT)<=root['max_bytes']
    assert used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']+131072<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15}
    for path,h in (root['inputs'] if full else root['protected_files']).items():
        assert sha(path)==h,path


def setup():
    R.HERE=HERE
    R.verify=verify
    R.setup(OUT)
    Base.verify=verify


def describe(row,a):
    dt=a['native_counts'].astype(float)*.002
    np.testing.assert_allclose(np.diff(a['history_times'][:,-1]),dt,atol=2e-8,rtol=0)
    poses=np.vstack([a['poses'],a['next_poses'][-1]])
    d=np.linalg.norm(poses[:,:2]-a['task_goal_xy'],axis=1)
    assert abs(float(d[-1])-row['remaining_m'])<1e-8
    ds=np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1)
    speed=ds/dt
    radial=(d[:-1]-d[1:])/dt
    t=np.r_[0.,np.cumsum(dt)]
    angle=np.arctan2(a['task_goal_xy'][1]-a['poses'][:,1],a['task_goal_xy'][0]-a['poses'][:,0])
    heading=np.abs(np.arctan2(np.sin(angle-a['poses'][:,2]),np.cos(angle-a['poses'][:,2])))
    return dict(job_id=row['job_id'],policy_hash=row['policy_hash'],
        success=row['success'],contact=row['collision'],timeout=row['timeout'],
        censored=row['administrative_censor'],remaining_m=float(d[-1]),
        min_decision_distance_m=float(d.min()),closest_time_s=float(t[d.argmin()]),
        elapsed_native_seconds=float(t[-1]),initial_distance_m=float(d[0]),
        endpoint_path_m=float(ds.sum()),mean_endpoint_speed_m_s=float(ds.sum()/dt.sum()),
        net_radial_progress_m=float(d[0]-d[-1]),
        low_speed_time_fraction=float(dt[speed<.04].sum()/dt.sum()),
        moving_away_seconds=float(dt[radial<-.001].sum()),
        near_half_m_then_exit=bool(d.min()<=.5 and d[-1]>.5),
        mean_heading_abs_deg=float(np.average(heading,weights=dt)*180/np.pi),
        drive_command_time_mean=float(np.average(a['actions'][:,0],weights=dt)),
        steer_saturated_time_fraction=float(np.average(abs(a['actions'][:,1])>=.95,weights=dt)),
        sampling='decision endpoints, not continuous native arc/minimum',
        path=row['path'],sha256=row['sha256'])


def pairwise(p,rows):
    by={(r['model_sha256'],r['job_id']):r for r in rows};comparison=[]
    for job in p['jobs']:
        rr=[by[(m['sha256'],job['id'])] for m in p['models']]
        aa=[E.N.S.arrays(r['path']) for r in rr]
        for a in aa[1:]:
            for k in ['initial_native_integration','task_goal_xy','scene_sha256']:
                assert np.array_equal(a[k],aa[0][k]),(job['id'],k)
        deviations=[]
        for m,a in zip(p['models'][1:],aa[1:]):
            n=min(len(a['poses']),len(aa[0]['poses']))
            delta=a['poses'][:n]-aa[0]['poses'][:n]
            pos=np.linalg.norm(delta[:,:2],axis=1)
            yaw=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])))
            hits=np.flatnonzero((pos>=.03)|(yaw>=np.deg2rad(3)))
            k=int(hits[0]) if len(hits) else None
            deviations.append(dict(model=m['label'],first_3cm_or_3deg_decision=k,
                time_s=None if k is None else float(a['native_counts'][:k].sum()*.002),
                max_common_time_position_difference_m=float(pos.max()),
                not_causal_or_hidden_state_similarity_certificate=True))
        comparison.append(dict(job_id=job['id'],models=[dict(label=m['label'],**describe(r,a))
            for m,r,a in zip(p['models'],rr,aa)],deviation_from_T550=deviations))
    return comparison


def main():
    p=read(OUT/'protocol.json');setup();verify(p,True)
    E.N.offline();E.L.install_stop_handlers()
    rt=Base.Runtime(p);state='FAILED';error=None;result=None
    try:
        rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for model in p['models']:
            vals=rt.request(p['jobs'],model,'FROZEN_'+model['label'],full=True)
            rr=[r for a,r in vals]
            for a,r in vals:
                F.M.audit_full_history(p,r,a)
                previous=next((x for x in p['historical_rows'] if x['model_sha256']==model['sha256'] and x['job_id']==r['job_id']),None)
                if previous is not None:
                    assert r['array_hashes']==previous['array_hashes']
                    assert r['integration_sha256']==previous['integration_sha256']
                    Base.append('historical_overlap.jsonl',dict(job_id=r['job_id'],model=model['label'],
                        exact_all_arrays_and_integration=True,new_independent_sample=False))
            # Same trace is generated in a fresh episode; original full file is never overwritten.
            rt.request(p['jobs'],model,'EXACT_'+model['label'],full=True,reference=rr)
        assert len(rt.rows)==len(rt.replays)==18
        result=dict(comparison=pairwise(p,rt.rows),new_Adam=0,new_RL=0,
            models_unchanged=all(sha(m['path'])==m['sha256'] for m in p['models']),
            full407=None,fresh16=None,reload78=None,no_model_selection_or_promotion=True,
            selected_development_cases_NOT_generalization=True)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists()
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);replays=G.rows(OUT,'replays.jsonl')
    assert len(rows)==len(replays)==18
    by={r['path']:r for r in rows}
    for row in rows:
        a=E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256']
        assert int(a['native_counts'].sum())==row['physics']
        assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        assert not a['ownership'].any() and not a['stochastic'].any()
        F.M.audit_full_history(p,row,a)
    for r in replays:
        old=by[r['reference_path']]
        assert r['exact'] and r['array_hashes']==old['array_hashes']
        assert r['integration_sha256']==old['integration_sha256']
    assert pairwise(p,rows)==s['result']['comparison']
    assert s['physics']==sum(r['physics'] for r in rows+replays)<=p['max_physics']
    assert s['commands']==40<=p['max_commands']
    assert not (OUT/'optimizer_calls.jsonl').exists()
    verify(p,True)
    save(OUT/'audit.json',dict(state='PASS',new_native=0,new_Adam=0,
        full18_history_hidden_density_rewards_rebuilt=True,native18_exact=True,
        identical_tasks_initial_native_scene_goals=True,all_model_and_source_SHA_unchanged=True,
        physics=s['physics'],commands=s['commands'],no_generalization_claim=True))
    print('T579_AUDIT_PASS',flush=True)


if __name__=='__main__':globals()[sys.argv[1]]()
