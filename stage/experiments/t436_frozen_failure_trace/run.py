"""Stage-only bounded frozen-policy differential trace; no optimization."""
from pathlib import Path
import sys, json, time, hashlib, shutil, os, copy, subprocess, fcntl, traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t436_frozen_failure_trace'
DOC=ROOT/'stage/docs/review_20261109'
sys.path.insert(0,str(ROOT/'stage/experiments/t432_offline_night'))
import t432_core as C
E,np,torch=C.E,C.np,C.torch
sha,read,save=C.sha,C.read,C.save
IDS=['wide117_80_dev6_LEFT_00','wide117_80_dev6_LEFT_07','dev_task6_09','dev_task6_24']

def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]

def used():
    return E.N.K.S.used(ROOT/'stage/runs')

def verify(p):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<p['all_stop']
    assert used()<=2**30 and E.N.K.S.used(OUT)<=p['max_bytes']
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f

def prepare():
    assert not OUT.exists()
    old=read(ROOT/'stage/runs/t432_offline_night/protocol.json')
    sources=[ROOT/'stage/runs/t427_fixed_representation/learn',ROOT/'stage/runs/t432_offline_night/c']
    models={k:read(v/'summary.json')['result']['proposed_model'] for k,v in zip(['source','C'],sources)}
    rows={k:{r['job_id']:r for r in jsonl(v/'cases.jsonl')} for k,v in zip(['source','C'],sources)}
    extra={r['job_id']:r for r in jsonl(ROOT/'stage/runs/t432_offline_night/data/cases.jsonl')}
    existing={j:(extra[j] if j=='dev_task6_09' else rows['source'][j]) for j in IDS[:3]}
    jobs={j['id']:j for j in old['main_jobs']+old['old275_jobs']}
    requests=[dict(label='C',job=jobs[j],model=models['C']['path'],mode='student',seed=None) for j in IDS]
    requests.append(dict(label='source',job=jobs[IDS[-1]],model=models['source']['path'],mode='student',seed=None))
    assert 2*sum(x['job']['case']['step_cap'] for x in requests)==300004
    inputs=dict(old['inputs'])
    for f in [*HERE.glob('*.py'),DOC/'T436_PLAN.md',ROOT/'stage/runs/t432_offline_night/protocol.json',*[v/'cases.jsonl' for v in sources],ROOT/'stage/runs/t432_offline_night/data/cases.jsonl',*[Path(m['path']) for m in models.values()]]:
        inputs[str(f)]=sha(f)
    for row in existing.values():inputs[row['path']]=row['sha256']
    now=time.time()
    p=dict(id='T436_FROZEN_FAILURE_TRACE',authorization='2026-11-09 user entrusted Stage review and bounded experiment advancement; latest continue. New diagnostic only, no revival of expired night budget.',
        source_protocol=str(ROOT/'stage/runs/t432_offline_night/protocol.json'),models=models,
        requests=requests,existing=existing,expected={k:{j:v[j] for j in IDS} for k,v in rows.items()},
        inputs=inputs,max_bytes=3*2**20,max_physics=300004,max_commands=14,
        full_bytes=384*1024,main_seconds=600,audit_seconds=300,all_stop=now+1500,
        new_optimizer=0,new_teacher=0,new_RL=0,unique_policy_task_pairs=8,
        newly_regenerated_historical_pairs=5,exact_repeats=5,not_independent_new_navigation_samples=True,
        stable_six_gate=False,eight_started=False,source_C_rejected_diagnostic_only=True)
    assert used()+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    verify(p)
    save(OUT/'preflight.json',dict(state='PASS_STATIC_AND_PINNED_RUNTIME_CONTRACT',
        native=0,Actor_forward=0,optimizer=0,all_inputs_hash_verified=True,
        worker_runtime='unchanged T432 worker, source bytes pinned',full_worst_bytes=used()+p['max_bytes'],
        no_claim_of_new_numeric_training_contract=True))
    print('T436_REGISTERED_ZERO_NATIVE_PREFLIGHT_PASS',flush=True)

def historical(row,reference):
    for key in ['job_id','array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','remaining_m','policy_hash','loaded_model_sha256']:
        assert row[key]==reference[key],(key,row['job_id'])

def main():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'main_started.json').exists()
    C.ban_learning();E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.L.install_stop_handlers()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();deadline=min(start+p['main_seconds'],p['all_stop']-p['audit_seconds']-300)
    counts=C.mp.get_context('spawn').RawArray('q',5);pool=None;rows=[];replays=[];commands=0;state='FAILED';error=None
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline))
    def guard():
        E.L.check_stop();assert time.time()<deadline-5 and sum(counts)<=p['max_physics'] and commands<=p['max_commands']
        verify(p);save(OUT/'status.json',dict(state='FROZEN_TRACE',pid=os.getpid(),epoch=time.time(),physics=sum(counts),commands=commands,full_traces=len(rows),exact_replays=len(replays)))
    try:
        cfg=read(p['source_protocol']);cfg.update(out=str(OUT),native_stop_epoch=deadline,max_physics=p['max_physics'])
        E.set_out(OUT);pool=C.Pool(cfg,counts,guard);commands=4
        batches=[p['requests'][:4],p['requests'][4:]];values=[]
        for requests in batches:
            guard();commands+=len(requests)
            for req,(a,r) in zip(requests,pool.request(requests)):
                row=dict(r,job_id=req['job']['id'],role='FROZEN_'+req['label'],label=req['label'],request=req,seed=None,
                    full=True,physics_steps=int(a['native_counts'].sum()),loaded_model_sha256=sha(req['model']))
                historical(row,p['expected'][req['label']][row['job_id']])
                path=OUT/'episodes'/f'{len(rows):02}.npz';path.parent.mkdir(exist_ok=True)
                E.N.write_npz(path,E.N.S.packed(a),p['full_bytes']);row.update(path=str(path),sha256=sha(path));rows.append(row);values.append((a,row))
                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        for offset,requests in [(0,p['requests'][:4]),(4,p['requests'][4:])]:
            guard();commands+=len(requests)
            for k,(a,r) in enumerate(pool.request(requests)):
                old,row=values[offset+k];assert set(a)==set(old)
                for key in a:assert E.equal(a[key],old[key]),key
                assert r['array_digest']==row['array_digest'] and r['integration_sha256']==row['integration_sha256']
                replay=dict(reference_path=row['path'],reference_sha256=row['sha256'],job_id=row['job_id'],label=row['label'],
                    physics=r['physics'],array_digest=r['array_digest'],integration_sha256=r['integration_sha256'],all_arrays_exact=True,independent_sample=False)
                replays.append(replay)
                with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(replay)+'\n')
        assert commands==14 and len(rows)==len(replays)==5
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,partial_preserved=True,no_restart=True))
    finally:
        if pool:pool.close()
        save(OUT/'summary.json',dict(state=state,error=error,elapsed_s=time.time()-start,physics=sum(counts),commands=commands,
            full_traces=len(rows),exact_replays=len(replays),new_optimizer=0,new_RL=0,
            all_workers_stopped=not any(c.is_alive() for c in pool.children) if pool else True,stable_six_gate=False))
        save(OUT/'status.json',dict(state=state,epoch=time.time(),physics=sum(counts),commands=commands))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE',error
    print('T436_FROZEN_TRACE_COMPLETE',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p);assert not (OUT/'audit.json').exists()
    start=time.time();assert not Path('/proc',str(read(OUT/'main_started.json')['pid'])).exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in read(OUT/'workers.json')['workers'])
    C.ban_learning();E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    import importlib.util
    spec=importlib.util.spec_from_file_location('t436_pinned_policy_audit',ROOT/'stage/experiments/t427_fixed_representation/audit.py')
    H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and s['new_optimizer']==0
    rows=jsonl(OUT/'cases.jsonl');replays=jsonl(OUT/'replays.jsonl');arrays={}
    base=read(p['source_protocol'])
    for row in list(p['existing'].values())+rows:
        assert time.time()<min(start+p['audit_seconds'],p['all_stop'])
        assert sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path'])
        assert C.digest(a)==row['array_digest'] and {k:C.digest(v) for k,v in a.items()}==row['array_hashes']
        H.full_policy(C,a,row,base)
        assert not a['ownership'].any() and not a['stochastic'].any()
        label=row.get('label','source');arrays[label,row['job_id']]=a
    assert len(arrays)==8 and len(rows)==len(replays)==5
    assert s['physics']==sum(r['physics'] for r in rows)+sum(r['physics'] for r in replays)<=p['max_physics']
    assert s['commands']==14
    facts=[]
    for identity in IDS:
        a,b=arrays['source',identity],arrays['C',identity]
        for key in ['initial_native_integration','task_goal_xy','scene_sha256']:
            assert E.equal(a[key],b[key]),(identity,key)
        n=min(len(a['actions']),len(b['actions']));assert np.array_equal(a['native_counts'][:n-1],b['native_counts'][:n-1])
        delta=np.max(np.abs(a['actions'][:n]-b['actions'][:n]),axis=1)
        xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
        yaw=np.abs((a['poses'][:n,2]-b['poses'][:n,2]+np.pi)%(2*np.pi)-np.pi)
        clock=np.r_[0,np.cumsum(a['native_counts'])][:-1]*.002
        first=lambda condition:float(clock[np.flatnonzero(condition)[0]]) if condition.any() else None
        facts.append(dict(job_id=identity,common_decisions=n,
            action_first_nonzero_s=first(delta>0),action_first_gt_1e_3_s=first(delta>1e-3),
            first_xy_gt_1cm_s=first(xy>.01),first_xy_gt_5cm_s=first(xy>.05),first_yaw_gt_5deg_s=first(yaw>np.deg2rad(5)),
            max_action_difference=float(delta.max()),max_xy_difference_m=float(xy.max()),
            rms_action_difference_common_states_NOT_measured=True,
            source_end_flags=a['end_flags'][-1].tolist(),C_end_flags=b['end_flags'][-1].tolist(),
            source_end_goal_distance_m=float(np.linalg.norm(a['next_poses'][-1,:2]-a['task_goal_xy'])),
            C_end_goal_distance_m=float(np.linalg.norm(b['next_poses'][-1,:2]-b['task_goal_xy']))))
    save(OUT/'audit.json',dict(state='PASS',native=0,optimizer=0,all8_policy_history_density_reward_checked=True,
        all5_regenerated_historical_array_hashes_exact=True,all5_native_replays_exact=True,
        physics=s['physics'],commands=14,models_unmodified=True,source_C_not_promoted=True,
        evidence_limits='Different visited-state actions conflate parameter and feedback drift; first divergence is descriptive, not a causal isolation or clearance proof.',facts=facts))
    print(json.dumps(dict(state='T436_AUDIT_PASS',physics=s['physics'],facts=facts)),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
