"""Frozen current-student occupancy/teacher-label diagnosis; zero optimizers.

Every native trial is autonomous. Teachers are evaluated only after complete
student trajectories have been stored, and never act in this diagnostic.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t467_actual_occupancy_audit';DOC=ROOT/'stage/docs/review_20261110'
spec=importlib.util.spec_from_file_location('stage467_pinned465',ROOT/'stage/experiments/t465_matched_safe_recovery/run.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
J,H,M,P,E,R,F,Y,torch,np=V.J,V.H,V.M,V.P,V.E,V.R,V.F,V.Y,V.torch,V.np
read,save,sha,digest,frozen=V.read,V.save,V.sha,V.digest,V.frozen
JOB_IDS=['dev_task6_00','dev_task6_21','dev_task6_17','dev_task6_22','dev_task6_26','wide117_80_dev6_LEFT_13']


def verify(p,full=False):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert M.used(OUT)<=p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f


def worker(pipe,index,p,counts):
    P.frozen=frozen
    M.worker(pipe,index,p,counts)


class Pool(M.Pool):
    def __init__(self,p,counts,guard):
        self.guard,self.children,self.pipes=guard,[],[];ctx=R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close()
                self.children.append(c);self.pipes.append(a)
            self.ready=[self.receive(i) for i in range(8)];assert all(v['state']=='READY' for v in self.ready)
            f=Path(p['out'])/'worker_generations.json';generations=read(f) if f.exists() else []
            generations.append(dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time()));save(f,generations)
        except BaseException:self.close();raise


def setup_runtime():
    M.OUT,M.verify,M.Pool=OUT,verify,Pool;M.setup_runtime()


def prepare():
    assert not OUT.exists(),'Never restart a registered or ended directory'
    prior=ROOT/'stage/runs/t465_matched_safe_recovery';report=read(DOC/'T465_FINAL_REPORT.json')
    assert read(prior/'summary.json')['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS'
    assert report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc',str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc',str(read(prior/'driver_started.json')['pid'])).exists()
    assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    p=copy.deepcopy(read(prior/'protocol.json'));jobs={j['id']:j for j in p['main_jobs']+p['old275_jobs']}
    history=ROOT/'stage/runs/t463_evening_2000/ppo/cases.jsonl'
    original={r['job_id']:r for r in map(json.loads,history.read_text().splitlines()) if r['role']=='FINAL329'}
    assert sorted(k for k in JOB_IDS if not original[k]['success'])==['dev_task6_00','dev_task6_21']
    p.update(id='T467_ACTUAL_CURRENT_STUDENT_OCCUPANCY_DIAGNOSIS',out=str(OUT),
        diagnostic_jobs=[jobs[k] for k in JOB_IDS],diagnostic_reference={k:original[k] for k in JOB_IDS},
        max_bytes=10*2**20,max_physics=1000000,max_commands=64,preflight_commands=24,
        full_bytes=448*1024,compact_bytes=4096,main_seconds=600,learning_seconds=1,audit_seconds=300,
        backup_seconds=600,all_seconds=1500,maximum_unique_trials=6,maximum_exact_replays=6,
        model_bytes=0,Actor_Adam=0,Value_Adam=0,teacher_executed_native=0,
        prior_window_physics=report['window_physics'],prior_window_commands=report['window_commands'],
        actual_all_commands=52,registered_native_commands=28,
        onset_min_decision=20,onset_consecutive=3,onset_abs_steering_error=.02,
        onset_semantics='First causal detection index with3consecutive abs(action_steer minus fixedteacher)> .02, each index>=20. Detection index is a candidate future handover, NOT a validated recovery.',
        comparison='Same frozenPPO327 policy/job exact historical repeats, full actual student history now stored. Teacher label statistics only; not learning or independent navigation generalization.',
        user_authorization='Nov10 continue next step and prepare overnight; bounded zero-optimizer actual-state diagnosis under original remaining126commands and20:00stop.',
        eight_started=False,six_gate=False,fresh16=None,reload78=None,
        scientific_stop='Any contract failure preserves raw and stops; no teacher-control recovery, threshold/time/seed grid, policy update, or restart.')
    assert p['learning_initial']['policy_hash']==original[JOB_IDS[0]]['policy_hash']
    p['worst_physics']=2*sum(j['case']['step_cap'] for j in p['diagnostic_jobs'])
    assert p['worst_physics']<=p['max_physics'] and p['prior_window_physics']+p['worst_physics']<=p['window_total_physics_max']
    assert p['prior_window_commands']+52<=p['window_total_commands_max'] and p['window_total_commands_max']-p['prior_window_commands']==126
    assert time.time()+p['all_seconds']<p['absolute_all_stop'] and time.time()+p['main_seconds']<p['absolute_native_stop']
    plan=DOC/'T467_PLAN.md';assert plan.exists(),'Write exact bounded plan before registering'
    files=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(plan),str(history),str(prior/'summary.json'),str(prior/'audit.json'),str(prior/'protocol.json'),str(DOC/'T465_FINAL_REPORT.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T465.json'),str(ROOT/'stage/runs/t464_two_failure_recovery/summary.json')}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    assert M.used()+p['max_bytes']<=2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);verify(p,True)
    save(DOC/'T467_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',own_max_bytes=p['max_bytes'],global_full=M.used()-M.used(OUT)+p['max_bytes'],
        max_physics=p['max_physics'],worst_physics=p['worst_physics'],actual_all_commands=52,worker_main_commands=28,preflight_commands=24,
        prior_window_commands=p['prior_window_commands'],maximum_window_commands=p['prior_window_commands']+52,
        prior_window_physics=p['prior_window_physics'],maximum_window_physics=p['prior_window_physics']+p['worst_physics'],
        max_commands_internal_guard64_NOT_extra_requests=True,Actor_Adam=0,Value_Adam=0,teacher_native_actions=0,main_seconds=600,audit_seconds=300,all_seconds=1500,
        absolute_native_stop=p['absolute_native_stop'],absolute_all_stop=p['absolute_all_stop'],no_full329_or_fresh_or_reload_scheduled=True))
    print('T467_REGISTERED',flush=True)


def preflight():
    p=read(OUT/'protocol.json');verify(p,True);assert not (OUT/'preflight.json').exists();setup_runtime()
    counts=R.M.mp.get_context('spawn').RawArray('q',9);folder=OUT/'preflight_native';folder.mkdir()
    pool=Pool(dict(p,out=str(folder),native_stop_epoch=min(time.time()+180,p['absolute_native_stop']),max_physics=1),counts,lambda:None)
    try:
        replies=pool.request([dict(model=p['learning_initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']])
        assert all(v['state']=='CANCEL_PASS' for v in replies) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,optimizer=0,
        actual_frozen_model_cpu8workers_cancelled_before_first_native=True,
        six_historical_hash_contract_NOT_actual_reproduction_yet=True))
    print('T467_PREFLIGHT_PASS',flush=True)


def main():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p);status,error='FAILED',None;source_rows=[]
    try:
        rt.pool=Pool(dict(p,native_stop_epoch=min(rt.deadline,p['absolute_native_stop'])),rt.counts,rt.guard)
        values=rt.request(p['diagnostic_jobs'],p['learning_initial'],'CURRENT_STUDENT_OCCUPANCY6',full=True)
        F.check_parent_reproduction(values,p['diagnostic_reference']);source_rows=[r for a,r in values]
        assert all(r['full'] and not r['teacher'] and not r['administrative_censor'] for r in source_rows)
        assert [r['success'] for r in source_rows]==[False,False,True,True,True,True]
        rt.request(p['diagnostic_jobs'],p['learning_initial'],'CURRENT_STUDENT_OCCUPANCY6_EXACT',full=True,references=source_rows)
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True,optimizer=0))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        save(OUT/'summary.json',dict(state=status,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+p['preflight_commands'],error=error,
            source_rows=source_rows,same_policy_job_history_repeated_not_independent=True,optimizer=0,Actor_Adam=0,Value_Adam=0,teacher_executed_native=0,
            full_329=None,fresh16=None,reload78=None,six_gate=False,eight_started=False,all_native_workers_stopped=True))
        save(OUT/'status.json',dict(state=status,epoch=time.time()));R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    print('T467_MAIN_COMPLETE',flush=True)


def label_statistics(p,rows):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    safe=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False)
    carry=frozen(p,p['recovery_teacher_model']['path']).eval().requires_grad_(False)
    results=[];raw={}
    with torch.no_grad():
        for r in rows:
            if r['job_id'] not in ['dev_task6_00','dev_task6_21']:continue
            a=E.N.S.arrays(r['path']);labels=[];hidden=None
            for x in a['states']:
                xx=torch.tensor(x[None])
                if r['job_id']=='dev_task6_00':command=safe.base_actions(xx.to(torch.float32)).float()
                else:command,_,hidden,_=carry.step(xx,hidden)
                labels.append(command[0].numpy().copy())
            labels=np.asarray(labels,np.float32);actions=a['actions'];assert labels.shape==actions.shape and np.isfinite(labels).all()
            error=np.abs(actions.astype(np.float64)-labels.astype(np.float64));start=p['onset_min_decision'];n=p['onset_consecutive'];threshold=p['onset_abs_steering_error']
            detected=None
            for i in range(start+n-1,len(error)):
                if np.all(error[i-n+1:i+1,1]>threshold):detected=i;break
            regions={}
            for name,part in [('prefix_first21',slice(0,21)),('suffix21onward',slice(21,None)),('all',slice(None))]:
                e=error[part]
                regions[name]=dict(decisions=len(e),RMSE_per_axis=np.sqrt((e**2).mean(0)).tolist(),max_abs_per_axis=e.max(0).tolist(),mean_abs_per_axis=e.mean(0).tolist(),
                    steering_gt02_count=int((e[:,1]>.02).sum()),steering_gt05_count=int((e[:,1]>.05).sum()))
            key=r['job_id'];raw[key+'_teacher_actions']=labels;raw[key+'_action_abs_error']=error.astype(np.float32)
            raw[key+'_source_states_sha256']=np.frombuffer(bytes.fromhex(r['array_hashes']['states']),np.uint8).copy()
            results.append(dict(job_id=key,source_path=r['path'],source_sha256=r['sha256'],source_array_digest=r['array_digest'],actual_student_success=r['success'],actual_student_collision=r['collision'],
                teacher='safeT255reset' if key.endswith('_00') else 'T461_actual_history_carry',teacher_only_offline_never_executed=True,
                regions=regions,onset_candidate_detection_decision=detected,onset_candidate_first_streak_decision=None if detected is None else detected-n+1,
                onset_candidate_time_s=None if detected is None else detected*.2,
                onset_rule=dict(min_decision=start,consecutive=n,abs_steering_error_threshold=threshold),
                candidate_NOT_physically_validated_recovery=True,statistic_NOT_causal_or_safety_proof=True))
    assert len(results)==2
    return results,raw


def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    ids=[s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()));replays=list(map(json.loads,(OUT/'replays.jsonl').read_text().splitlines()))
    assert len(rows)==len(replays)==6 and {r['job_id'] for r in rows}==set(JOB_IDS)
    assert s['physics']==sum(r['physics'] for r in rows+replays)<=p['max_physics'] and s['commands']==24+16+len(rows)+len(replays)==52
    assert p['prior_window_commands']+s['commands']<=p['window_total_commands_max'] and p['prior_window_physics']+s['physics']<=p['window_total_physics_max']
    reference=p['diagnostic_reference'];bypath={r['path']:r for r in rows}
    for r in rows:
        assert sha(r['path'])==r['sha256'] and r['policy_hash']==p['learning_initial']['policy_hash'] and r['loaded_model_sha256']==p['learning_initial']['sha256']
        a=E.N.S.arrays(r['path']);assert digest(a)==r['array_digest'] and {k:digest(v) for k,v in a.items()}==r['array_hashes']
        assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and not a['ownership'].any() and not a['stochastic'].any()
        assert int(a['native_counts'].sum())==r['physics'] and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
        assert r['array_digest']==reference[r['job_id']]['array_digest'] and r['array_hashes']==reference[r['job_id']]['array_hashes'] and r['integration_sha256']==reference[r['job_id']]['integration_sha256']
        P.frozen=frozen
        try:M.audit_full_history(p,r,a)
        finally:P.frozen=H.ORIGINAL_FROZEN
    for r in replays:
        original=bypath[r['reference_path']];assert r['array_digest']==original['array_digest'] and r['array_hashes']==original['array_hashes'] and r['integration_sha256']==original['integration_sha256']
    statistics,arrays=label_statistics(p,rows)
    blob=E.N.R.blob(arrays);assert len(blob)<=128*1024
    path=OUT/'offline_teacher_labels.npz';assert not path.exists();path.write_bytes(blob)
    # Independent CPU second pass checks all labels, statistics and serialized bytes.
    again,other=label_statistics(p,rows);assert again==statistics and all(np.array_equal(arrays[k],other[k]) for k in arrays)
    assert E.N.R.blob(other)==blob
    report=dict(state='PASS_FROZEN_ACTUAL_OCCUPANCY_DIAGNOSIS_ONLY',source_rows=rows,labels=statistics,
        labels_npz=dict(path=str(path),bytes=len(blob),sha256=sha(path)),optimizer=0,new_teacher_native=0,
        original_full6_historical_repeats=True,six_exact_replays_NOT_independent_new_data=True,
        teacher_labels_on_actual_student_histories_not_teacher_occupancy=True,
        no_physical_recovery_validated=True,no_candidate_model_trained=True,fresh16=None,reload78=None,six_gate=False,eight_started=False)
    save(OUT/'occupancy_diagnosis.json',report)
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,optimizer=0,backward=0,
        all_full120_history_hidden_density_rewards_and_integration_exact=True,
        every_native_trial_and_replay_matches_frozen_original_policy_job=True,
        offline_teacher_labels_independently_exact_twice=True,source_model_SHA=sha(p['learning_initial']['path']),
        physics=s['physics'],commands=s['commands'],new_unique_policy_job_data=0,
        onset_candidates=[dict(job_id=r['job_id'],decision=r['onset_candidate_detection_decision']) for r in statistics],
        onset_candidates_NOT_recovery_or_navigation_success=True,six_gate=False,eight_started=False))
    print('T467_AUDIT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p,True);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+p['all_seconds']<p['absolute_all_stop'] and now+p['main_seconds']<p['absolute_native_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T467','--why=Frozen current occupancy diagnosis',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T467_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
