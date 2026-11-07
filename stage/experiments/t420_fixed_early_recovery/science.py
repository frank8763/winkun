"""One fixed LEFT hold17 handoff and one exact native replay, no learning."""
from pathlib import Path
import sys,os,json,time,copy,hashlib,shutil,traceback,fcntl,multiprocessing as mp
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t420_fixed_early_recovery';DOC=ROOT/'stage/docs/day_20261107'
sys.path.insert(0,str(ROOT/'stage/experiments/t418_evening_repair_chain'))
import repair_core as E
torch,np=E.torch,E.np
read,save,sha,digest=E.read,E.save,E.sha,E.digest
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert E.N.K.S.used(OUT)<=p['final_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
def ban_optimization():
    def prohibited(*a,**k):raise AssertionError('T420 forbids optimization/backward')
    torch.optim.Adam.step=prohibited;torch.optim.AdamW.step=prohibited;torch.optim.SGD.step=prohibited;torch.Tensor.backward=prohibited
def prefix(a,p):
    previous=E.N.S.arrays(p['parent_history']['path']);assert sha(p['parent_history']['path'])==p['parent_history']['sha256']
    n=p['teacher_first'];assert len(a['states'])>n and int(a['native_counts'][:n].sum())==1700
    keys=['states','next_states','poses','next_poses','history_poses','history_goals','actions','raw_actions','locations','native_counts','rewards','end_flags','hidden_before','ownership','stochastic','innovations','atom','logp']
    keys += [key for key in ['hold_sha256','future_locations'] if key in a and key in previous]
    for key in keys:assert E.equal(a[key][:n],previous[key][:n]),('Changed actual student prefix',key)
    old_query=E.N.S.arrays(p['old_failed_query']['path']);assert sha(p['old_failed_query']['path'])==p['old_failed_query']['sha256']
    for key in ['states','teacher_actions','teacher_locations','teacher_hidden_before','teacher_hidden_after']:
        assert E.equal(a[key][:n+1],old_query[key][:n+1]),('Teacher reset before new handoff',key)
    assert np.all(a['ownership'][:n]==0) and np.all(a['ownership'][n:]==1)
    assert abs(float(a['states'][n,112])-(40000-1700)/40000)<2e-6
    return dict(state='PARENT_PREFIX17_ALL_RECORDED_ARRAYS_EXACT',keys=keys,holds=17,native=1700,seconds=3.4,no_clock_goal_history_reset=True,teacher_query_hidden_through_handoff_matches_existing_actual_prefix=True)
def main():
    E.N.offline();ban_optimization();p=read(OUT/'protocol.json');verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    E.set_out(OUT);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);base=E.N.science(p);q=E.safe_initial(base,p).eval().requires_grad_(False)
    assert q.phash()==p['student_policy_hash'] and sha(p['student_start_model'])==p['student_start_sha256'];fixed=digest(q.state_dict())
    start=time.time();deadline=min(start+p['main_seconds'],p['native_stop_absolute']);config=dict(p,native_stop_epoch=deadline)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start,deadline=deadline,Actor_updates=0,optimizer_steps=0))
    counts=mp.get_context('spawn').RawArray('q',5);pool=None;commands=0;state='FAILED';results=[];last=0.
    def guard(physical=False):
        nonlocal last
        E.L.check_stop();assert time.time()<deadline-5 and sum(counts)<=80000 and commands<=6
        if not physical and time.time()-last>3:
            last=time.time();verify(p);save(OUT/'status.json',dict(state='FIXED_HANDOFF_NATIVE',pid=os.getpid(),epoch=last,physics=sum(counts),main_commands=commands,Actor_updates=0,optimizer_steps=0))
    try:
        pool=E.Pool(config,counts,guard);commands=4
        for i,row in enumerate(pool.ready):
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=i+1,role='worker_setup',physics=0,pid=row['pid']))+'\n')
        first=None
        for role in ['handoff17','handoff17_exact']:
            guard();assert sum(counts)+p['job']['case']['step_cap']<=80000;commands+=1
            request=dict(model=p['student_start_model'],job=p['job'],first=17,seed=None,query=True,case=p['case'])
            values=pool.request([request]);assert len(values)==1;a,row=values[0]
            row=dict(row,role=role,job_id=p['job']['id'],physics_steps=int(a['native_counts'].sum()),loaded_model_sha256=sha(p['student_start_model']),seed=None)
            assert row['policy_hash']==p['student_policy_hash'] and row['teacher_policy_hash']==p['teacher_policy_hash']
            assert row['physics_steps']==row['physics'] and digest(a)==row['array_digest'];prefix(a,p)
            assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
            if first is None:
                path=OUT/'mixed.npz';E.N.write_npz(path,E.N.S.packed(a),p['query_full_bytes']);row.update(path=str(path),sha256=sha(path));first=(a,row)
                save(OUT/'case.json',row)
            else:
                assert a.keys()==first[0].keys()
                for key in a:assert E.equal(a[key],first[0][key]),('Native exact replay differs',key)
                for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==first[1][key],key
                row.update(exact=True,independent_sample=False,reference_sha256=first[1]['sha256']);save(OUT/'replay.json',row)
            results.append(row)
            with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(dict(id=commands,role=role,physics=row['physics'],pid=row['pid']))+'\n')
        assert digest(q.state_dict())==fixed;state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),physics=sum(counts),commands=commands,no_training_restart=True))
    finally:
        if pool:pool.close()
        result=dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,physics=sum(counts),main_commands=commands,preflight_zero_commands=8,all_commands=commands+8,Actor_updates=0,optimizer_steps=0,student_policy_hash=q.phash(),student_unchanged=digest(q.state_dict())==fixed,
            valid_recovery_label=bool(state=='COMPLETE' and first is not None and first[1]['success'] and not first[1]['collision'] and not first[1]['timeout']),result=results,no_six_metre_acceptance=True,seven_eight_entered=False)
        save(OUT/'summary.json',result);save(OUT/'status.json',dict(state=state,epoch=time.time(),physics=sum(counts),all_commands=commands+8,Actor_updates=0,optimizer_steps=0));E.N.manifest(OUT);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE';print('T420_FIXED_HANDOFF_COMPLETE',result['valid_recovery_label'],flush=True)
def audit():
    E.N.offline();ban_optimization();p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists() and not Path('/proc',str(s['pid'])).exists()
    for pid in read(OUT/'workers.json')['workers']:assert not Path('/proc',str(pid)).exists()
    for row in read(OUT/'local_manifest.json')['files']:assert sha(ROOT/row['path'])==row['sha256']
    start=time.time();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=E.safe_initial(E.N.science(p),p).eval().requires_grad_(False);fixed=digest(q.state_dict());a=E.N.S.arrays(OUT/'mixed.npz');row=read(OUT/'case.json');rep=read(OUT/'replay.json')
    assert sha(OUT/'mixed.npz')==row['sha256'] and digest(a)==row['array_digest'] and digest(a['final_native_integration'])==row['integration_sha256'];pr=prefix(a,p)
    E.audit_query(q,a,p['case'],p);assert digest(q.state_dict())==fixed and q.phash()==p['student_policy_hash']
    for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout']:assert row[key]==rep[key],key
    assert rep['exact'] and rep['independent_sample'] is False
    commands=[json.loads(line) for line in (OUT/'commands.jsonl').read_text().splitlines()]
    assert [r['id'] for r in commands]==list(range(1,7)) and len(commands)==s['main_commands']==6
    assert sum(r['physics'] for r in commands)==s['physics']==2*int(a['native_counts'].sum())<=80000 and s['all_commands']==14
    assert bool(a['end_flags'][-1,1])==row['success'] and bool(a['end_flags'][:,0].any())==row['collision'] and bool(a['end_flags'][-1,2])==row['timeout']
    valid=bool(row['success'] and not row['collision'] and not row['timeout']);assert valid==s['valid_recovery_label'] and s['student_unchanged']
    assert not list(OUT.rglob('partial*.npz')) and time.time()-start<p['audit_seconds']
    verify(p);save(OUT/'audit.json',dict(state='PASS',elapsed_s=time.time()-start,native=0,optimizer_steps=0,Actor_updates=0,all_source_and_data_SHA=True,full_student_and_teacher_actual_history_hidden_actions_rewards_exact=True,original_goal_clock_and_handoff_prefix=pr,all_live_and_exact_physics_commands_charged=True,physics=s['physics'],all_commands=14,valid_recovery_label=valid,no_six_metre_acceptance=True));E.N.manifest(OUT)
    print('T420_INDEPENDENT_AUDIT_PASS',valid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
