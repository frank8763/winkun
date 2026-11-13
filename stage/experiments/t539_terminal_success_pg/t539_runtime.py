"""One whole-episode success update -> original-start old8/field24 -> exact."""
from pathlib import Path
import copy,fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t539_terminal_success_pg';DOC=ROOT/'stage/docs/day_20261113'
sys.path.insert(0,str(HERE));import t539_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_runtime as Prior
F,G,E,P,np,torch=Prior.F,L.G,Prior.E,L.P,L.np,L.torch
read,save,sha,digest=L.read,L.save,L.sha,L.digest
def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert time.time()<root['absolute_all_stop'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
    assert used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
def configuration(p):
    c=read(p['actor_configuration']);c.update(p)
    c['training_rows']=G.rows(Path(p['training_cases']).parent)[:4]
    c['old_success_rows']=[r for r in c['keep_rows'] if not r['job_id'].startswith('t502_')]
    c['evaluation_jobs']=c['old8_jobs']+c['field_jobs']
    c['initial_field_success_ids']=c['baseline_field_success_ids']
    return c
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;P.frozen=L.A.frozen
def main():
    root=read(OUT/'protocol.json');p=configuration(root);F.R.setup(OUT);verify(p,True);E.N.offline();E.L.install_stop_handlers();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rt=G.B.Runtime(p);state='FAILED';error=None;obj=shadow=before=None;result=None
    try:
        obj=L.make(p);shadow=L.make(p);before=L.state(obj);items=L.arrays(p);rt.role='TERMINAL_SUCCESS_REINFORCE'
        aa=L.fit(obj,items,p,lambda:rt.guard(True),F.R.emit('live'));bb=L.fit(shadow,items,p,lambda:rt.guard(True),F.R.emit('shadow'));assert aa==bb and digest(L.state(obj))==digest(L.state(shadow))
        control=[E.N.S.arrays(r['path']) for r in p['old_success_rows']];trust=L.trust(p,obj[0],items+control)
        model=L.export(obj,p,OUT/'proposal.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash']
        result=dict(update=aa,trust=trust,model=model,Actor_SGD=1,Value_updates=0,new_policy_gradient_steps=1,old8=None,field24=None,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
        if trust['passed']:
            rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
            rows,_=Prior.evaluation(rt,p,model);Prior.evaluation(rt,p,model,rows);result.update(Prior.assess(p,rows))
        if not result['accepted']:
            L.load(obj,before);L.load(shadow,before)
        result['retained_state_sha256']=digest(L.state(obj));result['reason']='FULL_EPISODE_SUCCESS_NEW_FIELD_AND_ALL_PRIOR_IDS_RETAINED' if result['accepted'] else 'ONE_STEP_SUCCESS_PG_REJECTED_ROLLBACK';state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for o in [obj,shadow]:L.load(o,before)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error
def audit():
    root=read(OUT/'protocol.json');p=configuration(root);F.R.setup(OUT);verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT)
    obj=L.make(p);before=L.state(obj);items=L.arrays(p)
    # All behavioral arrays are existing immutable full episodes; read and
    # reconstruct the likelihood/hidden/reward with the identical source Actor.
    for row in p['training_rows']:
        audit_row=copy.deepcopy(row);assert row['policy_hash']==p['initial']['policy_hash'];audit_row['request']['model']=p['initial']['path']
        F.M.audit_full_history(p,audit_row,E.N.S.arrays(row['path']))
    u=L.fit(obj,items,p,lambda:None,lambda *_:None);assert u==s['result']['update'] and digest(L.state(obj))==s['result']['model']['joint_hash']
    assert sha(s['result']['model']['path'])==s['result']['model']['sha256']
    controls=[E.N.S.arrays(r['path']) for r in p['old_success_rows']];assert L.trust(p,obj[0],items+controls)==s['result']['trust']
    rows,reps=G.rows(OUT),G.rows(OUT,'replays.jsonl')
    if s['result']['trust']['passed']:
        assert len(rows)==len(reps)==32
        by={r['path']:r for r in rows}
        for r in rows:
            assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert set(a)=={'actions','native_counts','end_flags'}
            assert all(digest(v)==r['array_hashes'][k] for k,v in a.items()) and int(a['native_counts'].sum())==r['physics']
        for r in reps:assert r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256'] and r['exact']
        for k,v in Prior.assess(p,rows).items():assert v==s['result'][k]
    else:assert not rows and not reps
    if not s['result']['accepted']:L.load(obj,before)
    assert digest(L.state(obj))==s['result']['retained_state_sha256']
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and s['commands']==len(rows)+len(reps)+(4 if rows else 2)
    assert G.rows(OUT,'optimizer_calls.jsonl')==[[w,t,0] for w in ['live','shadow'] for t in ['BEFORE','AFTER']]
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_Actor_SGD=1,Actor_Adam=0,Value_Adam=0,existing4full_trajectories_history_density_and_success_score_rebuilt=True,Actor_SGD_state_and_rollback_exact=True,all_native_evaluation_full_array_hashes_and_integration_match_replays=True,compact_evaluation_NOT_full120_reconstructed_offline=True,physics=s['physics'],commands=s['commands']));print('T539_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
