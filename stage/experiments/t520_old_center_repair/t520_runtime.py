"""Fresh full-Actor RL; acceptance on real native navigation, not action proxy."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t520_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t520_old_center_repair'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify
def event(which):return F.R.emit(which)

def evaluation(rt,p,model,reference=None):
    """Full arrays generated twice; compact evidence saved, all full hashes match.

    Explicit disk contract: evaluator full states are NOT saved/rebuilt offline.
    Native replay equality checks every full array/integration before discard.
    """
    rows=[];details=[];jobs=p['evaluation_jobs'];rt.role='EXACT_FULL32' if reference else 'AUTONOMOUS_FULL32'
    for off in range(0,len(jobs),2):
        batch=jobs[off:off+2];rt.guard()
        assert sum(rt.counts)+sum(j['case']['step_cap'] for j in batch)<=p['max_physics'] and rt.commands+len(batch)+2<=p['max_commands']
        rt.commands+=len(batch)
        requests=[dict(model=model['path'],job=j,seed=None,query=False,full_path=None) for j in batch]
        for i,((a,r),j,req) in enumerate(zip(rt.pool.request(requests),batch,requests)):
            assert r['policy_hash']==model['policy_hash'] and int(a['native_counts'].sum())==r['physics']
            r.update(job_id=j['id'],role=rt.role,seed=None,request=req,loaded_model_sha256=sha(model['path']),full=False,full_arrays_generated_but_not_saved=True)
            detail=F.detail(p,r,a) if j['id'].startswith('t502_') else None
            r['field_detail']=detail
            if reference:
                old=reference[off+i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail']
                assert [r[k] for k in ['success','collision','timeout','remaining_m']]==[old[k] for k in ['success','collision','timeout','remaining_m']]
                r.update(exact=True,reference_path=old['path'],independent_new_sample=False);rt.replays.append(r);filename='replays.jsonl'
            else:
                path=Path(p['out'])/'episodes'/('eval_'+str(off+i)+'.npz');path.parent.mkdir(exist_ok=True)
                E.N.write_npz(path,{k:a[k] for k in ['actions','native_counts','end_flags']},p['compact_bytes'])
                r.update(path=str(path),sha256=sha(path));rt.rows.append(r);filename='cases.jsonl'
            with (Path(p['out'])/filename).open('a') as f:f.write(json.dumps(r)+'\n')
            rows.append(r);details.append(detail)
    return rows,details

def assess(p,rows):
    assert len(rows)==32;old=G.R.M.tally(rows[:8]);field=G.R.M.tally(rows[8:])
    ids={r['job_id'] for r in rows[8:] if r['field_detail']['field_qualified_success']}
    ok=old['success']==8 and old['invalid']==old['censored']==0 and set(p['initial_field_success_ids'])<=ids and len(ids)>=12 and field['contact']<=9 and field['invalid']==field['censored']==0
    return dict(old8=old,field24=field,field_qualified_goals=len(ids),field_success_ids=sorted(ids),gained_ids=sorted(ids-set(p['initial_field_success_ids'])),lost_ids=sorted(set(p['initial_field_success_ids'])-ids),accepted=ok)

def main(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['training_jobs'],p['initial'],'FOUR_FRESH_FULL_ACTOR_ONPOLICY',seeds=p['training_seeds'],full=True)
        samples=[r for a,r in vv];rt.request(p['training_jobs'],p['initial'],'EXACT_FOUR_FRESH_ONPOLICY',seeds=p['training_seeds'],full=True,references=samples)
        eligible=any(F.detail(p,r,a)['field_qualified_success'] for a,r in vv)
        result=dict(training=G.R.M.tally(samples),learning_eligible=eligible,training_qualified_successes=sum(F.detail(p,r,a)['field_qualified_success'] for a,r in vv),Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,
                    old8=None,field24=None,accepted=False,full_old275_six54_eight54_NOT_executed=True,fresh16=None,reload78=None,under8_gate=False)
        if eligible:
            obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p)
            aa=L.fit(obj,[a for a,r in vv],p,refs,lambda:rt.guard(True),event('live'));bb=L.fit(shadow,[a for a,r in vv],p,refs,lambda:rt.guard(True),event('shadow'))
            assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
            model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
            rows,details=evaluation(rt,p,model);evaluation(rt,p,model,rows)
            result.update(assess(p,rows),model=model,update=aa,Actor_Adam=aa['Actor_Adam'],new_policy_gradient_updates=aa['Actor_Adam'])
            save(folder/'diagnosis.json',details[8:])
            if not result['accepted']:
                for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
            result['final_joint_hash']=digest(P.state(*obj))
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for o in [obj,shadow]:P.load(o,before)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=L.A.frozen;rows,reps=G.rows(folder),G.rows(folder,'replays.jsonl');result=s['result'];train=rows[:4];arrays=[]
    for r in train:
        assert r['full'] and sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a);arrays.append(a)
    assert G.R.M.tally(train)==result['training'];eligible=any(F.detail(p,r,a)['field_qualified_success'] for r,a in zip(train,arrays));assert eligible==result['learning_eligible']
    by={r['path']:r for r in rows}
    for r in reps:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    steps=0
    if eligible:
        obj=L.make(p);before=P.state(*obj);u=L.fit(obj,arrays,p,L.references(p),lambda:None,lambda *_:None);steps=u['Actor_Adam']
        assert u==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
        erows=rows[4:];assert len(erows)==32
        for r in erows:
            assert not r['full'] and r['full_arrays_generated_but_not_saved'] and sha(r['path'])==r['sha256']
            a=E.N.S.arrays(r['path']);assert set(a)=={'actions','native_counts','end_flags'}
            assert int(a['native_counts'].sum())==r['physics'] and a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])]
            assert not a['end_flags'][:-1].any() and not r['administrative_censor']
            for k,v in a.items():assert digest(v)==r['array_hashes'][k]
        for k,v in assess(p,erows).items():assert result[k]==v
        if not result['accepted']:P.load(obj,before)
        assert digest(P.state(*obj))==result['final_joint_hash']
    assert len(rows)==len(reps)==(36 if eligible else 4)
    assert G.rows(folder,'optimizer_calls.jsonl')==[[w,st,k] for w in ['live','shadow'] for k in range(steps) for st in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and s['commands']==len(rows)+len(reps)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=steps,Value_Adam=0,
         all_fresh_training_full120_history_density_reward_rebuilt=True,full_actor_Adam_and_rollback_rebuilt=True,
         compact_evaluation_NOT_full120_reconstructed_offline=True,all_native_evaluation_full_array_hashes_and_integration_match_replays=True,
         physics=s['physics'],commands=s['commands']))
    print('T520_FRESH_FULL_ACTOR_RL_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
