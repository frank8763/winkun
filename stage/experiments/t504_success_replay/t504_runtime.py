"""Success-likelihood replay and autonomous full-field evaluation."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t504_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t504_success_replay'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def event(which):return F.R.emit(which)

def accepted(p,qualified,oldcount,fieldcount):
    return (set(p['initial_field_success_ids'])<=qualified and len(qualified)>6
            and oldcount['success']==8 and fieldcount['contact']<=10
            and fieldcount['invalid']==fieldcount['censored']==0)

def main(phase):
    assert phase=='pilot';folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p);data=L.training_data(p)
        save(folder/'data_manifest.json',dict(training=[dict(id=g['id'],source_sha256=g['source_sha256'],near_decisions=int(g['near'].sum()),states=len(g['near'])) for g in data],keep=[dict(id=g['id'],family=g['family'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),target_sha256=digest(g['target'])) for g in refs]))
        rt.role='HISTORICAL_SUCCESS_LIKELIHOOD_REPLAY_NOT_ONPOLICY'
        aa=L.fit(obj,data,p,refs,lambda:rt.guard(True),event('live'));bb=L.fit(shadow,data,p,refs,lambda:rt.guard(True),event('shadow'))
        assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
        model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
        rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        old=rt.request(p['old8_jobs'],model,'AUTONOMOUS_OLD8_SENTINELS',full=True)
        field=rt.request(p['field_jobs'],model,'AUTONOMOUS_FULLFIELD24_AFTER_REPLAY',full=True);rows=[r for a,r in field]
        detail=[F.detail(p,r,a) for a,r in field];save(folder/'diagnosis.json',detail)
        rt.request(p['field_jobs'],model,'EXACT_FULLFIELD24_AFTER_REPLAY',full=True,references=rows)
        qualified={m['id'] for m in detail if m['field_qualified_success']};initial=set(p['initial_field_success_ids'])
        oldcount=G.R.M.tally([r for a,r in old]);fieldcount=G.R.M.tally(rows);ok=accepted(p,qualified,oldcount,fieldcount)
        if not ok:
            for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
        result=dict(replayed_historical_successes=2,new_training_native_episodes=0,new_policy_gradient_updates=0,
                    model=model,update=aa,Actor_Adam=aa['Actor_Adam'],Value_Adam=0,field24=fieldcount,old8=oldcount,
                    field_qualified_goals=len(qualified),field_success_ids=sorted(qualified),gained_ids=sorted(qualified-initial),lost_ids=sorted(initial-qualified),
                    accepted=ok,rollback_joint_hash=digest(P.state(*obj)),full_old275_six54_eight54_NOT_executed=True,fresh16=None,under8_gate=False)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if before is not None:
            for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=L.A.frozen;rows,rep=G.rows(folder),G.rows(folder,'replays.jsonl');result=s['result'];assert len(rows)==32 and len(rep)==24
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
    by={r['path']:r for r in rows}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    obj=L.make(p);before=P.state(*obj);update=L.fit(obj,L.training_data(p),p,L.references(p),lambda:None,lambda *_:None)
    assert update==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
    metrics=[F.detail(p,r,E.N.S.arrays(r['path'])) for r in rows[8:]];assert metrics==read(folder/'diagnosis.json')
    ids={d['id'] for d in metrics if d['field_qualified_success']};oldcount=G.R.M.tally(rows[:8]);count=G.R.M.tally(rows[8:]);ok=accepted(p,ids,oldcount,count)
    assert count==result['field24'] and oldcount==result['old8'] and ok==result['accepted'] and sorted(ids)==result['field_success_ids']
    if not ok:P.load(obj,before)
    assert digest(P.state(*obj))==result['rollback_joint_hash']
    assert G.rows(folder,'optimizer_calls.jsonl')==[[w,stage,k] for w in ['live','shadow'] for k in range(update['Actor_Adam']) for stage in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=update['Actor_Adam'],Value_Adam=0,new_policy_gradient_updates=0,all_arrays_history_rewards_and_mature_Adam_exact=True,physics=s['physics'],commands=s['commands']))
    print('T504_SUCCESS_REPLAY_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
