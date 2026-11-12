"""One double-command repair candidate and complete383 autonomous validation."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t501_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'));import speed_runtime as R
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t501_joint_command_repair'
# The existing spawn path looks up this function in every fresh child process.
R.S.frozen=L.frozen

def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
R.OUT=OUT;R.HERE=HERE;R.verify=verify

def acceptance(p,rows):
    count=G.R.M.tally(rows);ids={r['job_id'] for r in rows if r['success'] and not r['collision']}
    valid=len(rows)==383 and count['invalid']==count['censored']==0
    retained=valid and set(p['initial_success_ids'])<=ids and len(ids)>340
    return count,ids,retained

def main(phase):
    assert phase=='joint';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        obj=L.make(p);shadow=L.make(p);before=P.state(*obj);assert digest(before)==p['expanded_initial_joint_hash'];data=L.groups(p)
        save(folder/'training_data_manifest.json',[dict(id=g['id'],kind=g['kind'],family=g['family'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),target_sha256=digest(g['target'])) for g in data])
        rt.role='JOINT_DRIVE_STEER_REPAIR_WITH340_OWN_SUCCESS_KEEP'
        aa=L.fit(obj,data,p,lambda:rt.guard(True),R.emit('live'));bb=L.fit(shadow,data,p,lambda:rt.guard(True),R.emit('shadow'))
        assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
        model=L.export(obj,folder/'proposal.pt.gz',p['model_bytes']);rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        jobs=p['old275_jobs']+p['six_jobs']+p['wide8_jobs'];vv=rt.request(jobs,model,'AUTONOMOUS_JOINT_COMMAND_FULL383',full=True);rows=[r for a,r in vv]
        count,ids,accepted=acceptance(p,rows)
        if not accepted:
            for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
        result=dict(model=model,update=aa,accepted=accepted,old275=G.R.M.tally(rows[:275]),wide6=G.R.M.tally(rows[275:329]),wide8=G.R.M.tally(rows[329:]),all383=count,all383_pass=count['success']==383 and count['invalid']==count['censored']==0,success_ids=sorted(ids),gained_ids=sorted(ids-set(p['initial_success_ids'])),lost_ids=sorted(set(p['initial_success_ids'])-ids),rollback_joint_hash=digest(P.state(*obj)),fresh16=None,reload78=None,under8_gate=False,new_policy_gradient_steps=0,new_Value_Adam=0)
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
    assert phase=='joint';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=L.frozen;rows=G.rows(folder);assert len(rows)==383 and not G.rows(folder,'replays.jsonl')
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
        assert r['loaded_model_sha256']==s['result']['model']['sha256'] and r['policy_hash']==s['result']['model']['policy_hash']
    assert [r['job_id'] for r in rows]==[j['id'] for j in p['old275_jobs']+p['six_jobs']+p['wide8_jobs']]
    obj=L.make(p);before=P.state(*obj);update=L.fit(obj,L.groups(p),p,lambda:None,lambda *_:None);result=s['result']
    assert update==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
    count,ids,accepted=acceptance(p,rows);assert count==result['all383'] and sorted(ids)==result['success_ids'] and accepted==result['accepted']
    assert result['old275']==G.R.M.tally(rows[:275]) and result['wide6']==G.R.M.tally(rows[275:329]) and result['wide8']==G.R.M.tally(rows[329:])
    assert result['all383_pass']==(count['success']==383 and count['invalid']==count['censored']==0)
    if not accepted:P.load(obj,before)
    assert digest(P.state(*obj))==result['rollback_joint_hash']
    calls=G.rows(folder,'optimizer_calls.jsonl');assert calls==[[which,stage,k] for which in ['live','shadow'] for k in range(p['fit_steps']) for stage in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rows)<=p['max_physics'] and s['commands']==387
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=p['fit_steps'],Value_Adam=0,all383_full_history_density_rewards_model_and_mapped_Adam_exact=True,physics=s['physics'],commands=s['commands']))
    print('T501_JOINT_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
