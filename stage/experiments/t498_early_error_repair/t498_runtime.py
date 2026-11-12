"""One fixed early-error repair candidate, complete autonomous native test."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t498_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'));import speed_runtime as R
G,S,P,E,M,torch,np=L.G,L.S,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t498_early_error_repair'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
R.OUT=OUT;R.HERE=HERE;R.verify=verify

def acceptance(p,rows):
    count=G.R.M.tally(rows);ids={r['job_id'] for r in rows if r['success'] and not r['collision']}
    accepted=len(rows)==54 and count['success']==54 and count['contact']==count['invalid']==count['censored']==0 and set(p['initial_success_ids'])<=ids
    return count,ids,accepted

def main(phase):
    assert phase=='repair';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        obj=L.make(p);shadow=L.make(p);before=P.state(*obj);data=L.groups(p)
        save(folder/'training_data_manifest.json',[dict(id=g['id'],kind=g['kind'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),target_sha256=digest(g['target'])) for g in data])
        rt.role='EARLY_WORST_ERROR_WITH_53_OWN_SUCCESS_RETENTION'
        aa=L.fit(obj,data,p,lambda:rt.guard(True),R.emit('live'));bb=L.fit(shadow,data,p,lambda:rt.guard(True),R.emit('shadow'))
        assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
        model=S.export(obj,folder/'proposal.pt.gz',p['model_bytes']);rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vals=rt.request(p['wide8_jobs'],model,'AUTONOMOUS_REPAIR_FULL54',full=True);rows=[r for a,r in vals];count,ids,accepted=acceptance(p,rows)
        if accepted:
            rt.pool.close();rt.commands+=2;dest=folder/'independent_model.pt.gz';shutil.copyfile(model['path'],dest);assert sha(dest)==model['sha256']
            rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
            rt.request(p['wide8_jobs'],dict(model,path=str(dest)),'INDEPENDENT_FULL54_MODEL_COPY',full=True,references=rows)
        else:
            by={r['job_id']:r for r in rows};jobs=[j for j in p['wide8_jobs'] if j['id'] in p['replay_failure_ids']]
            rt.request(jobs,model,'EXACT_ORIGINAL14_AND_RIGHT18',full=True,references=[by[j['id']] for j in jobs])
            for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
        result=dict(model=model,accepted=accepted,update=aa,wide8=count,sides={s:sum(s in x for x in ids) for s in ['LEFT','RIGHT']},success_ids=sorted(ids),gained_ids=sorted(ids-set(p['initial_success_ids'])),lost_ids=sorted(set(p['initial_success_ids'])-ids),right18_GOAL=p['repair_row']['job_id'] in ids,eight_development_allpass=accepted,independent54_exact=accepted,rollback_joint_hash=digest(P.state(*obj)),old275=None,wide6=None,fresh16=None,under8_gate=False,new_policy_gradient_steps=0)
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
    assert phase=='repair';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=S.frozen;rr,rep=G.rows(folder),G.rows(folder,'replays.jsonl');assert len(rr)==54 and len(rep)==(54 if s['result']['accepted'] else 15)
    for r in rr:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
    by={r['path']:r for r in rr}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    obj=L.make(p);before=P.state(*obj);data=L.groups(p);aa=L.fit(obj,data,p,lambda:None,lambda *_:None);result=s['result']
    assert aa==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
    count,ids,accepted=acceptance(p,rr);assert count==result['wide8'] and sorted(ids)==result['success_ids'] and accepted==result['accepted']
    if not accepted:P.load(obj,before)
    assert digest(P.state(*obj))==result['rollback_joint_hash']
    calls=G.rows(folder,'optimizer_calls.jsonl');assert calls==[[which,stage,k] for which in ['live','shadow'] for k in range(p['repair_steps']) for stage in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rr+rep) and s['commands']==len(rr)+len(rep)+4+(4 if accepted else 0)
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=p['repair_steps'],full_history_density_rewards_model_and_mature_Adam_exact=True,physics=s['physics'],commands=s['commands']))
    print('T498_REPAIR_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
