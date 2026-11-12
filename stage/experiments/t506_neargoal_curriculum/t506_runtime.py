"""Three-task fresh own-policy pilot with native success retention."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t506_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t506_neargoal_curriculum'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def event(which):return F.R.emit(which)

def qualified(p,vals):
    return {r['job_id'] for a,r in vals if (F.detail(p,r,a)['field_qualified_success'] if r['job_id'].startswith('t502_') else r['success'] and not r['collision'])}

def main(phase):
    assert phase=='pilot';folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None;obj=shadow=before=None
    try:
        rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['training_jobs'],p['initial'],'FRESH_OWN_POLICY_NEARGOAL12',seeds=p['training_seeds'],full=True)
        arrays=[a for a,r in vv];samples=[r for a,r in vv]
        rt.request(p['training_jobs'],p['initial'],'EXACT_FRESH_OWN_POLICY12',seeds=p['training_seeds'],full=True,references=samples)
        eligible=bool(qualified(p,vv));result=dict(training=G.R.M.tally(samples),training_qualified_successes=sum(F.detail(p,r,a)['field_qualified_success'] for a,r in vv),learning_eligible=eligible,
            Actor_Adam=0,Value_Adam=0,guard15=None,field24=None,old8=None,accepted=False,full_old275_six54_eight54_NOT_executed=True,fresh16=None,under8_gate=False)
        if eligible:
            obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p)
            save(folder/'keep_manifest.json',[dict(id=g['id'],family=g['family'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),target_sha256=digest(g['target'])) for g in refs])
            rt.role='CLIPPED_REAL_POLICY_GRADIENT_WITH_15_SUCCESS_KEEP'
            aa=L.fit(obj,arrays,p,refs,lambda:rt.guard(True),event('live'));bb=L.fit(shadow,arrays,p,refs,lambda:rt.guard(True),event('shadow'))
            assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
            model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes'])
            gv=rt.request(p['guard_jobs'],model,'AUTONOMOUS_GUARD15',full=True);gr=[r for a,r in gv]
            rt.request(p['guard_jobs'],model,'EXACT_GUARD15',full=True,references=gr)
            gc=G.R.M.tally(gr);ids=qualified(p,gv);guard_ok=set(p['initial_protected_ids'])<=ids and gc['invalid']==gc['censored']==0
            result.update(model=model,update=aa,Actor_Adam=aa['Actor_Adam'],guard15=gc,guard_success_ids=sorted(ids),guard_pass=guard_ok,old8=G.R.M.tally(gr[:8]))
            if guard_ok:
                rest=rt.request(p['remaining_field_jobs'],model,'FINAL_REMAINING_FIELD17',full=True)
                by={r['job_id']:(a,r) for a,r in gv if r['job_id'].startswith('t502_')};by.update({r['job_id']:(a,r) for a,r in rest})
                field=[by[j['id']] for j in p['field_jobs']];rows=[r for a,r in field];details=[F.detail(p,r,a) for a,r in field];save(folder/'diagnosis.json',details)
                rt.request(p['field_jobs'],model,'FINAL_EXACT_FIELD24',full=True,references=rows)
                field_ids=qualified(p,field);fc=G.R.M.tally(rows);accepted=set(p['initial_field_success_ids'])<=field_ids and len(field_ids)>7 and fc['contact']<=10 and fc['invalid']==fc['censored']==0
                result.update(field24=fc,field_qualified_goals=len(field_ids),field_success_ids=sorted(field_ids),gained_ids=sorted(field_ids-set(p['initial_field_success_ids'])),accepted=accepted,field7_guard_reuse_same_policy_NOT_independent_repeat=True)
            if not result['accepted']:
                for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
            result['final_joint_hash']=digest(P.state(*obj))
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
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=L.A.frozen;rows,rep=G.rows(folder),G.rows(folder,'replays.jsonl');result=s['result']
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
    by={r['path']:r for r in rows}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    train=[(E.N.S.arrays(r['path']),r) for r in rows[:12]];eligible=bool(qualified(p,train));assert eligible==result['learning_eligible'] and G.R.M.tally(rows[:12])==result['training'];steps=0
    if eligible:
        obj=L.make(p);before=P.state(*obj);u=L.fit(obj,[a for a,r in train],p,L.references(p),lambda:None,lambda *_:None);steps=u['Actor_Adam']
        assert u==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
        gv=[(E.N.S.arrays(r['path']),r) for r in rows[12:27]];gc=G.R.M.tally([r for a,r in gv]);guard_ok=set(p['initial_protected_ids'])<=qualified(p,gv) and gc['invalid']==gc['censored']==0
        assert guard_ok==result['guard_pass'] and gc==result['guard15']
        assert len(rows)==(44 if guard_ok else 27) and len(rep)==(51 if guard_ok else 27)
        if guard_ok:
            by_id={r['job_id']:(a,r) for a,r in gv if r['job_id'].startswith('t502_')};by_id.update({r['job_id']:(E.N.S.arrays(r['path']),r) for r in rows[27:]});field=[by_id[j['id']] for j in p['field_jobs']]
            details=[F.detail(p,r,a) for a,r in field];assert details==read(folder/'diagnosis.json');fc=G.R.M.tally([r for a,r in field]);ids=qualified(p,field)
            ok=set(p['initial_field_success_ids'])<=ids and len(ids)>7 and fc['contact']<=10 and fc['invalid']==fc['censored']==0
            assert ok==result['accepted'] and fc==result['field24']
        else:assert result['field24'] is None and not result['accepted']
        if not result['accepted']:P.load(obj,before)
        assert digest(P.state(*obj))==result['final_joint_hash'] and G.rows(folder,'optimizer_calls.jsonl')==[[w,st,k] for w in ['live','shadow'] for k in range(steps) for st in ['BEFORE','AFTER']]
    else:assert len(rows)==len(rep)==12 and not (folder/'proposal.pt.gz').exists() and not G.rows(folder,'optimizer_calls.jsonl')
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=steps,Value_Adam=0,all_own_policy_density_peer_credit_history_arrays_ActorAdam_and_native_guards_exact=True,physics=s['physics'],commands=s['commands']))
    print('T506_OWN_POLICY_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])

