"""Success-likelihood replay and autonomous full-field evaluation."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t510_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t510_overnight_command_credit'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def event(which):return F.R.emit(which)

def native_successes(p,rows,arrays):
    ids=set()
    for r,a in zip(rows,arrays):
        if r['success'] and not r['collision']:
            if r['job_id'].startswith('t502_'):
                if F.detail(p,r,a)['field_qualified_success']:ids.add(r['job_id'])
            else:ids.add(r['job_id'])
    return ids

def emit_block(which,b):return event(which+'_block'+str(b))

def qualified_samples(p,vals):
    return [r for a,r in vals if F.detail(p,r,a)['field_qualified_success']]

def repair_mean(update,data,which):
    return sum(sum(update[which][g['record_id']] for g in data if g['kind']=='repair' and g['family']==i)/sum(g['kind']=='repair' and g['family']==i for g in data) for i in range(2))/2

def gate(p,qualified,fc,oc):
    return set(p['initial_field_success_ids'])<=qualified and len(qualified)>11 and oc['success']==8 and fc['contact']<=9 and fc['invalid']==fc['censored']==0

def main(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None
    try:
        obj=L.make(p);shadow=L.make(p);initial=P.state(*obj);data=L.groups(p);rt.role=phase+'_SUPERVISED_PROPOSAL'
        aa=L.fit(obj,data,p,lambda:rt.guard(True),event('live'));bb=L.fit(shadow,data,p,lambda:rt.guard(True),event('shadow'));assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
        model=L.A.export(obj,folder/'proposal.pt.gz',p['model_bytes']);rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vals=rt.request(p['guard_jobs'],model,'AUTONOMOUS_GUARD21',full=True);rr=[r for a,r in vals]
        rt.request(p['guard_jobs'],model,'EXACT_GUARD21',full=True,references=rr)
        ids=native_successes(p,rr,[a for a,r in vals]);tally=G.R.M.tally(rr)
        safe=set(p['initial_protected_ids'])<=ids and tally['invalid']==tally['censored']==0
        proxy=repair_mean(aa,data,'final')<repair_mean(aa,data,'initial')-1e-12
        result=dict(arm=phase,update=aa,model=model,Actor_Adam=aa['Actor_Adam'],Value_Adam=0,new_policy_gradient_updates=0,guard21=tally,guard_success_ids=sorted(ids),protected19_preserved=safe,repair_proxy_improved=proxy,
            old8=G.R.M.tally(rr[:8]),field24=None,accepted=False,full_old275_six54_eight54_NOT_executed=True,fresh16=None,under8_gate=False)
        if safe and proxy and set(p['new_task_ids'])&ids:
            rest=rt.request(p['remaining_field_jobs'],model,'REMAINING_FIELD11',full=True)
            known={r['job_id']:(a,r) for a,r in vals if r['job_id'].startswith('t502_')};known.update({r['job_id']:(a,r) for a,r in rest});field=[known[j['id']] for j in p['field_jobs']]
            details=[F.detail(p,r,a) for a,r in field];save(folder/'diagnosis.json',details)
            rt.request(p['field_jobs'],model,'FINAL_EXACT_FIELD24',full=True,references=[r for a,r in field])
            qualified={z['id'] for z in details if z['field_qualified_success']};fc=G.R.M.tally([r for a,r in field]);result.update(field24=fc,field_qualified_goals=len(qualified),field_success_ids=sorted(qualified),accepted=gate(p,qualified,fc,result['old8']),field13_reused_same_policy_guard_NOT_independent_repeat=True)
        if not result['accepted']:
            for o in [obj,shadow]:P.load(o,initial);assert digest(P.state(*o))==digest(initial)
        result['final_joint_hash']=digest(P.state(*obj));state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=L.A.frozen;rows,rep=G.rows(folder),G.rows(folder,'replays.jsonl');result=s['result']
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
    by={r['path']:r for r in rows}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    obj=L.make(p);initial=P.state(*obj);data=L.groups(p);u=L.fit(obj,data,p,lambda:None,lambda *_:None)
    assert u==result['update'] and digest(P.state(*obj))==result['model']['joint_hash'] and sha(result['model']['path'])==result['model']['sha256']
    gv=[(E.N.S.arrays(r['path']),r) for r in rows[:21]];ids=native_successes(p,[r for a,r in gv],[a for a,r in gv]);t=G.R.M.tally(rows[:21]);safe=set(p['initial_protected_ids'])<=ids and t['invalid']==t['censored']==0;proxy=repair_mean(u,data,'final')<repair_mean(u,data,'initial')-1e-12
    assert safe==result['protected19_preserved'] and proxy==result['repair_proxy_improved'] and t==result['guard21'] and sorted(ids)==result['guard_success_ids']
    final=safe and proxy and bool(set(p['new_task_ids'])&ids);assert len(rows)==(32 if final else 21) and len(rep)==(45 if final else 21)
    if final:
        known={r['job_id']:(a,r) for a,r in gv if r['job_id'].startswith('t502_')};known.update({r['job_id']:(E.N.S.arrays(r['path']),r) for r in rows[21:]});field=[known[j['id']] for j in p['field_jobs']];details=[F.detail(p,r,a) for a,r in field];assert details==read(folder/'diagnosis.json');q={z['id'] for z in details if z['field_qualified_success']};fc=G.R.M.tally([r for a,r in field]);assert fc==result['field24'] and gate(p,q,fc,result['old8'])==result['accepted']
    else:assert result['field24'] is None and not result['accepted']
    if not result['accepted']:P.load(obj,initial)
    assert digest(P.state(*obj))==result['final_joint_hash'] and G.rows(folder,'optimizer_calls.jsonl')==[[w,st,k] for w in ['live','shadow'] for k in range(u['Actor_Adam']) for st in ['BEFORE','AFTER']]
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=u['Actor_Adam'],Value_Adam=0,actual_parameter_subset_mature_Adam_projection_and_all_arrays_native_acceptance_verified=True,physics=s['physics'],commands=s['commands']));print('T510_ARM_AUDIT_PASS',phase,flush=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])

