"""Success-likelihood replay and autonomous full-field evaluation."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t509_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t509_two_goal_joint'
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
    state,error,result='FAILED',None,None;obj=shadow=initial=None
    try:
        rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        train=rt.request(p['training_jobs'],p['initial'],'FRESH_TWO_GOAL_OWN_EXPLORATION8',seeds=p['training_seeds'],full=True);tr=[r for a,r in train]
        rt.request(p['training_jobs'],p['initial'],'EXACT_FRESH_EXPLORATION8',seeds=p['training_seeds'],full=True,references=tr)
        repair=qualified_samples(p,train);eligible=all(any(r['job_id']==j for r in repair) for j in p['new_task_ids'])
        result=dict(training=G.R.M.tally(tr),training_qualified_successes=len(repair),learning_eligible=eligible,blocks_executed=0,blocks=[],Actor_Adam=0,Value_Adam=0,new_policy_gradient_updates=0,old8=None,field24=None,accepted=False,full_old275_six54_eight54_NOT_executed=True,fresh16=None,under8_gate=False)
        if eligible:
            p=dict(p,repair_rows=repair);obj=L.make(p);shadow=L.make(p);initial=P.state(*obj);data=L.groups(p);protected=set(p['initial_protected_ids']);blocks=[];last=None
            save(folder/'data_manifest.json',[dict(id=g['id'],record_id=g['record_id'],kind=g['kind'],family=g['family'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),target_sha256=digest(g['target'])) for g in data])
            for b in range(p['max_blocks']):
                before=P.state(*obj);rt.role='JOINT_TWO_GOAL_BLOCK_'+str(b)
                aa=L.fit(obj,data,p,lambda:rt.guard(True),emit_block('live',b));bb=L.fit(shadow,data,p,lambda:rt.guard(True),emit_block('shadow',b));assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow))
                model=L.A.export(obj,folder/('block'+str(b)+'_proposal.pt.gz'),p['model_bytes'])
                vals=rt.request(p['guard_jobs'],model,'BLOCK_'+str(b)+'_NATIVE_GUARD21',full=True);rr=[r for a,r in vals]
                rt.request(p['guard_jobs'],model,'BLOCK_'+str(b)+'_EXACT_GUARD21',full=True,references=rr)
                ids=native_successes(p,rr,[a for a,r in vals]);tally=G.R.M.tally(rr)
                ok=protected<=ids and tally['invalid']==tally['censored']==0 and repair_mean(aa,data,'final')<repair_mean(aa,data,'initial')-1e-12
                block=dict(block=b,before_joint_hash=digest(before),model=model,update=aa,protected_before=sorted(protected),native_success_ids=sorted(ids),native_tally=tally,accepted=ok)
                if ok:protected|=ids;last=(vals,model)
                else:
                    for o in [obj,shadow]:P.load(o,before);assert digest(P.state(*o))==digest(before)
                block['after_accept_or_rollback_joint_hash']=digest(P.state(*obj));blocks.append(block)
                with (folder/'blocks.jsonl').open('a') as f:f.write(__import__('json').dumps(block,ensure_ascii=False)+'\n')
                if not ok or len(ids)==21:break
            result.update(blocks_executed=len(blocks),blocks=blocks,Actor_Adam=sum(z['update']['Actor_Adam'] for z in blocks),retained_supervised_steps=sum(z['update']['Actor_Adam'] for z in blocks if z['accepted']))
            if last is not None:
                vals,model=last;ids=native_successes(p,[r for a,r in vals],[a for a,r in vals]);result.update(model=model,old8=G.R.M.tally([r for a,r in vals[:8]]))
                if set(p['new_task_ids'])&ids:
                    rest=rt.request(p['remaining_field_jobs'],model,'FINAL_REMAINING_FIELD11',full=True)
                    known={r['job_id']:(a,r) for a,r in vals if r['job_id'].startswith('t502_')};known.update({r['job_id']:(a,r) for a,r in rest});field=[known[j['id']] for j in p['field_jobs']]
                    detail=[F.detail(p,r,a) for a,r in field];save(folder/'diagnosis.json',detail)
                    rt.request(p['field_jobs'],model,'FINAL_EXACT_FIELD24',full=True,references=[r for a,r in field])
                    q={d['id'] for d in detail if d['field_qualified_success']};fc=G.R.M.tally([r for a,r in field]);result.update(field24=fc,field_qualified_goals=len(q),field_success_ids=sorted(q),accepted=gate(p,q,fc,result['old8']),field13_same_policy_guard_reuse_NOT_independent_repeat=True)
            if not result['accepted']:
                for o in [obj,shadow]:P.load(o,initial);assert digest(P.state(*o))==digest(initial)
            result['final_joint_hash']=digest(P.state(*obj))
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if initial is not None:
            for o in [obj,shadow]:P.load(o,initial);assert digest(P.state(*o))==digest(initial)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
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
    train=[(E.N.S.arrays(r['path']),r) for r in rows[:8]];repair=qualified_samples(p,train);eligible=all(any(r['job_id']==j for r in repair) for j in p['new_task_ids']);assert eligible==result['learning_eligible'] and G.R.M.tally(rows[:8])==result['training'];steps=0
    if eligible:
        p=dict(p,repair_rows=repair);obj=L.make(p);initial=P.state(*obj);data=L.groups(p);protected=set(p['initial_protected_ids']);last=None;expected=[];blocks=G.rows(folder,'blocks.jsonl');assert blocks==result['blocks']
        for b,block in enumerate(blocks):
            before=P.state(*obj);assert digest(before)==block['before_joint_hash'];u=L.fit(obj,data,p,lambda:None,lambda *_:None);assert u==block['update'] and digest(P.state(*obj))==block['model']['joint_hash'] and sha(block['model']['path'])==block['model']['sha256'];steps+=u['Actor_Adam']
            rr=rows[8+21*b:8+21*(b+1)];aa=[E.N.S.arrays(r['path']) for r in rr];ids=native_successes(p,rr,aa);t=G.R.M.tally(rr);ok=protected<=ids and t['invalid']==t['censored']==0 and repair_mean(u,data,'final')<repair_mean(u,data,'initial')-1e-12
            assert ok==block['accepted'] and sorted(protected)==block['protected_before'] and sorted(ids)==block['native_success_ids'] and t==block['native_tally']
            if ok:protected|=ids;last=list(zip(aa,rr))
            else:P.load(obj,before);assert b==len(blocks)-1
            assert digest(P.state(*obj))==block['after_accept_or_rollback_joint_hash'];expected.extend([[w+'_block'+str(b),st,k] for w in ['live','shadow'] for k in range(u['Actor_Adam']) for st in ['BEFORE','AFTER']])
        final=last is not None and bool(set(p['new_task_ids'])&native_successes(p,[r for a,r in last],[a for a,r in last]));n=len(blocks)
        assert len(rows)==8+21*n+(11 if final else 0) and len(rep)==8+21*n+(24 if final else 0)
        if final:
            known={r['job_id']:(a,r) for a,r in last if r['job_id'].startswith('t502_')};known.update({r['job_id']:(E.N.S.arrays(r['path']),r) for r in rows[8+21*n:]});field=[known[j['id']] for j in p['field_jobs']];d=[F.detail(p,r,a) for a,r in field];assert d==read(folder/'diagnosis.json');q={z['id'] for z in d if z['field_qualified_success']};fc=G.R.M.tally([r for a,r in field]);oc=G.R.M.tally([r for a,r in last[:8]]);assert fc==result['field24'] and oc==result['old8'] and gate(p,q,fc,oc)==result['accepted']
        else:assert result['field24'] is None and not result['accepted']
        if not result['accepted']:P.load(obj,initial)
        assert digest(P.state(*obj))==result['final_joint_hash'] and G.rows(folder,'optimizer_calls.jsonl')==expected
    else:assert len(rows)==len(rep)==8 and not G.rows(folder,'optimizer_calls.jsonl')
    assert steps==result['Actor_Adam'] and s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==len(rows)+len(rep)+4
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,discarded_historical_Actor_Adam=steps,Value_Adam=0,fresh8_and_success_selection_joint_ActorAdam_guard_and_all_full_history_arrays_verified=True,physics=s['physics'],commands=s['commands']));print('T509_JOINT_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])

