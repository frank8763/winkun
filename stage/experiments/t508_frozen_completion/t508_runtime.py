"""Success-likelihood replay and autonomous full-field evaluation."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as L
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=L.G,L.P,L.G.E,L.G.M,L.torch,L.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t508_frozen_completion'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def event(which):return F.R.emit(which)

def full_field(p,new):
    known={r['job_id']:(E.N.S.arrays(r['path']),r) for r in p['retained18'] if r['job_id'].startswith('t502_')}
    known.update({r['job_id']:(a,r) for a,r in new});assert len(known)==24
    return [known[j['id']] for j in p['field_jobs']]

def evaluate(p,field):
    details=[F.detail(p,r,a) for a,r in field];ids={d['id'] for d in details if d['field_qualified_success']};fc=G.R.M.tally([r for a,r in field]);old=G.R.M.tally(p['retained18'][:8])
    ok=set(p['initial_field_success_ids'])<=ids and len(ids)>7 and old['success']==8 and fc['contact']<=10 and fc['invalid']==fc['censored']==0
    return details,dict(old8=old,field24=fc,field_qualified_goals=len(ids),field_success_ids=sorted(ids),accepted=ok,model=p['initial'],new_Adam=0,Value_Adam=0,new_training_native=0,new_policy_gradient_updates=0,
        imported_same_policy_cases=18,imported_field_cases=10,new_field_cases=14,old18_NOT_independent_new_repeat=True,full_old275_six54_eight54_NOT_executed=True,fresh16=None,under8_gate=False)

def main(phase):
    assert phase=='pilot';folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None
    try:
        rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        new=rt.request(p['remaining_field_jobs'],p['initial'],'FROZEN_REMAINING14_AFTER_POWERLOSS',full=True)
        field=full_field(p,new);details,result=evaluate(p,field);save(folder/'diagnosis.json',details)
        save(folder/'combined_field24_rows.json',[r for a,r in field])
        rt.request(p['field_jobs'],p['initial'],'EXACT_COMPLETE_FIELD24_AFTER_POWERLOSS',full=True,references=[r for a,r in field])
        assert sha(p['initial']['path'])==p['initial']['sha256'];state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=L.frozen;rows,rep=G.rows(folder),G.rows(folder,'replays.jsonl');assert len(rows)==14 and len(rep)==24
    allrows=p['retained18']+rows
    for r in allrows:
        assert sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==p['initial']['sha256']
        a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
    field=full_field(p,[(E.N.S.arrays(r['path']),r) for r in rows]);details,result=evaluate(p,field)
    assert result==s['result'] and details==read(folder/'diagnosis.json') and [r for a,r in field]==read(folder/'combined_field24_rows.json')
    by={r['path']:r for r in allrows}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    assert not G.rows(folder,'optimizer_calls.jsonl') and sha(p['initial']['path'])==p['initial']['sha256'] and digest(P.checkpoint(p['initial']['path'])['joint_state'])==p['initial']['joint_hash']
    assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==42
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,imported18_exact_same_model_SHA_verified=True,new14_and24exact_full_arrays_history_density_reward_verified=True,model_ActorAdam_unchanged=True,physics=s['physics'],commands=s['commands']))
    print('T508_FROZEN_COMPLETION_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])

