"""Complete five historical successes, then reuse the unchanged T496 student.

Success-source selection is an offline training-data decision. Deployment has
one learned Actor and never selects teachers, task IDs or time profiles.
"""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'))
import speed_runtime as R
G,S,P,E,M,torch,np=R.G,R.S,R.P,R.E,R.M,R.torch,R.np
read,save,sha,digest=R.read,R.save,R.sha,R.digest
OUT=ROOT/'stage/runs/t497_success_union_student'

def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30 and not torch.cuda.is_initialized()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15}
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
    for row in p.get('teacher_rows',[]):assert sha(row['path'])==row['sha256']
R.OUT=OUT;R.HERE=HERE;R.verify=verify

def main(phase):
    if phase=='student':return R.main(phase)
    assert phase=='teacher'
    folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True)
    E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None
    try:
        # phase != teacher explicitly suppresses the old training-only pulse.
        rt.pool=R.Pool(dict(p,phase='baseline_reconstruction',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['reconstruction_jobs'],p['initial'],'HISTORICAL_BASE40_SUCCESS_RECONSTRUCTION',full=True)
        rr=[r for a,r in vv]
        for r in rr:
            old=p['reconstruction_references'][r['job_id']]
            assert r['success'] and not r['collision'] and not r.get('teacher',False)
            assert r['array_hashes']==old['array_hashes'] and r['array_digest']==old['array_digest'] and r['integration_sha256']==old['integration_sha256']
        rt.request(p['reconstruction_jobs'],p['initial'],'EXACT_RECONSTRUCTED_FIVE',full=True,references=rr)
        data=p['verified_success_rows']+rr
        assert len(data)==54 and {r['job_id'] for r in data}=={j['id'] for j in p['wide8_jobs']}
        assert all(r['success'] and not r['collision'] for r in data)
        # Actual-history label compatibility, not a new optimizer step.
        groups=S.groups(p,data);assert len(groups)==54
        result=dict(eligible_for_student=True,reconstructed_five=G.R.M.tally(rr),rows=data,reconstruction_rows=rr,historical_reconstruction_NOT_independent_new_samples=True,source_union_NOT_single_policy_navigation=True,Actor_Adam=0,Value_Adam=0)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),phase=phase,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    if phase=='student':return R.audit(phase)
    assert phase=='teacher'
    folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);s=read(folder/'summary.json')
    assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    rr,rep=G.rows(folder),G.rows(folder,'replays.jsonl');assert len(rr)==len(rep)==5
    for r in rr:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        M.audit_full_history(p,r,a)
        old=p['reconstruction_references'][r['job_id']]
        assert r['success'] and not r['collision'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    by={r['path']:r for r in rr}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rr+rep) and s['commands']==14
    assert s['result']['rows']==p['verified_success_rows']+rr
    assert len(S.groups(p,s['result']['rows']))==54
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Actor_Adam=0,Value_Adam=0,physics=s['physics'],commands=s['commands'],five_original_hashes_and_full_history_density_rewards_exact=True,source_union_NOT_single_actor_navigation=True))
    print('T497_SOURCE_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
