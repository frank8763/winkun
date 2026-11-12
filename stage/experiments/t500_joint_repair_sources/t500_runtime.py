"""Frozen speed-adapter ablation and verified recovery-source reconstruction.

No training. The existing eight-metre model and all unique evidence stay fixed.
"""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'));import speed_runtime as R
G,S,P,E,M,torch,np=R.G,R.S,R.P,R.E,R.M,R.torch,R.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t500_joint_repair_sources'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
R.OUT=OUT;R.HERE=HERE;R.verify=verify

def main(phase):
    assert phase in ['ablation','sources'];folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None
    try:
        model=p['ablation_model'] if phase=='ablation' else p['success_source_model'];jobs=p['ablation_jobs'] if phase=='ablation' else p['source_jobs']
        rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(jobs,model,'ZERO_DRIVE_ADAPTER_CAUSAL_ABLATION' if phase=='ablation' else 'HISTORICAL_SUCCESS_SOURCE_RECONSTRUCTION',full=True);rr=[r for a,r in vv]
        if phase=='sources':
            for r in rr:
                old=p['source_references'][r['job_id']]
                assert r['success'] and not r['collision'] and r['array_hashes']==old['array_hashes'] and r['array_digest']==old['array_digest'] and r['integration_sha256']==old['integration_sha256']
        rt.request(jobs,model,'EXACT_SAME_FROZEN_TRAJECTORIES',full=True,references=rr)
        result=dict(model=model,tally=G.R.M.tally(rr),rows=rr,Actor_Adam=0,Value_Adam=0)
        if phase=='sources':
            assert len(rr)==42;result.update(successful_repair_sources=rr+[p['existing_recovery_row']],all43_sources_available=True,historical_reconstruction_NOT_new_independent_data=True,source_selection_training_only_NOT_deployment_router=True)
        else:result.update(frozen_initial_conditions_only=True,baseline_existing_failures={r['job_id']:p['failed_references'][r['job_id']]['collision'] for r in rr},cannot_extrapolate_three_to_all43=True)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=S.frozen;rr,rep=G.rows(folder),G.rows(folder,'replays.jsonl');n=3 if phase=='ablation' else 42;assert len(rr)==len(rep)==n
    for r in rr:
        a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
        if phase=='sources':
            old=p['source_references'][r['job_id']];assert r['success'] and not r['collision'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    by={r['path']:r for r in rr}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rr+rep) and s['commands']==2*n+4
    if phase=='sources':
        row=p['existing_recovery_row'];a=E.N.S.arrays(row['path']);assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'] and all(digest(v)==row['array_hashes'][k] for k,v in a.items())
        M.audit_full_history(dict(p,teacher_first=5),row,a)
        assert len(s['result']['successful_repair_sources'])==43
    else:
        before=P.checkpoint(p['initial']['path'])['joint_state'];after=P.checkpoint(p['ablation_model']['path'])['joint_state']
        for key in ['value','actor_opt','value_opt']:assert digest(before[key])==digest(after[key])
        for key in before['actor']:
            if key in ['drive_adapter.2.weight','drive_adapter.2.bias']:assert not torch.count_nonzero(after['actor'][key])
            else:assert torch.equal(before['actor'][key],after['actor'][key])
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,all_arrays_hidden_density_rewards_and_exact_replays=True,physics=s['physics'],commands=s['commands'],full43_sources_verified=phase=='sources'))
    print('T500_'+phase+'_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
