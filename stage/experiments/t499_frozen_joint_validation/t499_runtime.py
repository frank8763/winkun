"""Frozen single-Actor old275 + wide6 full-history validation; no learning."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t496_neural_speed_profile'))
import speed_runtime as R
G,S,P,E,M,torch,np=R.G,R.S,R.P,R.E,R.M,R.torch,R.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t499_frozen_joint_validation'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
R.OUT=OUT;R.HERE=HERE;R.verify=verify

def main(phase):
    assert phase=='regression';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p)
    state,error,result='FAILED',None,None
    try:
        model=p['initial'];before=sha(model['path']);rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        # Task definitions and native simulator remain the original Stage ones.
        vv=rt.request(p['old275_jobs']+p['six_jobs'],model,'FROZEN_SINGLE_ACTOR_FULL329',full=True);rows=[r for a,r in vv]
        assert len(rows)==329 and len({r['job_id'] for r in rows})==329 and sha(model['path'])==before
        old=G.R.M.tally(rows[:275]);six=G.R.M.tally(rows[275:]);allpass=all(r['success'] and not r['collision'] and r['complete'] and not r['administrative_censor'] for r in rows)
        result=dict(model=model,old275=old,wide6=six,all329_pass=allpass,failures=[dict(id=r['job_id'],contact=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],complete=r['complete'],censored=r['administrative_censor']) for r in rows if not r['success'] or r['collision'] or not r['complete']],Actor_Adam=0,Value_Adam=0,model_bytes_unchanged=True,wide8_prior54=read(p['eight_summary'])['result']['wide8'],fresh16=None,reload78=None,under8_gate=False)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit(phase):
    assert phase=='regression';folder=OUT/phase;p=R.child_plan(folder);R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder)
    P.frozen=S.frozen;rr=G.rows(folder);assert len(rr)==329 and not G.rows(folder,'replays.jsonl')
    for r in rr:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());M.audit_full_history(p,r,a)
        assert r['loaded_model_sha256']==p['initial']['sha256'] and r['policy_hash']==p['initial']['policy_hash']
        assert int(a['native_counts'].sum())==r['physics'] and r['request']['seed'] is None
    expected=[j['id'] for j in p['old275_jobs']+p['six_jobs']];assert [r['job_id'] for r in rr]==expected
    result=s['result'];assert result['old275']==G.R.M.tally(rr[:275]) and result['wide6']==G.R.M.tally(rr[275:])
    assert result['all329_pass']==all(r['success'] and not r['collision'] and r['complete'] and not r['administrative_censor'] for r in rr)
    assert s['physics']==sum(r['physics'] for r in rr)<=p['max_physics'] and s['commands']==333
    assert sha(p['initial']['path'])==p['initial']['sha256'] and digest(P.checkpoint(p['initial']['path'])['joint_state'])==p['initial']['joint_hash']
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Actor_Adam=0,Value_Adam=0,all329_full_arrays_hidden_history_density_rewards_verified=True,model_joint_and_bytes_unchanged=True,physics=s['physics'],commands=s['commands']))
    print('T499_FULL329_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
