"""Offline-only compatibility pilot; no simulator construction or step."""
from pathlib import Path
import fcntl,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t516_learning as L
G,P,E,M,torch=L.G,L.P,L.G.E,L.G.M,L.torch
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t516_nonlinear_compatibility'
def verify(p,full=False):
    root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
    for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
def event(folder,which):
    def emit(stage,k):
        with (folder/'optimizer_calls.jsonl').open('a') as f:f.write(json.dumps([which,stage,k])+'\n')
    return emit
def ban_native():
    import mujoco
    def denied(*_a,**_kw):raise AssertionError("T516 forbids native setup/physics")
    mujoco.mj_step=denied;G.B.native_setup=denied;E.N.S.P.C.B.setup=denied

def main(phase):
    assert phase=='fit';folder=OUT/phase;p=read(OUT/'protocol.json');verify(p,True);E.N.offline();E.L.install_stop_handlers();ban_native()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();deadline=min(start+p['main_seconds'],p['absolute_science_stop']);state,error,result='FAILED',None,None
    obj=shadow=initial=None;last=0.
    def guard():
        nonlocal last
        E.L.check_stop();assert time.time()<deadline-5
        if time.time()-last>10:
            verify(p);last=time.time();save(folder/'status.json',dict(state='OFFLINE_NONLINEAR_COMPATIBILITY',epoch=last,pid=os.getpid()))
    try:
        rows=L.data(p);obj=L.make(p);shadow=L.make(p);initial=P.state(*obj)
        save(folder/'data_manifest.json',[dict(id=g['id'],record_id=g['record_id'],kind=g['kind'],source_sha256=g['source_sha256'],features_sha256=digest(g['features']),targets_sha256=digest(g['target']),train_mask_sha256=digest(g['train_mask']) if g['kind']=='repair' else None) for g in rows])
        a=L.fit(obj,rows,p,guard,event(folder,'live'));b=L.fit(shadow,rows,p,guard,event(folder,'shadow'))
        assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
        model=L.export(obj,folder/'offline_candidate.pt.gz',p['model_bytes']);save(folder/'metrics.json',a['final'])
        result=dict(update=a,candidate=model,offline_joint_gate=a['final']['offline_joint_gate'],Actor_Adam=a['Actor_Adam'],shadow_Actor_Adam=b['Actor_Adam'],Value_Adam=0,new_policy_gradient_updates=0,
                    promoted_navigation_model=False,field24=None,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
        for o in [obj,shadow]:P.load(o,initial);assert digest(P.state(*o))==p['initial']['joint_hash']
        result['active_research_state_rolled_back_to_zero_residual']=True
        result['final_joint_hash']=digest(P.state(*obj));state='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if initial is not None:
            for o in [obj,shadow]:P.load(o,initial)
        save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=0,commands=0,elapsed_s=time.time()-start,error=error,result=result))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert state=='COMPLETE',error
def audit(phase):
    folder=OUT/phase;p=read(OUT/'protocol.json');verify(p,True);s=read(folder/'summary.json')
    ban_native();assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (folder/'audit.json').exists()
    deadline=min(time.time()+p['audit_seconds'],p['absolute_science_stop'])
    last=[0.]
    def guard():
        assert time.time()<deadline-5
        if time.time()-last[0]>10:verify(p);last[0]=time.time()
    obj=L.make(p);initial=P.state(*obj);rows=L.data(p);update=L.fit(obj,rows,p,guard,lambda *_:None)
    result=s['result'];assert update==result['update'] and digest(P.state(*obj))==result['candidate']['joint_hash']
    assert sha(result['candidate']['path'])==result['candidate']['sha256'] and L.frozen(p,result['candidate']['path']).phash()==result['candidate']['policy_hash']
    assert update['final']==read(folder/'metrics.json') and update['final']['offline_joint_gate']==result['offline_joint_gate']
    calls=G.rows(folder,'optimizer_calls.jsonl');expected=[[w,stage,k] for w in ['live','shadow'] for k in range(p['fit_steps']) for stage in ['BEFORE','AFTER']];assert calls==expected
    P.load(obj,initial);assert digest(P.state(*obj))==result['final_joint_hash']
    assert s['physics']==s['commands']==0 and sha(p['parent_model']['path'])==p['parent_model']['sha256']
    save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,commands=0,discarded_historical_Actor_Adam=p['fit_steps'],all_training_and_blocked_label_metrics_and_ActorAdam_reconstructed=True,parent_frozen_and_rollback_exact=True,offline_joint_gate=result['offline_joint_gate']))
    print('T516_OFFLINE_COMPATIBILITY_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
