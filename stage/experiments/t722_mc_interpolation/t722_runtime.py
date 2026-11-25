"""Exclusive scientific child, complete fresh blocks and independent replay."""
from pathlib import Path
import sys,os,fcntl,gzip,json,time,copy,traceback
import importlib.util
_spec=importlib.util.spec_from_file_location('t722_runtime_controller',Path(__file__).resolve().parent/'run.py');C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
import t722_learning as L
import acquisition_native as RN
A,N=C.A,C.M.N
torch,np,digest=L.torch,L.np,L.digest

def setup(name):
    RN.Base.G._real_affinity(0,set(range(8,16)));p=dict(C.root_plan());child=C.read(C.OUT/name/'protocol.json');assert child['_root_sha256']==C.sha(C.OUT/'protocol.json');p.update(child)
    N.OUT,N.DOC,N.HERE=C.OUT,C.DOC,C.HERE;N.verify=C.verify;N.worker=RN.worker;N.A.frozen=A.frozen;N.BASE_INSPECT=RN.inspect
    N.setup(p);N.S.verify=C.verify;N.S.N.inspect=RN.inspect
    torch.set_num_threads(1);return p

def pool(rt):
    folder=Path(rt.p['out']);file=folder/'worker_generations.json';old=C.read(file) if file.exists() else []
    rt.pool=N.T.R.N.Pool(dict(rt.p,out=str(folder),native_stop_epoch=rt.deadline),rt.counts,rt.guard)
    new=C.read(file);C.save(file,old+new)
    if old:rt.commands+=14

def event(who,rnd):
    def emit(kind,stage,k):
        with (C.OUT/'fit_calls.jsonl').open('a') as f:f.write(json.dumps(dict(who=who,round=rnd,kind=kind,stage=stage,k=k))+'\n')
    return emit

def rows(folder):return [json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
def checkrows(p,folder):
    values=rows(folder);by={r['path']:r for r in values};assert len(by)==len(values)
    for r in values:
        C.verify(p);assert C.sha(r['path'])==r['sha256'] and C.sha(r['request']['model'])==r['loaded_model_sha256']
        RN.inspect(p,RN.Base.C.E.N.S.arrays(r['path']),r)
        if r.get('exact'):assert r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    return values

def evaluate(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);N.T.R.E.L.install_stop_handlers();model=p['evaluation_model']
    assert C.sha(model['path'])==model['sha256'];lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=N.S.NativeRuntime(p);error=None;fresh=[];started=time.time()
    try:
        pool(rt)
        for off in range(0,len(p['jobs']),28):fresh +=[r for a,r in rt.request(p['jobs'][off:off+28],model,'fresh',True)]
        by={r['job_id']:r for r in fresh};rt.request(p['exact_jobs'],model,'exact',True,[by[j['id']] for j in p['exact_jobs']])
    except BaseException:error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        N.T.close(rt);safe=[r['job_id'] for r in fresh if N.T.Prior.safe(r)]
        C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,model=model,physics=sum(rt.counts),commands=rt.commands+7,success=len(safe),success_ids=sorted(safe),total=len(p['jobs']),collision_ids=[r['job_id'] for r in fresh if r['collision']],timeout_ids=[r['job_id'] for r in fresh if not r['success'] and not r['collision']],boundary_ids=[r['job_id'] for r in fresh if 'fullfield_start_pose' in r['request']['job'] and r.get('field_boundary')],elapsed_seconds=time.time()-started));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def evaluate_audit(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);s=C.read(folder/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    rr=checkrows(p,folder);fresh=[r for r in rr if not r.get('exact')];assert len(fresh)==len(p['jobs']) and {r['job_id'] for r in fresh}=={j['id'] for j in p['jobs']}
    assert len(rr)==len(p['jobs'])+len(p['exact_jobs']);assert sum(r['physics'] for r in rr)==s['physics']<=p['max_physics'];assert s['commands']==len(rr)+14<=p['max_commands']
    assert all(r['loaded_model_sha256']==s['model']['sha256'] for r in rr)
    base={r['job_id']:r for r in p.get('baseline_rows',[])}
    for r in fresh:
        a=RN.Base.C.E.N.S.arrays(r['path'])
        if r['job_id'] in base:
            b=RN.Base.C.E.N.S.arrays(base[r['job_id']]['path'])
            for k in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[k][0],b[k][0]),('initial',r['job_id'],k)
            for k in ['initial_native_integration','task_goal_xy','scene_sha256']:assert np.array_equal(a[k],b[k]),('initial',r['job_id'],k)
        if 'prospective_initial' in r['request']['job']:
            c=r['request']['job']['prospective_initial'];assert digest(a['states'][0])==c['packet_hash'] and digest(a['initial_native_integration'])==c['native_hash'] and digest(a['scene_sha256'])==c['scene_hash'] and digest(a['task_goal_xy'])==c['goal_hash']
    assert sorted(r['job_id'] for r in fresh if N.T.Prior.safe(r))==s['success_ids'];C.verify(p,force=True)
    C.save(folder/'audit.json',dict(state='PASS',rows=len(rr),full_owncarry_prefix_density_all_payload_exact=True,all_initial_scene_goal_packet_baseline_pairing=True,new_native=0,new_learning=0))
