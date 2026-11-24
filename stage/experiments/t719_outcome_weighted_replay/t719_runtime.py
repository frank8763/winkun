"""Exclusive scientific child, complete fresh blocks and independent replay."""
from pathlib import Path
import sys,os,fcntl,gzip,json,time,copy,traceback
import importlib.util
_spec=importlib.util.spec_from_file_location('t719_runtime_controller',Path(__file__).resolve().parent/'run.py');C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
import t719_learning as L
import acquisition_native as RN
A,N=C.A,C.M.N
torch,np,digest=L.torch,L.np,L.digest

def setup(name):
    p=dict(C.root_plan());child=C.read(C.OUT/name/'protocol.json');assert child['_root_sha256']==C.sha(C.OUT/'protocol.json');p.update(child)
    N.OUT,N.DOC,N.HERE=C.OUT,C.DOC,C.HERE;N.verify=C.verify;N.worker=RN.worker;N.A.frozen=A.frozen;N.BASE_INSPECT=RN.inspect
    N.setup(p);N.S.verify=C.verify;N.S.N.inspect=RN.inspect
    return p

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
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);assert C.read(C.OUT/'fit/audit.json')['state']=='PASS';model=C.read(C.OUT/'fit/summary.json')['model']
    N.T.R.E.L.install_stop_handlers();lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=N.S.NativeRuntime(p);error=None;fresh=[]
    try:
        pool(rt)
        for kind in ['recovery','original']:
            jobs=p[kind+'_jobs'];values=rt.request(jobs,model,kind,True);refs=[r for a,r in values];fresh+=refs
            rt.request(jobs,model,'exact_'+kind,True,refs)
    except BaseException:error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        N.T.close(rt);C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,model=model,physics=sum(rt.counts),commands=rt.commands+7,fresh=fresh,elapsed_seconds=time.time()-rt.start));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def evaluate_audit(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);sm=C.read(folder/'summary.json');assert sm['state']=='COMPLETE' and not Path('/proc',str(sm['pid'])).exists()
    for g in C.read(folder/'worker_generations.json'):assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rr=checkrows(p,folder);assert len(rr)==72 and sum(r['physics'] for r in rr)==sm['physics']<=p['max_physics'] and sm['commands']==86
    assert all(r['loaded_model_sha256']==sm['model']['sha256'] for r in rr)
    for role in ['recovery','original']:
        fresh=[r for r in rr if r['role']==role];assert len(fresh)==18 and {r['job_id'] for r in fresh}=={j['id'] for j in p[role+'_jobs']}
    C.verify(p,force=True);C.save(folder/'audit.json',dict(state='PASS',all72_histories_full_prefix_owncarry_density_native_arrays_exact=True,new_native=0))
