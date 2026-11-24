"""Exclusive scientific child, complete fresh blocks and independent replay."""
from pathlib import Path
import sys,os,fcntl,gzip,json,time,copy,traceback
import importlib.util
_spec=importlib.util.spec_from_file_location('t717_runtime_controller',Path(__file__).resolve().parent/'run.py');C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
import t717_learning as L
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

def train(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);N.T.R.E.L.install_stop_handlers();lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rt=N.S.NativeRuntime(p);records=[];current=p['learner_initial'];obj=None;error=None;fitstart=time.time()
    C.save(folder/'science_started.json',dict(pid=os.getpid(),epoch=fitstart))
    try:
        torch.set_num_threads(1);obj=L.make(p);body=L.fixed(obj[0])
        for rnd in range(p['rounds']):
            assert time.time()+p['round_reserve_seconds']<p['absolute_learning_stop'] and time.time()+p['round_reserve_seconds']<rt.deadline,'Next complete fresh block time admission denied'
            pool(rt);jobs=[j for j in p['training_jobs'] for _ in range(2)];seeds=[[p['seed'],rnd,i] for i in range(len(jobs))]
            values=rt.request(jobs,current,f'fresh{rnd}',True,seeds=seeds)
            rt.request(jobs,current,f'exact{rnd}',True,[r for a,r in values],seeds=seeds);N.T.close(rt)
            torch.set_num_threads(p['fit_threads']);before=L.state(obj);start=L.write_state(folder/f'start_{rnd}.pt.gz',before)
            def guard():C.verify(p);assert time.time()<p['absolute_learning_stop']-5 and time.time()<rt.deadline-5
            u=L.fit(obj,values,p,guard,event('live',rnd));shadow=L.make(p);L.load(shadow,before);v=L.fit(shadow,values,p,guard,event('shadow',rnd))
            assert u==v and digest(L.state(obj))==digest(L.state(shadow));del shadow
            trained=L.write_state(folder/f'trained_{rnd}.pt.gz',L.state(obj));current=L.export(obj[0],p,folder/f'model_{rnd}.pt.gz')
            assert A.frozen(p,current['path']).phash()==obj[0].phash() and L.fixed(obj[0])==body
            record=dict(round=rnd,start=start,trained=trained,model=current,fit=u,training_paths=[r['path'] for a,r in values]);records.append(record);C.save(folder/'ROUNDS.json',records)
            C.save(C.OUT/'progress.json',dict(rounds=len(records),model=current,physics=sum(rt.counts),commands=rt.commands));torch.set_num_threads(1)
    except BaseException:error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        N.T.close(rt);C.verify(p,force=True);C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands+7,rounds=records,model=current,elapsed_seconds=time.time()-fitstart,new_RL=True,no_oldtask_guard=True));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def train_audit(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);s=C.read(folder/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists();assert len(s['rounds'])==p['rounds']
    for g in C.read(folder/'worker_generations.json'):assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    allrows=checkrows(p,folder);by={r['path']:r for r in allrows};torch.set_num_threads(p['fit_threads']);obj=L.make(p);previous=digest(L.state(obj));started=time.time()
    def guard():C.verify(p);assert time.time()<min(started+p['audit_seconds']-5,p['absolute_all_stop']-5)
    for rec in s['rounds']:
        before=L.read_state(rec['start']);assert digest(before)==previous;L.load(obj,before)
        values=[(RN.Base.C.E.N.S.arrays(path),by[path]) for path in rec['training_paths']]
        assert {r['job_id'] for a,r in values}==set(p['training_ids']) and len(values)==24
        u=L.fit(obj,values,p,guard,event('independent',rec['round']));after=L.read_state(rec['trained']);assert u==rec['fit'] and digest(L.state(obj))==digest(after);previous=digest(after)
        saved=L.checkpoint(rec['model']['path']);assert digest(saved['joint_state']['actor'])==digest(obj[0].state_dict()) and obj[0].phash()==rec['model']['policy_hash']
    assert sum(r['physics'] for r in allrows)==s['physics']<=p['max_physics'];assert len(allrows)==144 and s['commands']==len(allrows)+14*p['rounds']<=p['max_commands']
    C.verify(p,force=True);C.save(folder/'audit.json',dict(state='PASS',rows=len(allrows),independent_Actor_Adam=12,independent_Value_Adam=12,full_arrays_prefix_owncarry_density_rewards_native_exact=True,all_optimizer_state_exact=True,new_native=0))

def evaluate(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);assert C.read(C.OUT/'training/audit.json')['state']=='PASS';model=C.read(C.OUT/'training/summary.json')['model']
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
