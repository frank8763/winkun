"""Exclusive scientific child, complete fresh blocks and independent replay."""
from pathlib import Path
import sys,os,fcntl,gzip,json,time,copy,traceback
import importlib.util
_spec=importlib.util.spec_from_file_location('t721_runtime_controller',Path(__file__).resolve().parent/'run.py');C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
import t721_learning as L
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

def fit_affinity(folder,stage):
    for task in Path('/proc/self/task').iterdir():
        try:RN.Base.G._real_affinity(int(task.name),set(range(8,16)))
        except ProcessLookupError:pass
    torch.set_num_threads(8)
    torch.mm(torch.ones((256,256),dtype=torch.float64),torch.ones((256,256),dtype=torch.float64))
    masks={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in Path('/proc/self/task').iterdir()}
    assert masks and all(v==list(range(8,16)) for v in masks.values())
    C.save(folder/(stage+'_THREAD_AFFINITY.json'),dict(threads=masks,fit_threads=torch.get_num_threads(),cpu_set=list(range(8,16))))

def train(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);N.T.R.E.L.install_stop_handlers()
    lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rt=N.S.NativeRuntime(p);records=[];obj=L.make(p);error=None;started=time.time()
    current=p['learner_initial'];prior=None
    if p.get('previous_child'):
        prior=C.read(C.OUT/p['previous_child']/'summary.json');assert C.read(C.OUT/p['previous_child']/'audit.json')['state']=='PASS'
        L.load(obj,L.read_state(prior['final_state']));current=prior['model']
    initial_hash=digest(L.state(obj));body=L.fixed(obj[0]);C.save(folder/'science_started.json',dict(pid=os.getpid(),epoch=started,previous_child=p.get('previous_child'),initial_state_hash=initial_hash))
    try:
        for rnd in range(p['round_start'],p['round_start']+p['rounds']):
            assert time.time()+p['round_reserve_seconds']<min(p['absolute_learning_stop'],rt.deadline),'Next complete block admission denied'
            pool(rt);jobs=[j for pair in zip(p['training_original_jobs'],p['training_jobs']) for j in pair];seeds=[[p['seed'],rnd,i] for i in range(len(jobs))]
            values=rt.request(jobs,current,f'fresh{rnd}',True,seeds=seeds)
            indices=list(range(0 if rnd%2==0 else 1,24,2));exactjobs=[jobs[i] for i in indices]
            rt.request(exactjobs,current,f'exact{rnd}',True,[values[i][1] for i in indices],seeds=[seeds[i] for i in indices]);N.T.close(rt)
            assert all(Path(r['path']).stat().st_size<=p['train_file_bound'] for a,r in values)
            fit_affinity(folder,f'round{rnd}');before=L.state(obj);start=L.write_state(folder/f'start_{rnd}.pt.gz',before)
            def guard():C.verify(p);assert time.time()<min(p['absolute_learning_stop'],rt.deadline)-5
            u=L.fit(obj,values,p,guard,event('live',rnd));shadow=L.make(p);L.load(shadow,before);v=L.fit(shadow,values,p,guard,event('shadow',rnd))
            assert u==v and digest(L.state(obj))==digest(L.state(shadow));del shadow
            trained=L.write_state(folder/f'trained_{rnd}.pt.gz',L.state(obj));current=L.export(obj[0],p,folder/f'model_{rnd}.pt.gz')
            assert A.frozen(p,current['path']).phash()==obj[0].phash() and L.fixed(obj[0])==body
            rec=dict(round=rnd,start=start,trained=trained,model=current,fit=u,training_paths=[r['path'] for a,r in values]);records.append(rec);C.save(folder/'ROUNDS.json',records)
            C.save(C.OUT/'progress.json',dict(arm=p['arm'],child=name,rounds_in_child=len(records),completed_global_round=rnd,model=current,physics=sum(rt.counts),commands=rt.commands));torch.set_num_threads(1)
    except BaseException:error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        N.T.close(rt);C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands+7,rounds=records,model=current,final_state=records[-1]['trained'] if records else None,initial_state_hash=initial_hash,elapsed_seconds=time.time()-started,new_RL=True,no_oldtask_guard=True));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def train_audit(name):
    p=setup(name);folder=Path(p['out']);C.verify(p,force=True);sm=C.read(folder/'summary.json');assert sm['state']=='COMPLETE' and not Path('/proc',str(sm['pid'])).exists()
    assert len(sm['rounds'])==p['rounds'];rr=checkrows(p,folder);by={r['path']:r for r in rr};fit_affinity(folder,'audit');obj=L.make(p);previous=digest(L.state(obj));started=time.time()
    if p.get('previous_child'):
        prior=C.read(C.OUT/p['previous_child']/'summary.json');L.load(obj,L.read_state(prior['final_state']));previous=digest(L.state(obj))
    assert previous==sm['initial_state_hash']
    def guard():C.verify(p);assert time.time()<min(started+p['audit_seconds'],p['absolute_all_stop'])-5
    for rec in sm['rounds']:
        before=L.read_state(rec['start']);assert digest(before)==previous;L.load(obj,before)
        values=[(RN.Base.C.E.N.S.arrays(path),by[path]) for path in rec['training_paths']]
        assert len(values)==24 and {r['job_id'] for a,r in values}==set(p['training_ids'])
        assert sum(int(a['recovery_first'])==0 for a,r in values)==12
        u=L.fit(obj,values,p,guard,event('independent',rec['round']));after=L.read_state(rec['trained']);assert u==rec['fit'] and digest(L.state(obj))==digest(after);previous=digest(after)
        saved=L.checkpoint(rec['model']['path']);assert digest(saved['joint_state']['actor'])==digest(obj[0].state_dict()) and obj[0].phash()==rec['model']['policy_hash']
    assert len(rr)==36*p['rounds'] and sum(r['physics'] for r in rr)==sm['physics']<=p['max_physics'] and sm['commands']==len(rr)+14*p['rounds']<=p['max_commands']
    C.verify(p,force=True);C.save(folder/'audit.json',dict(state='PASS',rows=len(rr),Actor_Adam_independent=p['rounds']*p['epochs'],Value_Adam_independent=p['rounds']*p['epochs'],full_arrays_prefix_owncarry_density_rewards_native_exact=True,all_optimizer_state_exact=True,previous_child_exact=True,new_native=0))

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
