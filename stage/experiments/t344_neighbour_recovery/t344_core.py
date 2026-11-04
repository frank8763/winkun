"""Bounded acquisition on two fixed neighbouring start families.

Standard CAPG/PPO with successful trajectory supervision; no novelty claim.
"""
from pathlib import Path
import sys,copy,gzip,io,json,os,time,traceback,shutil,math,contextlib,importlib.util
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t340_demo_supported_recovery'))
import t340_core as T
K,N,np,torch=T.K,T.N,T.np,T.torch
read,save,digest,arrays,packed=T.read,T.save,T.digest,T.arrays,T.packed
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t344_neighbour_recovery';DOC=ROOT/'stage/docs/review_20261104'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

class Learner(T.Learner):
    def __init__(self,p,arm='capg'):
        super().__init__(p,arm)
        v=torch.load(io.BytesIO(gzip.decompress(Path(p['student_initial_model']).read_bytes())),weights_only=False,map_location='cpu')
        self.actor.load_state_dict(dict(self.actor.state_dict(),**v['state']['actor']),strict=True)
        assert digest(self.actor.state_dict())==p['student_initial_policy_hash']
        self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0

def targets(p,q,source,successes=()):
    assert source['row']['success'] and source['row']['complete'] and not source['row']['collision']
    return T.demos(p,q,[source,*successes])

def setup(folder,p):
    spec=importlib.util.spec_from_file_location('stage344_real_history_factory',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ACTIVE=read(p['course_protocol'])
    native,create,witness,reward=m.setup(folder);base=m.original_task_env
    def factory(create,job):
        env,obs,actual=base(create,job)
        if job.get('wide_course'):obs=m.widen(env,witness)
        return env,obs,actual
    K.R.task_env=factory
    return m,native,create,witness,reward

def geometry(env,obs,witness,cap):
    x=K.T.L.packet(env,obs,0,cap,np.zeros(2,np.float32),0);hp,hg,ht=witness.get(env)
    assert np.max(np.abs(K.H.rebuild(x,np.asarray(env._pose()[:3]),hp,env.goal_xy,hg)-x))<=2e-6
    return dict(initial120_sha256=digest(x),integration_sha256=K.R.arrays_hash(K.R.integration(env)),scene_sha256=K.C.model_hash(env),
        full120=x.tolist(),history_poses=hp.tolist(),history_goals=hg.tolist(),history_times=ht.tolist(),
        initial_native_integration=K.R.integration(env).tolist(),valid=not env._collision(),max_wheel=float(env.max_wheel))

def worker(pipe,index,p,physical,preflight=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=Learner(p).actor.eval().requires_grad_(False)
    if preflight:
        cmd=pipe.recv();q.load_state_dict({k:torch.tensor(v) for k,v in cmd['policy_np'].items()},strict=True)
        with torch.no_grad():out=q.draw(torch.tensor(cmd['states']),torch.tensor(cmd['z']),False)
        pipe.send(dict(outputs={k:v.numpy() for k,v in out.items()},policy_hash=digest(q.state_dict()),pid=os.getpid()));pipe.close();return
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log
    total=0;local=0;s=None;witness=None;draws=[];innovations=[]
    try:
        torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('No native worker optimizer'))
        m,native,create,witness,reward=setup(folder/'native',p)
        geo=read(OUT/'geometry.json');course=read(p['course_protocol'])
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            job=pipe.recv()
            if job is None:break
            q.load_state_dict({k:torch.tensor(v) for k,v in job['policy_np'].items()},strict=True)
            if job.get('readback_model'):
                v=torch.load(io.BytesIO(gzip.decompress(Path(job['readback_model']).read_bytes())),weights_only=False,map_location='cpu')
                q.load_state_dict(dict(q.state_dict(),**v['state']['actor']),strict=True)
            assert digest(q.state_dict())==job['policy_hash']
            if not job.get('wide_course'):
                item=N.generate(job,q,native,create,job['deadline_epoch'],physical,index,total);total+=item['row']['physics_steps']
                item['row'].update(role=job['role'],job_id=job['source_reset_id'],policy_hash=job['policy_hash']);witness.events.clear();pipe.send(item);continue
            local=0;draws=[];innovations=[];lastcheck=0.
            def guard():
                nonlocal lastcheck
                assert time.time()<job['deadline_epoch']
                if time.time()-lastcheck>1:
                    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes'];lastcheck=time.time()
            def count():
                nonlocal local
                local+=1
                if local%100==0:
                    with physical.get_lock():physical[index]=total+local
            cp=copy.deepcopy(course);cp['job']={k:v for k,v in job.items() if k not in ['policy_np','replay_arrays']}
            cp['initial_geometry']=geo[job['source_reset_id']]
            s=m.R.Session(cp,native,create,witness,reward,guard,count)
            assert s.steps==0 and s.env.max_steps==40000 and s.env.max_wheel==5.
            assert np.array_equal(K.R.integration(s.env),np.array(geo[job['source_reset_id']]['initial_native_integration']))
            rng=np.random.default_rng(job['seed']);source=job.get('replay_arrays');k=0
            while not(s.term or s.trunc):
                x=K.T.L.packet(s.env,s.obs,s.steps,40000,s.previous,s.pn)
                with torch.no_grad():
                    z=torch.tensor(rng.normal(size=(1,2)),dtype=torch.float64);draw=q.draw(torch.tensor(x[None]),z,job['deterministic']);action=draw['actions'][0].numpy()
                if source is not None:assert np.array_equal(source['actions'][k],action)
                s.step(action);draws.append({name:v[0].numpy() for name,v in draw.items() if name!='actions'});innovations.append(z[0].numpy());k+=1
            a=s.arrays();a.update(initial_native_integration=s.initial.copy(),final_native_integration=K.R.integration(s.env).copy(),innovations=np.asarray(innovations))
            a.update({name:np.asarray([v[name] for v in draws]) for name in draws[0]})
            r=dict(id=job['id'],job_id=job['source_reset_id'],role=job['role'],seed=job['seed'],deterministic=job['deterministic'],policy_hash=job['policy_hash'],
                physics_steps=local,complete=bool(s.term or s.trunc),success=bool(s.info.get('reach_goal')) and not bool(s.info.get('collision')),collision=bool(s.info.get('collision')),timeout=bool(s.trunc),
                remaining_m=float(s.env._goal_state()[1]),scene_sha256=s.scene,initial120_sha256=digest(s.initial120),integration_sha256=K.R.arrays_hash(K.R.integration(s.env)),
                goal_xy=s.goal.tolist(),zero_human_prefix=True,array_hashes={name:digest(v) for name,v in a.items()},independent_model_readback=bool(job.get('readback_model')))
            if source is not None:assert a.keys()==source.keys() and all(np.array_equal(v,source[name]) for name,v in a.items())
            total+=local
            with physical.get_lock():physical[index]=total
            s.close();s=None;pipe.send(dict(row=r,arrays=a))
    except EOFError:pass
    except BaseException:
        error=traceback.format_exc()
        if s is not None:
            try:
                a=s.arrays();a.update(final_native_integration=K.R.integration(s.env).copy())
                if draws and len(draws)==len(a['states']):a.update({name:np.asarray([v[name] for v in draws]) for name in draws[0]});a['innovations']=np.asarray(innovations)
                b=io.BytesIO();np.savez_compressed(b,**packed(a));blob=b.getvalue();assert len(blob)<=p['partial_bytes'];(folder/'partial.npz').write_bytes(blob)
            except BaseException:save(folder/'partial_save_failure.json',dict(error=traceback.format_exc()))
            with physical.get_lock():physical[index]=total+local
        failure=dict(error=error,actual_physics=physical[index]);save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        if s is not None:s.close()
        if witness is not None:witness.close()
        pipe.close();log.close()
