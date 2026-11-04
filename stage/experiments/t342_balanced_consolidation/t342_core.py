"""One balanced functional-replay consolidation, not new online RL.

Teachers exist only while making fixed supervised targets. One Actor deploys.
"""
from pathlib import Path
import sys,copy,gzip,io,json,os,time,traceback,shutil,contextlib,importlib.util,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t340_demo_supported_recovery'))
import t340_core as T
K,N,np,torch=T.K,T.N,T.np,T.torch
read,save,digest,arrays,packed=T.read,T.save,T.digest,T.arrays,T.packed
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t342_balanced_consolidation';DOC=ROOT/'stage/docs/review_20261104'

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

class Learner:
    def __init__(self,p):
        self.actor=T.Actor(p,'capg')
        v=torch.load(io.BytesIO(gzip.decompress(Path(p['student_initial_model']).read_bytes())),weights_only=False,map_location='cpu')
        self.actor.load_state_dict(dict(self.actor.state_dict(),**v['state']['actor']),strict=True)
        assert digest(self.actor.state_dict())==p['student_initial_policy_hash']
        self.opt=torch.optim.Adam(self.actor.correction.parameters(),lr=p['consolidation_lr']);self.calls=0
    def state(self):return dict(actor={k:v.detach().clone() for k,v in self.actor.state_dict().items() if not k.startswith('fixed.')},optimizer=copy.deepcopy(self.opt.state_dict()))
    def load(self,s):self.actor.load_state_dict(dict(self.actor.state_dict(),**s['actor']),strict=True);self.opt.load_state_dict(s['optimizer'])
    def hashes(self):return {k:digest(v) for k,v in self.state().items()}

def dataset(p,q,guard):
    with np.load(p['old_states']) as z:
        old=z['states'].copy();off=z['offsets'].copy();tasks=z['task'].copy()
    assert old.shape==(28629,120) and len(tasks)==211
    safe=K.policy(read(p['base_protocol'])).eval().requires_grad_(False)
    assert digest(safe.state_dict())==p['safe_policy_hash']
    labels=[]
    with torch.no_grad():
        for i in range(0,len(old),512):
            guard();labels.append(torch.cat([safe.step(torch.tensor(v[None]))[0] for v in old[i:i+512]]))
    labels=torch.cat(labels).double();groups=[];allx=[];ally=[]
    for t in range(8):
        ids=np.concatenate([np.arange(off[k],off[k+1]) for k in range(211) if tasks[k]==t])
        x=torch.tensor(old[ids]);y=labels[ids];allx.append(x);ally.append(y);groups.append(dict(name='old_task'+str(t),x=x,y=y,weight=.5/8))
    fresh=arrays(p['new_success_states']);assert fresh['states'].shape[1]==120
    assert fresh['end_flags'][-1,1] and not fresh['end_flags'][:,0].any() and not fresh['end_flags'][:-1].any()
    x=torch.tensor(fresh['states']);guard()
    with torch.no_grad():
        loc,_=q.stats(x);mode=q.step(x)[0]
        assert np.array_equal(mode.numpy(),fresh['actions'])
    groups.append(dict(name='new_wide6_deterministic_teacher',x=x,y=loc.detach(),weight=.5))
    for g in groups:
        guard();g['cache']=q.cache(g['x']);g['norm']=((g['x']-q.mean)/q.scale).clamp(-10,10)
    # Fixed teacher-state replay. New targets are continuous teacher locations;
    # old targets are executed safe modes, all in the same location coordinates.
    return groups

def predict(q,g,idx=None):
    # Training-only vectorized dense backward. Deployment and every evaluation
    # use the canonical batch1 path. The numerical tolerance is preflight-locked;
    # this supervised fit has no on-policy probability-density identity claim.
    norm=g['norm'] if idx is None else g['norm'][idx]
    b,sd=g['cache'] if idx is None else tuple(v[idx] for v in g['cache'])
    delta=q.correction(norm).double();margin=1+sd
    return margin*torch.tanh(torch.atanh(b/margin)+delta)

def fit(p,obj,groups,guard,event,steps=None):
    rng=np.random.default_rng(p['consolidation_seed']);trace=[]
    for k in range(p['consolidation_steps'] if steps is None else steps):
        guard();obj.opt.zero_grad(set_to_none=True);losses=[0.,0.]
        for j,g in enumerate(groups):
            idx=rng.integers(len(g['x']),size=p['new_rows'] if j==8 else p['old_rows_each'])
            loss=((predict(obj.actor,g,idx)-g['y'][idx]).square().mean())*g['weight']
            assert torch.isfinite(loss);loss.backward();losses[1 if j==8 else 0]+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(obj.actor.correction.parameters(),1.,error_if_nonfinite=True))
        event(dict(stage='BEFORE_ADAM',step=k));obj.opt.step();obj.calls+=1
        assert all(v.grad is None for v in obj.actor.fixed.parameters()) and obj.actor.logscale.grad is None
        row=dict(step=k,old_loss=losses[0],new_loss=losses[1],gradient_norm=norm)
        if k%p['hash_interval']==0 or k==(p['consolidation_steps'] if steps is None else steps)-1:row['hashes']=obj.hashes()
        event(dict(stage='AFTER_ADAM',**row));trace.append(row)
    return trace

def metrics(q,groups,guard):
    result=[]
    with torch.no_grad():
        for g in groups:
            error=[]
            for i in range(0,len(g['x']),512):
                guard();idx=np.arange(i,min(i+512,len(g['x'])));loc,_=q.stats(g['x'][idx],tuple(v[idx] for v in g['cache']))
                error.append((loc-g['y'][idx]).detach().numpy())
            e=np.concatenate(error);result.append(dict(name=g['name'],rows=len(e),location_MSE=float(np.mean(e**2)),location_max=float(np.max(np.abs(e)))))
    return result

def export(p,obj,path):
    b=io.BytesIO();torch.save(dict(state=obj.state(),hashes=obj.hashes(),training='fixed-balanced-supervised-replay',RL_updates=0),b)
    blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes'];Path(path).write_bytes(blob)
    return dict(path=str(path),bytes=len(blob),sha256=K.C.sha(path),hashes=obj.hashes())

def worker(pipe,index,p,counts,preflight=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=Learner(p).actor.eval().requires_grad_(False)
    if preflight:
        cmd=pipe.recv();q.load_state_dict({k:torch.tensor(v) for k,v in cmd['policy_np'].items()},strict=True)
        with torch.no_grad():v=q.step(torch.tensor(cmd['states']))[0].numpy()
        pipe.send(dict(policy_hash=digest(q.state_dict()),actions=v,pid=os.getpid()));pipe.close();return
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;w=None
    try:
        torch.optim.Adam.step=lambda *_a,**_kw:(_ for _ in ()).throw(AssertionError('No worker optimizer'))
        spec=importlib.util.spec_from_file_location('stage342_course_factory',ROOT/'stage/experiments/t331_policy_prefix_retry/recorder.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ACTIVE=read(p['course_protocol'])
        native,create,w,reward=m.setup(folder/'native');base=m.original_task_env
        def factory(create,job):
            env,obs,actual=base(create,job)
            if job.get('wide_course'):obs=m.widen(env,w)
            return env,obs,actual
        K.R.task_env=factory
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            job=pipe.recv()
            if job is None:break
            q.load_state_dict({k:torch.tensor(v) for k,v in job['policy_np'].items()},strict=True)
            if job.get('readback_model'):
                v=torch.load(io.BytesIO(gzip.decompress(Path(job['readback_model']).read_bytes())),weights_only=False,map_location='cpu')
                q.load_state_dict(dict(q.state_dict(),**v['state']['actor']),strict=True)
            assert digest(q.state_dict())==job['policy_hash']
            item=N.generate(job,q,native,create,job['deadline_epoch'],counts,index,total);total+=item['row']['physics_steps']
            item['row'].update(policy_hash=job['policy_hash'],role=job['role'],independent_model_readback=bool(job.get('readback_model')))
            w.events.clear();pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),physical_steps=counts[index]);save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        if w is not None:w.close()
        pipe.close();log.close()
