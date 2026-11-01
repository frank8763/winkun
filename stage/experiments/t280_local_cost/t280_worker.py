"""Canonical batch1 Actor and conditional AR; complete/partial evidence returned."""
from t280_core import *
import contextlib,hashlib,io,json,os,shutil,time,traceback


class LimitedLog:
    def __init__(self,path):self.f=path.open('w');self.written=0;self.total=0
    def write(self,text):
        self.total+=len(text.encode());part=text[:max(0,65536-self.written)];self.f.write(part);self.f.flush();self.written+=len(part.encode());return len(text)
    def flush(self):self.f.flush()
    def close(self):self.f.close()


def generate(job,p,native,create,deadline,counts,index,previous_total):
    env=None;records=[];steps=0;error=None;term=trunc=False;info={};remaining=None;integration=None;initial=None
    try:
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job)
        assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();cap=job['case']['step_cap']
        previous=np.zeros(2,dtype=np.float32);previous_count=0;latent=None;prevmean=prevact=None;rng=np.random.default_rng(job.get('seed',0))
        initial=L.packet(env,obs,0,cap,previous,0);assert S.digest(initial)==job['expected_initial120_sha256'] if job.get('expected_initial120_sha256') else True
        while not(term or trunc) and steps<cap and time.time()<deadline:
            assert shutil.disk_usage(ROOT).free>=40*1024**3
            x=L.packet(env,obs,steps,cap,previous,previous_count);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():
                mode,mean,_,_=p.step(torch.tensor(x[None]));mean=mean[0]
                if job['mode']=='deterministic':action=mode[0].numpy();lp=0.;storedmean=np.zeros(2,dtype=np.float32)
                else:
                    first=latent is None;innovation=rng.normal(size=2);latent=innovation if first else L.AR.RHO*latent+math.sqrt(1-L.AR.RHO**2)*innovation
                    a=L.AR.sample(mean,latent);lp=float(L.AR.logp(mean[None],a[None],(mean if first else prevmean)[None],(a if first else prevact)[None],torch.tensor([first])))
                    action=a.numpy();storedmean=mean.numpy();prevmean,prevact=mean,a
            try:
                for _ in range(min(100,cap-steps)):
                    if time.time()>=deadline:break
                    obs,_,term,trunc,info=native.step_checked(env,action);steps+=1
                    if term or trunc:break
            except BaseException:error=traceback.format_exc()
            n=steps-before
            if n:
                d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc
                event=float(info.get('reach_goal',False))-float(info.get('collision',False))-.5*float(trunc)
                reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
                nxt=L.packet(env,obs,steps,cap,action,n)
                records.append((x,action,storedmean,lp,n,nxt,pose,np.asarray(env._pose()[:3]),[info.get('collision',False),info.get('reach_goal',False),trunc],reward))
                previous=action.copy();previous_count=n
            with counts.get_lock():counts[index]=previous_total+steps
            if error or not n:break
        remaining=float(env._goal_state()[1]);integration=R.arrays_hash(R.integration(env))
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
    finally:
        with counts.get_lock():counts[index]=previous_total+steps
        if env is not None:env.close()
    names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards']
    a={k:np.asarray([r[i] for r in records]) for i,k in enumerate(names)}
    a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
    assert len(records)>0,'Zero-decision initialization/deadline failure; no training'
    a['environment_rewards']=a['rewards'].copy();a['aux_costs']=endpoint_cost(a,cap);a['rewards']=a['environment_rewards']-job['aux_eta']*a['aux_costs']
    row=dict(id=job['id'],source_reset_id=job.get('source_reset_id',job['id']),group=job['group'],mode=job['mode'],seed=job.get('seed',0),
        complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),
        physics_steps=steps,remaining_m=remaining,array_hashes={k:S.digest(v) for k,v in a.items()},integration_sha256=integration,
        initial120_sha256=S.digest(initial),scene_sha256=scene,goal_xy=goal.tolist(),aux_eta=job['aux_eta'],
        unit_discounted_endpoint_cost=discounted_cost(a['aux_costs'],a['native_counts']),error=error,
        native_factory_stdout_sha256=hashlib.sha256(capture.getvalue().encode()).hexdigest())
    return dict(row=row,arrays=a)


def worker(pipe,index,out,plan,counts):
    folder=Path(out)/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log
    total=0
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);p=policy(plan).eval().requires_grad_(False);ph=S.digest(p.state_dict())
        pipe.send(dict(ready=True,pid=os.getpid(),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            assert S.digest(j['policy_np'])==j['policy_hash']
            if ph!=j['policy_hash']:S.load_np(p,j['policy_np']);ph=j['policy_hash']
            item=generate(j,p,native,create,j['deadline_epoch'],counts,index,total);total+=item['row']['physics_steps']
            item['row']['policy_hash']=ph;pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),completed_worker_physics=total);C.save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        C.save(folder/'log_accounting.json',dict(bytes_observed=log.total,bytes_written=log.written,limit=65536));pipe.close();log.close()
