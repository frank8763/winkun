"""Complete causal episodes, actual time budget, identical Stage physics."""
from mem_actor import *
import os,json,time,math,io,gzip,traceback
sys.path.insert(0,str(ROOT/'stage/experiments/t238_ar_density_recovery'))
import copula as AR
digest,used,state_np=S.digest,S.used,S.state_np
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)
def generate(job,p,native,create,deadline,frame_hook=None):
    env=None
    try:
        env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job);assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();steps=0;cap=job['case']['step_cap'];previous=np.zeros(2,dtype=np.float32);previous_count=0;hidden=None;latent=None;prevmean=prevact=None;term=trunc=False;info={};records=[];rng=np.random.default_rng(job.get('seed',0))
        while not(term or trunc) and steps<cap and time.time()<deadline:
            x=packet(env,obs,steps,cap,previous,previous_count);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():
                mode,mean,hidden,raw=p.step(torch.tensor(x[None]),hidden);mean=mean[0]
                if job['mode']=='deterministic':action=mode[0].numpy();lp=0.;storedmean=np.zeros(2,dtype=np.float32)
                else:
                    first=latent is None;innovation=rng.normal(size=2);latent=innovation if first else AR.RHO*latent+math.sqrt(1-AR.RHO**2)*innovation;a=AR.sample(mean,latent);lp=float(AR.logp(mean[None],a[None],(mean if first else prevmean)[None],(a if first else prevact)[None],torch.tensor([first])));action=a.numpy();storedmean=mean.numpy();prevmean,prevact=mean,a
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc);reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            nxt=packet(env,obs,steps,cap,action,n);records.append((x,action,storedmean,lp,n,nxt,pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward));previous=action.copy();previous_count=n
            if frame_hook is not None:frame_hook(env,steps,info)
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'];a={k:np.asarray([r[i] for r in records]) for i,k in enumerate(names)};a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
        base={k:(v[:,:117] if k in ['states','next_states'] else v) for k,v in a.items()};row=dict(id=job['id'],source_reset_id=job.get('source_reset_id',job['id']),group=job['group'],mode=job['mode'],seed=job.get('seed',0),complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,remaining_m=d1,array_hashes={k:digest(v) for k,v in a.items()},base_array_hashes={k:digest(v) for k,v in base.items()},integration_sha256=R.arrays_hash(R.integration(env)),job={k:v for k,v in job.items() if k not in ['policy_np']})
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy);return dict(row=row,arrays=a)
    finally:
        if env is not None:env.close()
def worker(pipe,index,out,checkpoint,deadline):
    folder=Path(out)/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);parent=read(checkpoint)['policy'];p=None;ph=None;pipe.send(dict(ready=True,pid=os.getpid()))
        while True:
            j=pipe.recv()
            if j is None:break
            assert digest(j['policy_np'])==j['policy_hash']
            if ph!=(j['policy_hash'],j['memory_on']):
                p=Actor(parent,j['memory_on'],j['model_seed']);S.load_np(p,j['policy_np']);p.eval().requires_grad_(False);ph=(j['policy_hash'],j['memory_on'])
            pipe.send(generate(j,p,native,create,deadline))
    except EOFError:pass
    except BaseException:
        err=dict(error=traceback.format_exc());C.save(folder/'failure.json',err)
        try:pipe.send(err)
        except OSError:pass
        raise
    finally:pipe.close();log.close()
def pad(arrays):
    lengths=[len(a['states']) for a in arrays];t=max(lengths);b=len(arrays);x=torch.zeros(t,b,DIM);actions=torch.zeros(t,b,2);mask=torch.zeros(t,b,dtype=torch.bool)
    for j,a in enumerate(arrays):n=lengths[j];x[:n,j]=torch.tensor(a['states']);actions[:n,j]=torch.tensor(a['actions']);mask[:n,j]=True
    ti,bi=mask.nonzero(as_tuple=True);prev_full=(ti-1).clamp_min(0)*b+bi;reset=ti==0
    return dict(x=x,actions=actions,mask=mask,lengths=lengths,prev_full=prev_full,reset=reset)
def likelihood(p,data):
    mode,mean,raw=p.sequence(data['x']);flat=mean.reshape(-1,2);act=data['actions'];lp=AR.logp(mean[data['mask']],act[data['mask']],flat[data['prev_full']],act.reshape(-1,2)[data['prev_full']],data['reset']);return lp,mode
def training_batch(items,p):
    arrays=[i['arrays'] for i in items];d=pad(arrays);t,b=d['mask'].shape;record=torch.zeros(t,b,2);old=torch.zeros(t,b,dtype=torch.float64);adv=torch.zeros_like(old);weights=torch.zeros_like(old);tasks=torch.zeros(t,b,dtype=torch.long);ret=[S.monte_carlo(a['rewards'],a['native_counts']) for a in arrays];ids=sorted({i['row']['source_reset_id'] for i in items})
    for j,a in enumerate(arrays):
        n=len(a['states']);baseline=[ret[k][0] for k,r in enumerate(items) if k!=j and r['row']['source_reset_id']==items[j]['row']['source_reset_id']];assert len(baseline)==3;record[:n,j]=torch.tensor(a['means']);old[:n,j]=torch.tensor(a['logp']);adv[:n,j]=torch.tensor(ret[j]-np.mean(baseline));weights[:n,j]=torch.tensor(R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]);tasks[:n,j]=ids.index(items[j]['row']['source_reset_id'])
    mask=d['mask'];recorded=AR.logp(record[mask],d['actions'][mask],record.reshape(-1,2)[d['prev_full']],d['actions'].reshape(-1,2)[d['prev_full']],d['reset']);err=float((recorded-old[mask]).abs().max());assert err<1e-10
    with torch.no_grad():_,mean,_=p.sequence(d['x']);lp,_=likelihood(p,d);me=float((mean[mask]-record[mask]).abs().max());le=float((lp-old[mask]).abs().max());assert me<3e-6 and le<.005,(me,le)
    a=adv[mask];a/=a.square().mean().sqrt().clamp_min(.05);d.update(oldlp=lp.detach(),advantage=a,weights=weights[mask],task=tasks[mask],density=dict(recorded_error=err,sequence_mean_error=me,sequence_logp_error=le,same_parameter_ratio_error=0.));return d
