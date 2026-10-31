"""Stage single-Actor fresh conditional-AR policy gradients, no old critic."""
from pathlib import Path
import sys,os,json,time,math,io,gzip,copy,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t239_ar_policy_gradient'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
sys.path.insert(0,str(ROOT/'stage/experiments/t238_ar_density_recovery'))
import copula as AR
sys.path.insert(0,str(ROOT/'stage/experiments'))
from native_update_guard import guarded_adam_step
digest,used,state_np=S.digest,S.used,S.state_np
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def generate(job,p,native,create,deadline):
    if job['mode']=='deterministic':return V.B.generate(job,p,native,create,deadline)
    env=None
    try:
        env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job);assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();rng=np.random.default_rng(job['seed']);latent=None;prevmean=prevact=None
        steps=0;cap=job['case']['step_cap'];term=trunc=False;info={};records=[];nearest=initial=float(env._goal_state()[1])
        while not(term or trunc) and steps<cap and time.time()<deadline:
            state=C.observation(obs,1-steps/cap);first=latent is None;innovation=rng.normal(size=2);latent=innovation if first else AR.RHO*latent+math.sqrt(1-AR.RHO**2)*innovation
            with torch.no_grad():
                mean=p.physical_mean(torch.tensor(state[None]))[0];act=AR.sample(mean,latent)
                lp=float(AR.logp(mean[None],act[None],(mean if first else prevmean)[None],(act if first else prevact)[None],torch.tensor([first])))
            action=act.numpy();before=steps;d0=float(env._goal_state()[1]);pose=np.asarray(env._pose()[:3])
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            records.append((state,action,mean.numpy(),lp,n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward,latent.copy()));prevmean,prevact=mean,act
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards','latents'];arrays={k:np.asarray([r[i] for r in records]) for i,k in enumerate(names)};arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
        row={k:job[k] for k in ['id','arm','seed','mode','group','round','source_reset_id']};row.update(complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,initial_distance_m=initial,remaining_m=d1,nearest_m=nearest,array_hashes={k:digest(a) for k,a in arrays.items()},integration_sha256=R.arrays_hash(R.integration(env)),environment_return=float(S.monte_carlo(arrays['rewards'],arrays['native_counts'])[0]))
        assert C.model_hash(env)==scene and np.array_equal(goal,env.goal_xy);return dict(row=row,arrays=arrays)
    finally:
        if env is not None:env.close()
def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);pipe.send(dict(ready=True,pid=os.getpid()));p=None;ph=None
        while True:
            j=pipe.recv()
            if j is None:break
            assert digest(j['policy_np'])==j['policy_hash']
            if ph!=j['policy_hash']:p=S.load_np(R.Policy(),j['policy_np']).eval().requires_grad_(False);ph=j['policy_hash']
            item=generate(j,p,native,create,deadline);item['row'].update(policy_hash=ph,job={k:v for k,v in j.items() if k!='policy_np'});pipe.send(item)
    except EOFError:pass
    except BaseException:
        error=dict(error=traceback.format_exc());C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
        raise
    finally:pipe.close();log.close()
def anchors(plan,p):
    with np.load(plan['retention_targets']) as z:x=torch.tensor(z['states']);offset=z['offsets']
    groups=[[x[offset[i]:offset[i+1]]] for i in range(7)]
    with np.load(plan['original_left_states']) as z:groups.append([torch.tensor(z['states'])])
    for r in plan['retained_sources']:
        with np.load(r['path']) as z:groups[r['task']].append(torch.tensor(z['states']))
    groups=[torch.cat(g)[::plan['anchor_stride']] for g in groups];x=torch.cat(groups);offset=np.cumsum([0]+[len(g) for g in groups]).tolist()
    with torch.no_grad():y=p(x).detach()
    return x,y,offset
def batch(items,p,plan):
    aa=[r['arrays'] for r in items];x=torch.tensor(np.concatenate([a['states'] for a in aa]));actions=torch.tensor(np.concatenate([a['actions'] for a in aa]));record=torch.tensor(np.concatenate([a['means'] for a in aa]));old=torch.tensor(np.concatenate([a['logp'] for a in aa]));reset=[];prev=[];off=0;returns=[];weights=[];g0=[]
    for a in aa:
        n=len(a['states']);prev.extend([off]+list(range(off,off+n-1)));reset.extend([True]+[False]*(n-1));off+=n;g=S.monte_carlo(a['rewards'],a['native_counts']);returns.append(g);g0.append(float(g[0]));weights.extend((R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]).tolist())
    advantage=[]
    for i,item in enumerate(items):
        other=[g0[j] for j,r in enumerate(items) if j!=i and r['row']['source_reset_id']==item['row']['source_reset_id']];assert len(other)==3
        advantage.extend((returns[i]-np.mean(other)).tolist())
    advantage=torch.tensor(advantage,dtype=torch.float64);advantage/=advantage.square().mean().sqrt().clamp_min(.05)
    prev=torch.tensor(prev);reset=torch.tensor(reset);weights=torch.tensor(weights,dtype=torch.float64)
    with torch.no_grad():
        check=AR.logp(record,actions,record[prev],actions[prev],reset);error=float((check-old).abs().max());assert error<1e-10
        mean=p.physical_mean(x);lp=AR.logp(mean,actions,mean[prev],actions[prev],reset);mean_error=float((mean-record).abs().max());lp_error=float((lp-old).abs().max());assert mean_error<=plan['batch_mean_tolerance'] and lp_error<=plan['batch_logp_tolerance'],(mean_error,lp_error)
    return dict(x=x,actions=actions,prev=prev,reset=reset,oldlp=lp.detach(),advantage=advantage,weights=weights,episodes=len(items),density_check=dict(recorded_density_max_error=error,batch_mean_max_error=mean_error,batch_logp_max_error=lp_error,same_parameter_ratio_error=0.))
def update(p,opt,data,ax,ay,offset,plan,round_index,guard):
    rng=np.random.default_rng(plan['seed']+round_index);inds=np.concatenate([rng.integers(offset[i],offset[i+1],plan['anchor_batch_each']) for i in range(8)]);sx=ax[inds];sy=ay[inds];reports=[]
    def measure():
        m=p.physical_mean(data['x']);lp=AR.logp(m,data['actions'],m[data['prev']],data['actions'][data['prev']],data['reset']);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];surrogate=(torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights']).sum()/data['episodes']/234
        mse=(p(sx)-sy).square().mean();return surrogate-plan['retention_weight']*mse,surrogate,delta
    for k in range(plan['actor_attempts_per_round']):
        guard();objective,_,_=measure();before=float(objective.detach());opt.zero_grad();(-objective).backward();first_grad=float(p.body.net[0].weight.grad.norm());gn=float(torch.nn.utils.clip_grad_norm_([q for q in p.parameters() if q.requires_grad],1.,error_if_nonfinite=True))
        def evaluate(q):
            try:obj,surr,delta=measure()
            except AssertionError as e:return dict(accept=False,numeric_domain_rejection=str(e))
            errors=(q(ax)-ay).abs();mse=torch.stack([errors[offset[i]:offset[i+1]].square().mean() for i in range(8)]).mean();q99=torch.quantile(errors.flatten(),.99);mx=errors.max();k3=(delta.exp()-1-delta).mean();dl=delta.abs().max()
            accept=bool(torch.isfinite(obj)) and float(obj)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio'] and float(mse)<=plan['max_anchor_MSE'] and float(q99)<=plan['max_anchor_q99'] and float(mx)<=plan['max_anchor_error']
            return dict(accept=accept,objective=float(obj),surrogate=float(surr),improvement=float(obj)-before,conditional_empirical_k3=float(k3),max_abs_log_ratio=float(dl),balanced_anchor_MSE=float(mse),anchor_q99=float(q99),anchor_max=float(mx))
        result=guarded_adam_step(p,opt,evaluate,plan['step_factors']);result.update(step=k,first_layer_gradient_norm=first_grad,gradient_norm_before_clip=gn,policy_hash=digest(p.state_dict()),optimizer_hash=digest(opt.state_dict()));reports.append(result)
        if not result['accepted']:break
    return reports
