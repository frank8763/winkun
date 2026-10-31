"""Actual measured-steering policy gradients, same native commands and reward."""
from pathlib import Path
import sys,os,json,time,math,io,gzip,copy,traceback,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t250_actuator_policy_gradient'
sys.path.insert(0,str(ROOT/'stage/experiments/t251_actuator_recovery'))
import actuator_support as AS
S,C,R,np,torch=AS.S,AS.C,AS.R,AS.np,AS.torch
sys.path.insert(0,str(ROOT/'stage/experiments/t239_ar_policy_gradient'))
import ar_support as A
AR=A.AR;guarded_adam_step=A.guarded_adam_step
digest,used,state_np=S.digest,S.used,S.state_np
S.policy=AS.policy
read=AS.read

def anchors(plan,p):
    with np.load(plan['induced_states']) as z:x=z['states'].copy();off=z['offsets'];ok=z['success'];task=z['task']
    with np.load(plan['steering_context']) as z:e=z['encoders'].copy();assert np.array_equal(off,z['offsets'])
    x=np.concatenate([x,e],axis=1);groups=[torch.tensor(np.concatenate([x[off[k]:off[k+1]] for k in range(211) if ok[k] and task[k]==t])[::plan['anchor_stride']]) for t in range(8)];xx=torch.cat(groups);offset=np.cumsum([0]+[len(g) for g in groups]).tolist()
    with torch.no_grad():y=p(xx).detach()
    return xx,y,offset

def batch(items,p,plan):
    d=A.batch(items,p,plan);names=sorted({i['row']['source_reset_id'] for i in items});d['task_labels']=torch.tensor(np.concatenate([np.full(len(i['arrays']['states']),names.index(i['row']['source_reset_id']),dtype=np.int64) for i in items]));d['task_count']=len(names);return d

def generate(job,p,native,create,deadline):
    if job['mode']=='deterministic':return AS.generate(dict(job,hold_steps=100),p,native,create,deadline)
    env=None
    try:
        env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job);assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();rng=np.random.default_rng(job['seed']);latent=None;prevmean=prevact=None
        steps=0;cap=job['case']['step_cap'];term=trunc=False;info={};records=[];nearest=initial=float(env._goal_state()[1])
        while not(term or trunc) and steps<cap and time.time()<deadline:
            state=AS.packet(env,obs,steps,cap);first=latent is None;innovation=rng.normal(size=2);latent=innovation if first else AR.RHO*latent+math.sqrt(1-AR.RHO**2)*innovation
            with torch.no_grad():
                mean=p.physical_mean(torch.tensor(state[None]))[0];act=AR.sample(mean,latent)
                lp=float(AR.logp(mean[None],act[None],(mean if first else prevmean)[None],(act if first else prevact)[None],torch.tensor([first])))
            action=act.numpy();before=steps;d0=float(env._goal_state()[1]);pose=np.asarray(env._pose()[:3])
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            records.append((state,action,mean.numpy(),lp,n,AS.packet(env,obs,steps,cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward,latent.copy()));prevmean,prevact=mean,act
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
            if ph!=j['policy_hash']:p=S.load_np(AS.policy(),j['policy_np']).eval().requires_grad_(False);ph=j['policy_hash']
            item=generate(j,p,native,create,deadline);item['row'].update(policy_hash=ph,job={k:v for k,v in j.items() if k!='policy_np'});pipe.send(item)
    except EOFError:pass
    except BaseException:
        error=dict(error=traceback.format_exc());C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
        raise
    finally:pipe.close();log.close()

def project(gradients,seed):
    rng=np.random.default_rng(seed);result=[];conflicts=0
    for i,gi in enumerate(gradients):
        q=gi.clone()
        for j in rng.permutation(len(gradients)):
            if i==j:continue
            g=gradients[j];dot=torch.dot(q,g)
            if float(dot)<0:q-=dot/g.square().sum().clamp_min(1e-30)*g;conflicts+=1
        result.append(q)
    return torch.stack(result).mean(0),conflicts
def update(p,opt,data,ax,ay,offset,plan,round_index,guard):
    rng=np.random.default_rng(plan['seed']+round_index);ind=np.concatenate([rng.integers(offset[i],offset[i+1],plan['anchor_batch_each']) for i in range(8)]);sx=ax[ind];sy=ay[ind];params=[q for q in p.parameters() if q.requires_grad];reports=[]
    flat=lambda gg:torch.cat([g.reshape(-1) for g in gg])
    def measure():
        m=p.physical_mean(data['x']);lp=AR.logp(m,data['actions'],m[data['prev']],data['actions'][data['prev']],data['reset']);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];terms=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights'];surrogates=[terms[data['task_labels']==t].sum()/4/234 for t in range(data['task_count'])];surr=torch.stack(surrogates).mean();ret=plan['retention_weight']*(p(sx)-sy).square().mean();return surr-ret,surr,delta,surrogates,ret
    for k in range(plan['actor_attempts_per_round']):
        guard();obj,_,_,surrogates,ret=measure();before=float(obj.detach());grads=[flat(torch.autograd.grad(-s,params,retain_graph=True)) for s in surrogates];retg=flat(torch.autograd.grad(ret,params));combined,conflicts=project(grads,plan['seed']+100*round_index+k);combined+=retg;opt.zero_grad();pos=0
        for q in params:q.grad=combined[pos:pos+q.numel()].view_as(q).clone();pos+=q.numel()
        sensor_grad=float(p.body.net[0].proprio.weight.grad.norm());first=float(p.body.net[0].weight.grad.norm());gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        def evaluate(q):
            try:o,s,d,_,_=measure()
            except AssertionError as e:return dict(accept=False,numeric_domain_rejection=str(e))
            err=(q(ax)-ay).abs();mse=torch.stack([err[offset[i]:offset[i+1]].square().mean() for i in range(8)]).mean();q99=torch.quantile(err.flatten(),.99);mx=err.max();k3=(d.exp()-1-d).mean();dl=d.abs().max();ok=bool(torch.isfinite(o)) and float(o)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio'] and float(mse)<=plan['max_anchor_MSE'] and float(q99)<=plan['max_anchor_q99'] and float(mx)<=plan['max_anchor_error']
            return dict(accept=ok,objective=float(o),surrogate=float(s),improvement=float(o)-before,conditional_empirical_k3=float(k3),max_abs_log_ratio=float(dl),balanced_anchor_MSE=float(mse),anchor_q99=float(q99),anchor_max=float(mx))
        result=guarded_adam_step(p,opt,evaluate,plan['step_factors']);result.update(step=k,pcgrad_negative_projection_count=conflicts,encoder_gradient_norm=sensor_grad,first_layer_gradient_norm=first,gradient_norm_before_clip=gn,policy_hash=digest(p.state_dict()),optimizer_hash=digest(opt.state_dict()));reports.append(result)
        if not result['accepted']:break
    return reports
