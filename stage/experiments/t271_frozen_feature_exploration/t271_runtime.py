"""Paired frozen exploration: existing AR versus normalized feature-copula noise."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import *
sys.path.insert(0,str(ROOT/'stage/experiments/t270_feature_noise_contract'))
import feature_noise as F
import math,os,time,traceback


def generate(j,p,native,create,deadline):
    if j['mode']!='feature_copula':return L.generate(j,p,native,create,deadline)
    env=None
    try:
        env,obs,_=R.task_env(create,j);obs=C.perturb_anchor(env,obs,j);assert not env._collision()
        scene=C.model_hash(env);goal=env.goal_xy.copy();steps=0;cap=j['case']['step_cap']
        previous=np.zeros(2,dtype=np.float32);previous_count=0;term=trunc=False;info={};records=[];extra=[];matrices=[]
        rng=np.random.default_rng(j['seed'])
        while not(term or trunc) and steps<cap and time.time()<deadline:
            x=L.packet(env,obs,steps,cap,previous,previous_count);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():
                _,mean,hidden,_=p.step(torch.tensor(x[None]));mean=mean[0];feature=F.unit_feature(hidden)
                if len(records)%F.BLOCK==0:
                    W=torch.tensor(rng.normal(size=(32,2)),dtype=torch.float64)*math.sqrt(F.LATENT_VAR)
                    matrices.append(W.numpy().copy());pm,P=F.prior(feature)
                innovation=rng.normal(size=2);latent=F.latent_sample(feature,W,innovation)
                action=F.AR.sample(mean,latent);lp,pm,P=F.logp_step(mean,action,feature,pm,P)
                a=action.numpy();stored=mean.numpy()
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,a);steps+=1
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc
            event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            nxt=L.packet(env,obs,steps,cap,a,n)
            records.append((x,a,stored,float(lp),n,nxt,pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
            extra.append((feature.numpy().copy(),latent.numpy().copy(),innovation.copy()));previous=a.copy();previous_count=n
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards']
        arrays={key:np.asarray([r[i] for r in records]) for i,key in enumerate(names)}
        arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
        arrays.update(feature_vectors=np.asarray([e[0] for e in extra]),latent_draws=np.asarray([e[1] for e in extra]),independent_innovations=np.asarray([e[2] for e in extra]),block_weights=np.asarray(matrices))
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
        recovered=F.sequence_logp(torch.tensor(arrays['means']),torch.tensor(arrays['actions']),torch.tensor(arrays['feature_vectors']))
        error=float((recovered-torch.tensor(arrays['logp'])).abs().max());assert error==0
        row=dict(id=j['id'],source_reset_id=j['source_reset_id'],group=j['group'],mode=j['mode'],seed=j['seed'],
            complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,remaining_m=d1,
            array_hashes={k:S.digest(v) for k,v in arrays.items()},integration_sha256=R.arrays_hash(R.integration(env)),
            job={k:v for k,v in j.items() if k!='policy_np'},conditional_density_reconstruction_error=error,
            no_optimizer_or_actor_change=True,training_time_stochastic_behavior_not_deterministic_acceptance=True)
        return dict(row=row,arrays=arrays)
    finally:
        if env is not None:env.close()


def worker(pipe,index,output,plan,deadline):
    folder=Path(output)/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True)
    log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        native,create=S.native_setup(folder);p=policy(plan).eval().requires_grad_(False)
        assert S.digest(p.state_dict())==plan['policy_hash'];pipe.send(dict(ready=True,pid=os.getpid()))
        while True:
            j=pipe.recv()
            if j is None:break
            if j.get('geometry_only'):
                env=None
                try:
                    env,obs,_=R.task_env(create,j);obs=C.perturb_anchor(env,obs,j)
                    pipe.send(dict(id=j['source_reset_id'],valid=not env._collision(),physics_steps=0,
                        pose=list(env._pose()[:3]),goal_xy=env.goal_xy.tolist(),scene_sha256=C.model_hash(env),integration_sha256=R.arrays_hash(R.integration(env))))
                finally:
                    if env is not None:env.close()
            else:
                item=generate(j,p,native,create,deadline);item['row']['policy_hash']=plan['policy_hash'];pipe.send(item)
    except EOFError:pass
    except BaseException:
        error=dict(error=traceback.format_exc());C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
    finally:pipe.close();log.close()
