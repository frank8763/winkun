"""Read-only model/replay audit plus exact-state native soft-Q witnesses."""
import os
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU','10'))})
from pathlib import Path
import sys,json,time,math,hashlib,shutil,traceback,fcntl,inspect
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t202_cause_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t200_residual_placement'))
import common as C
from replay import Replay,GAMMA,build
import numpy as np
import torch

def used():return sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
def own():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
def save(name,v):C.save(OUT/name,v)
def dot(a,b):return float(torch.dot(a.double(),b.double()))
def cosine(a,b):return dot(a,b)/max(1e-30,float(a.double().norm()*b.double().norm()))
def vector(loss,params):
    g=torch.autograd.grad(loss,params,allow_unused=True)
    pieces=[]
    for p,x in zip(params,g):
        x=torch.zeros_like(p) if x is None else x.clone()
        if p.shape==(800,113):
            x[:,[57,58,74,75,86,87,98,99,110,111]]=0
        pieces.append(x.reshape(-1))
    return torch.cat(pieces)

def load_policy(plan,arm):
    ck=torch.load(plan['checkpoints'][arm],map_location='cpu',weights_only=False)
    p=C.Policy(arm).eval();p.load_state_dict(ck['policy'])
    q=C.Critic().eval().requires_grad_(False);q.load_state_dict(ck['critic'])
    qt=C.Critic().eval().requires_grad_(False);qt.load_state_dict(ck['critic_target'])
    return p,q,qt,ck

def offline(plan):
    result={};torch.manual_seed(plan['seed'])
    with np.load(plan['retention_targets']) as z:anchors=torch.tensor(z['states']);targets=torch.tensor(z['actions']);offsets=z['offsets']
    with np.load(ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz') as z:common=torch.tensor(z['states'])
    for arm in plan['checkpoints']:
        p,q,qt,ck=load_policy(plan,arm);replay=Replay(plan['seed']);receipts=json.loads((ROOT/f'stage/runs/t200_residual_placement/{arm}/replay_receipts.json').read_text());episode_info=[]
        for r in receipts:
            assert C.sha(r['path'])==r['sha256'];replay.add(r['path'])
            with np.load(r['path']) as z:
                a={k:z[k].copy() for k in z.files}
            flags=a.get('end_flags')
            if flags is None:
                ix=(a['steps']+a['native_counts']).astype(int)-1
                flags=a['physical_flags'][ix] if 'physical_flags' in a else a['physical'][ix,7:10]
            episode_info.append({'path':r['path'],'n':len(a['states']),'initial_distance_m':float(np.linalg.norm(a['poses'][0,:2]-a['goal_xy'])),'event':np.asarray(flags[-1],dtype=int).tolist()})
        replay.materialize();data=replay.orig;params=list(p.parameters());gradients=[]
        for repeat in range(4):
            torch.manual_seed(plan['seed']+repeat);x=replay.sample(512,'cpu');a,lp=p.sample(x['states']);qa,qb=q(x['states'],a);lq=-torch.minimum(qa,qb).mean();gq=vector(lq,params)
            torch.manual_seed(plan['seed']+repeat);_,lp=p.sample(x['states']);le=plan['alpha']*lp.mean();ge=vector(le,params)
            lk=sum((p(anchors[offsets[j]:offsets[j+1]])-targets[offsets[j]:offsets[j+1]]).square().mean() for j in range(7))/7*10.;gk=vector(lk,params)
            total=gq+ge+gk
            gradients.append({'repeat':repeat,'q_loss':float(lq.detach()),'alpha_logp_loss':float(le.detach()),'weighted_BC_loss':float(lk.detach()),'q_gradient_norm':float(gq.norm()),'entropy_gradient_norm':float(ge.norm()),'BC_gradient_norm':float(gk.norm()),'BC_Q_cosine':cosine(gk,gq),'entropy_Q_cosine':cosine(ge,gq),'BC_over_Q_gradient_norm':float(gk.norm()/gq.norm().clamp_min(1e-30)),'entropy_over_Q_gradient_norm':float(ge.norm()/gq.norm().clamp_min(1e-30)),'combined_gradient_Q_cosine':cosine(total,gq)})
        with torch.no_grad():
            samples=torch.tensor(data['states']);actions=torch.tensor(data['actions']);u,v=q(samples,actions);term=data['terminal'][:,0]>0
            deterministic=p(samples);drive=5*((deterministic[:,0]+1)/2).square();mean_change=(p(anchors)-targets).square().mean().sqrt()
            # True one-step TD residual, not a substitute for full value calibration.
            torch.manual_seed(plan['seed']);na,nlp=p.sample(torch.tensor(data['next_states']));ta,tb=qt(torch.tensor(data['next_states']),na);target=torch.tensor(data['rewards'])+torch.tensor(data['discounts'])*(torch.minimum(ta,tb)-plan['alpha']*nlp)
            quant=lambda x:np.quantile(np.asarray(x),[0,.1,.5,.9,1]).tolist()
            summary={'transitions':len(samples),'terminal_fraction_uniform':float(term.mean()),'terminal_probability_actual_sampler':.25+.75*float(term.mean()),'episode_start_events':episode_info,'sigma_latent':p.logstd.clamp(-3,0).exp().tolist(),'terminal_Q1_MAE':float((u[term]-torch.tensor(data['rewards'][term])).abs().mean()),'nonterminal_Q1_TD_MAE':float((u[~term]-target[~term]).abs().mean()),'Q1_Q2_abs_difference_mean':float((u-v).abs().mean()),'deterministic_native_throttle_quantiles':quant(deterministic[:,0]),'deterministic_wheel_command_quantiles':quant(drive),'deterministic_wheel_command_below_0_05_fraction':float((drive<.05).float().mean()),'retention_all_state_action_RMSE':float(mean_change),'common_parent_LEFT_throttle_quantiles':quant(p(common)[:,0])}
        ca=p(common).detach().requires_grad_(True);u,v=q(common,ca);ga=torch.autograd.grad(torch.minimum(u,v).sum(),ca)[0]
        summary['common_parent_LEFT_Q_prefers_lower_throttle_fraction']=float((ga[:,0]<0).float().mean());summary['common_parent_LEFT_Q_action_gradient_quantiles']=np.quantile(ga.detach().numpy(),[0,.1,.5,.9,1],axis=0).tolist()
        summary['gradient_decomposition']=gradients;result[arm]=summary
        assert all(torch.equal(v,p.state_dict()[k]) for k,v in ck['policy'].items())
    save('offline.json',{'state':'COMPLETE','physics_steps':0,'optimizer_updates':0,'models':result,'scope':'Checkpoint-local gradients, not exact old training batches or Adam contributions; TD and terminal errors alone do not certify Q action ranking.'})
    return result

def main():
    lock=(OUT/'audit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate'
    plan=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
    start=time.monotonic();deadline=start+plan['wall_seconds'];spent=0;env=None;rows=[];contracts=[]
    def guard():
        assert time.monotonic()<deadline,'Wall budget'
        assert spent<plan['max_physics'],'Physics budget'
        assert used()+plan['pending_branch_bytes']<1024**3 and own()+plan['pending_branch_bytes']<plan['peak_bytes']
        assert shutil.disk_usage(ROOT).free>40*1024**3
    def step(env,action):
        nonlocal spent
        assert spent<plan['max_physics']
        result=native.step_checked(env,action);spent+=1
        return result
    try:
        save('status.json',{'state':'OFFLINE_AUDIT','optimizer_updates':0});offline_result=offline(plan)
        native,create=C.native_setup(OUT/'native');case=plan['case'];cap=case['step_cap']
        for arm in plan['checkpoints']:
            p,q,_,ck=load_policy(plan,arm);p.requires_grad_(False);snapshots={};elapsed=0;records=[];env,obs,actual=create(case);env.max_steps=cap
            contracts.append({'arm':arm,'actual_seed':int(actual),'step_owner':env.step.__qualname__,'step_source':inspect.getsourcefile(env.step),'step_source_sha256':C.sha(inspect.getsourcefile(env.step)),'command_owner':env._policy_to_commands.__qualname__,'command_source':inspect.getsourcefile(env._policy_to_commands),'command_source_sha256':C.sha(inspect.getsourcefile(env._policy_to_commands)),'success_radius':float(env.success_radius),'frame_skip':int(env.frame_skip),'timestep':float(env.model.opt.timestep),'initial_goal':env.goal_xy.tolist(),'initial_pose':list(env._pose())})
            assert env.success_radius==.2 and env.frame_skip==1 and abs(env.model.opt.timestep-.002)<1e-12
            snapshots[0]=C.capture(env,obs);C.write_snapshot(OUT/f'{arm}_anchor_0.npz',snapshots[0])
            while elapsed<14000:
                guard();state=C.observation(obs,1-elapsed/cap)
                with torch.no_grad():action=np.tanh(p.latent_mean(torch.tensor(state[None])).numpy()[0]).astype(np.float32)
                records.append([elapsed,*env._pose(),*action])
                for _ in range(100):
                    obs,_,t,u,_=step(env,action);elapsed+=1;assert not(t or u),'No late anchor after early termination'
            snapshots[14000]=C.capture(env,obs);C.write_snapshot(OUT/f'{arm}_anchor_14000.npz',snapshots[14000]);env.close();env=None
            np.savez_compressed(OUT/f'{arm}_prefix.npz',trace=np.asarray(records))
            for absolute,snapshot in snapshots.items():
                state=C.observation(snapshot['obs'],1-absolute/cap);x=torch.tensor(state[None]);base=p(x).detach();aa=base.clone().requires_grad_(True);qu,qv=q(x,aa);grad=torch.autograd.grad(torch.minimum(qu,qv).sum(),aa)[0][0].numpy();direction=grad/max(1e-30,float(np.linalg.norm(grad)))
                actions={'baseline':base.numpy()[0],'q_positive':np.clip(base.numpy()[0]+plan['action_gradient_step']*direction,-.999999,.999999),'q_negative':np.clip(base.numpy()[0]-plan['action_gradient_step']*direction,-.999999,.999999)}
                for label,a0 in actions.items():
                    with torch.no_grad():qu,qv=q(x,torch.tensor(a0[None],dtype=torch.float32))
                    for seed in plan['rollout_noise_seeds']:
                        guard();env,_,_=create(case);obs=C.restore(env,snapshot);assert np.array_equal(obs,snapshot['obs']);env.max_steps=cap
                        # Verify the complete integration snapshot before every branch.
                        assert np.array_equal(C.capture(env,obs)['physics'],snapshot['physics'])
                        rng=np.random.default_rng(seed);sigma=p.logstd.clamp(-3,0).exp().numpy();steps=absolute;term=trunc=False;records=[];soft=environment=0.;id=f'{arm}_s{absolute}_{label}_n{seed}'
                        goal=env.goal_xy.copy();model=C.model_hash(env);initial_d=float(env._goal_state()[1]);entropy=0.
                        while not(term or trunc):
                            guard();before=steps;state=C.observation(obs,1-steps/cap);pose=np.asarray(env._pose()[:3]);distance=float(env._goal_state()[1]);logp=0.
                            if steps==absolute:action=np.asarray(a0,dtype=np.float32)
                            else:
                                with torch.no_grad():mu=p.latent_mean(torch.tensor(state[None])).numpy()[0]
                                eps=rng.normal(size=2);z=mu+sigma*eps;action=np.tanh(z).astype(np.float32)
                                logp=float(np.sum(-.5*eps**2-np.log(sigma)-.5*math.log(2*math.pi)-2*(math.log(2)-z-np.logaddexp(0,-2*z))))
                            for _ in range(min(100,cap-steps)):
                                obs,_,term,trunc,info=step(env,action);steps+=1
                                if term or trunc:break
                            n=steps-before;d1=float(env._goal_state()[1]);discount=GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
                            reward=-.000002*(1-discount)/(1-GAMMA)+.03*distance-discount*.03*d1*(not terminal)+(GAMMA**(n-1)*event if terminal else 0.)
                            w=GAMMA**(before-absolute);environment+=w*reward;entropy-=plan['alpha']*w*logp
                            records.append((before,state,action.copy(),n,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward,logp,pose))
                        expected=-.000002*(1-GAMMA**(steps-absolute))/(1-GAMMA)+GAMMA**(steps-absolute-1)*event+.03*initial_d
                        assert abs(environment-expected)<1e-8
                        assert C.model_hash(env)==model and np.array_equal(env.goal_xy,goal)
                        path=OUT/(id+'.npz');np.savez_compressed(path,steps=np.asarray([r[0] for r in records]),states=np.asarray([r[1] for r in records]),actions=np.asarray([r[2] for r in records]),native_counts=np.asarray([r[3] for r in records]),next_poses=np.asarray([r[4] for r in records]),end_flags=np.asarray([r[5] for r in records],dtype=np.uint8),rewards=np.asarray([r[6] for r in records]),continuation_logp=np.asarray([r[7] for r in records]),poses=np.asarray([r[8] for r in records]),goal_xy=goal)
                        assert path.stat().st_size<256*1024
                        row={'id':id,'arm':arm,'anchor_absolute_steps':absolute,'first_action_kind':label,'first_action':a0.tolist(),'Q1':float(qu),'Q2':float(qv),'minQ':min(float(qu),float(qv)),'noise_seed':seed,'physics_steps':steps-absolute,'environment_MC':environment,'continuation_entropy_MC':entropy,'soft_Q_MC':environment+entropy,'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc),'remaining_m':d1,'trace_sha256':C.sha(path),'anchor_physics_sha256':hashlib.sha256(snapshot['physics'].tobytes()).hexdigest()}
                        rows.append(row);save('status.json',{'state':'FROZEN_BRANCH_AUDIT','completed_branches':len(rows),'physics_steps':spent,'optimizer_updates':0})
                        with (OUT/'branches.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        print(json.dumps(row),flush=True);env.close();env=None
            assert all(torch.equal(v,p.state_dict()[k]) for k,v in ck['policy'].items())
        groups=[];pairs=[]
        for arm in plan['checkpoints']:
            for absolute in plan['anchor_elapsed_steps']:
                for label in plan['first_actions']:
                    subset=[r for r in rows if (r['arm'],r['anchor_absolute_steps'],r['first_action_kind'])==(arm,absolute,label)];means=np.asarray([r['soft_Q_MC'] for r in subset]);groups.append({'arm':arm,'anchor':absolute,'action':label,'n':len(subset),'minQ':subset[0]['minQ'],'mean_soft_MC':float(means.mean()),'sample_std_soft_MC':float(means.std(ddof=1)),'Q_minus_MC_mean':float(subset[0]['minQ']-means.mean()),'successes':sum(r['success'] for r in subset)})
                baseline=[r for r in rows if (r['arm'],r['anchor_absolute_steps'],r['first_action_kind'])==(arm,absolute,'baseline')]
                for label in ['q_positive','q_negative']:
                    subset=[r for r in rows if (r['arm'],r['anchor_absolute_steps'],r['first_action_kind'])==(arm,absolute,label)];delta=np.asarray([r['soft_Q_MC']-b['soft_Q_MC'] for r,b in zip(subset,baseline)]);pred=subset[0]['minQ']-baseline[0]['minQ'];pairs.append({'arm':arm,'anchor':absolute,'action':label,'predicted_delta':pred,'paired_MC_deltas':delta.tolist(),'mean_MC_delta':float(delta.mean()),'sample_std_MC_delta':float(delta.std(ddof=1)),'mean_sign_agrees':bool(pred*delta.mean()>0),'scope':'Four common-random-number samples, no significance or independent-map claim'})
        assert len(rows)==48 and spent<=plan['max_physics'] and own()<plan['max_bytes']
        for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
        save('contract_audit.json',contracts)
        result={'state':'COMPLETE','physics_steps':spent,'optimizer_updates':0,'training_physics_steps':0,'branches':len(rows),'groups':groups,'paired_differences':pairs,'all_input_hashes_unchanged':True,'all_new_runs_bytes':used(),'own_bytes_before_summary':own(),'scope':plan['value_contract']}
        save('summary.json',result);save('status.json',result);print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        save('failure.json',{'state':'FAILED','physics_steps':spent,'completed_branches':len(rows),'optimizer_updates':0,'traceback':traceback.format_exc()});raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
