"""Stage native-domain truncated Gaussian and losslessly replayable episodes."""
import os
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU','0'))})
from pathlib import Path
import sys,time,json,math,hashlib,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t200_residual_placement'))
import common as C
import numpy as np
import torch
import mujoco
GAMMA=math.exp(-.002/60)
HEAD_KEYS=('body.net.4.weight','body.net.4.bias','native_head.weight','native_head.bias','direct_head.weight','direct_head.bias')
SIGMA=(.35,.30)

def center(raw):
    # Identity inside the action bounds; smooth bounded tails, no hard clip gradient.
    return torch.where(raw>1,1+.75*torch.tanh((raw-1)/.75),torch.where(raw < -1,-1+.75*torch.tanh((raw+1)/.75),raw))

def log_density(mean,action):
    mean=mean.double();action=action.double();sigma=torch.tensor(SIGMA,dtype=torch.float64,device=mean.device)
    lo=torch.special.ndtr((-1-mean)/sigma);hi=torch.special.ndtr((1-mean)/sigma);z=hi-lo
    assert bool((z>.001).all()) and bool(torch.isfinite(z).all())
    return (-.5*((action-mean)/sigma).square()-torch.log(sigma)-.5*math.log(2*math.pi)-torch.log(z)).sum(-1)

class Policy(C.OriginalPolicy):
    def physical_mean(self,x):return center(C.ParentPolicy.forward(self,x))
    def physical_logp(self,x,action):return log_density(self.physical_mean(x),action)
    def heads_only(self):
        self.requires_grad_(False)
        for k,p in self.named_parameters():
            if k in HEAD_KEYS:p.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

def heads(p):return {k:p.state_dict()[k].detach().cpu().numpy().copy() for k in HEAD_KEYS}
def set_heads(p,state):
    assert set(state)==set(HEAD_KEYS)
    with torch.no_grad():
        for k,v in state.items():p.state_dict()[k].copy_(torch.as_tensor(v,device=p.state_dict()[k].device))
def save_heads(path,p):np.savez_compressed(path,**heads(p));return C.sha(path)
def read_heads(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}
def integration(env):
    a=np.zeros(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,a,mujoco.mjtState.mjSTATE_INTEGRATION)
    return a
def arrays_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:h.update(np.asarray(a).tobytes())
    return h.hexdigest()

def draw(p,state,mode,rng):
    x=torch.tensor(state[None])
    with torch.no_grad():
        if mode=='deterministic':return p(x).numpy()[0].astype(np.float32),np.zeros(2,dtype=np.float32),0.
        if mode=='legacy_gaussian':
            mean=p.latent_mean(x).numpy()[0];eps=rng.normal(size=2);z=mean+.5*eps;a=np.tanh(z).astype(np.float32)
            lp=float(np.sum(-.5*eps**2-math.log(.5)-.5*math.log(2*math.pi)-2*(math.log(2)-z-np.logaddexp(0,-2*z))))
            return a,mean,lp
        assert mode=='physical_gaussian'
        mean=p.physical_mean(x).double()[0];sigma=torch.tensor(SIGMA,dtype=torch.float64)
        lo=torch.special.ndtr((-1-mean)/sigma);hi=torch.special.ndtr((1-mean)/sigma)
        u=torch.tensor(np.clip(rng.random(2),1e-12,1-1e-12),dtype=torch.float64)
        a=(mean+sigma*torch.special.ndtri(lo+u*(hi-lo))).clamp(-1,1).float()
        return a.numpy(),mean.float().numpy(),float(log_density(mean[None],a[None]))

def task_env(create,spec):
    env,obs,actual=create(spec['case']);env.max_steps=spec['case']['step_cap']
    assert env.success_radius==.2 and env.frame_skip==1 and abs(env.model.opt.timestep-.002)<1e-12
    if spec.get('exact_snapshot'):
        obs=C.restore(env,C.read_snapshot(spec['exact_snapshot']))
    elif spec.get('near_snapshot'):
        obs=C.restore(env,C.read_snapshot(spec['near_snapshot']))
        dx,dy,deg=spec.get('delta',[0,0,0]);env.data.qpos[:2]+=[dx,dy]
        rot=np.array([math.cos(math.radians(deg)/2),0,0,math.sin(math.radians(deg)/2)]);q=env.data.qpos[3:7].copy();mujoco.mju_mulQuat(env.data.qpos[3:7],rot,q)
        mujoco.mj_forward(env.model,env.data)
        env.step_count=0;env.prev_action=np.zeros(2,dtype=np.float32);env.history.clear();env.prev_goal_dist=float(env._goal_state()[1]);obs=np.asarray(env._observation(update_history=False),dtype=np.float32)
    env.max_steps=spec['case']['step_cap']
    return env,np.asarray(obs,dtype=np.float32),actual

def worker(pipe,index,root,checkpoint,deadline):
    out=Path(root);folder=out/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=C.native_setup(folder)
        p=Policy().eval().requires_grad_(False);p.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'])
        while True:
            command=pipe.recv()
            if command is None:break
            kind,job=command
            if kind=='make_task':
                env,obs,actual=task_env(create,job);valid=not env._collision();row={'id':job['id'],'valid':valid,'physics_steps':0,'pose':list(env._pose()),'goal_xy':env.goal_xy.tolist(),'remaining_m':float(env._goal_state()[1]),'model_hash':C.model_hash(env)}
                if valid:
                    path=out/'tasks'/(job['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();C.write_snapshot(path,C.capture(env,obs));row.update(snapshot=str(path),snapshot_sha256=C.sha(path))
                env.close();env=None;pipe.send(row);continue
            assert kind in ['rollout','replay']
            if kind=='rollout':
                set_heads(p,read_heads(job['heads']));env,obs,actual=task_env(create,job);assert not env._collision();goal=env.goal_xy.copy();model=C.model_hash(env);steps=0;term=trunc=False;records=[];rng=np.random.default_rng(job['seed']);cap=job['case']['step_cap'];nearest=float(env._goal_state()[1]);initial_d=nearest;info={};mode=job['mode']
                while not(term or trunc) and steps<cap and time.time()<deadline:
                    state=C.observation(obs,1-steps/cap);action,mean,lp=draw(p,state,mode,rng);before=steps;pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1])
                    for _ in range(min(100,cap-steps)):
                        obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                        if term or trunc:break
                    n=steps-before;d1=float(env._goal_state()[1]);discount=GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
                    reward=-.000002*(1-discount)/(1-GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(GAMMA**(n-1)*event if terminal else 0)
                    records.append((before,state,action,mean,lp,n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
                assert model==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
                complete=bool(term or trunc);a={k:np.asarray([r[i] for r in records]) for i,k in enumerate(['steps','states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'])}
                if records:
                    a['end_flags']=a['end_flags'].astype(np.uint8);a['native_counts']=a['native_counts'].astype(np.uint8);a['goal_xy']=goal
                retain=bool(records) and (not job.get('summary_only',False) or not info.get('reach_goal',False))
                if retain:
                    path=out/'pending'/(job['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists();np.savez_compressed(path,**a);assert path.stat().st_size<256*1024
                else:path=None
                g=float(np.sum(GAMMA**a['steps']*a['rewards'])) if complete else None
                row={'id':job['id'],'task':job['task'],'mode':mode,'seed':job['seed'],'heads':job['heads'],'heads_sha256':C.sha(job['heads']),'snapshot':job['exact_snapshot'],'snapshot_sha256':C.sha(job['exact_snapshot']),'physics_steps':steps,'decisions':len(records),'complete':complete,'success':bool(info.get('reach_goal',False)),'collision':bool(info.get('collision',False)),'timeout':bool(trunc),'remaining_m':float(env._goal_state()[1]),'nearest_m':nearest,'initial_distance_m':initial_d,'environment_return':g,'states_sha256':arrays_hash(a['states'],a['next_states']) if records else None,'trajectory_sha256':arrays_hash(a['actions'],a['poses'],a['next_poses'],a['native_counts'],a['end_flags']) if records else None,'final_integration_sha256':arrays_hash(integration(env)),'path':str(path) if path else None,'full_sha256':C.sha(path) if path else None,'initial_pose':job.get('initial_pose'),'arm':job.get('arm'),'round':job.get('round'),'role':job.get('role'),'case':job['case']}
                env.close();env=None;pipe.send(row)
            else:
                row=job;assert C.sha(row['path'])==row['full_sha256'];set_heads(p,read_heads(row['heads']))
                with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
                spec={'case':row['case'],'exact_snapshot':row['snapshot']};env,obs,_=task_env(create,spec);steps=0;states=[];next_states=[];pose_error=action_error=lp_error=0.;rng=np.random.default_rng(row['seed']);cap=row['case']['step_cap'];finished=True
                for i,action in enumerate(a['actions']):
                    if time.time()>=deadline:finished=False;break
                    state=C.observation(obs,1-steps/cap);states.append(state);expected,_,lp=draw(p,state,row['mode'],rng);action_error=max(action_error,float(np.abs(expected-action).max()))
                    if row['mode']!='legacy_gaussian':lp_error=max(lp_error,abs(lp-float(a['logp'][i])))
                    for _ in range(int(a['native_counts'][i])):obs,_,t,u,info=native.step_checked(env,action);steps+=1
                    next_states.append(C.observation(obs,1-steps/cap));pose_error=max(pose_error,float(np.abs(np.asarray(env._pose()[:3])-a['next_poses'][i]).max()))
                if finished:
                    assert arrays_hash(np.asarray(states),np.asarray(next_states))==row['states_sha256']
                    assert arrays_hash(integration(env))==row['final_integration_sha256']
                    assert pose_error==0 and action_error<2e-6 and lp_error<1e-5
                env.close();env=None;pipe.send({'id':row['id'],'verified':finished,'physics_steps':steps,'pose_error':pose_error,'action_error':action_error,'logp_error':lp_error})
    except(EOFError,SystemExit):pass
    except BaseException:
        try:pipe.send({'error':traceback.format_exc()})
        except OSError:pass
        raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()

def compact(row,verification,root):
    assert verification['verified'];path=Path(row['path']);assert C.sha(path)==row['full_sha256']
    with np.load(path) as z:a={k:z[k].copy() for k in z.files if k not in ['states','next_states']}
    target=Path(root)/'episodes'/path.name;target.parent.mkdir(exist_ok=True);assert not target.exists();np.savez_compressed(target,**a)
    receipt={'id':row['id'],'full_sha256':row['full_sha256'],'full_bytes':path.stat().st_size,'compact_path':str(target),'compact_sha256':C.sha(target),'compact_bytes':target.stat().st_size,'removed_fields':['states','next_states'],'replay_verified':verification,'states_sha256':row['states_sha256'],'restore_contract':'All actions, initial complete integration snapshot, frozen backbone, exact behavior heads, seed, counters and output hashes retained. Replay matched every recorded state/next_state and final integration hash before removing only owned redundant observation arrays.'}
    with (Path(root)/'compaction.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
    path.unlink();return receipt
