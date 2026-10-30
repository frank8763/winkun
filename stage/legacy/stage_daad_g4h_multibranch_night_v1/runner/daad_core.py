#!/usr/bin/env python3
"""STAGE ONLY: experimental Dynamics-Aware Adaptive-Duration SAC (DAAD-SAC).

Trains true joint continuous (drive,steering) + categorical duration policy with
semi-Markov soft Bellman backups. Observation-derived dynamics history only.
This is an ALGORITHM FEASIBILITY PILOT; not a validated journal contribution.
No planner, no privileged observation, no qpos override, no Phase writes.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HOME = Path.home()
REL = Path('training/e2e_phase10_lidar_v7_curriculum')
ENTRY = 't12a0m27_frontwheel_risk_sac.py'
PARENT = HOME / 'mkmini_stage32e5_b0d3t12a0m10_heldout_long_v1/strictly_improved_research_best.zip'
PARENT_HASH = '35c301288bb4bd2fabdaffd4d17d5fd03b5d6788ea35a03a169d2c30316a3f5a'
OUT = HOME / 'mkmini_stage_daad_sac_native_v1' / 'run'
TIPS = (-2.17, -2.40)
DURATIONS = (1, 8, 24)
OBS_N = 112
ACT_N = 2
FEAT_N = 2*OBS_N + ACT_N + 1


def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()


def seedall(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def feat(obs, prev_obs, prev_action, prev_len):
    o=np.asarray(obs,dtype=np.float32).reshape(OBS_N)
    po=np.asarray(prev_obs,dtype=np.float32).reshape(OBS_N)
    pa=np.asarray(prev_action,dtype=np.float32).reshape(ACT_N)
    x=np.concatenate((o,o-po,pa,np.array([prev_len/max(DURATIONS)],np.float32)))
    if x.shape!=(FEAT_N,) or not np.isfinite(x).all():
        raise RuntimeError('BAD_CAUSAL_FEATURES')
    return x.astype(np.float32)


def clipped_atanh(a):
    x=torch.clamp(a,-0.995,0.995)
    return 0.5*(torch.log1p(x)-torch.log1p(-x))


class Actor(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(FEAT_N,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.mean=nn.Linear(width,len(DURATIONS)*ACT_N)
        self.logstd=nn.Linear(width,len(DURATIONS)*ACT_N)
        self.logits=nn.Linear(width,len(DURATIONS))
        # initially prefer short commitments, not blind 24-frame holds
        with torch.no_grad():
            self.logits.bias.copy_(torch.tensor([0., -0.2, -1.3]))
            self.logstd.bias.fill_(-1.2)
    def forward(self,x):
        h=self.encoder(x)
        mean=self.mean(h).view(-1,len(DURATIONS),ACT_N).clamp(-5,5)
        logstd=self.logstd(h).view(-1,len(DURATIONS),ACT_N).clamp(-5,0.5)
        return mean,logstd,self.logits(h)
    def sample_all(self,x,deterministic=False):
        mu,ls,logits=self.forward(x)
        z=mu if deterministic else mu+ls.exp()*torch.randn_like(mu)
        action=torch.tanh(z)
        # normal log p transformed by tanh, stable Jacobian
        logp=(-.5*((z-mu)/(ls.exp()+1e-8)).square()-ls-.5*math.log(2*math.pi)
              -2*(math.log(2)-z-F.softplus(-2*z))).sum(-1)
        p=F.softmax(logits,dim=-1)
        lp=F.log_softmax(logits,dim=-1)
        return action,logp,p,lp
    @torch.no_grad()
    def decide(self,x,device,mode,stochastic):
        self.eval()
        xt=torch.as_tensor(x,device=device).view(1,-1)
        a,_,p,_=self.sample_all(xt,deterministic=not stochastic)
        if mode=='fixed8': idx=1
        elif stochastic: idx=int(torch.multinomial(p[0],1).item())
        else: idx=int(torch.argmax(p[0]).item())
        return np.asarray(a[0,idx].cpu(),np.float32),idx


class Critic(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEAT_N+ACT_N+len(DURATIONS),width),nn.ReLU(),
             nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,s,a,idx):
        one=F.one_hot(idx.long(),num_classes=len(DURATIONS)).to(s.dtype)
        return self.net(torch.cat([s,a,one],dim=-1)).squeeze(-1)


class Agent:
    def __init__(self,device,seed,mode,lr=2e-4,gamma=.99,tau=.005):
        seedall(seed);self.device=device;self.mode=mode;self.gamma=gamma;self.tau=tau
        self.actor=Actor().to(device);self.q1=Critic().to(device);self.q2=Critic().to(device)
        self.t1=Critic().to(device);self.t2=Critic().to(device)
        self.t1.load_state_dict(self.q1.state_dict());self.t2.load_state_dict(self.q2.state_dict())
        self.ap=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.cp=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.alpha_c=.12;self.alpha_d=.025
        self.mean=torch.zeros(FEAT_N,dtype=torch.float32,device=device)
        self.std=torch.ones(FEAT_N,dtype=torch.float32,device=device)
        self.updates=0
    def normalize(self,x):return torch.clamp((x-self.mean)/self.std,-10.,10.)
    def state(self,x):return self.normalize(torch.as_tensor(x,dtype=torch.float32,device=self.device))
    def choose(self,x,stochastic=True):
        return self.actor.decide(self.state(x),self.device,self.mode,stochastic)
    def _value(self,states,tar):
        actions,lp_a,p,lp_d=self.actor.sample_all(states)
        if self.mode=='fixed8':
            k=1
            idx=torch.full((states.shape[0],),k,device=self.device,dtype=torch.long)
            return (torch.minimum(tar[0](states,actions[:,k],idx),
                                  tar[1](states,actions[:,k],idx))
                    -self.alpha_c*lp_a[:,k])
        qterms=[]
        for k in range(len(DURATIONS)):
            idx=torch.full((states.shape[0],),k,device=self.device,dtype=torch.long)
            q1=tar[0](states,actions[:,k],idx)
            q2=tar[1](states,actions[:,k],idx)
            qterms.append(torch.minimum(q1,q2)-self.alpha_c*lp_a[:,k]-self.alpha_d*lp_d[:,k])
        return (p*torch.stack(qterms,dim=-1)).sum(-1)
    def update(self,batch):
        sx,ix,act,ret,nx,discount,terminal=batch
        s=self.state(sx);n=self.state(nx)
        a=torch.as_tensor(act,dtype=torch.float32,device=self.device)
        i=torch.as_tensor(ix,dtype=torch.long,device=self.device)
        r=torch.as_tensor(ret,dtype=torch.float32,device=self.device)
        g=torch.as_tensor(discount,dtype=torch.float32,device=self.device)
        done=torch.as_tensor(terminal,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            nxt=self._value(n,(self.t1,self.t2))
            target=r+g*(1.-done)*nxt
        lq=(self.q1(s,a,i)-target).square().mean()+(self.q2(s,a,i)-target).square().mean()
        self.cp.zero_grad();lq.backward();torch.nn.utils.clip_grad_norm_(list(self.q1.parameters())+list(self.q2.parameters()),5.)
        self.cp.step()
        # Make q frozen while actor gradients flow through reparameterized actions.
        for q in (self.q1,self.q2):
            for p in q.parameters():p.requires_grad_(False)
        if self.mode=='adaptive':
            a_all,lp_a,p,lp_d=self.actor.sample_all(s)
            qterms=[]
            for k in range(len(DURATIONS)):
                ids=torch.full((s.shape[0],),k,dtype=torch.long,device=self.device)
                qm=torch.minimum(self.q1(s,a_all[:,k],ids),self.q2(s,a_all[:,k],ids))
                qterms.append(self.alpha_c*lp_a[:,k]+self.alpha_d*lp_d[:,k]-qm)
            lp=(p*torch.stack(qterms,dim=-1)).sum(-1).mean()
        else:
            a_all,lp_a,_,_=self.actor.sample_all(s)
            ids=torch.full((s.shape[0],),1,dtype=torch.long,device=self.device)
            qm=torch.minimum(self.q1(s,a_all[:,1],ids),self.q2(s,a_all[:,1],ids))
            lp=(self.alpha_c*lp_a[:,1]-qm).mean()
        self.ap.zero_grad();lp.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),5.);self.ap.step()
        for q in (self.q1,self.q2):
            for p in q.parameters():p.requires_grad_(True)
        with torch.no_grad():
            for src,tgt in ((self.q1,self.t1),(self.q2,self.t2)):
                for p,tp in zip(src.parameters(),tgt.parameters()):tp.lerp_(p,self.tau)
        self.updates+=1
        return float(lq.detach().cpu()),float(lp.detach().cpu())


class Replay:
    def __init__(self,maxsize=50000):self.data=deque(maxlen=maxsize)
    def __len__(self):return len(self.data)
    def add(self,*items):self.data.append(tuple(items))
    def sample(self,batch=64):
        inds=np.random.randint(0,len(self.data),size=batch)
        rows=[self.data[i] for i in inds]
        return (np.stack([r[0] for r in rows]),
                np.array([r[1] for r in rows],np.int64),
                np.stack([r[2] for r in rows]),
                np.array([r[3] for r in rows],np.float32),
                np.stack([r[4] for r in rows]),
                np.array([r[5] for r in rows],np.float32),
                np.array([r[6] for r in rows],np.float32))


def with_native(root):
    # M27 imports sibling Stage modules by their short module name.
    # Both project root AND Stage package directory are required.
    root=Path(root).resolve()
    if not (root/REL/ENTRY).is_file():raise RuntimeError('STAGE_M27_SOURCE_MISSING')
    for p in (root,root/REL):
        s=str(p)
        if s not in sys.path:sys.path.insert(0,s)
    from stable_baselines3 import SAC
    from training.e2e_phase10_lidar_v7_curriculum import t12a0m27_frontwheel_risk_sac as m27
    runtime=m27.runtime(root)
    return m27,runtime,SAC


def new_env(rt,m27,tip,seed):
    env=rt['make_env'](float(tip))
    try:
        obs,reset_info=env.reset(seed=int(seed))
        m27.physics_lock(env,rt)
        ob=np.asarray(obs,dtype=np.float32).reshape(-1)
        if ob.shape!=(OBS_N,) or tuple(env.action_space.shape)!=(ACT_N,):
            raise RuntimeError('OBS_ACTION_CONTRACT_VIOLATED')
        return env,ob
    except Exception:
        env.close();raise


def flags(info):
    if 'stage27_goal_m' not in info:raise RuntimeError('GOAL_METRIC_MISSING_AFTER_NATIVE_STEP')
    goal_m=float(info['stage27_goal_m'])
    if not math.isfinite(goal_m):raise RuntimeError('NONFINITE_GOAL_M')
    contact=bool(info.get('stage27_raw_vehicle_obstacle_contact',False))
    goal=bool(info.get('stage27_goal_reached',False))
    return goal_m,contact,goal


def macro(env,action,requested,gamma,limit):
    rew=0.;total=0;terminal=False;contact=False;goal=False;goals=[];native_rewards=[]
    last_obs=None;env_trunc=False;env_term=False
    for t in range(min(requested,limit)):
        last_obs,r,term,trunc,info=env.step(np.asarray(action,dtype=np.float32))
        dist,c,g=flags(info);goals.append(dist);native_rewards.append(float(r))
        contact|=c;goal|=g
        rew+=(gamma**t)*float(r)
        total+=1;env_term=bool(term);env_trunc=bool(trunc)
        if term or trunc:terminal=True;break
    # Manual episode horizon is a training termination, not a navigation success.
    if total==limit:terminal=True
    if total<1:raise RuntimeError('ZERO_PRIMITIVE_FRAMES')
    return np.asarray(last_obs,dtype=np.float32).reshape(OBS_N),float(rew),total,terminal,contact,goal,goals[-1],min(goals),env_term,env_trunc


def teacher_data(m27,rt,teacher,steps,seed):
    examples=[];signatures=[];contacts=0;goals=0;episode=0;env=None
    try:
        while len(examples)<steps:
            if env is None:
                tip=TIPS[episode%2];env,o=new_env(rt,m27,tip,seed+episode)
                if episode<4:signatures.append(dict(tip=tip,seed=seed+episode,obs_hash=hashlib.sha256(o.tobytes()).hexdigest()))
                prev=o.copy();prev_a=np.zeros(ACT_N,np.float32);prev_d=1
            f=feat(o,prev,prev_a,prev_d)
            a,_=teacher.predict(o,deterministic=True);a=np.asarray(a,dtype=np.float32).reshape(ACT_N)
            examples.append((f,a))
            no,_,done,trunc,info=env.step(a)
            _,c,g=flags(info);contacts+=int(c);goals+=int(g)
            prev=o.copy();o=np.asarray(no,dtype=np.float32);prev_a=a;prev_d=1
            if done or trunc:
                env.close();env=None;episode+=1
    finally:
        if env is not None:env.close()
    return examples,dict(primitive_steps=len(examples),contacts=contacts,goals=goals,reset_signatures=signatures)


def bootstrap_agent(mode,example,device,seed,bc_epochs=14):
    ag=Agent(device,seed,mode)
    xx=np.stack([e[0] for e in example]).astype(np.float32)
    yy=np.stack([e[1] for e in example]).astype(np.float32)
    ag.mean=torch.as_tensor(xx.mean(0),device=device)
    ag.std=torch.as_tensor(np.maximum(xx.std(0),.05),device=device)
    x=ag.state(xx);y=torch.as_tensor(yy,device=device)
    target=clipped_atanh(y)
    for _ in range(bc_epochs):
        perm=torch.randperm(len(x),device=device)
        for j in range(0,len(x),128):
            idx=perm[j:j+128];mu,_,_=ag.actor(x[idx]);loss=(mu-target[idx,None,:]).square().mean()
            ag.ap.zero_grad();loss.backward();ag.ap.step()
    with torch.no_grad():
        pred=ag.actor(x)[0].tanh()
        mse=float((pred-y[:,None,:]).square().mean().cpu())
    return ag,mse


def evaluate(agent,m27,rt,tip,seed,horizon):
    env,o=new_env(rt,m27,tip,seed)
    prev=o.copy();prev_a=np.zeros(ACT_N,np.float32);prev_d=1
    steps=0;g0=None;gfinal=None;gmin=float('inf');contact=False;goal=False;dur_count={str(d):0 for d in DURATIONS}
    try:
        while steps<horizon:
            state=feat(o,prev,prev_a,prev_d)
            action,idx=agent.choose(state,stochastic=False)
            requested=DURATIONS[idx];remaining=horizon-steps
            no,ret,actual,terminal,c,g,last,minimum,envterm,envtrunc=macro(env,action,requested,agent.gamma,remaining)
            if g0 is None:
                # Post-step initial distance (environment does not expose metric at reset).
                g0=last
            steps+=actual;gmin=min(gmin,minimum);gfinal=last;contact|=c;goal|=g
            dur_count[str(requested)]+=1
            prev=o.copy();prev_a=action;prev_d=actual;o=no
            if terminal:break
    finally:env.close()
    return dict(tip=tip,seed=seed,steps=steps,goal=goal,contact=contact,timeout=bool(not goal and not contact and steps>=horizon),
                first_step_goal_proxy_m=round(g0,6),final_goal_m=round(gfinal,6),min_goal_m=round(gmin,6),
                duration_decisions=dur_count)


def train_arm(mode,base_examples,m27,rt,steps,eval_horizon,seed,device,bc_epochs):
    ag,bc=bootstrap_agent(mode,base_examples,device,seed,bc_epochs)
    replay=Replay();ep=0;seen_steps=0;episodes=0;contacts=0;goals=0;last_losses=[];trace=[];used={str(d):0 for d in DURATIONS}
    # Train on independent native physical steps, not decision counts.
    env=None;tip=None
    try:
        while seen_steps<steps:
            if env is None:
                tip=TIPS[ep%2];env,o=new_env(rt,m27,tip,seed+2300+ep)
                prev=o.copy();prev_a=np.zeros(ACT_N,np.float32);prev_d=1;ep_steps=0
            s=feat(o,prev,prev_a,prev_d)
            action,idx=ag.choose(s,stochastic=True)
            asked=DURATIONS[idx];left=min(steps-seen_steps,1200-ep_steps)
            no,ret,d,terminal,c,g,gd,gm,env_term,env_trunc=macro(env,action,asked,ag.gamma,left)
            ns=feat(no,o,action,d)
            replay.add(s,idx,action,ret,ns,ag.gamma**d,int(terminal))
            seen_steps+=d;ep_steps+=d;used[str(asked)]+=1
            contacts+=int(c);goals+=int(g)
            # Equal updates per physical-frame budget across fixed/adaptive.
            target_updates=seen_steps//16
            while len(replay)>=96 and ag.updates<target_updates:
                lq,la=ag.update(replay.sample(64));last_losses=[round(lq,4),round(la,4)]
            if seen_steps//1500>len(trace) or seen_steps>=steps:
                item=dict(primitive_steps=seen_steps,macro_transitions=len(replay),updates=ag.updates,
                          training_contacts=contacts,training_goals=goals,tip=tip,goal_m=round(gd,4),
                          duration_decisions=used.copy(),losses=last_losses)
                trace.append(item);print('TRAIN',mode,json.dumps(item),flush=True)
            if terminal:
                episodes+=1;ep+=1;env.close();env=None
            else:
                prev=o.copy();prev_a=action;prev_d=d;o=no
    finally:
        if env is not None:env.close()
    evaluations=[evaluate(ag,m27,rt,tip,seed+70101,eval_horizon) for tip in TIPS]
    print('EVAL',mode,json.dumps(evaluations),flush=True)
    return dict(arm=mode,native_primitive_steps=seen_steps,macro_transitions=len(replay),actor_critic_updates=ag.updates,
       demonstrations= len(base_examples),behavior_cloning_mse=round(bc,6),training_episodes=episodes,
       training_contacts=contacts,training_goals=goals,duration_decisions=used,evaluations=evaluations,trace=trace)


def self_test():
    seedall(1729);torch.set_num_threads(2);d=torch.device('cpu')
    assert FEAT_N==227
    f=feat(np.ones(112),np.zeros(112),np.zeros(2),8);assert f.shape==(227,)
    a=Actor();q1=Critic();x=torch.randn(5,227)
    av,lpa,p,lpd=a.sample_all(x)
    assert av.shape==(5,3,2) and lpa.shape==(5,3) and p.shape==(5,3)
    assert bool(torch.isfinite(lpa).all()) and bool(torch.allclose(p.sum(-1),torch.ones(5)))
    idx=torch.full((5,),1);assert q1(x,av[:,1],idx).shape==(5,)
    ag=Agent(d,1729,'adaptive');rp=Replay()
    for j in range(140):
        st=np.random.normal(size=227).astype('float32');ac=np.random.uniform(-1,1,size=2).astype('float32')
        dt=DURATIONS[j%3];rp.add(st,j%3,ac,.1,np.random.normal(size=227).astype('float32'),.99**dt,int(j%37==0))
    before=ag.actor.mean.weight.detach().clone();ql=ag.q1.net[0].weight.detach().clone()
    err,pol=ag.update(rp.sample(64));assert math.isfinite(err) and math.isfinite(pol)
    assert not torch.equal(before,ag.actor.mean.weight) and not torch.equal(ql,ag.q1.net[0].weight)
    # SMDP discount and within-macro cumulative rewards, including termination.
    class Dummy:
        def __init__(self):self.n=0
        def step(self,a):
            self.n+=1
            return np.ones(112,np.float32),1.0,self.n>=3,False,dict(stage27_goal_m=5.0,stage27_goal_reached=False,stage27_raw_vehicle_obstacle_contact=self.n>=3)
    ob,ret,k,done,c,g,*_=macro(Dummy(),np.zeros(2),24,.9,24)
    assert k==3 and done and c and not g and abs(ret-(1+.9+.81))<1e-6
    # fixed-duration option samples and critic gradients must work as well.
    af=Agent(d,1729,'fixed8');af.update(rp.sample(64))
    print('SELFTEST=PASS SMDP_GAMMA_DURATION=PASS DISCRETE_CONTINUOUS_SAC=PASS',flush=True)


def main(args):
    if args.self_test:self_test();return
    root=Path(args.project).resolve()
    if not PARENT.is_file() or filehash(PARENT)!=PARENT_HASH:raise RuntimeError('STAGE_FROZEN_PARENT_SHA_MISMATCH')
    if (OUT/'result.json').exists():
        print('RUN_ALREADY_COMPLETE_NO_DUPLICATE='+str(OUT/'result.json'));return
    torch.set_num_threads(min(4,torch.get_num_threads()))
    self_test()
    m27,rt,SAC=with_native(root)
    dev=torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    if args.device=='cuda' and dev.type!='cuda':raise RuntimeError('CUDA_NOT_AVAILABLE')
    teacher=SAC.load(str(PARENT),device=str(dev))
    seedall(args.seed)
    demos,demostats=teacher_data(m27,rt,teacher,args.demo_steps,args.seed)
    print('PARENT_SHA256=PASS TRAIN_NATIVE_ENV=PASS',flush=True)
    print('TEACHER_DEMO',json.dumps(demostats),flush=True)
    del teacher
    results=dict(gate='RUNNING',stage_only=True,planner_used=False,algorithm='DAAD-SAC SMDP learned duration feasibility',
       frozen_parent_sha256=PARENT_HASH,physical_plant='CONDIM3_ACKERMANN_REAR_DIFF_FRONT_FREE_KP_X2',
       duration_choices=list(DURATIONS),train_tips=list(TIPS),independent_layouts_n=0,
       seed_randomized_resets_not_verified=True,eval_horizon=args.eval_steps,steps_per_arm=args.steps,
       demo=demostats,arms=[],model_saved=False,phase_modified=False,parent_modified=False,
       limitations=['Not yet a T-RO/IJRR-level novel method.','Only two approved tip geometries, not a long-range generalization evaluation.',
                    'Duration choices repeat controls without a learned early-termination/risk model.',
                    'Offline teacher behavior initialization uses frozen Stage SAC only.',
                    'Do not count unchanged seed resets as independent environments.'])
    t0=time.monotonic()
    for mode in ('fixed8','adaptive'):
        seedall(args.seed)
        row=train_arm(mode,demos,m27,rt,args.steps,args.eval_steps,args.seed,dev,args.bc_epochs)
        results['arms'].append(row)
    if filehash(PARENT)!=PARENT_HASH:raise RuntimeError('PARENT_SHA_CHANGED_AFTER_TRAINING')
    results['elapsed_seconds']=round(time.monotonic()-t0,2)
    results['gate']='DAAD_NATIVE_SMDP_FEASIBILITY_COMPLETE_NO_PROMOTION'
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'result.json').open('x',encoding='utf8') as f:
        json.dump(results,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('GATE='+results['gate']);print('RESULT='+str(OUT/'result.json'))
    print('MODEL_SAVED=FALSE PARENT_MODIFIED=FALSE PHASE_WRITE=FALSE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('project',nargs='?')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--demo-steps',type=int,default=1024)
    parser.add_argument('--eval-steps',type=int,default=1200)
    parser.add_argument('--bc-epochs',type=int,default=12)
    parser.add_argument('--seed',type=int,default=20261012)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    a=parser.parse_args()
    if not a.self_test and (a.project is None or not 1000<=a.steps<=40000 or not 128<=a.demo_steps<=5000 or not 300<=a.eval_steps<=3000):
        parser.error('PROJECT_AND_BOUNDED_BUDGET_REQUIRED')
    try:main(a)
    except Exception as exc:
        print('GATE=ERROR',type(exc).__name__,str(exc),flush=True)
        traceback.print_exc();sys.exit(2)
