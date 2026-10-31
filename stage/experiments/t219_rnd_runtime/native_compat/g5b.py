#!/usr/bin/env python3
"""STAGE ONLY: Frozen continuous heads + learnable duration selector pilot.

G5A established C 24-head action potential, not generalization. This experiment
freezes every continuous-action parameter and learns only the categorical
selector with off-policy semi-Markov SAC, on entirely new training seeds.
Original M27, old 150k, G4G blind pass, G4H branch source remain read-only.
No planner, route, goal proxy, manual action teacher, Phase writes or promotion.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

NAME='stage_daad_g5b_selector_recovery_v1'
ROOT=Path.home() / ('mkmini_'+NAME) / 'run'
INSTALL=Path.home()/'stage_daad_g4h_multibranch_night_v1'/'runner'
G4H_RUN=Path.home()/'stage_daad_g4h_multibranch_night_v1'/'run'
G5A=Path.home()/'mkmini_stage_daad_g5a_counterfactual_duration_v1'/'run/partial.json'
BRANCH='C_medium_75_entropy'
VAL_SEEDS={'short':tuple(range(202635101,202635113)), 'medium':tuple(range(202635301,202635313))}


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def atomic(path, obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')
 os.replace(temp,path)


def prepare(project):
 if not INSTALL.is_dir():raise RuntimeError('G4H_INSTALLED_RUNNER_MISSING:'+str(INSTALL))
 sys.path.insert(0,str(INSTALL)); os.chdir(project)
 import g4h as H
 import daad_core as C
 import native_env as N
 proof=H.verify()
 final=json.loads((G4H_RUN/'final.json').read_text())
 if final.get('status')!='EXITED' or final.get('error') is not None:raise RuntimeError('G4H_NOT_CLEANLY_STOPPED')
 if not G5A.is_file():raise RuntimeError('G5A_REPORT_MISSING:'+str(G5A))
 g5a=json.loads(G5A.read_text());cases=g5a.get('cases',[])
 if len(cases)!=88 or g5a.get('gate')!='G5A_COUNTERFACTUAL_DURATION_COMPLETE_NO_TRAINING':
  raise RuntimeError('G5A_CAUSAL_WITNESS_INCOMPLETE')
 source=G4H_RUN/BRANCH/'resume.pt'
 if not source.is_file():raise RuntimeError('C_SOURCE_MISSING:'+str(source))
 ck=torch.load(source,map_location='cpu',weights_only=False) # trusted user-owned local Stage checkpoint
 if ck.get('branch')!=BRANCH or ck.get('source_sha')!=proof['source_sha']:
  raise RuntimeError('C_BRANCH_LINEAGE_INVALID')
 frozen={str(k):sha(k) for k in (H.SOURCE,H.PARENT,H.SRC_OLD,H.SRC_BLIND,H.SRC_FINAL,source,G5A)}
 return H,C,N,proof,ck,frozen


def frozen_assert(proof,frozen,H):
 H.assert_frozen(proof)
 for p,h in frozen.items():
  if sha(p)!=h:raise RuntimeError('FROZEN_FILE_CHANGED:'+p)


def new_agent(H,C,ck,device,seed,selector_reset):
 C.seedall(seed)
 ag=C.Agent(device,seed,'adaptive',gamma=H.P.GAMMA)
 ag.actor.load_state_dict(ck['agent']['actor'],strict=True)
 ag.mean=ck['mean'].to(device); ag.std=ck['std'].to(device)
 ag.alpha_d=float(ck.get('alpha_d',0.080))
 if not math.isclose(ag.alpha_d,.080,rel_tol=1e-7,abs_tol=1e-8):
  raise RuntimeError('C_ALPHA_NOT_008')
 for name,p in ag.actor.named_parameters():
  p.requires_grad_(name.startswith('logits.'))
 if selector_reset:
  # Remove nearly-deterministic k=1 decision, not the 3 continuous controls.
  with torch.no_grad():
   ag.actor.logits.weight.zero_()
   ag.actor.logits.bias.zero_()
 ag.ap=torch.optim.Adam(list(ag.actor.logits.parameters()),lr=1e-4)
 return ag


def fixed_digest(agent):
 h=hashlib.sha256()
 for n,p in agent.actor.named_parameters():
  if not n.startswith('logits.'):
   h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def choice(C,ag,f,mode,stochastic=False,epsilon=0.0):
 with torch.no_grad():
  s=ag.state(f).reshape(1,-1)
  aa,_,pp,_=ag.actor.sample_all(s,deterministic=not stochastic)
  probs=pp[0].cpu().numpy()
  if mode=='head24':idx=2
  elif stochastic and random.random()<epsilon:idx=random.randrange(3)
  elif stochastic:idx=int(torch.multinomial(pp[0],1).item())
  else:idx=int(np.argmax(probs))
  a=aa[0,idx].cpu().numpy().astype('float32')
  return a,idx,probs


def rollout(H,C,N,ag,project,task,seed,mode,deadline):
 env=None
 try:
  env,obs,goal=H.P.environment(project,task,N.TEST_LAYOUT,seed)
  start=float(np.linalg.norm(np.asarray(env.data.qpos[:2],float)-goal))
  prev=obs.copy();prev_a=np.zeros(2,np.float32);prev_k=1
  max_steps=H.P.TRAIN_HORIZON[task]
  n=0;minimum=start;got=False;contact=False;counts={1:0,8:0,24:0}
  complete=False
  while n<max_steps:
   if time.monotonic()>deadline:raise TimeoutError('EVAL_TIMED_OUT')
   f=C.feat(obs,prev,prev_a,prev_k)
   a,i,p=choice(C,ag,f,mode)
   d=C.DURATIONS[i];counts[d]+=1
   before=obs.copy();k=0;term=False;tr=False
   for _ in range(min(d,max_steps-n)):
    ob,rew,term,tr,info=N.step_checked(env,a)
    k+=1;n+=1
    minimum=min(minimum,float(info['goal_dist']))
    got|=bool(info['reach_goal']);contact|=bool(info['collision'])
    if term or tr:break
   if k==0:raise RuntimeError('NATIVE_ZERO_STEPS')
   prev=before;obs=np.asarray(ob,np.float32);prev_a=a;prev_k=k
   if term or tr:break
  complete=bool(got or contact or tr or n>=max_steps)
  return {'task':task,'seed':seed,'mode':mode,'complete':complete,'safe_goal':bool(got and not contact),
   'contact':bool(contact),'steps':n,'start_goal_m':round(start,6),
   'min_goal_m':round(minimum,6),'progress_m':round(start-minimum,6),
   'durations':{str(k):v for k,v in counts.items()}}
 finally:
  if env is not None:env.close()


def evaluate(H,C,N,agents,project,deadline):
 output={}
 for label,agent,mode in agents:
  rows=[]
  for task,seeds in VAL_SEEDS.items():
   for seed in seeds:
    row=rollout(H,C,N,agent,project,task,seed,mode,deadline)
    if not row['complete']:raise RuntimeError('INCOMPLETE_VALIDATION')
    rows.append(row)
   print('G5B_EVAL',label,task,'SAFE',sum(r['safe_goal'] for r in rows if r['task']==task),
    'CONTACTS',sum(r['contact'] for r in rows if r['task']==task),flush=True)
  summary={}
  for task in VAL_SEEDS:
   s=[r for r in rows if r['task']==task]
   summary[task]={'safe_goals':sum(r['safe_goal'] for r in s),'contacts':sum(r['contact'] for r in s),
    'n':len(s),'mean_progress_m':round(float(np.mean([r['progress_m'] for r in s])),6)}
  output[label]={'summary':summary,'rows':rows}
 return output


def gate(candidate,baseline):
 a=candidate['summary'];b=baseline['summary']
 return bool(a['medium']['safe_goals']>b['medium']['safe_goals']
  and a['medium']['contacts']<=b['medium']['contacts']
  and a['short']['safe_goals']>=b['short']['safe_goals']
  and a['short']['contacts']<=b['short']['contacts'])


def self_test():
 assert gate({'summary':{'medium':{'safe_goals':5,'contacts':2},'short':{'safe_goals':8,'contacts':1}}},
             {'summary':{'medium':{'safe_goals':4,'contacts':3},'short':{'safe_goals':8,'contacts':2}}})
 assert not gate({'summary':{'medium':{'safe_goals':5,'contacts':4},'short':{'safe_goals':8,'contacts':1}}},
             {'summary':{'medium':{'safe_goals':4,'contacts':3},'short':{'safe_goals':8,'contacts':2}}})
 sys.path.insert(0,str(INSTALL)) if INSTALL.is_dir() else None
 try:
  import daad_core as C
 except ImportError:
  # Selftest runs even without Ubuntu project via packaged fallback when available.
  return print('G5B_SELFTEST=PASS CORE_IMPORT_ON_HOST_REQUIRED')
 ag=C.Agent('cpu',42,'adaptive',gamma=math.exp(math.log(.5)*.002/10.))
 for n,p in ag.actor.named_parameters():p.requires_grad_(n.startswith('logits.'))
 ag.ap=torch.optim.Adam(ag.actor.logits.parameters(),lr=1e-4)
 before=fixed_digest(ag)
 b0=ag.actor.logits.weight.detach().clone()
 rp=C.Replay(256)
 for i in range(160):
  x=np.random.randn(C.FEAT_N).astype('float32')*.1
  nx=x+np.random.randn(C.FEAT_N).astype('float32')*.001
  k=random.randrange(3)
  rp.add(x,k,np.random.uniform(-.6,.6,2).astype('float32'),float(np.random.randn()),
   nx,ag.gamma**C.DURATIONS[k],0)
 for i in range(4):
  loss=ag.update(rp.sample(64))
  assert all(math.isfinite(v) for v in loss)
 assert fixed_digest(ag)==before,'FROZEN_CONTINUOUS_HEADS_CHANGED'
 assert not torch.equal(ag.actor.logits.weight.detach(),b0),'SELECTOR_NOT_LEARNING'
 print('G5B_SELFTEST=PASS FROZEN_HEADS=PASS SELECTOR_GRADIENT=PASS SMDP=PASS',flush=True)


def pilot(H,C,N,ck,project,deadline):
 ag=new_agent(H,C,ck,'cpu',66521,True)
 frozen=fixed_digest(ag)
 replay=C.Replay(1024)
 steps=0;updates=0;per_task={}
 # Legitimate obstacle contact may terminate a run after just 1-3 steps.
 # Try multiple fresh training-only reset seeds, do not waive the native contract.
 for task in ('short','medium'):
  task_steps=0;task_episodes=0
  for attempt in range(5):
   if task_steps>=80 and len(replay)>=128:break
   if time.monotonic()>deadline:raise TimeoutError('PILOT_BUDGET')
   env=None
   try:
    env,obs,goal=H.P.environment(project,task,N.TRAIN_LAYOUTS[attempt%2],202636019+attempt*10+(task=='medium'))
    prev=obs.copy();prev_a=np.zeros(2,np.float32);prev_k=1
    epsteps=0
    while epsteps<360 and time.monotonic()<deadline:
     f=C.feat(obs,prev,prev_a,prev_k)
     a,i,pp=choice(C,ag,f,'natural',True,0.33)
     before=obs.copy();term=False;tr=False;ret=0;k=0;nx=None
     for j in range(min(C.DURATIONS[i],360-epsteps)):
      nx,r,term,tr,info=N.step_checked(env,a);ret+=(ag.gamma**j)*float(r);k+=1
      if term or tr:break
     if k==0:raise RuntimeError('PILOT_NO_NATIVE_STEPS')
     replay.add(f,i,a,ret,C.feat(nx,obs,a,k),ag.gamma**k,int(term))
     obs=nx;prev=before;prev_a=a;prev_k=k;epsteps+=k;steps+=k;task_steps+=k
     if len(replay)>=128:
      loss=ag.update(replay.sample(64));updates+=1
      if not all(math.isfinite(x) for x in loss):raise RuntimeError('PILOT_LOSS_NONFINITE')
     if term or tr:break
    if epsteps>0:task_episodes+=1
   finally:
    if env is not None:env.close()
  per_task[task]={'steps':task_steps,'episodes':task_episodes}
  print('G5B_PILOT_TASK',task,'STEPS',task_steps,'EPISODES',task_episodes,flush=True)
  if task_steps<15 or task_episodes<1:raise RuntimeError('PILOT_NATIVE_TASK_STEPS_TOO_FEW:'+task)
 if len(replay)<20:raise RuntimeError('PILOT_REPLAY_TOO_SHORT')
 while updates<8:
  loss=ag.update(replay.sample(min(len(replay),64)));updates+=1
 if fixed_digest(ag)!=frozen:raise RuntimeError('PILOT_CONTINUOUS_HEAD_MODIFIED')
 print('GATE=G5B_FOREGROUND_NATIVE_SELECTOR_PILOT_PASS STEPS='+str(steps)+' UPDATES='+str(updates),flush=True)


def train(H,C,N,ck,project,deadline,max_frames):
 device='cuda:0' if torch.cuda.is_available() else 'cpu'
 print('G5B_DEVICE',device,flush=True)
 ag=new_agent(H,C,ck,device,202636011,True)
 frozen=fixed_digest(ag)
 replay=C.Replay(18000)
 steps=0;updates=0;episodes=0;goals={'short':0,'medium':0};contacts={'short':0,'medium':0};counts={'1':0,'8':0,'24':0}
 while steps<max_frames and time.monotonic()<deadline-120:
  task=('short','medium')[episodes%2]
  layout=N.TRAIN_LAYOUTS[(episodes//2)%len(N.TRAIN_LAYOUTS)]
  env=None;episode_steps=0;hit=False;coll=False
  try:
   env,obs,goal=H.P.environment(project,task,layout,202636000+episodes)
   prev=obs.copy();prev_a=np.zeros(2,np.float32);prev_k=1
   horizon=H.P.TRAIN_HORIZON[task]
   while episode_steps<horizon and steps<max_frames and time.monotonic()<deadline-120:
    f=C.feat(obs,prev,prev_a,prev_k)
    eps=.35 if steps<max_frames//2 else .15
    a,i,prob=choice(C,ag,f,'natural',True,eps)
    before=obs.copy();k=0;ret=0.;term=False;tr=False;nx=None
    for j in range(min(C.DURATIONS[i],horizon-episode_steps,max_frames-steps)):
     if time.monotonic()>deadline-120:break
     nx,r,term,tr,info=N.step_checked(env,a)
     ret+=(ag.gamma**j)*float(r);k+=1
     hit|=bool(info['reach_goal']);coll|=bool(info['collision'])
     if term or tr:break
    if k==0:break
    replay.add(f,i,a,ret,C.feat(nx,obs,a,k),ag.gamma**k,int(term))
    episode_steps+=k;steps+=k
    key=str(C.DURATIONS[i]);counts[key]+=1
    if len(replay)>=128 and ag.updates<steps//16:
     losses=ag.update(replay.sample(64));updates+=1
     if not all(math.isfinite(x) for x in losses):raise RuntimeError('NONFINITE_SELECTOR_LOSS')
    prev=before;obs=nx;prev_a=a;prev_k=k
    if term or tr:break
   if episode_steps>0:
    episodes+=1;goals[task]+=int(hit);contacts[task]+=int(coll)
  finally:
   if env is not None:env.close()
  if episodes%10==0 or steps>=max_frames:
   print('G5B_PROGRESS',json.dumps({'frames':steps,'episodes':episodes,'goals':goals,'contacts':contacts,
    'updates':updates,'durations':counts},ensure_ascii=False),flush=True)
   atomic(ROOT/'progress.json',{'frames':steps,'episodes':episodes,'goals':goals,'contacts':contacts,
    'updates':updates,'durations':counts})
 if fixed_digest(ag)!=frozen:raise RuntimeError('CONTINUOUS_HEAD_DRIFT_DURING_TRAINING')
 if steps<min(2500,max_frames//3):raise RuntimeError('INSUFFICIENT_TRAINING_PROGRESS')
 return ag,{'frames':steps,'episodes':episodes,'goals':goals,'contacts':contacts,'updates':updates,'durations':counts,
  'nonlogit_hash_unchanged':True,'initial_nonlogit_hash':frozen}


def main():
 p=argparse.ArgumentParser()
 p.add_argument('project',nargs='?',default=str(Path.home()/'catkin_ws/src/mk-mini'))
 p.add_argument('--self-test',action='store_true')
 p.add_argument('--frames',type=int,default=60000)
 p.add_argument('--max-seconds',type=int,default=5400)
 args=p.parse_args();self_test()
 if args.self_test:return
 if not 6000<=args.frames<=180000:raise ValueError('INVALID_FRAME_BUDGET')
 if not 1200<=args.max_seconds<=7200:raise ValueError('INVALID_WALL_BUDGET')
 project=Path(args.project).resolve();deadline=time.monotonic()+args.max_seconds
 ROOT.mkdir(parents=True,exist_ok=True)
 H,C,N,proof,ck,frozen=prepare(project)
 if sha(H.PARENT)!=proof['parent_sha']:raise RuntimeError('PARENT_CHANGED_BEFORE_RUN')
 device='cuda:0' if torch.cuda.is_available() else 'cpu'
 g4g=H.agent_new(device,2026101407,.025,proof)
 c0=new_agent(H,C,ck,device,621,False)
 selector0=new_agent(H,C,ck,device,621,True)
 frozen_h=fixed_digest(c0)
 print('G5B_INITIAL_FROZEN_CONTINUOUS_HASH',frozen_h,flush=True)
 print('G5B_PILOT_START',flush=True)
 pilot(H,C,N,ck,project,deadline)
 print('G5B_TRAIN_START',flush=True)
 sel,run=train(H,C,N,ck,project,deadline,args.frames)
 print('G5B_PAIRED_VALIDATION_START new seeds; DO NOT USE AS PROMOTION',flush=True)
 ev=evaluate(H,C,N,[('g4g_frozen_blind',g4g,'natural'),('C_final_natural',c0,'natural'),
    ('C_final_forced24_diagnostic',c0,'head24'),('selector_initial_uniform',selector0,'natural'),('selector_after_training',sel,'natural')],project,deadline)
 result={'gate':'G5B_SELECTOR_RECOVERY_FOREGROUND_COMPLETE_NO_PROMOTION','stage_only':True,'no_planner':True,
  'training':run,'comparison':ev,'selection_vs_g4g':gate(ev['selector_after_training'],ev['g4g_frozen_blind']),
  'selection_vs_C':gate(ev['selector_after_training'],ev['C_final_natural']),
  'frozen':frozen,'phase_write':False,'promotion':False,'validation_only':True,
  'note':'This is a selector-only foreground feasibility experiment; heldout must not be used for automatic promotion.'}
 frozen_assert(proof,frozen,H)
 atomic(ROOT/'result.json',result)
 # New research artifact only; never overwrite any source/checkpoint
 if fixed_digest(sel)!=frozen_h:raise RuntimeError('FINAL_FIXED_CONTINUOUS_SHA_CHANGED')
 ck_out={'format':'G5B_FOREGROUND_RESEARCH_ONLY','source_c_sha':frozen[str(G4H_RUN/BRANCH/'resume.pt')],
  'actor':{n:v.detach().cpu().clone() for n,v in sel.actor.state_dict().items()},
  'mean':sel.mean.detach().cpu(),'std':sel.std.detach().cpu(),
  'run':run,'paired_validation_summary':{k:v['summary'] for k,v in ev.items()},
  'promotion':False}
 path=ROOT/'selector_pilot_only.pt'
 tmp=ROOT/'selector_pilot_only.pt.tmp';torch.save(ck_out,tmp);os.replace(tmp,path)
 print('GATE='+result['gate'],'FRAMES='+str(run['frames']),'TRAINING_STARTED=True','PROMOTION=False',flush=True)
 for tag,d in ev.items():
  print('G5B_EVAL_FINAL',tag,json.dumps(d['summary'],ensure_ascii=False),flush=True)
 print('G5B_SELECTOR_STRICT_IMPROVED_ON_VALIDATION',result['selection_vs_g4g'],flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print('GATE=G5B_BLOCKED ERROR='+repr(exc),flush=True)
  traceback.print_exc();raise
