#!/usr/bin/env python3
"""Stage-only G4H: 3 serial DAAD-SAC branches, shared paired validation, blind-once.

No external navigation planner, no manual waypoint, no Phase write. The Stage
legacy source has 'phase10' in its historical module path; no Phase line is run.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, signal, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import g4g as P
import daad_core as C
import native_env as N

NAME='stage_daad_g4h_multibranch_night_v1'
ROOT=Path.home()/NAME/'run'
G4G_ROOT=Path.home()/'stage_daad_g4g_medium_exposure_night_v1/run'
SOURCE=G4G_ROOT/'blind_validated_candidate.pt'
SRC_BLIND=G4G_ROOT/'blind.json'
SRC_FINAL=G4G_ROOT/'final.json'
SRC_OLD=P.OLD
PARENT=N.PARENT
CFG={
 'A_equal_mix': {'pattern':('short','medium'), 'alpha_d':0.025},
 'B_medium_75':{'pattern':('short','medium','medium','medium'),'alpha_d':0.025},
 'C_medium_75_entropy':{'pattern':('short','medium','medium','medium'),'alpha_d':0.080},
}
VAL_SEEDS={'short':tuple(202620101+i for i in range(12)),
           'medium':tuple(202620301+i for i in range(12))}
BLIND_SEEDS={'short':tuple(202620701+i for i in range(20)),
             'medium':tuple(202620901+i for i in range(20))}
FRAMES_PER_ROTATION=1200000
VAL_INTERVAL=150000
TRAIN_RESERVE_SEC=3600
HARD_END_CST='2026-10-14 08:00:00 +0800'
STOP=False

def stopp(*_):
 global STOP
 STOP=True

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def atomic_json(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n');os.replace(t,p)

def torch_atomic(p,obj):
 t=p.with_name(p.name+'.tmp');torch.save(obj,t);os.replace(t,p)

def verify():
 for p in (SOURCE,SRC_BLIND,SRC_FINAL,SRC_OLD,PARENT):
  if not p.is_file():raise RuntimeError('REQUIRED_FROZEN_FILE_MISSING:'+str(p))
 if sha(PARENT)!=N.PARENT_SHA:raise RuntimeError('M27_PARENT_SHA_CHANGED')
 b=json.loads(SRC_BLIND.read_text());f=json.loads(SRC_FINAL.read_text())
 if b.get('status')!='COMPLETE' or b.get('strictly_improved') is not True:
  raise RuntimeError('G4G_BLIND_NOT_STRICTLY_IMPROVED')
 if f.get('status')!='EXITED' or f.get('error') is not None or not f.get('source_150k_unchanged') or not f.get('parent_unchanged'):
  raise RuntimeError('G4G_FINAL_NOT_A_CLEAN_FINISH')
 ck=torch.load(SOURCE,map_location='cpu',weights_only=False) # trusted local Stage source
 if ck.get('format')!='STAGE_G4G_BLIND_PASS':raise RuntimeError('G4G_BLIND_FORMAT_INVALID')
 if ck.get('parent_sha')!=N.PARENT_SHA or ck.get('source_sha')!=sha(SRC_OLD):
  raise RuntimeError('G4G_BLIND_LINEAGE_INVALID')
 if ck.get('blind_result',{}).get('strictly_improved') is not True:
  raise RuntimeError('BLIND_CKPT_DOES_NOT_ATTEST_PASS')
 P.checks() # Recheck Stage-native physics evidence and original 150k lineage
 return {'source_sha':sha(SOURCE),'parent_sha':sha(PARENT),'old_sha':sha(SRC_OLD),
         'g4g_final_sha':sha(SRC_FINAL),'g4g_blind_sha':sha(SRC_BLIND)}

def assert_frozen(proof):
 for key,path in [('source_sha',SOURCE),('parent_sha',PARENT),('old_sha',SRC_OLD),
                  ('g4g_final_sha',SRC_FINAL),('g4g_blind_sha',SRC_BLIND)]:
  if sha(path)!=proof[key]:raise RuntimeError('FROZEN_SHA_CHANGED:'+str(path))

def agent_new(device,seed,alpha,proof):
 assert_frozen(proof)
 s=torch.load(SOURCE,map_location='cpu',weights_only=False)
 ag=C.Agent(device,seed,'adaptive',gamma=P.GAMMA)
 ag.actor.load_state_dict(s['actor'],strict=True)
 ag.mean=s['mean'].to(device);ag.std=s['std'].to(device)
 ag.alpha_d=alpha
 if not torch.isfinite(ag.mean).all() or not torch.isfinite(ag.std).all():
  raise RuntimeError('BAD_ACTOR_NORMALIZER')
 return ag

def macro(env,action,duration,remaining,stop_epoch):
 r=0.; n=0;ob=None;term=False;trunc=False;goal=False;contact=False;dist=None
 for j in range(min(duration,remaining)):
  if STOP or time.time()>=stop_epoch:break
  ob,reward,t,tr,info=N.step_checked(env,action)
  r+=(P.GAMMA**j)*float(reward);n+=1
  goal=bool(info['reach_goal']);contact=bool(info['collision'])
  term=bool(t or goal or contact);trunc=bool(tr);dist=float(info['goal_dist'])
  if term or trunc:break
 return ob,r,n,term,trunc,goal,contact,dist

def evaluation(ag,project,seedset,tag,cutoff):
 tasks={}
 for task in ('short','medium'):
  rows=[];horizon=P.TRAIN_HORIZON[task]
  for seed in seedset[task]:
   if STOP or time.time()>cutoff:break
   env=None
   try:
    env,obs,goal=P.environment(project,task,N.TEST_LAYOUT,seed)
    startdist=float(np.linalg.norm(np.asarray(env.data.qpos[:2],float)-goal))
    prev=obs.copy();prev_act=np.zeros(2,np.float32);prev_k=1
    n=0;closest=startdist;last=startdist;got=False;col=False;timeout=False
    while n<horizon and not STOP and time.time()<cutoff:
     f=C.feat(obs,prev,prev_act,prev_k)
     a,i=ag.choose(f,stochastic=False)
     next_obs,ret,k,term,tr,hit,contact,dist=macro(env,a,C.DURATIONS[i],horizon-n,cutoff)
     if k==0:break
     n+=k;closest=min(closest,dist);last=dist
     got|=hit;col|=contact
     prev=obs.copy();obs=next_obs;prev_act=a;prev_k=k
     if term or tr:
      timeout=bool(tr);break
    rows.append({'seed':seed,'goal':bool(got),'contact':bool(col),'steps':n,
      'complete':bool(got or col or timeout or n>=horizon),
      'initial_goal_m':round(startdist,5),'min_goal_m':round(closest,5),
      'final_goal_m':round(last,5),'progress_m':round(startdist-closest,5)})
   finally:
    if env is not None:env.close()
  tasks[task]={'n':len(rows),'complete':bool(len(rows)==len(seedset[task]) and all(r['complete'] for r in rows)),
    'safe_goals':sum(r['goal'] and not r['contact'] for r in rows),
    'contacts':sum(r['contact'] for r in rows),
    'mean_progress_m':round(float(np.mean([r['progress_m'] for r in rows])),6) if rows else 0.,
    'rows':rows}
 return {'tag':tag,'complete':all(v['complete'] for v in tasks.values()),'tasks':tasks}

def compact(ev):
 return {'tag':ev['tag'],'complete':ev['complete'],'tasks':{
  t:{k:v for k,v in ev['tasks'][t].items() if k!='rows'} for t in ('short','medium')}}

def qualifies(ev,baseline):
 return P.winner(ev,baseline)

def score(ev):
 t=ev['tasks'];return (t['medium']['safe_goals'],-t['medium']['contacts'],
  t['short']['safe_goals'],-t['short']['contacts'],
  t['medium']['mean_progress_m'],t['short']['mean_progress_m'])

def save_branch(name,ag,replay,state,proof):
 ck={'format':'G4H_BRANCH_V1','branch':name,'configuration':CFG[name],
  'source_sha':proof['source_sha'],'parent_sha':proof['parent_sha'],
  'agent':{k:getattr(ag,k).state_dict() for k in ('actor','q1','q2','t1','t2','ap','cp')},
  'mean':ag.mean.detach().cpu(),'std':ag.std.detach().cpu(),'alpha_d':ag.alpha_d,
  'updates':ag.updates,'replay':list(replay.data),'state':state,
  'np_rng':np.random.get_state(),'py_rng':random.getstate(),
  'torch_rng':torch.get_rng_state(),
  'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
 d=ROOT/name;d.mkdir(parents=True,exist_ok=True)
 torch_atomic(d/'resume.pt',ck);atomic_json(d/'progress.json',state)

def load_branch(name,device,proof):
 C.seedall(2026101407);torch.set_num_threads(3)
 ag=agent_new(device,2026101407,CFG[name]['alpha_d'],proof)
 replay=C.Replay(20000)
 p=ROOT/name/'resume.pt'
 if not p.is_file():
  return ag,replay,{'frames':0,'episodes':0,'goals':{'short':0,'medium':0},
   'contacts':{'short':0,'medium':0},'task_episodes':{'short':0,'medium':0},
   'durations':{},'last_eval_frames':0,'validation':[],'best_score':None,
   'best_checkpoint':False,'name':name}
 ck=torch.load(p,map_location='cpu',weights_only=False)
 if ck.get('format')!='G4H_BRANCH_V1' or ck.get('branch')!=name or ck.get('source_sha')!=proof['source_sha'] or ck.get('configuration')!=CFG[name]:
  raise RuntimeError('BRANCH_RESUME_CONTRACT_INVALID:'+name)
 for k in ('actor','q1','q2','t1','t2','ap','cp'):
  getattr(ag,k).load_state_dict(ck['agent'][k])
 for optim in (ag.ap,ag.cp):
  for param_state in optim.state.values():
   for key,v in list(param_state.items()):
    if torch.is_tensor(v):param_state[key]=v.to(device)
 ag.mean=ck['mean'].to(device);ag.std=ck['std'].to(device);ag.updates=int(ck['updates'])
 replay.data.clear();replay.data.extend(ck['replay'])
 np.random.set_state(ck['np_rng']);random.setstate(ck['py_rng']);torch.set_rng_state(ck['torch_rng'])
 if torch.cuda.is_available() and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
 return ag,replay,ck['state']

def save_validation_candidate(name,ag,ev,proof,state):
 if not ev['complete']:raise RuntimeError('CANNOT_SAVE_INCOMPLETE_EVAL')
 assert_frozen(proof)
 obj={'format':'G4H_VALIDATION_ONLY','branch':name,'config':CFG[name],
  'source_sha':proof['source_sha'],'parent_sha':proof['parent_sha'],
  'gamma':P.GAMMA,'mean':ag.mean.detach().cpu(),'std':ag.std.detach().cpu(),
  'actor':{k:v.detach().cpu().clone() for k,v in ag.actor.state_dict().items()},
  'validation':ev,'frames':state['frames'],'updates':ag.updates}
 torch_atomic(ROOT/name/'validation_best.pt',obj)
 state['best_checkpoint']=True;state['best_score']=list(score(ev))

def training_run(name,project,quota,train_end,proof,baseline):
 global STOP
 ag,replay,s=load_branch(name,'cuda:0',proof)
 start_frames=int(s['frames']);episodes_here=0
 if start_frames>=quota:return {'branch':name,'skipped':'quota_already_reached','frames':start_frames}
 print('G4H_BRANCH_START',json.dumps({'branch':name,'frames':start_frames,'quota':quota,
  'pattern':CFG[name]['pattern'],'alpha_d':ag.alpha_d}),flush=True)
 try:
  while not STOP and time.time()<train_end and s['frames']<quota:
   ep=int(s['episodes']);task=CFG[name]['pattern'][ep%len(CFG[name]['pattern'])]
   layout=N.TRAIN_LAYOUTS[(ep//len(CFG[name]['pattern']))%len(N.TRAIN_LAYOUTS)]
   horizon=P.TRAIN_HORIZON[task]
   env=None;epsteps=0;hit=False;contact=False
   try:
    env,obs,goal=P.environment(project,task,layout,2026220000+ep)
    prev=obs.copy();prev_a=np.zeros(2,np.float32);prev_k=1
    while not STOP and epsteps<horizon and s['frames']<quota and time.time()<train_end:
     f=C.feat(obs,prev,prev_a,prev_k);a,i=ag.choose(f,stochastic=True)
     remaining=min(horizon-epsteps,quota-s['frames'])
     nx,ret,k,terminal,truncated,gh,ct,dist=macro(env,a,C.DURATIONS[i],remaining,train_end)
     if k==0:break
     replay.add(f,i,a,ret,C.feat(nx,obs,a,k),P.GAMMA**k,int(terminal))
     epsteps+=k;s['frames']+=k;hit|=gh;contact|=ct
     key=str(C.DURATIONS[i]);s['durations'][key]=int(s['durations'].get(key,0))+1
     if len(replay)>=128 and ag.updates<s['frames']//16:
      loss=ag.update(replay.sample(64))
      if not all(math.isfinite(float(v)) for v in loss):raise RuntimeError('NONFINITE_SAC_LOSS:'+name)
     prev=obs.copy();obs=nx;prev_a=a;prev_k=k
     if terminal or truncated:break
    if epsteps>0:
     s['episodes']+=1;episodes_here+=1;s['goals'][task]+=int(hit)
     s['contacts'][task]+=int(contact);s['task_episodes'][task]+=1
    elif not STOP and time.time()<train_end:
     raise RuntimeError('ZERO_NATIVE_PHYSICS_STEPS:'+name)
   finally:
    if env is not None:env.close()
   if s['frames']-s.get('last_saved_frames',0)>=50000 or s['frames']>=quota or time.time()>=train_end:
    s['last_saved_frames']=s['frames']
    save_branch(name,ag,replay,s,proof)
    print('G4H_PROGRESS',json.dumps({'branch':name,'frames':s['frames'],
      'episodes':s['episodes'],'goals':s['goals'],'contacts':s['contacts'],
      'updates':ag.updates,'durations':s['durations']}),flush=True)
   if s['frames']-int(s['last_eval_frames'])>=VAL_INTERVAL and time.time()<train_end-700:
    ev=evaluation(ag,project,VAL_SEEDS,s['frames'],min(train_end-60,time.time()+700))
    if not ev['complete']:
     print('G4H_VALIDATION_INCOMPLETE_NOT_USED',name,s['frames'],flush=True)
    else:
     ce=compact(ev);s['validation'].append(ce);s['last_eval_frames']=s['frames']
     print('G4H_VALIDATION',json.dumps({'branch':name,'result':ce}),flush=True)
     best_before=s['best_score']
     if qualifies(ev,baseline) and (best_before is None or score(ev)>tuple(best_before)):
      save_validation_candidate(name,ag,ev,proof,s)
      print('G4H_BRANCH_STRICT_VALIDATION_IMPROVED',name,score(ev),flush=True)
    save_branch(name,ag,replay,s,proof)
 finally:
  save_branch(name,ag,replay,s,proof)
 return {'branch':name,'initial_frames':start_frames,'frames':s['frames'],
  'episodes':s['episodes'],'updates':ag.updates,'qualified_candidate':s['best_checkpoint']}

def select_candidate(proof):
 eligible=[]
 for name in CFG:
  ckpath=ROOT/name/'validation_best.pt'
  if not ckpath.is_file():continue
  ck=torch.load(ckpath,map_location='cpu',weights_only=False)
  if ck['source_sha']!=proof['source_sha'] or ck['parent_sha']!=proof['parent_sha']:
   raise RuntimeError('NOMINEE_LINEAGE_CHANGED')
  eligible.append((score(ck['validation']),name,ckpath,ck['frames']))
 eligible.sort(key=lambda r:(r[0],-r[3],r[1]),reverse=True)
 return eligible[0] if eligible else None

def blind_once(nom,project,cutoff,proof):
 assert_frozen(proof)
 if not nom:return {'status':'NOT_RUN_NO_QUALIFIED_VALIDATION_CANDIDATE'}
 if (ROOT/'blind.json').is_file():
  return json.loads((ROOT/'blind.json').read_text()) # NEVER repeat/look at blind for reselection
 sc,name,path,frames=nom
 print('G4H_BLIND_START',json.dumps({'selected_branch':name,'frames':frames}),flush=True)
 ck=torch.load(path,map_location='cpu',weights_only=False)
 cand=agent_new('cuda:0',2026101407,CFG[name]['alpha_d'],proof)
 cand.actor.load_state_dict(ck['actor'],strict=True)
 cand.mean=ck['mean'].to('cuda:0');cand.std=ck['std'].to('cuda:0')
 base=agent_new('cuda:0',2026101407,.025,proof)
 cand_ev=evaluation(cand,project,BLIND_SEEDS,'blind_candidate',cutoff)
 base_ev=evaluation(base,project,BLIND_SEEDS,'blind_baseline',cutoff)
 both=bool(cand_ev['complete'] and base_ev['complete'])
 passed=bool(both and qualifies(cand_ev,base_ev))
 out={'status':'COMPLETE' if both else 'INCOMPLETE','strictly_improved':passed,
  'nominated_branch':name,'nomination_frames':frames,'candidate':compact(cand_ev),
  'baseline':compact(base_ev),'limitations':'Single same-topology layout; not long-range/topology generalization'}
 atomic_json(ROOT/'blind.json',out)
 if passed:
  torch_atomic(ROOT/'blind_pass_candidate.pt',{'format':'G4H_SINGLE_BLIND_PASS',
   'branch':name,'actor':ck['actor'],'mean':ck['mean'],'std':ck['std'],
   'source_sha':proof['source_sha'],'parent_sha':proof['parent_sha'],'blind':out})
  print('G4H_BLIND_PASS',str(ROOT/'blind_pass_candidate.pt'),flush=True)
 else:print('G4H_BLIND_NO_STRICT_IMPROVEMENT',flush=True)
 return out

def pilot(project):
 proof=verify();C.self_test();C.seedall(2026101407)
 ag=agent_new('cpu',2026101407,.025,proof);replay=C.Replay(512)
 totals={}
 for task in ('short','medium'):
  nsteps=0;macros=0;trials=[];before=ag.updates
  for attempt in range(8):
   env=None;steps=0;terminal_kind='HORIZON'
   try:
    env,obs,goal=P.environment(project,task,N.TRAIN_LAYOUTS[attempt%2],2026230000+attempt*2+(task=='medium'))
    prev=obs.copy();pa=np.zeros(2,np.float32);pk=1
    while steps<300:
     f=C.feat(obs,prev,pa,pk);a,i=ag.choose(f,stochastic=True)
     nx,ret,k,term,tr,hit,contact,dist=macro(env,a,C.DURATIONS[i],300-steps,time.time()+300)
     if k==0:raise RuntimeError('PILOT_NO_REAL_STEPS')
     replay.add(f,i,a,ret,C.feat(nx,obs,a,k),P.GAMMA**k,int(term))
     macros+=1;steps+=k
     if len(replay)>=48 and ag.updates<before+8:
      loss=ag.update(replay.sample(32))
      if not all(math.isfinite(float(v)) for v in loss):raise RuntimeError('NONFINITE_PILOT_GRADIENT')
     prev=obs.copy();obs=nx;pa=a;pk=k
     if term or tr:
      terminal_kind='GOAL' if hit else 'CONTACT' if contact else 'OTHER_TERMINAL'
      break
   finally:
    if env is not None:env.close()
   nsteps+=steps;trials.append({'steps':steps,'event':terminal_kind})
   if attempt>=1 and nsteps>=80 and macros>=8 and ag.updates-before>=8:break
  totals[task]={'steps':nsteps,'macros':macros,'updates':ag.updates-before,'trials':trials}
  if nsteps<80 or macros<8 or ag.updates-before<8:
   atomic_json(ROOT/'pilot_rejected.json',totals)
   raise RuntimeError('G4H_REAL_PHYSICS_PILOT_FAIL:'+task)
 assert_frozen(proof)
 atomic_json(ROOT/'pilot.json',{'gate':'G4H_NATIVE_PILOT_PASS','proof':proof,'totals':totals})
 print('GATE=G4H_NATIVE_PILOT_PASS',json.dumps({k:(v['steps'],v['updates']) for k,v in totals.items()}),flush=True)

def run_night(project,deadline):
 global STOP
 signal.signal(signal.SIGTERM,stopp);signal.signal(signal.SIGINT,stopp)
 proof=verify();pilot_info=json.loads((ROOT/'pilot.json').read_text())
 if pilot_info.get('gate')!='G4H_NATIVE_PILOT_PASS' or pilot_info.get('proof')!=proof:
  raise RuntimeError('PILOT_PROOF_MISMATCH')
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_FOR_LONG_TRAIN')
 train_end=deadline-TRAIN_RESERVE_SEC
 if train_end-time.time()<3600:raise RuntimeError('INSUFFICIENT_TRAIN_WINDOW')
 C.seedall(2026101407)
 baseline_p=ROOT/'baseline.json'
 if baseline_p.is_file():
  baseline_full=json.loads(baseline_p.read_text())
 else:
  atomic_json(ROOT/'phase.json',{'phase':'BASELINE_EVAL','at_epoch':int(time.time())})
  base=agent_new('cuda:0',2026101407,.025,proof)
  baseline_full=evaluation(base,project,VAL_SEEDS,'g4g_blind_actor_baseline',train_end-700)
  if not baseline_full['complete']:raise RuntimeError('BASELINE_EVAL_INCOMPLETE')
  atomic_json(baseline_p,baseline_full)
 baseline=baseline_full
 print('G4H_BASELINE',json.dumps(compact(baseline)),flush=True)
 sched=ROOT/'schedule.json';history=json.loads(sched.read_text()) if sched.is_file() else {'round':0,'completed':[]}
 reason='TRAIN_WINDOW_FINISHED';err=None;blind={'status':'NOT_RUN'}
 try:
  while not STOP and time.time()<train_end:
   r=int(history['round'])+1
   target=r*FRAMES_PER_ROTATION
   for name in CFG:
    if STOP or time.time()>=train_end:break
    assert_frozen(proof)
    atomic_json(ROOT/'phase.json',{'phase':'TRAINING','branch':name,'round':r,'quota':target,'at_epoch':int(time.time())})
    result=training_run(name,project,target,train_end,proof,baseline)
    history['completed'].append({'round':r,**result,'ended_at_epoch':int(time.time())})
    atomic_json(sched,history)
   history['round']=r;atomic_json(sched,history)
  if not STOP and time.time()<deadline-60:
   atomic_json(ROOT/'phase.json',{'phase':'BLIND_EVAL','at_epoch':int(time.time())})
   nom=select_candidate(proof)
   blind=blind_once(nom,project,deadline-60,proof)
 except Exception as e:
  err=repr(e);reason='ERROR';atomic_json(ROOT/'error.json',{'error':err,'traceback':traceback.format_exc()})
 finally:
  assert_frozen(proof)
  out={'gate':'G4H_FINISHED_'+reason,'status':'EXITED','error':err,
    'completed_rounds':history['round'],'selected_branch':blind.get('nominated_branch'),
    'blind_status':blind.get('status'),'blind_strictly_improved':blind.get('strictly_improved',False),
    'parent_unchanged':True,'g4g_blind_source_unchanged':True,'source_150k_unchanged':True,
    'phase_write':False,'model_promoted':False,'deadline_epoch':deadline,
    'finished_epoch':int(time.time())}
  atomic_json(ROOT/'final.json',out)
  print('G4H_FINAL',json.dumps(out),flush=True)
 if err:raise RuntimeError(err)

def self_test():
 C.self_test()
 assert math.isclose(P.GAMMA**(10/.002),.5,rel_tol=1e-7)
 a={'complete':True,'tasks':{'short':{'safe_goals':8,'contacts':1,'mean_progress_m':.3},
                             'medium':{'safe_goals':4,'contacts':3,'mean_progress_m':.5}}}
 b={'complete':True,'tasks':{'short':{'safe_goals':8,'contacts':1,'mean_progress_m':.3},
                             'medium':{'safe_goals':5,'contacts':3,'mean_progress_m':.5}}}
 assert qualifies(b,a) and not qualifies(a,b)
 assert all(CFG[k]['pattern'] for k in CFG)
 assert set(CFG)=={'A_equal_mix','B_medium_75','C_medium_75_entropy'}
 print('SELFTEST=PASS G4H_SMDP=PASS BLIND_ONCE=PASS 3_BRANCHES=PASS',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('project',nargs='?',default=str(Path.home()/'catkin_ws/src/mk-mini'))
 p.add_argument('--mode',choices=('self-test','pilot','night'),required=True)
 p.add_argument('--deadline-epoch',type=int)
 a=p.parse_args()
 if a.mode=='self-test':self_test()
 elif a.mode=='pilot':pilot(Path(a.project))
 else:
  if not a.deadline_epoch:raise RuntimeError('EXPLICIT_DEADLINE_REQUIRED')
  run_night(Path(a.project),a.deadline_epoch)

if __name__=='__main__':main()
