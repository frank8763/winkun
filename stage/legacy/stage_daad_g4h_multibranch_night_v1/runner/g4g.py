#!/usr/bin/env python3
"""Stage-only G4G: mixed short/medium DAAD-SAC with paired safety evaluation.

Research change vs G4E: expose policy to *real* medium goal tasks (1:1
short/medium), rather than block all medium episodes behind a short-goal gate.
All observations and actions remain native 112D/2D; no trajectory teacher,
planner, human-defined waypoint, heuristic turning, or model/scene writes.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, signal, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import daad_core as C
import native_env as N
import native_factory_utils as U

NAME='stage_daad_g4g_medium_exposure_night_v1'
ROOT=Path.home()/NAME/'run'
OLD=Path.home()/'mkmini_stage_daad_g2e_overnight_gate_v1/run/resume_checkpoint.pt'
NIGHT=OLD.parent/'night_result.json'
G4E=Path.home()/'stage_daad_g4e_collision_curriculum_v1/run/final.json'
G4F=Path.home()/'mkmini_stage_daad_g4f_medium_collision_probe_v1/run/physics.json'
G4FA=G4F.parent/'audit.json'
GAMMA=math.exp(math.log(.5)*.002/10.)
DURATIONS=(1,8,24)
TRAIN_HORIZON={'short':7000,'medium':14000}
VAL_SEEDS={'short':tuple(202617101+i for i in range(12)),
           'medium':tuple(202617301+i for i in range(12))}
BLIND_SEEDS={'short':tuple(202618101+i for i in range(12)),
             'medium':tuple(202618301+i for i in range(12))}
EVAL_EVERY=75000
MAX_FRAMES=10000000
STOP=False

def stop_signal(*_):
    global STOP
    STOP=True

def atomic(path,value):
    N.atomic_json(path,value)

def sha(p):return U.sha(Path(p))

def checks():
    required=[OLD,NIGHT,G4E,G4F,G4FA,N.PARENT]
    for p in required:
        if not p.is_file():raise RuntimeError('PREREQUISITE_MISSING:'+str(p))
    if sha(N.PARENT)!=N.PARENT_SHA:raise RuntimeError('FROZEN_PARENT_DRIFT')
    night=json.loads(NIGHT.read_text()); ge=json.loads(G4E.read_text())
    if night.get('gate')!='OVERNIGHT_STOPPED_FRAME_BUDGET_REACHED' or int(night['state']['physical_steps'])!=150000:
        raise RuntimeError('SOURCE_150K_NOT_COMPLETE')
    if ge.get('gate')!='G4E_FINISHED_EARLY_STOP_NO_INDEPENDENT_IMPROVEMENT' or not ge.get('parent_unchanged') or not ge.get('source_150k_unchanged'):
        raise RuntimeError('G4E_FINAL_NOT_APPROVED')
    pf=json.loads(G4F.read_text()); pa=json.loads(G4FA.read_text())
    if pf.get('gate')!='G4F_PHYSICAL_STEERING_WITNESS_COMPLETE_NO_TRAINING' or not pf.get('complete') or int(pf.get('count',0))!=20:
        raise RuntimeError('G4F_PHYSICAL_WITNESS_INCOMPLETE')
    if pa.get('gate')!='G4F_OFFLINE_EVIDENCE_AUDIT_PASS' or pa.get('training_started') or pa.get('model_writes'):
        raise RuntimeError('G4F_AUDIT_INVALID')
    # Verify reachable medium tasks are witnessed, without using the witness action in training.
    right=[r for r in pf['results'] if r['witness'].startswith('right_')]
    if len(right)!=8 or sum(bool(r['goal']) and not bool(r['contact']) for r in right)<4:
        raise RuntimeError('MEDIUM_RIGHT_STEERING_PHYSICAL_REACHABILITY_MISSING')
    if not math.isclose(GAMMA**(10/.002),.5,rel_tol=1e-7):raise RuntimeError('TIME_DISCOUNT_INVALID')
    return {'source_sha':sha(OLD),'parent_sha':sha(N.PARENT),'g4e_sha':sha(G4E),'g4f_sha':sha(G4F),
            'medium_physics_witness_goals':sum(bool(r['goal']) and not r['contact'] for r in right)}

def new_agent(device,source_sha):
    if sha(OLD)!=source_sha:raise RuntimeError('OLD_CHANGED_AT_ACTOR_LOAD')
    ck=torch.load(OLD,map_location='cpu',weights_only=False) # trusted local experiment artifact
    ag=C.Agent(device,2026101401,'adaptive',gamma=GAMMA)
    # Modified gamma versus original 150k: start fresh Q, targets, optimizers and replay.
    ag.actor.load_state_dict(ck['agent']['actor'],strict=True)
    ag.mean=ck['mean'].to(device);ag.std=ck['std'].to(device)
    if not torch.isfinite(ag.mean).all() or not torch.isfinite(ag.std).all():
        raise RuntimeError('SOURCE_NORMALIZER_NONFINITE')
    return ag

def environment(project,task,layout,seed):
    env=None
    try:
        env,obs,_,goal=N.create(project,layout,task,seed)
        dt=float(env.model.opt.timestep)*int(env.frame_skip)
        if not math.isclose(dt,.002,abs_tol=1e-10):raise RuntimeError('PHYSICAL_DT_CHANGED')
        old_expected=700 if task=='short' else 1100
        if int(env.max_steps)!=old_expected:raise RuntimeError('NATIVE_HORIZON_DRIFT')
        if not math.isclose(float(env.collision_penalty),12.,abs_tol=1e-8):
            raise RuntimeError('NATIVE_COLLISION_PENALTY_DRIFT')
        env.max_steps=TRAIN_HORIZON[task]
        env.collision_penalty=24.0  # held CONSTANT relative to G4E; ONLY curriculum changes
        if bool(env._collision()):raise RuntimeError('CONTACT_AT_RESET')
        return env,obs,goal
    except Exception:
        if env is not None:env.close()
        raise

def macro(env,action,duration,remaining,deadline):
    rew=0.;n=0;obs=None;terminal=False;truncated=False;goal=False;contact=False;dist=None
    for j in range(min(duration,remaining)):
        if STOP or time.time()>=deadline-120:break
        obs,r,t,tr,info=N.step_checked(env,action)
        rew+=(GAMMA**j)*float(r)
        n+=1
        goal=bool(info['reach_goal']);contact=bool(info['collision'])
        terminal=bool(t or goal or contact)
        truncated=bool(tr)
        dist=float(info['goal_dist'])
        if terminal or truncated:break
    return obs,rew,n,terminal,truncated,goal,contact,dist

def evaluate(agent,project,deadline,tag,seedset):
    summaries={}
    for task in ('short','medium'):
        rows=[]; horizon=TRAIN_HORIZON[task]
        for seed in seedset[task]:
            if STOP or time.time()>deadline-900:break
            env=None
            try:
                env,obs,goal=environment(project,task,N.TEST_LAYOUT,seed)
                ini=float(np.linalg.norm(np.asarray(env.data.qpos[:2],float)-goal))
                prev=obs.copy();prev_a=np.zeros(2,np.float32);prev_k=1
                step=0;mini=ini;last=ini;ok=False;col=False
                while step<horizon and not STOP and time.time()<deadline-900:
                    f=C.feat(obs,prev,prev_a,prev_k)
                    act,idx=agent.choose(f,stochastic=False)
                    nxt,rew,k,term,tr,hit,contact,dist=macro(env,act,C.DURATIONS[idx],horizon-step,deadline-600)
                    if k==0:break
                    step+=k;mini=min(mini,dist);last=dist
                    ok|=hit;col|=contact
                    prev=obs.copy();obs=nxt;prev_a=act;prev_k=k
                    if term or tr:break
                rows.append({'task':task,'seed':seed,'steps':step,'goal':bool(ok),'contact':bool(col),
                             'complete':bool(ok or col or step>=horizon),'initial_goal_m':round(ini,6),
                             'min_goal_m':round(mini,6),'final_goal_m':round(last,6),
                             'progress_m':round(ini-mini,6)})
            finally:
                if env is not None:env.close()
        complete=len(rows)==len(seedset[task]) and all(x['complete'] for x in rows)
        summaries[task]={'n':len(rows),'complete':complete,
                         'safe_goals':sum(r['goal'] and not r['contact'] for r in rows),
                         'contacts':sum(r['contact'] for r in rows),
                         'mean_progress_m':round(float(np.mean([r['progress_m'] for r in rows])),6) if rows else 0.,
                         'rows':rows}
    return {'tag':tag,'complete':all(x['complete'] for x in summaries.values()),'tasks':summaries}

def compact(ev):
    return {'tag':ev['tag'],'complete':ev['complete'],**{t:{k:v for k,v in ev['tasks'][t].items() if k!='rows'} for t in ('short','medium')}}

def winner(new,old):
    if not new['complete'] or not old['complete']:return False
    n=new['tasks']; b=old['tasks']
    return (n['medium']['safe_goals']>b['medium']['safe_goals']
            and n['medium']['contacts']<=b['medium']['contacts']
            and n['short']['safe_goals']>=b['short']['safe_goals']
            and n['short']['contacts']<=b['short']['contacts'])

def save(agent,replay,state,source_sha):
    ck={'format':'STAGE_G4G_V1','gamma':GAMMA,'source_sha':source_sha,
        'agent':{k:getattr(agent,k).state_dict() for k in ('actor','q1','q2','t1','t2','ap','cp')},
        'mean':agent.mean.detach().cpu(),'std':agent.std.detach().cpu(),
        'updates':agent.updates,'replay':list(replay.data),'state':state,
        'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'torch_rng':torch.get_rng_state(),
        'torch_cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    ROOT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'resume.pt';tmp=ROOT/'resume.pt.tmp'
    torch.save(ck,tmp);os.replace(tmp,p)
    atomic(ROOT/'progress.json',state)

def resume(agent,replay,source_sha):
    p=ROOT/'resume.pt'
    if not p.is_file():return None
    ck=torch.load(p,map_location='cpu',weights_only=False)
    if ck.get('format')!='STAGE_G4G_V1' or ck.get('source_sha')!=source_sha or abs(float(ck.get('gamma',0))-GAMMA)>1e-12:
        raise RuntimeError('RESUME_FORMAT_OR_LINEAGE_MISMATCH')
    for k in ('actor','q1','q2','t1','t2','ap','cp'):
        getattr(agent,k).load_state_dict(ck['agent'][k])
    for obj in (agent.ap,agent.cp):
        for v in obj.state.values():
            for key,t in list(v.items()):
                if torch.is_tensor(t):v[key]=t.to(agent.device)
    agent.mean=ck['mean'].to(agent.device);agent.std=ck['std'].to(agent.device)
    agent.updates=int(ck['updates'])
    replay.data.clear();replay.data.extend(ck['replay'])
    np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);torch.set_rng_state(ck['torch_rng'])
    if torch.cuda.is_available() and ck.get('torch_cuda_rng'):
        torch.cuda.set_rng_state_all(ck['torch_cuda_rng'])
    return ck['state']

def save_candidate(agent,proof,ev):
    if sha(OLD)!=proof['source_sha'] or sha(N.PARENT)!=proof['parent_sha']:
        raise RuntimeError('FROZEN_MODEL_SHA_CHANGED')
    obj={'format':'STAGE_G4G_SELECTION_CANDIDATE_V1',
         'actor':{k:v.detach().cpu() for k,v in agent.actor.state_dict().items()},
         'mean':agent.mean.detach().cpu(),'std':agent.std.detach().cpu(),'gamma':GAMMA,
         'validation':ev,'source_sha':proof['source_sha'],'parent_sha':proof['parent_sha']}
    p=ROOT/'validation_candidate.pt';tmp=ROOT/'validation_candidate.pt.tmp'
    torch.save(obj,tmp);os.replace(tmp,p)

def pilot(project):
    """Native smoke test: accept genuine early terminals, not unusable rollouts.

    An early collision is a real task outcome, *not* a failure of the simulator
    API.  However, repeatedly returning only a handful of valid physics steps
    is insufficient to launch an unattended training run.  Exercise BOTH
    training layouts, accumulate measured physics steps and actual SAC updates,
    and report exact termination causes on every attempt.
    """
    proof=checks();C.self_test();C.seedall(20261014)
    torch.set_num_threads(2);ag=new_agent('cpu',proof['source_sha'])
    replay=C.Replay(2048);tests=[]
    for task in ('short','medium'):
        trials=[];task_steps=0;task_macro=0;task_updates_start=ag.updates
        for attempt in range(8):
            # Match the two actual TRAIN layouts and the real train seed family.
            ep=2*attempt+(0 if task=='short' else 1)
            seed=2026190000+ep
            layout=N.TRAIN_LAYOUTS[attempt%len(N.TRAIN_LAYOUTS)]
            env=None;step=0;macros=0;last_dist=None
            term_kind='STEP_LIMIT';hit_any=False;contact_any=False
            try:
                env,obs,goal=environment(project,task,layout,seed)
                prev=obs.copy();pa=np.zeros(2,np.float32);pk=1
                while step<400:
                    f=C.feat(obs,prev,pa,pk);a,i=ag.choose(f,stochastic=True)
                    nxt,rew,k,term,tr,hit,cont,dist=macro(env,a,C.DURATIONS[i],400-step,time.time()+600)
                    if k==0:raise RuntimeError('PILOT_NO_NATIVE_STEPS:'+task+':'+str(seed))
                    if k<1 or k>min(C.DURATIONS[i],400-step):
                        raise RuntimeError('PILOT_INVALID_DURATION:'+task+':'+str(seed))
                    replay.add(f,i,a,rew,C.feat(nxt,obs,a,k),GAMMA**k,int(term))
                    step+=k;macros+=1;last_dist=dist
                    hit_any|=bool(hit);contact_any|=bool(cont)
                    if len(replay)>=48 and ag.updates<min(80,task_updates_start+12):
                        loss=ag.update(replay.sample(32))
                        if not all(math.isfinite(float(v)) for v in loss):
                            raise RuntimeError('PILOT_NONFINITE_GRADIENT')
                    prev=obs.copy();obs=nxt;pa=a;pk=k
                    if term or tr:
                        term_kind=('GOAL' if hit else 'CONTACT' if cont else
                                   'OTHER_TERMINAL' if term else 'TRUNCATED')
                        break
            finally:
                if env is not None:env.close()
            task_steps+=step;task_macro+=macros
            row={'task':task,'layout':layout,'seed':seed,'steps':step,
                 'macros':macros,'termination':term_kind,'goal':hit_any,
                 'contact':contact_any,'last_goal_m':last_dist}
            trials.append(row)
            print('G4G_PILOT_EPISODE',json.dumps(row,allow_nan=False),flush=True)
            # Repeated genuine early terminals are expected in an obstacle
            # task.  Require enough cumulative real MuJoCo steps instead of
            # an arbitrary 30-step threshold on one episode.
            if attempt>=1 and task_steps>=80 and task_macro>=8 and (ag.updates-task_updates_start)>=8:
                break
        result={'task':task,'steps':task_steps,'macros':task_macro,
                'episodes':len(trials),'updates_total':ag.updates,
                'updates_in_task':ag.updates-task_updates_start,'trials':trials}
        tests.append(result)
        if task_steps<80 or task_macro<8 or (ag.updates-task_updates_start)<8:
            ROOT.mkdir(parents=True,exist_ok=True)
            atomic(ROOT/'pilot_diagnostics.json',{'gate':'BLOCKED_G4G_PILOT_INSUFFICIENT_NATIVE_ROLLOUT',
                                                  'tests':tests,'training_started':False})
            raise RuntimeError('PILOT_INSUFFICIENT_NATIVE_ROLLOUT:'+task+':'+
                               str(task_steps)+'_STEPS_'+str(task_macro)+'_MACROS_'+
                               str(ag.updates-task_updates_start)+'_UPDATES')
    if ag.updates<8:
        ROOT.mkdir(parents=True,exist_ok=True)
        atomic(ROOT/'pilot_diagnostics.json',{'gate':'BLOCKED_G4G_PILOT_INSUFFICIENT_GRADIENT',
                                              'tests':tests,'training_started':False})
        raise RuntimeError('PILOT_NOT_ENOUGH_GRADIENT:'+str(ag.updates))
    if sha(OLD)!=proof['source_sha'] or sha(N.PARENT)!=proof['parent_sha']:
        raise RuntimeError('PILOT_MUTATED_SOURCE')
    ROOT.mkdir(parents=True,exist_ok=True)
    out={'gate':'G4G_NATIVE_MIXED_TASK_PILOT_V1_1_PASS','old_sha':proof['source_sha'],
         'parent_sha':proof['parent_sha'],'g4f_sha':proof['g4f_sha'],
         'tests':tests,'updates':ag.updates,'training_started':False,
         'phase_write':False,'pilot_version':'v1.1_early_terminal_safe'}
    atomic(ROOT/'pilot.json',out)
    print('GATE=G4G_NATIVE_MIXED_TASK_PILOT_V1_1_PASS',
          'SHORT_STEPS='+str(tests[0]['steps']),
          'MEDIUM_STEPS='+str(tests[1]['steps']),
          'UPDATES='+str(ag.updates),flush=True)

def train(project,deadline):
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    proof=checks()
    p=json.loads((ROOT/'pilot.json').read_text())
    if p.get('gate')!='G4G_NATIVE_MIXED_TASK_PILOT_V1_1_PASS' or p.get('old_sha')!=proof['source_sha'] or p.get('g4f_sha')!=proof['g4f_sha'] or p.get('pilot_version')!='v1.1_early_terminal_safe':
        raise RuntimeError('G4G_PILOT_NOT_VALID')
    if not torch.cuda.is_available():raise RuntimeError('GPU_REQUIRED')
    C.seedall(20261014);torch.set_num_threads(3)
    device='cuda:0';ag=new_agent(device,proof['source_sha']);replay=C.Replay(20000)
    state=resume(ag,replay,proof['source_sha'])
    if state is None:
        atomic(ROOT/'phase.json',{'phase':'BASELINE_EVAL','at_epoch':int(time.time())})
        baseline=evaluate(ag,project,deadline,'baseline',VAL_SEEDS)
        if not baseline['complete']:raise RuntimeError('BASELINE_INCOMPLETE')
        state={'frames':0,'episodes':0,'training_goals':0,'training_contacts':0,
               'task_episodes':{'short':0,'medium':0},'task_goals':{'short':0,'medium':0},
               'task_contacts':{'short':0,'medium':0},'duration_counts':{},
               'baseline':baseline,'evaluations':[baseline],'last_eval_frames':0,
               'best_medium_goals':baseline['tasks']['medium']['safe_goals'],
               'best_candidate':False,'no_medium_gain_evals':0}
        print('G4G_BASELINE',json.dumps(compact(baseline)),flush=True)
        save(ag,replay,state,proof['source_sha'])
    else:
        print('G4G_RESUME',state['frames'],flush=True)
    atomic(ROOT/'phase.json',{'phase':'TRAINING','at_epoch':int(time.time()),'frames':state['frames']})
    print('G4G_TRAINING_ENTERED FRAMES='+str(state['frames']),flush=True)
    start=time.monotonic();reason='NOT_FINISHED';error=None
    try:
        while not STOP and time.time()<deadline-900 and state['frames']<MAX_FRAMES:
            ep=int(state['episodes']);task='short' if ep%2==0 else 'medium'
            layout=N.TRAIN_LAYOUTS[(ep//2)%len(N.TRAIN_LAYOUTS)]
            horizon=TRAIN_HORIZON[task]
            env=None;epsteps=0;hit=False;col=False
            try:
                env,obs,goal=environment(project,task,layout,2026190000+ep)
                prev=obs.copy();pa=np.zeros(2,np.float32);pk=1
                while epsteps<horizon and not STOP and time.time()<deadline-900 and state['frames']<MAX_FRAMES:
                    f=C.feat(obs,prev,pa,pk);a,i=ag.choose(f,stochastic=True)
                    left=min(horizon-epsteps,MAX_FRAMES-state['frames'])
                    nxt,ret,k,term,tr,gh,ct,dist=macro(env,a,C.DURATIONS[i],left,deadline-600)
                    if k==0:break
                    replay.add(f,i,a,ret,C.feat(nxt,obs,a,k),GAMMA**k,int(term))
                    epsteps+=k;state['frames']+=k;hit|=gh;col|=ct
                    d=str(C.DURATIONS[i]);state['duration_counts'][d]=int(state['duration_counts'].get(d,0))+1
                    if len(replay)>=128 and ag.updates<state['frames']//16:
                        losses=ag.update(replay.sample(64))
                        if not all(math.isfinite(float(v)) for v in losses):raise RuntimeError('NONFINITE_LOSS')
                    prev=obs.copy();obs=nxt;pa=a;pk=k
                    if term or tr:break
                if epsteps==0 and not STOP:raise RuntimeError('TRAIN_EPISODE_ZERO_STEPS')
                state['episodes']+=1;state['training_goals']+=int(hit);state['training_contacts']+=int(col)
                state['task_episodes'][task]+=1;state['task_goals'][task]+=int(hit);state['task_contacts'][task]+=int(col)
                state['last_episode']={'task':task,'layout':layout,'steps':epsteps,'goal':hit,'contact':col}
            finally:
                if env is not None:env.close()
            save(ag,replay,state,proof['source_sha'])
            print('G4G_PROGRESS',json.dumps({'frames':state['frames'],'episodes':state['episodes'],
              'goals':state['task_goals'],'contacts':state['task_contacts'],
              'updates':ag.updates,'durations':state['duration_counts']}),flush=True)
            if state['frames']-int(state['last_eval_frames'])>=EVAL_EVERY and time.time()<deadline-1500:
                ev=evaluate(ag,project,deadline,state['frames'],VAL_SEEDS)
                if not ev['complete']:reason='EVAL_INCOMPLETE';break
                state['evaluations'].append(ev);state['last_eval_frames']=state['frames']
                print('G4G_VALIDATION',json.dumps(compact(ev)),flush=True)
                if winner(ev,state['baseline']) and ev['tasks']['medium']['safe_goals']>state['best_medium_goals']:
                    save_candidate(ag,proof,ev)
                    state['best_medium_goals']=ev['tasks']['medium']['safe_goals']
                    state['best_candidate']=True
                    state['no_medium_gain_evals']=0
                    print('G4G_VALIDATION_SAFE_MEDIUM_CANDIDATE',state['best_medium_goals'],flush=True)
                else:
                    state['no_medium_gain_evals']+=1
                save(ag,replay,state,proof['source_sha'])
                if state['frames']>=600000 and state['no_medium_gain_evals']>=8:
                    reason='EARLY_STOP_NO_MEDIUM_VALIDATION_GAIN';break
        if reason=='NOT_FINISHED':
            reason='STOP_SIGNAL' if STOP else 'LOCAL_MORNING_DEADLINE' if time.time()>=deadline-900 else 'FRAME_BUDGET'
    except Exception as exc:
        error=repr(exc);reason='ERROR'
        atomic(ROOT/'error.json',{'error':error,'traceback':traceback.format_exc()})
    finally:
        save(ag,replay,state,proof['source_sha'])
        src_ok=sha(OLD)==proof['source_sha'];parent_ok=sha(N.PARENT)==proof['parent_sha']
        blind={'status':'NOT_RUN_NO_VALIDATION_CANDIDATE'}
        if state['best_candidate'] and not STOP and time.time()<deadline-1200:
            try:
                ck=torch.load(ROOT/'validation_candidate.pt',map_location='cpu',weights_only=False)
                if ck['source_sha']!=proof['source_sha'] or ck['parent_sha']!=proof['parent_sha']:
                    raise RuntimeError('BLIND_CANDIDATE_LINEAGE')
                candidate=new_agent(device,proof['source_sha'])
                candidate.actor.load_state_dict(ck['actor'],strict=True)
                candidate.mean=ck['mean'].to(device);candidate.std=ck['std'].to(device)
                base=new_agent(device,proof['source_sha'])
                cv=evaluate(candidate,project,deadline,'blind_candidate',BLIND_SEEDS)
                bv=evaluate(base,project,deadline,'blind_baseline',BLIND_SEEDS)
                passed=winner(cv,bv)
                blind={'status':'COMPLETE' if cv['complete'] and bv['complete'] else 'INCOMPLETE',
                       'strictly_improved':bool(passed),'candidate':cv,'baseline':bv}
                atomic(ROOT/'blind.json',blind)
                if passed:
                    dest=ROOT/'blind_validated_candidate.pt';tmp=ROOT/'blind_validated_candidate.pt.tmp'
                    torch.save({'format':'STAGE_G4G_BLIND_PASS','actor':ck['actor'],
                        'mean':ck['mean'],'std':ck['std'],'gamma':GAMMA,
                        'source_sha':proof['source_sha'],'parent_sha':proof['parent_sha'],
                        'blind_result':blind},tmp)
                    os.replace(tmp,dest)
                    print('G4G_BLIND_CANDIDATE_PASS',str(dest),flush=True)
            except Exception as exc:
                blind={'status':'BLIND_ERROR','error':repr(exc)}
                atomic(ROOT/'blind.json',blind)
        out={'gate':'G4G_FINISHED_'+reason,'status':'EXITED','frames':state['frames'],
             'episodes':state['episodes'],'training_goals':state['training_goals'],
             'training_contacts':state['training_contacts'],'task_goals':state['task_goals'],
             'task_contacts':state['task_contacts'],'updates':ag.updates,
             'validation_baseline':compact(state['baseline']),
             'validation_last':compact(state['evaluations'][-1]),
             'best_candidate':state['best_candidate'],'blind_status':blind['status'],
             'source_150k_unchanged':src_ok,'parent_unchanged':parent_ok,
             'phase_write':False,'model_promoted':False,'elapsed_wall_s':round(time.monotonic()-start,2),
             'error':error,'limitation':'Medium exposure on existing scene topology, NOT long-range or multi-topology proof'}
        atomic(ROOT/'final.json',out)
        print('G4G_FINAL',json.dumps(out),flush=True)
        if not src_ok or not parent_ok:raise RuntimeError('FROZEN_SHA_DRIFT_AFTER_TRAIN')
    if error:raise RuntimeError(error)

def self_test():
    C.self_test()
    assert math.isclose(GAMMA**(10/.002),.5,rel_tol=1e-7)
    assert C.DURATIONS==DURATIONS
    b={'complete':True,'tasks':{'short':{'safe_goals':5,'contacts':2},'medium':{'safe_goals':0,'contacts':12}}}
    n={'complete':True,'tasks':{'short':{'safe_goals':5,'contacts':2},'medium':{'safe_goals':1,'contacts':12}}}
    assert winner(n,b) and not winner(b,n)
    print('G4G_SELFTEST=PASS SMDP=PASS SAFETY_PAIRED=PASS',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('project',nargs='?',default=str(Path.home()/'catkin_ws/src/mk-mini'))
    ap.add_argument('--mode',choices=('self-test','pilot','train'),required=True)
    ap.add_argument('--deadline-epoch',type=int)
    args=ap.parse_args()
    if args.mode=='self-test':self_test();return
    if args.mode=='pilot':pilot(Path(args.project));return
    if not args.deadline_epoch:raise RuntimeError('G4G_HARD_DEADLINE_REQUIRED')
    train(Path(args.project),int(args.deadline_epoch))

if __name__=='__main__':main()
