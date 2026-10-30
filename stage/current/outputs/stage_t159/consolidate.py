"""Consolidate six successful Stage neural-policy tasks into one Actor."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
MASK=[57,58,74,75,86,87,98,99,110,111]

def save(name,obj):
    p=ROOT/name;tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)

def main():
    lock=(ROOT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(), 'Do not duplicate or implicitly restart'
    plan=json.loads((ROOT/'plan.json').read_text())
    os.environ['C98_JOB']=os.environ['TD3_JOB']=str(ROOT)
    os.environ['C22_SCREEN_ONLY']='0'
    torch.set_num_threads(1);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
    spec=importlib.util.spec_from_file_location('t159_env',ROOT.parent/'stage_t148/train.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    for p,h in plan['input_hashes'].items():assert t.sha(p)==h,p
    core=t.load_module(t.CORE,'t159_core')
    teachers={}
    for name,p in plan['teachers'].items():
        ck=torch.load(p['checkpoint'],map_location='cpu',weights_only=False)
        net=core.Actor(112,2);net.load_state_dict(ck['actor']);net.eval();net.requires_grad_(False)
        teachers[name]=net
    helper=t.load_module(t.HELPER,'t159_helper');legacy=t.load_module(t.STAGE,'t159_legacy')
    for p,h in legacy.EXPECTED.items():assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'t159_runner')
    _,v,mod,prep,_=runner.load_native();N=prep[3];make=legacy.make_short_task(runner.make_task)
    observations=[];targets=[];offsets=[0];evidence=[]
    save('status.json',{'state':'COLLECTING_NEURAL_DEMONSTRATIONS','completed_tasks':0,'total_tasks':6})
    for task in plan['tasks']:
        env=renderer=None
        try:
            env,renderer,obs,actual=helper.make_goal_task(make,v,mod,N,task['seed'],task['angle_deg'],task['distance_m'])
            env.max_steps=task['step_cap'];teacher=teachers[task['teacher']]
            assert abs(float(env._goal_state()[1])-task['distance_m'])<1e-4
            with torch.no_grad():
                for step in range(1,task['step_cap']+1):
                    state=np.asarray(obs,dtype=np.float32).copy()
                    raw=teacher(torch.from_numpy(state).reshape(1,-1)).numpy()[0]
                    observations.append(state);targets.append(raw.copy())
                    action=np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32)
                    obs,_,term,trunc,info=N.step_checked(env,action)
                    if term or trunc or info['reach_goal'] or info['collision']:break
            row={**task,'actual_seed':int(actual),'frames':step,'success':bool(info['reach_goal']),
                'collision':bool(info['collision']),'remaining_m':float(env._goal_state()[1])}
            evidence.append(row)
            assert row['success'] and not row['collision'],('TEACHER_FAILED',row)
            offsets.append(len(observations))
            save('demonstrations.json',{'neural_rollouts_only':True,'tasks':evidence,'offsets':offsets})
            save('status.json',{'state':'COLLECTING_NEURAL_DEMONSTRATIONS','completed_tasks':len(evidence),
                'total_tasks':6,'frames_collected':len(observations)})
            print('T159_NEURAL_DEMONSTRATION',json.dumps(row),flush=True)
        finally:
            if renderer is not None:renderer.close()
            if env is not None:env.close()
    states=np.asarray(observations,dtype=np.float32);actions=np.asarray(targets,dtype=np.float32)
    np.savez_compressed(ROOT/'six_task_success.npz',states=states,raw_actions=actions,episode_offsets=np.asarray(offsets))
    s=torch.from_numpy(states);a=torch.from_numpy(actions)
    actor=core.Actor(112,2);actor.load_state_dict(teachers['T157'].state_dict())
    optimizer=torch.optim.Adam(actor.parameters(),lr=plan['learning_rate'])
    started=time.monotonic()
    for update in range(1,plan['updates']+1):
        ids=np.concatenate([rng.integers(offsets[i],offsets[i+1],64) for i in range(6)])
        pred=actor(s[ids]);loss=(pred-a[ids]).square().mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();actor.net[0].weight.grad[:,MASK]=0
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0);optimizer.step()
        if update%250==0:
            state={'state':'DISTILLING_SINGLE_ACTOR','updates':update,'target_updates':plan['updates'],
                'mse':float(loss.detach()),'seconds':time.monotonic()-started}
            save('status.json',state);print('T159_DISTILLATION',json.dumps(state),flush=True)
    fit=[]
    with torch.no_grad():
        for i in range(6):
            sq=[]
            for first in range(offsets[i],offsets[i+1],512):
                last=min(first+512,offsets[i+1]);sq.append((actor(s[first:last])-a[first:last]).square())
            fit.append(float(torch.cat(sq).mean()))
    ck={'algorithm':'STAGE_MULTI_TASK_POLICY_DISTILLATION','actor':actor.state_dict(),
        'inference_only':True,'td3_resume_compatible':False,'single_network_inference':True,
        'task_identity_input':False,'teachers_used_only_for_training':True,
        'updates':plan['updates'],'seed':plan['seed'],'task_fit_mse':fit,
        'plan_sha256':t.sha(ROOT/'plan.json'),'archive_sha256':t.sha(ROOT/'six_task_success.npz')}
    path=ROOT/'consolidated_actor.pt';tmp=path.with_suffix('.pt.partial')
    torch.save(ck,tmp);os.replace(tmp,path)
    save('status.json',{'state':'DISTILLATION_COMPLETED_NOT_YET_EVALUATED','updates':plan['updates'],
        'checkpoint':str(path),'checkpoint_sha256':t.sha(path),'task_fit_mse':fit,
        'automatic_promotion':False,'single_network_inference':True})
    print('T159_DISTILLATION_COMPLETE',path,flush=True)

if __name__=='__main__':main()
