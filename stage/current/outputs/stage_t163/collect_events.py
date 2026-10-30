"""Frozen Stage trajectory diagnostics, never a new policy search or training."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import time
import numpy as np
import torch
import mujoco
from event_labels import first_event_labels, check_boundary_cases

ROOT = Path(__file__).resolve().parent

def save(name, obj):
    p = ROOT/name; tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(obj, indent=2)+'\n'); os.replace(tmp,p)

def main():
    lock = (ROOT/'process.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(), 'Do not duplicate'
    plan = json.loads((ROOT/'plan.json').read_text())
    os.environ['C98_JOB'] = os.environ['TD3_JOB'] = str(ROOT)
    os.environ['C22_SCREEN_ONLY'] = '0'
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('t163_source', ROOT.parent/'stage_t148/train.py')
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    for p, h in plan['input_hashes'].items(): assert t.sha(p)==h, p
    helper=t.load_module(t.HELPER,'t163_helper'); legacy=t.load_module(t.STAGE,'t163_legacy')
    for p,h in legacy.EXPECTED.items(): assert t.sha(p)==h
    runner=legacy.module(legacy.SRC,'t163_runner')
    _,v,mod,prep,_=runner.load_native(); native=prep[3]
    make=legacy.make_short_task(runner.make_task); core=t.load_module(t.CORE,'t163_core')
    check_boundary_cases()
    agents={}; weights={}
    for name, model in plan['models'].items():
        agent=t.create_agent(core)
        ck=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        agent.actor.load_state_dict(ck['actor']); agent.actor.eval().requires_grad_(False)
        agents[name]=agent; weights[name]={k:x.clone() for k,x in agent.actor.state_dict().items()}
        del ck
    deadline=time.monotonic()+plan['wall_seconds']; rows=[]
    save('status.json',{'state':'RUNNING','completed':0,'total':len(plan['cases'])})
    for case in plan['cases']:
        env=renderer=None
        try:
            env,renderer,obs,actual=helper.make_goal_task(make,v,mod,native,case['seed'],case['angle_deg'],case['distance_m'])
            env.max_steps=case['step_cap']; start=float(env.data.time)
            states=[]; actions=[]; times=[]; poses=[]; kind='STEP_CAP'
            for step in range(1,case['step_cap']+1):
                if time.monotonic()>deadline: raise TimeoutError('NO_PARTIAL_SUCCESS')
                raw=agents[case['model']].act(obs)
                if (step-1)%plan['sample_every_physics_steps']==0:
                    states.append(np.asarray(obs).copy()); actions.append(raw.copy())
                    times.append(float(env.data.time)-start); poses.append(env._pose())
                obs,_,term,trunc,info=native.step_checked(env,np.array([(raw[0]+1)/2,raw[1]],dtype=np.float32))
                assert not (info['reach_goal'] and info['collision'])
                if term or trunc:
                    kind='COLLISION' if info['collision'] else 'GOAL' if info['reach_goal'] else 'TRUNCATED' if trunc else 'TERMINATED'
                    break
            duration=float(env.data.time)-start
            labels,valid=first_event_labels(times,duration,kind,plan['horizons_s'])
            contacts=[]
            for i in range(env.data.ncon):
                contact=env.data.contact[i]
                contacts.append({'geom1':int(contact.geom1),'geom2':int(contact.geom2),
                    'name1':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom1)),
                    'name2':mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom2)),
                    'distance':float(contact.dist)})
            path=ROOT/(case['id']+'.npz')
            np.savez_compressed(path,states=np.asarray(states,dtype=np.float32),
                raw_actions=np.asarray(actions,dtype=np.float32),sample_times_s=np.asarray(times),
                poses_diagnostic_only=np.asarray(poses),horizons_s=np.asarray(plan['horizons_s']),
                first_event_class=labels,label_observed=valid)
            row={**case,'actual_seed':int(actual),'steps':step,'duration_s':duration,'end':kind,
                'remaining_m':float(env._goal_state()[1]),'final_pose':list(env._pose()),
                'samples':len(times),'npz':str(path),'npz_sha256':t.sha(path),
                'final_contacts_including_floor':contacts,'unknown_labels':int((~valid).sum()),
                'label_counts_by_horizon':[{str(c):int((labels[:,j]==c).sum()) for c in (-1,0,1,2)} for j in range(len(plan['horizons_s']))]}
            rows.append(row)
            with (ROOT/'episodes.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            save('status.json',{'state':'RUNNING','completed':len(rows),'total':len(plan['cases']),'last':case['id']})
            print('T163_CASE',case['id'],kind,step,flush=True)
        finally:
            if renderer is not None: renderer.close()
            if env is not None: env.close()
    for name,agent in agents.items(): assert all(torch.equal(weights[name][k],x) for k,x in agent.actor.state_dict().items())
    for p,h in plan['input_hashes'].items(): assert t.sha(p)==h
    save('summary.json',{'experiment':plan['experiment'],'plan_sha256':t.sha(ROOT/'plan.json'),
        'episodes':rows,'actor_and_sources_unchanged':True,'training':False,
        'generalization_claim':False,'risk_prediction_trained':False,
        'limitation':'6 diagnostic episodes, correlated frames; split by episode/layout, never random frames. Labels describe the recorded behavior policy only. No counterfactual or safety guarantee.'})
    save('status.json',{'state':'COMPLETE','completed':len(rows),'total':len(plan['cases'])})

if __name__=='__main__': main()
