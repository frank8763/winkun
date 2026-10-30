"""Verified real-success BC consolidation and final closed-loop neural evaluation."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import time,fcntl,traceback,shutil
from common import *

def main():
    lock=(OUT/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate or implicit resume'
    plan=json.loads((HERE/'protocol.json').read_text())
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    assert shutil.disk_usage(OUT).free>40*1024**3
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);rng=np.random.default_rng(plan['seed'])
    start=time.monotonic();deadline=start+plan['wall_seconds'];native,create=native_setup(OUT/'native');env=None;spent=0;replays=[];groups=[]
    try:
        save(OUT/'status.json',{'state':'VERIFY_RL_SUCCESS_DATA','physics_steps':0})
        for source in plan['successful_sources']:
            case=next(c for c in plan['primary_cases'] if c['id']==source['case']);env,obs,actual=create(case);env.max_steps=case['step_cap']
            with np.load(source['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            state_error=physics_error=0.;steps=0;term=trunc=False;goal=env.goal_xy.copy()
            for i,action in enumerate(a['actions']):
                state_error=max(state_error,float(np.abs(observation(obs,1-steps/case['step_cap'])-a['states'][i]).max()))
                assert state_error<1e-6,state_error
                for _ in range(int(a['native_counts'][i])):
                    assert time.monotonic()<deadline
                    obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;spent+=1
                    physics_error=max(physics_error,float(np.abs(np.asarray(env._pose()[:2])-a['physical_xy'][steps-1]).max()))
                    assert physics_error<1e-9,physics_error
                    if term or trunc:break
                if term or trunc:break
            assert steps==source['physics_steps'] and info['reach_goal'] and not info['collision'] and np.array_equal(goal,env.goal_xy)
            row={'source_id':source['id'],'case':case['id'],'physics_steps':steps,'safe_goal':True,'max113_observation_error':state_error,'max_physics_xy_error':physics_error,'scope':'Saved-action data replay only, not policy deployment'}
            replays.append(row);groups.append((case['id'],a['states'].astype(np.float32),a['actions'].astype(np.float32)));env.close();env=None
        save(OUT/'data_replay_verification.json',{'state':'PASS','rows':replays,'physics_steps':spent})
        with np.load(plan['retention_demonstrations'],allow_pickle=False) as z:
            states=z['states'].copy();actions=z['actions'].copy();offsets=z['episode_offsets'].copy();counts=z['native_counts'].copy()
        actions[:,0]=.5*(actions[:,0]+1)
        for j in range(6):
            lo,hi=offsets[j:j+2];clock=1-np.r_[0,np.cumsum(counts[lo:hi-1])]/plan['primary_cases'][j]['step_cap'];groups.append((plan['primary_cases'][j]['id'],np.column_stack([states[lo:hi],clock]).astype(np.float32),actions[lo:hi]))
        tensors=[(name,torch.tensor(s,device='cuda'),torch.tensor(a,device='cuda')) for name,s,a in groups]
        policy=Policy().cuda();policy.load_state_dict(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']);policy.logstd.requires_grad_(False)
        optimizer=torch.optim.Adam([{'params':list(policy.body.parameters()),'lr':3e-5},{'params':list(policy.native_head.parameters())+list(policy.direct_head.parameters()),'lr':3e-4}])
        for update in range(plan['supervised_updates']):
            assert time.monotonic()<deadline
            losses=[]
            for name,s,a in tensors:
                ids=rng.integers(len(s),size=32);losses.append((policy(s[ids])-a[ids]).square().mean())
            loss=torch.stack(losses).mean();assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();policy.body.net[0].weight.grad[:,[57,58,74,75,86,87,98,99,110,111]]=0
            torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            if (update+1)%500==0:
                status={'state':'SUPERVISED_CONSOLIDATION','updates':update+1,'loss':float(loss.detach()),'elapsed_s':time.monotonic()-start};save(OUT/'status.json',status);print(json.dumps(status),flush=True)
        policy=policy.cpu().eval().requires_grad_(False);final=OUT/'final_policy.pt'
        torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'supervised_updates':plan['supervised_updates'],'rl_updates':0,'protocol_sha256':sha(HERE/'protocol.json'),'parent_sha256':sha(plan['checkpoint']),'data_replays':replays},final)
        losses={}
        with torch.no_grad():
            for name,s,a in groups:losses[name]=float((policy(torch.tensor(s))-torch.tensor(a)).square().mean())
        rows=[]
        save(OUT/'status.json',{'state':'FROZEN_EVALUATION','checkpoint_sha256':sha(final)})
        for case in plan['primary_cases']:
            env,obs,actual=create(case);env.max_steps=case['step_cap'];steps=0;term=trunc=False;nearest=float(env._goal_state()[1]);trace=[]
            while not(term or trunc):
                assert time.monotonic()<deadline
                state=observation(obs,1-steps/case['step_cap'])
                with torch.no_grad():action=policy(torch.tensor(state.reshape(1,-1))).numpy()[0]
                action=action_clip(action,'full_range');trace.append([steps,*env._pose(),*action])
                for _ in range(min(100,case['step_cap']-steps)):
                    obs,_,term,trunc,info=native.step_checked(env,action);obs=np.asarray(obs,dtype=np.float32);steps+=1;spent+=1
                    nearest=min(nearest,float(env._goal_state()[1]))
                    if term or trunc:break
            path=OUT/(case['id']+'_frozen_trace.npz');np.savez_compressed(path,trace=np.asarray(trace))
            row={'case':case['id'],'distance_m':case['distance_m'],'physics_steps':steps,'success':bool(info['reach_goal']),'collision':bool(info['collision']),'timeout':bool(trunc and not term),'nearest_m':nearest,'remaining_m':float(env._goal_state()[1]),'simulation_s':steps*.002,'trace_sha256':sha(path),'deployment':'Single closed-loop neural policy; no saved actions or routes'}
            rows.append(row)
            with (OUT/'evaluation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            env.close();env=None
        assert spent<=plan['max_physics']
        result={'state':'COMPLETE','scores':{'old6':{'safe_success':sum(r['success'] and not r['collision'] for r in rows[:6]),'n':6},'original5m':{'safe_success':sum(r['success'] and not r['collision'] for r in rows[6:]),'n':2}},'evaluation':rows,'training_sources':[g[0] for g in groups],'training_mse':losses,'supervised_updates':plan['supervised_updates'],'rl_updates':0,'new_training_physics':0,'data_replay_physics':sum(r['physics_steps'] for r in replays),'frozen_evaluation_physics':sum(r['physics_steps'] for r in rows),'checkpoint_sha256':sha(final),'automatic_promotion':False,'incumbent':'T159','scope':plan['scope']}
        save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        save(OUT/'failure.json',{'state':'FAILED','traceback':traceback.format_exc(),'physics_steps':spent});raise
    finally:
        if env is not None:env.close()

if __name__=='__main__':main()
