"""Same exact Stage native state, one precomputed action, frozen continuation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t259_closed_loop_recovery'))
from closed_loop_learning import L,S,C,R,np,torch,policy
import io,json,time


def simulate(plan,proposal,p,native,create,deadline,guard,on_step):
    env=None
    try:
        env,obs,_=R.task_env(create,plan['job']);obs=C.perturb_anchor(env,obs,plan['job'])
        saved=C.read_snapshot(plan['snapshot']);obs=C.restore(env,saved)
        begin=plan['start_step'];cap=plan['job']['case']['step_cap']
        context=json.loads(Path(plan['context']).read_text());previous=np.array(context['previous_action'],dtype=np.float32);previous_count=context['previous_count']
        assert np.array_equal(R.integration(env),saved['physics']) and env.step_count==begin
        x=L.packet(env,obs,begin,cap,previous,previous_count)
        assert S.digest(x)==context['packet120_sha256'] and np.array_equal(x,np.array(plan['proposals']['context120'],dtype=np.float32))
        scene=C.model_hash(env);goal=env.goal_xy.copy();steps=begin;term=trunc=False;records=[];info={}
        while not(term or trunc) and steps<cap:
            guard();assert time.time()<deadline
            x=L.packet(env,obs,steps,cap,previous,previous_count);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():action=p.step(torch.tensor(x[None]))[0][0].numpy()
            if steps==begin:
                assert np.array_equal(action,np.array(plan['proposals']['actions']['control'],dtype=np.float32))
                action=np.array(plan['proposals']['actions'][proposal],dtype=np.float32)
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;on_step()
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);disc=R.GAMMA**n;terminal=term or trunc
            event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-disc)/(1-R.GAMMA)+.03*d0-disc*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            nxt=L.packet(env,obs,steps,cap,action,n)
            records.append((x,action,np.zeros(2,dtype=np.float32),0.,n,nxt,pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
            previous=action.copy();previous_count=n
        keys=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards']
        a={k:np.asarray([r[i] for r in records]) for i,k in enumerate(keys)}
        a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
        assert C.model_hash(env)==scene and np.array_equal(env.goal_xy,goal)
        score16=float(info.get('reach_goal',False))-float(info.get('collision',False)) if steps-begin<=8000 and (term or trunc) else 0.
        row=dict(proposal=proposal,start_step=begin,absolute_end_step=steps,physics_steps=steps-begin,
            complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),remaining_m=d1,
            actual_score16=score16,predicted_score16=plan['proposals']['score16'][proposal],
            array_hashes={k:S.digest(v) for k,v in a.items()},integration_sha256=R.arrays_hash(R.integration(env)),
            policy_hash=plan['policy_hash'],diagnostic_intervention_only=True,not_on_policy_or_BC=True)
        return row,a
    finally:
        if env is not None:env.close()
