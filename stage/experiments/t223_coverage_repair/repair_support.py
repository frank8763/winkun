"""Stage-only reconstructible self-imitation coverage diagnostic."""
from pathlib import Path
import sys,os,json,time,math,traceback,copy
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t223_coverage_repair'
sys.path.insert(0,str(ROOT/'stage/experiments/t221_left_success'))
import left_support as L
S,C,R,np,torch=L.S,L.C,L.R,L.np,L.torch
TS=L.TS
native_setup=S.native_setup
digest,used,state_np,load_np=S.digest,S.used,S.state_np,S.load_np

def generate(job,p,native,create,deadline):
    env=None
    try:
        env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job);assert not env._collision()
        scene=C.model_hash(env);goal=env.goal_xy.copy();rng=np.random.default_rng(job['seed']);latent=None
        steps=0;cap=job['case']['step_cap'];term=trunc=False;info={};records=[];nearest=initial_d=float(env._goal_state()[1])
        while not(term or trunc) and steps<cap and time.time()<deadline:
            state=C.observation(obs,1-steps/cap)
            if job['mode']=='correlated_diagnostic':
                innovation=rng.normal(size=2);latent=innovation if latent is None else .98*latent+math.sqrt(1-.98**2)*innovation
                with torch.no_grad():m=p.physical_mean(torch.tensor(state[None])).double()[0];a=TS.copula_action(m,latent)
                action=a.numpy();mean=m.float().numpy();lp=0.
            else:action,mean,lp=R.draw(p,state,job['mode'],rng)
            before=steps;d0=float(env._goal_state()[1]);pose=np.asarray(env._pose()[:3])
            for _ in range(min(100,cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc
            event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
            reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            records.append((state,action,mean,lp,n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
        arrays={k:np.asarray([r[j] for r in records]) for j,k in enumerate(['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'])};arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
        row={k:job[k] for k in ('id','arm','seed','mode','delta_x_m','delta_y_m','delta_yaw_deg')};row.update(complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,initial_distance_m=initial_d,remaining_m=d1,nearest_m=nearest,array_hashes={k:digest(v) for k,v in arrays.items()},integration_sha256=R.arrays_hash(R.integration(env)))
        assert C.model_hash(env)==scene and np.array_equal(goal,env.goal_xy);return {'row':row,'arrays':arrays}
    finally:
        if env is not None:env.close()

def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=native_setup(folder);pipe.send({'ready':True,'pid':os.getpid()})
        p=None;ph=None
        while True:
            job=pipe.recv()
            if job is None:break
            assert digest(job['policy_np'])==job['policy_hash']
            if ph!=job['policy_hash']:p=load_np(R.Policy(),job['policy_np']).eval().requires_grad_(False);ph=job['policy_hash']
            pipe.send(generate(job,p,native,create,deadline))
    except EOFError:pass
    except BaseException:
        error={'error':traceback.format_exc()};C.save(folder/'failure.json',error)
        try:pipe.send(error)
        except OSError:pass
        raise
    finally:pipe.close();log.close()

def training_groups(plan,items):
    parent,groups=L.groups(plan);p=S.policy(parent)
    for side,index in [('RIGHT',6),('LEFT',7)]:
        extra=[i['arrays'] for i in items if i['row']['arm']==side and i['row']['complete'] and i['row']['success'] and not i['row']['collision']]
        for d in extra:
            x=torch.tensor(d['states'])
            with torch.no_grad():y=C.ParentPolicy.forward(p,x).detach() if side=='RIGHT' else torch.tensor(d['actions'])
            groups[index]=(torch.cat((groups[index][0],x)),torch.cat((groups[index][1],y)))
    return groups

def reconstruct():
    """Bounded rebuild: six frozen diagnostic episodes, then fixed3000 updates.

    Calls native physics (<=140004 steps/180s source reconstruction); must be
    separately budgeted by the caller. No network update is deployed until all
    source hashes and final model/optimizer hashes match. No model/trace export.
    """
    plan=json.loads((HERE/'protocol.json').read_text());recipe=json.loads((OUT/'recipe.json').read_text());assert recipe['independent_reconstruction_exact']
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=S.policy(torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy']).eval().requires_grad_(False)
    native,create=native_setup(OUT/'rebuild_native');items=[];deadline=time.time()+180;physics=0
    for job,source in zip(plan['jobs'],recipe['source_rows']):
        item=generate(job,p,native,create,deadline);physics+=item['row']['physics_steps'];assert item['row']==source;items.append(item)
    assert physics<=140004 and time.time()<deadline
    p,opt,count=L.train_recipe(plan,training_groups(plan,items),plan['updates'])
    assert digest(p.state_dict())==recipe['final_policy_sha256'] and digest(opt.state_dict())==recipe['final_optimizer_sha256']
    return p.eval().requires_grad_(False)
