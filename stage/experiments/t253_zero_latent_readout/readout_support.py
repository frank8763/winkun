"""One preregistered control-rate change, same frozen end-to-end Actor."""
from pathlib import Path
import sys,os,json,time,io,gzip,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t253_zero_latent_readout'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
sys.path.insert(0,str(ROOT/'stage/experiments/t238_ar_density_recovery'))
import copula as AR
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def generate(j,p,native,create,deadline):
    env=None
    try:
        env,obs,_=R.task_env(create,j);obs=C.perturb_anchor(env,obs,j);assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();steps=0;cap=j['case']['step_cap'];term=trunc=False;info={};records=[];nearest=initial=float(env._goal_state()[1])
        while not(term or trunc) and steps<cap and time.time()<deadline:
            state=C.observation(obs,1-steps/cap);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():
                x=torch.tensor(state[None])
                if j['readout']=='zero_latent':
                    m=p.physical_mean(x)[0];action=AR.sample(m,torch.zeros_like(m)).numpy()
                else:
                    assert j['readout']=='clipped_mode';action=p(x).numpy()[0].astype(np.float32)
            for _ in range(min(j['hold_steps'],cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc);reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            records.append((state,action,np.zeros(2,dtype=np.float32),0.,n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'];a={k:np.asarray([r[i] for r in records]) for i,k in enumerate(names)};a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
        r=dict(id=j['id'],group=j['group'],source_reset_id=j['source_reset_id'],hold_steps=j['hold_steps'],readout=j['readout'],complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,decisions=len(records),initial_distance_m=initial,remaining_m=d1,nearest_m=nearest,array_hashes={k:S.digest(v) for k,v in a.items()},integration_sha256=R.arrays_hash(R.integration(env)),job={k:v for k,v in j.items() if k not in ['policy_np']});assert C.model_hash(env)==scene and np.array_equal(goal,env.goal_xy);return dict(row=r,arrays=a)
    finally:
        if env is not None:env.close()
def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);pipe.send(dict(ready=True,pid=os.getpid()));p=None;ph=None
        while True:
            j=pipe.recv()
            if j is None:break
            assert S.digest(j['policy_np'])==j['policy_hash']
            if ph!=j['policy_hash']:p=S.load_np(R.Policy(),j['policy_np']).eval().requires_grad_(False);ph=j['policy_hash']
            pipe.send(generate(j,p,native,create,deadline))
    except EOFError:pass
    except BaseException:
        e=dict(error=traceback.format_exc());C.save(folder/'failure.json',e)
        try:pipe.send(e)
        except OSError:pass
        raise
    finally:pipe.close();log.close()
