"""One preregistered control-rate change, same frozen end-to-end Actor."""
from pathlib import Path
import sys,os,json,time,io,gzip,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t249_actuator_observation'
sys.path.insert(0,str(ROOT/'stage/experiments/t229_valid_start_acceptance'))
import valid_support as V
S,C,R,np,torch=V.S,V.C,V.R,V.np,V.torch
class InputFusion(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.proprio=torch.nn.Linear(4,600,bias=False);torch.nn.init.zeros_(self.proprio.weight)
    @property
    def weight(self):return self.base.weight
    def forward(self,x):
        assert x.shape[-1]==117
        return self.base(x[...,:113])+self.proprio(x[...,113:])
def policy(state=None):
    p=R.Policy()
    if state is not None and 'body.net.0.weight' in state:p.load_state_dict(state)
    p.body.net[0]=InputFusion(p.body.net[0])
    if state is not None and 'body.net.0.base.weight' in state:p.load_state_dict(state)
    p.requires_grad_(True);p.logstd.requires_grad_(False);return p
def packet(env,obs,steps,cap):
    ids=[int(env.model.actuator_trnid[getattr(env,n),0]) for n in ['id_lf_steer','id_rf_steer']];qa=[int(env.model.jnt_qposadr[j]) for j in ids];va=[int(env.model.jnt_dofadr[j]) for j in ids]
    extra=np.clip(np.r_[np.asarray(env.data.qpos[qa])/.5,np.asarray(env.data.qvel[va])/3.],-3,3).astype(np.float32)
    x=np.r_[C.observation(obs,1-steps/cap),extra].astype(np.float32);assert x.shape==(117,) and np.isfinite(x).all();return x
def read(path):return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)['policy']
def generate(j,p,native,create,deadline):
    env=None
    try:
        env,obs,_=R.task_env(create,j);obs=C.perturb_anchor(env,obs,j);assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();steps=0;cap=j['case']['step_cap'];term=trunc=False;info={};records=[];nearest=initial=float(env._goal_state()[1])
        while not(term or trunc) and steps<cap and time.time()<deadline:
            state=packet(env,obs,steps,cap);pose=np.asarray(env._pose()[:3]);d0=float(env._goal_state()[1]);before=steps
            with torch.no_grad():action=p(torch.tensor(state[None])).numpy()[0].astype(np.float32)
            for _ in range(min(j['hold_steps'],cap-steps)):
                obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                if term or trunc:break
            n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc;event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc);reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
            records.append((state,action,np.zeros(2,dtype=np.float32),0.,n,packet(env,obs,steps,cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward))
        names=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'];a={k:np.asarray([r[i] for r in records]) for i,k in enumerate(names)};a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
        base_arrays={k:(v[:,:113] if k in ['states','next_states'] else v) for k,v in a.items()}
        r=dict(id=j['id'],group=j['group'],source_reset_id=j['source_reset_id'],hold_steps=j['hold_steps'],complete=bool(term or trunc),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),physics_steps=steps,decisions=len(records),initial_distance_m=initial,remaining_m=d1,nearest_m=nearest,array_hashes={k:S.digest(v) for k,v in a.items()},base_array_hashes={k:S.digest(v) for k,v in base_arrays.items()},integration_sha256=R.arrays_hash(R.integration(env)),job={k:v for k,v in j.items() if k not in ['policy_np']});assert C.model_hash(env)==scene and np.array_equal(goal,env.goal_xy);return dict(row=r,arrays=a)
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
            if ph!=j['policy_hash']:p=S.load_np(policy(),j['policy_np']).eval().requires_grad_(False);ph=j['policy_hash']
            pipe.send(generate(j,p,native,create,deadline))
    except EOFError:pass
    except BaseException:
        e=dict(error=traceback.format_exc());C.save(folder/'failure.json',e)
        try:pipe.send(e)
        except OSError:pass
        raise
    finally:pipe.close();log.close()
