"""Compact new evidence with exact observation replay; Stage native Actor only."""
import os
os.sched_setaffinity(0,{int(os.environ.get('STAGE_PROCESS_CPU','0'))})
from pathlib import Path
import sys,json,time,hashlib,traceback,gzip,io,math,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t218_compact_rnd'
spec=importlib.util.spec_from_file_location('stage216_compact_source',ROOT/'stage/experiments/t216_left_specialist/support.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
R,C,np,torch=S.R,S.C,S.np,S.torch
policy,Value,monte_carlo,training_rewards=S.policy,S.Value,S.monte_carlo,S.training_rewards
sys.path.insert(0,str(ROOT/'stage/experiments/t217_rnd_preflight'))
from model import LidarRND

def digest(obj):
    h=hashlib.sha256()
    def put(x):
        if isinstance(x,torch.Tensor):put(x.detach().cpu().numpy());return
        if isinstance(x,np.ndarray):h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes());return
        if isinstance(x,dict):
            for k in sorted(x,key=lambda t:(type(t).__name__,repr(t))):put(k);put(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for v in x:put(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    put(obj);return h.hexdigest()
def state_np(p):return {k:v.detach().cpu().numpy().copy() for k,v in p.state_dict().items()}
def load_np(p,state):p.load_state_dict({k:torch.as_tensor(v) for k,v in state.items()});return p
def used(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def save_gzip(path,value,budget_guard=None):
    b=io.BytesIO();torch.save(value,b);data=gzip.compress(b.getvalue(),compresslevel=6,mtime=0);assert len(data)<2200*1024
    if budget_guard is not None:budget_guard(len(data))
    path.write_bytes(data);restored=torch.load(io.BytesIO(gzip.decompress(data)),map_location='cpu',weights_only=False)
    assert digest(restored)==digest(value)
    return {'path':str(path),'sha256':C.sha(path),'bytes':len(data),'content_sha256':digest(value)}
def worker(pipe,index,deadline):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=(folder/'worker.log').open('a',buffering=1);sys.stdout=sys.stderr=log;env=None;cache={}
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=C.native_setup(folder)
        while True:
            message=pipe.recv()
            if message is None:break
            kind,job=message
            if kind=='rollout':
                assert digest(job['policy_np'])==job['policy_hash']
                p=load_np(R.Policy(),job['policy_np']).eval().requires_grad_(False);cache={job['policy_hash']:p}
            else:
                assert kind=='replay';p=cache[job['policy_hash']]
            env,obs,_=R.task_env(create,job);assert not env._collision();model_hash=C.model_hash(env);goal=env.goal_xy.copy();rng=np.random.default_rng(job['seed'])
            steps=0;cap=job['case']['step_cap'];term=trunc=False;info={};records=[];nearest=float(env._goal_state()[1]);initial_d=nearest
            replay=None
            if kind=='replay':
                assert C.sha(job['trace'])==job['trace_sha256']
                with np.load(job['trace']) as z:replay={k:z[k].copy() for k in z.files}
            i=0
            while not(term or trunc) and steps<cap and time.time()<deadline:
                state=C.observation(obs,1-steps/cap);action,mean,lp=R.draw(p,state,job['mode'],rng)
                if replay is not None:
                    assert np.array_equal(action,replay['actions'][i]);assert np.array_equal(mean,replay['means'][i]);assert lp==float(replay['logp'][i])
                before=steps;d0=float(env._goal_state()[1]);pose=np.asarray(env._pose()[:3])
                count=min(100,cap-steps) if replay is None else int(replay['native_counts'][i])
                for _ in range(count):
                    obs,_,term,trunc,info=native.step_checked(env,action);steps+=1;nearest=min(nearest,float(env._goal_state()[1]))
                    if term or trunc:break
                n=steps-before;d1=float(env._goal_state()[1]);discount=R.GAMMA**n;terminal=term or trunc
                event=float(info['reach_goal'])-float(info['collision'])-.5*float(trunc)
                reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0)
                records.append((state,action,mean,lp,n,C.observation(obs,1-steps/cap),pose,np.asarray(env._pose()[:3]),[info['collision'],info['reach_goal'],trunc],reward));i+=1
            arrays={k:np.asarray([r[j] for r in records]) for j,k in enumerate(['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards'])}
            arrays['native_counts']=arrays['native_counts'].astype(np.uint8);arrays['end_flags']=arrays['end_flags'].astype(np.uint8)
            complete=bool(term or trunc);array_hashes={k:digest(v) for k,v in arrays.items()};integration_hash=R.arrays_hash(R.integration(env))
            if kind=='replay':
                assert complete and array_hashes==job['array_hashes'] and integration_hash==job['integration_sha256']
                row={'id':job['id'],'verified':True,'physics_steps':steps,'array_hashes':array_hashes,'integration_sha256':integration_hash}
            else:
                path=OUT/'episodes'/(job['id']+'.npz');path.parent.mkdir(exist_ok=True);assert not path.exists()
                compact={k:v for k,v in arrays.items() if k not in ('states','next_states')};compact['goal_xy']=goal
                np.savez_compressed(path,**compact);assert path.stat().st_size<32*1024
                row={k:job[k] for k in ('id','arm','round','role','seed','mode','case','exact_snapshot','policy_hash')}
                row.update(complete=complete,physics_steps=steps,decisions=len(records),success=bool(info.get('reach_goal')),collision=bool(info.get('collision')),timeout=bool(trunc),
                    remaining_m=float(env._goal_state()[1]),nearest_m=nearest,initial_distance_m=initial_d,trace=str(path),trace_sha256=C.sha(path),
                    array_hashes=array_hashes,integration_sha256=integration_hash)
            assert model_hash==C.model_hash(env) and np.array_equal(goal,env.goal_xy);env.close();env=None
            pipe.send({'row':row,'arrays':arrays})
    except (EOFError,SystemExit):pass
    except BaseException:
        pipe.send({'error':traceback.format_exc()});raise
    finally:
        if env is not None:env.close()
        pipe.close();log.close()
