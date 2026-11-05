"""Frozen recurrent feature-dependent exploration; no action-density shortcut."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t353_offline_night_chain'))
from night_core import *
import night_core as NC
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t359_coherent_narrow';DOC=ROOT/'stage/docs/narrow_20261105'
PHYSICAL_KEYS=['states','next_states','actions','native_counts','poses','next_poses','end_flags','rewards','history_poses','history_goals','history_times','hidden_before','last_hidden_after','initial_native_integration','final_native_integration']


class CoherentActor(NightActor):
    def __init__(self,p):
        super().__init__(p,0.);self.latent=np.zeros((16,2),np.float64)
    def phash(self):
        return digest(dict(parent=P.C.Actor.phash(self),sampler='unit_GRU_feature_episode_latent',sigma=self.p['raw_sigma'],latent=self.latent))
    def step(self,x,hidden=None):
        _,loc,h,_=P.C.Actor.step(self,x,hidden)
        feat=h[0,0].detach();norm=float(torch.linalg.vector_norm(feat));assert norm>1e-12
        feature=feat/norm;noise=feature@torch.tensor(self.latent,dtype=torch.float64)
        raw=loc.detach()+torch.tensor(self.p['raw_sigma'],dtype=torch.float64)*noise
        action=raw.clamp(-1,1).float()
        self.tape.append(dict(locations=loc[0].detach().numpy().copy(),features=feature.numpy().copy(),latent_action_noise=noise.numpy().copy(),raw_actions=raw[0].numpy().copy()))
        return action,loc,h,loc


def worker(pipe,index,p,counts):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);P.C.C.OUT=Path(p['out'])
    folder=Path(p['out'])/'workers'/str(index);folder.mkdir(parents=True);log=N.LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;w=None
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('T359 forbids all optimizers'))
    try:
        module,native,create,w,_=P.C.B.setup(folder/'native',p);original=copy.deepcopy(module.ACTIVE);q=CoherentActor(p).eval().requires_grad_(False)
        pipe.send(dict(ready=True,pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            q.load_learned(read_model(p['initial_model']));q.latent=np.asarray(j['latent'],np.float64);q.reset_noise();assert q.phash()==j['policy_hash']
            module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=j['circle_shift_x_m']
            item=P.C.rollout(p,q,j,native,create,w,j['deadline_epoch'],counts,index,total);a=item['arrays']
            assert len(q.tape)==len(a['states'])
            for name in q.tape[0]:a[name]=np.asarray([v[name] for v in q.tape])
            a['latent_matrix']=q.latent.copy();r=item['row'];r.update(latent_matrix_sha256=digest(q.latent),
                latent_matrix_logdensity=float(-.5*np.square(q.latent).sum()-16*np.log(2*np.pi)),
                latent_density_is_not_action_density=True,per_step_action_logp=None,circle_shift_x_m=j['circle_shift_x_m'],
                reset_family=j['reset_family'],matrix_family=j['matrix_family'],antithetic_sign=j['antithetic_sign'],
                physical_digest=digest({k:a[k] for k in PHYSICAL_KEYS}),array_hashes={k:digest(v) for k,v in a.items()})
            total+=r['physics_steps'];pipe.send(item)
    except EOFError:pass
    except BaseException:
        error=dict(error=traceback.format_exc(),actual_physics=counts[index]);save(folder/'failure.json',error)
        try:pipe.send(error)
        except (OSError,EOFError):pass
    finally:
        if w is not None:w.close()
        pipe.close();log.close()
