"""Untrained numerical repair prototype; original registered T338 is immutable."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t338_capg_recovery'))
from t338_core import Actor,torch

class CanonicalActor(Actor):
    def cache(self,x):
        if len(x)==1:return super().cache(x)
        rows=[self.cache(row[None]) for row in x]
        return tuple(torch.cat([v[k] for v in rows]) for k in range(2))
    def stats(self,x,cache=None):
        if len(x)==1:return super().stats(x,cache)
        rows=[self.stats(x[k:k+1],None if cache is None else tuple(v[k:k+1] for v in cache)) for k in range(len(x))]
        return tuple(torch.cat([v[k] for v in rows]) for k in range(2))
    def logp(self,loc,sigma,raw,atom):
        if len(loc)==1:return super().logp(loc,sigma,raw,atom)
        return torch.cat([self.logp(loc[k:k+1],sigma[k:k+1],raw[k:k+1],atom[k:k+1]) for k in range(len(loc))])
    def draw(self,x,z,deterministic=False):
        if len(x)==1:return super().draw(x,z,deterministic)
        rows=[self.draw(x[k:k+1],z[k:k+1],deterministic) for k in range(len(x))]
        return {name:torch.cat([v[name] for v in rows]) for name in rows[0]}

def numerical_worker(pipe,p,arm):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=CanonicalActor(p,arm).requires_grad_(False);cmd=pipe.recv()
    q.load_state_dict({k:torch.tensor(v) for k,v in cmd['policy_np'].items()},strict=True)
    with torch.no_grad():v=q.draw(torch.tensor(cmd['x']),torch.tensor(cmd['z']))
    pipe.send({k:x.numpy() for k,x in v.items()});pipe.close()
