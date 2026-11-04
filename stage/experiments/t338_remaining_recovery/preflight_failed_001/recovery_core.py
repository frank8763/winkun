from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t338_capg_recovery'))
from t338_core import *
sys.path.insert(0,str(ROOT/'stage/experiments/t338_failure_review'))
from canonical import CanonicalActor
HERE=Path(__file__).resolve().parent
OLD_OUT=ROOT/'stage/runs/t338_capg_recovery'
OUT=ROOT/'stage/runs/t338_remaining_recovery'
Actor=CanonicalActor
class Learner:
    def __init__(self,p,arm):
        self.actor=Actor(p,arm);self.opt=torch.optim.Adam(self.actor.trainable(),lr=p['actor_lr']);self.calls=0
    def state(self):return dict(actor={k:v.detach().clone() for k,v in self.actor.state_dict().items() if not k.startswith('fixed.')},optimizer=copy.deepcopy(self.opt.state_dict()))
    def load(self,s):
        self.actor.load_state_dict(dict(self.actor.state_dict(),**s['actor']),strict=True);self.opt.load_state_dict(s['optimizer'])
    def hashes(self):return {k:digest(v) for k,v in self.state().items()}
def cached(p):return [dict(row=r,arrays=arrays(r['path'])) for r in p['cached_training_rows']]
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    assert K.S.used(OUT)+K.S.used(OLD_OUT)<=p['combined_final_bytes']
    assert K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
