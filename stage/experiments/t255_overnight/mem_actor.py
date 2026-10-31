"""Matched-parameter recurrent/nonrecurrent end-to-end Stage Actor."""
from pathlib import Path
import sys,torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t251_actuator_recovery'))
import actuator_support as AS
C,R,S,np=AS.C,AS.R,AS.S,AS.np
DIM=120;HIDDEN=32
class Actor(torch.nn.Module):
    def __init__(self,parent_state,memory_on=True,seed=0):
        super().__init__();torch.manual_seed(seed);self.memory_on=bool(memory_on)
        self.base=AS.policy(parent_state);self.encoder=torch.nn.Sequential(torch.nn.Linear(DIM,64),torch.nn.Tanh());self.core=torch.nn.GRU(64,HIDDEN)
        with torch.no_grad():
            for block in self.core.weight_hh_l0.chunk(3):torch.nn.init.orthogonal_(block)
            torch.nn.init.xavier_uniform_(self.core.weight_ih_l0);self.core.bias_ih_l0.zero_();self.core.bias_hh_l0.zero_()
        self.residual=torch.nn.Linear(HIDDEN,2);torch.nn.init.zeros_(self.residual.weight);torch.nn.init.zeros_(self.residual.bias);self.base.logstd.requires_grad_(False)
    @staticmethod
    def mode(raw):return torch.tanh(torch.atanh(raw.clamp(-.999999,.999999)))
    def step(self,x,hidden=None):
        assert x.ndim==2 and x.shape[-1]==DIM
        if hidden is None or not self.memory_on:hidden=x.new_zeros((1,len(x),HIDDEN))
        h,new=self.core(self.encoder(x)[None],hidden);raw=C.ParentPolicy.forward(self.base,x[:,:117].contiguous())+self.residual(h[0]);return self.mode(raw),R.center(raw),new,raw
    def sequence(self,x):
        assert x.ndim==3 and x.shape[-1]==DIM
        t,b,_=x.shape;encoded=self.encoder(x)
        if self.memory_on:h,_=self.core(encoded)
        else:
            h,_=self.core(encoded.reshape(1,t*b,64));h=h.reshape(t,b,HIDDEN)
        base=C.ParentPolicy.forward(self.base,x[...,:117].reshape(t*b,117).contiguous()).reshape(t,b,2);raw=base+self.residual(h);return self.mode(raw),R.center(raw),raw
def packet(env,obs,steps,cap,previous_action,previous_count):
    a=np.asarray(previous_action,dtype=np.float32);assert a.shape==(2,) and previous_count>=0
    return np.r_[AS.packet(env,obs,steps,cap),a,np.float32(previous_count/100)].astype(np.float32)
