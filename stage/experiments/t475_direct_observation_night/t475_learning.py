"""One zero-initialized observation path inside the causal Actor.

The full input327 weights remain frozen. The new path sees the existing120
observations directly; it neither routes tasks nor calls a runtime teacher.
"""
from pathlib import Path
import copy,gzip,importlib.util,io
ROOT=Path(__file__).resolve().parents[3]
sp=importlib.util.spec_from_file_location('stage475_pinned474_learning',ROOT/'stage/experiments/t474_verified_joint_night/t474_learning.py')
B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
H,M,P,E,R,torch,np=B.H,B.M,B.P,B.E,B.R,B.torch,B.np
read,save,sha,digest=B.read,B.save,B.sha,B.digest
ARCH='frozen327_GRU64_plus_zero_initialized_current120_32_2_observation_logit_path'
class ObservationHead(torch.nn.Module):
    def __init__(self,base,seed):
        super().__init__();self.base=base.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);self.observation=torch.nn.Sequential(torch.nn.Linear(120,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).double()
            torch.nn.init.zeros_(self.observation[-1].weight);torch.nn.init.zeros_(self.observation[-1].bias)
        self.packet=None
    def forward(self,h):
        assert self.packet is not None and self.packet.shape[:-1]==h.shape[:-1]
        return self.base(h)+self.observation(self.packet)
class Actor(H.Actor):
    def __init__(self,p):
        super().__init__(p);c=P.checkpoint(p['learning_initial']['path']);assert digest(c['residual_state']['actor'])==p['frozen327_actor_digest']
        self.load_state_dict(c['residual_state']['actor']);self.requires_grad_(False);self.head=ObservationHead(self.head,p['observation_seed'])
    def sequence(self,x):
        packet=x[:,None] if x.ndim==2 else x
        self.head.packet=self.normalized(packet)
        try:return super().sequence(x)
        finally:self.head.packet=None
    def step(self,x,hidden=None):
        self.head.packet=self.normalized(x)
        try:return super().step(x,hidden)
        finally:self.head.packet=None
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))
def base_digest(q):
    d={}
    for name,v in q.state_dict().items():
        if name.startswith('head.observation.'):continue
        key='head.'+name[len('head.base.'):] if name.startswith('head.base.') else name
        d[key]=v
    return digest(d)
def check(q):
    assert sum(v.numel() for v in q.parameters() if v.requires_grad)==3938
    assert sum(v.numel() for v in q.parameters() if not v.requires_grad)==53940
    assert base_digest(q)==q.p['frozen327_actor_digest']
    assert all(v.grad is None for name,v in q.named_parameters() if not name.startswith('head.observation.'))
def make(p,device='cuda'):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    q=Actor(p).to(device);q.core.eval();opt=torch.optim.Adam(q.head.observation.parameters(),lr=p['observation_lr'],foreach=False,fused=False);check(q);return q,opt
def state(q,opt):return P.L.cpu(dict(actor=q.state_dict(),observation_opt=opt.state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(copy.deepcopy(s['observation_opt']));check(obj[0])
def export(obj,path,limit):
    s=state(*obj);mem=io.BytesIO();torch.save(dict(architecture=ARCH,observation_state=s,joint_hash=digest(s),policy_hash=obj[0].phash(),single_actor=True,frozen_parent=obj[0].p['learning_initial']),mem)
    assert len(mem.getvalue())+65536<=limit
    b=gzip.compress(mem.getvalue(),mtime=0);assert len(b)<=limit and not Path(path).exists();Path(path).write_bytes(b)
    return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(s),bytes=len(b),uncompressed_bytes=len(mem.getvalue()))
def frozen(p,path):
    c=P.checkpoint(path)
    if c.get('architecture')!=ARCH:return B.frozen(p,path)
    assert c['frozen_parent']==p['learning_initial'] and digest(c['observation_state'])==c['joint_hash']
    q=Actor(p);q.load_state_dict(c['observation_state']['actor']);assert q.phash()==c['policy_hash'];assert base_digest(q)==p['frozen327_actor_digest']
    return q.eval().requires_grad_(False)
def data(p,device='cuda'):return B.data(p,device)
def step(obj,batch):
    q,opt=obj;check(q);opt.zero_grad(set_to_none=True);result=dict(correction=0.,raw15=0.)
    for g in batch['corrections']:
        axis=P.action_loss(q.sequence(g['x']),g['y']);first=g['first'];n=first+1
        if g['raw_control_only']:loss=axis[first:].mean();result['raw15']=float(loss.detach())
        else:loss=axis[:n].mean()+.1*(axis[n:].mean() if n<len(axis) else axis.sum()*0)
        (loss/4).backward();result['correction']+=float(loss.detach())/4
    ref,own,extra=B._preservation(q,batch);result.update(reference=ref,own=own,extra=extra)
    result['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(q.head.observation.parameters(),1.,error_if_nonfinite=True));check(q);opt.step();check(q);return result
MODES=['direct']
bc_step=step
J=B.J
