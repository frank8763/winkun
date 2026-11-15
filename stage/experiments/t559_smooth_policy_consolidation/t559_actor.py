"""Replace all compact-memory corrections with one smooth fixed-capacity head.

Frozen encoder/causal GRU and original command adapters, no deployment teacher,
center bank, nearest-neighbor lookup or task/router input. Distillation only.
"""
from pathlib import Path
import copy,gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'));import t552_actor as Prior
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as Base
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
ARCH='T559_single120_64carry_frozen_base_smooth40450_no_compact_memory'

class Actor(Base.Actor):
    def __init__(self,p):
        super().__init__(p)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['student_seed'])
            self.student=torch.nn.Sequential(torch.nn.Linear(184,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,2)).double()
            torch.nn.init.zeros_(self.student[-1].weight);torch.nn.init.zeros_(self.student[-1].bias)
        self.requires_grad_(False)
    def location(self,h,n):return Base.Actor.location(self,h,n)+self.student(torch.cat([h,n],-1))
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def initial(p):
    teacher=Prior.Old.frozen(p,p['teacher']['path']);assert teacher.phash()==p['teacher']['policy_hash'];q=Actor(p)
    base={k:v for k,v in teacher.state_dict().items() if not k.startswith(('repair_','added_','layers.'))}
    missing,unexpected=q.load_state_dict(base,strict=False);assert not unexpected and set(missing)=={'student.'+k for k in q.student.state_dict()}
    assert not any(k.startswith(('repair_','added_','layers.')) for k in q.state_dict())
    for k,v in base.items():assert torch.equal(q.state_dict()[k],v)
    return q

def frozen(p,path):
    s=P.checkpoint(path)
    if s.get('architecture')!=ARCH:return Prior.Old.frozen(p,path)
    q=Actor(p);q.load_state_dict(s['actor']);assert q.phash()==s['policy_hash'];return q.eval()

def make(p):
    q=initial(p);q.student.requires_grad_(True);opt=torch.optim.Adam(q.student.parameters(),lr=p['fit_lr'],foreach=False,fused=False);assert sum(x.numel() for x in q.student.parameters())==40450;return q,opt
def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('student.')})

def dataset(p,limit=None):
    teacher=Prior.Old.frozen(p,p['teacher']['path']);q=initial(p);data=[]
    with torch.no_grad():
        for kind,rows in [('keep',p['keep_rows']),('neighbor',p['neighbor_rows'])]:
            for row in rows:
                assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);f=torch.cat([h[:,0],n],-1);base=Base.Actor.location(q,h[:,0],n)
                target=torch.tensor(a['locations']);assert float((teacher.sequence(x)-target).abs().max())<1e-10
                mask=torch.tensor(~a['ownership'].astype(bool));assert mask.sum()>=2
                if limit is not None:mask[limit:]=False
                data.append(dict(id=row['job_id'],kind=kind,f=f[mask],base=base[mask],target=target[mask],command=torch.tensor(a['actions'])[mask],source_sha256=row['sha256']))
    assert len(data)==27 and sum(g['kind']=='keep' for g in data)==23
    return data

def locations(q,g):return g['base']+q.student(g['f'])
def measure(q,data):
    rows=[]
    with torch.no_grad():
        for g in data:
            loc=locations(q,g);cmd=loc.clamp(-1,1).float().double();delta=cmd-g['command'].double();err=loc-g['target']
            rows.append(dict(id=g['id'],kind=g['kind'],location_RMS=float(err.square().mean().sqrt()),command_RMS=float(delta.square().mean().sqrt()),command_max=float(delta.abs().max())))
    return dict(rows=rows,worst_command_RMS=max(v['command_RMS'] for v in rows),worst_command_max=max(v['command_max'] for v in rows),mean_location_MSE=sum(v['location_RMS']**2 for v in rows)/len(rows))

def fit(obj,data,p,guard,event,steps=None):
    q,opt=obj;fixed=fixed_hash(q);trace=[];rng=np.random.default_rng(p['batch_seed']);before=measure(q,data)
    for k in range(p['fit_steps'] if steps is None else steps):
        guard();opt.zero_grad(set_to_none=True);features=[];base=[];targets=[]
        for g in data:
            ids=rng.integers(0,len(g['f']),size=p['rows_per_trajectory']);features.append(g['f'][ids]);base.append(g['base'][ids]);targets.append(g['target'][ids])
        loc=torch.cat(base)+q.student(torch.cat(features));e=(loc-torch.cat(targets)).square().mean(-1).reshape(len(data),p['rows_per_trajectory']);loss=(.5*e.mean(-1)+.5*e.topk(max(1,math.ceil(.2*p['rows_per_trajectory'])),dim=-1).values.mean(-1)).mean()
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.student.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([float(loss.detach()),norm])
    assert fixed_hash(q)==fixed;after=measure(q,data)
    return dict(steps=len(trace),trace=np.asarray(trace,dtype=np.float64),before=before,after=after,fit_gate=after['worst_command_RMS']<=p['fit_gate_RMS'] and after['worst_command_max']<=p['fit_gate_max'],new_RL=0,teacher_runtime=False)

def export(obj,p,path):
    q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,actor=s['actor'],actor_opt=s['actor_opt'],joint_hash=digest(s),policy_hash=q.phash(),single_actor=True,no_runtime_teacher=True,no_compact_banks=True),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
    with Path(path).open('xb') as h:h.write(blob);h.flush();__import__('os').fsync(h.fileno())
    return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),joint_hash=digest(s),bytes=len(blob))
