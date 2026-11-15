"""Matched Adam comparison: final linear readout versus trainable nonlinear head.

Same existing single Actor and initial function, frozen causal encoder/GRU,
same current-policy histories and verified full recovery; no new runtime module.
"""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t575_complete_neural_recovery'));import t575_learning as Prev
Prior=Prev.Prior
P,G,torch,np=Prev.P,Prev.G,Prev.torch,Prev.np
read,save,sha,digest=Prev.read,Prev.save,Prev.sha,Prev.digest
frozen,state,load,export=Prev.frozen,Prev.state,Prev.load,Prev.export

def make(p):
 q=frozen(p,p['initial']['path']);assert sha(p['initial']['path'])==p['initial']['sha256'];q.requires_grad_(False)
 (q.continual[-1] if p['arm']=='linear' else q.continual).requires_grad_(True)
 n=sum(x.numel() for x in q.parameters() if x.requires_grad);assert n==(66 if p['arm']=='linear' else 5986)
 return q,torch.optim.Adam([x for x in q.parameters() if x.requires_grad],lr=p['fit_lr'],foreach=False,fused=False)
def fixed(q,p):
 prefix='continual.2.' if p['arm']=='linear' else 'continual.'
 return digest({k:v for k,v in q.state_dict().items() if not k.startswith(prefix)})

def dataset(p):
 q=frozen(p,p['initial']['path']);data=[]
 with torch.no_grad():
  for kind,rows in [('keep',p['memory_rows']),('repair',[p['repair_row']])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);f,b=Prior.feature(q,torch.tensor(a['states']));original=b+q.continual(f)
    if kind=='keep':
     y=torch.tensor(a['locations']);assert float((original-y).abs().max())<1e-9;mask=torch.ones(len(f),dtype=torch.bool);target=y
    else:
     mask=torch.tensor(a['ownership'].astype(bool));assert mask[:89].sum()==0 and mask[89:].all();y=torch.tensor(a['teacher_actions'],dtype=torch.float64);target=torch.where(y>=1,torch.maximum(original,torch.ones_like(original)),torch.where(y<=-1,torch.minimum(original,-torch.ones_like(original)),y))
    data.append(dict(id=row['job_id'],kind=kind,f=f[mask],base=b[mask],target=target[mask],command=y[mask]))
 assert len(data)==23 and sum(g['kind']=='keep' for g in data)==22;return data

def measure(q,data):
 with torch.no_grad():
  return [dict(id=g['id'],kind=g['kind'],location_RMS=float((g['base']+q.continual(g['f'])-g['target']).square().mean().sqrt()),location_absmax=float((g['base']+q.continual(g['f'])-g['target']).abs().max()),command_absmax=float(((g['base']+q.continual(g['f'])).clamp(-1,1).float().double()-g['command'].clamp(-1,1)).abs().max())) for g in data]

def fit(obj,data,p,guard,event,steps=None):
 q,opt=obj;body=fixed(q,p);initial={k:v.detach().clone() for k,v in q.continual.named_parameters()};F=torch.cat([g['f'] for g in data]);B=torch.cat([g['base'] for g in data]);Y=torch.cat([g['target'] for g in data]);W=torch.cat([torch.full((len(g['f']),),1/((22 if g['kind']=='keep' else 1)*len(g['f'])),dtype=torch.float64) for g in data]);trace=[];before=measure(q,data)
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);prediction=B+q.continual(F);data_loss=((prediction-Y).square().mean(-1)*W).sum();penalty=sum((v-initial[name]).square().sum() for name,v in q.continual.named_parameters());loss=data_loss+p['parameter_penalty']*penalty;assert torch.isfinite(loss);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_([v for v in q.parameters() if v.requires_grad],1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([float(loss.detach()),float(data_loss.detach()),norm]);assert fixed(q,p)==body
 return dict(steps=len(trace),trace=trace,before=before,after=measure(q,data),frozen_body_hash=body,trainable_parameters=sum(v.numel() for v in q.parameters() if v.requires_grad),new_RL=0,new_Critic=0,actual_history_frozen_body_features=True)
