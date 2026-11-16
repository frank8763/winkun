"""Matched smooth students, frozen vs trainable encoder/GRU; full-history BC.

Reuse T562 research weights, not a promoted controller. No compact support banks.
"""
from pathlib import Path
import sys,copy,gzip,io,math,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t562_full_history_representation'))
import t562_actor as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
state,load=Prior.state,Prior.load

def initial(p):return Prior.initial(p)
def frozen(p,path):return Prior.frozen(p,path)
def variables(q,method):
 assert method in ['frozen_representation','trainable_representation']
 return list(q.student.parameters())+(list(q.encoder.parameters())+list(q.core.parameters()) if method=='trainable_representation' else [])
def make(p):
 q=initial(p);q.requires_grad_(False)
 for v in variables(q,p['method']):v.requires_grad_(True)
 return q,torch.optim.Adam(variables(q,p['method']),lr=p['fit_lr'],foreach=False,fused=False)
def fixed(q,method):
 prefixes=('student.',) if method=='frozen_representation' else ('student.','encoder.','core.')
 return digest({k:v for k,v in q.state_dict().items() if not k.startswith(prefixes)})

def dataset(p):
 q=initial(p);data=[]
 with torch.no_grad():
  for kind,rows in [('old',p['keep_rows']),('recovery',p['recovery_rows'])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);target=torch.tensor(a['actions'],dtype=torch.float64)
    if kind=='old':mask=torch.tensor(~a['ownership'].astype(bool))
    else:
     k=int(a['teacher_first_declared']);mask=torch.tensor(a['ownership'].astype(bool));assert not mask[:k].any() and mask[k:].all()
    assert mask.any() and torch.isfinite(target).all() and target.abs().max()<=1
    n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0];features=torch.cat([h,n],-1);base=Prior.Base.Actor.location(q,h,n)
    data.append(dict(id=row['job_id'],kind=kind,x=x,target=target,mask=mask,features=features,base=base,source_sha256=row['sha256']))
 assert len(data)==42 and len({g['id'] for g in data if g['kind']=='old'})==39
 return data

def locations(q,g,method):
 return g['base']+q.student(g['features']) if method=='frozen_representation' else q.sequence(g['x'])

def measure(q,data,method):
 rows=[]
 with torch.no_grad():
  for g in data:
   loc=locations(q,g,method)[g['mask']];cmd=loc.clamp(-1,1).float().double();err=cmd-g['target'][g['mask']]
   rows.append(dict(id=g['id'],kind=g['kind'],command_RMS=float(err.square().mean().sqrt()),command_MAE=float(err.abs().mean()),command_max=float(err.abs().max()),first_action_error=err[0].tolist(),labeled_decisions=len(err)))
 return dict(rows=rows,old_worst_RMS=max(z['command_RMS'] for z in rows if z['kind']=='old'),old_worst_max=max(z['command_max'] for z in rows if z['kind']=='old'),recovery_worst_RMS=max(z['command_RMS'] for z in rows if z['kind']=='recovery'),recovery_worst_max=max(z['command_max'] for z in rows if z['kind']=='recovery'),joint_descriptive_fit=all(z['command_RMS']<=.02 and z['command_max']<=.1 for z in rows))

def objective(q,data,method,packed):
 losses=dict(old=0.,recovery=0.)
 if method=='trainable_representation':pred=q.sequence(packed)
 else:
  allf=torch.cat([g['features'] for g in data]);allb=torch.cat([g['base'] for g in data]);pieces=(allb+q.student(allf)).split([len(g['x']) for g in data])
 total=packed.new_zeros(())
 for i,g in enumerate(data):
  loc=(pred[:len(g['x']),i] if method=='trainable_representation' else pieces[i])[g['mask']];target=g['target'][g['mask']]
  per=P.action_loss(loc,target).mean(-1);part=.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean();weight=.5/(39 if g['kind']=='old' else 3)
  total=total+weight*part;losses[g['kind']]+=weight*float(part.detach())
 return total,losses

def fit(obj,data,p,guard,event,steps=None):
 q,opt=obj;method=p['method'];body=fixed(q,method);before=measure(q,data,method);trace=[];curve=[]
 packed=torch.nn.utils.rnn.pad_sequence([g['x'] for g in data])
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);total,losses=objective(q,data,method,packed)
  total.backward();norm=float(torch.nn.utils.clip_grad_norm_(variables(q,method),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([losses['old'],losses['recovery'],norm])
  if (k+1)%100==0:curve.append(dict(step=k+1,metrics=measure(q,data,method)))
 assert fixed(q,method)==body
 # Timing is separately logged by caller, excluded from exact-shadow comparisons.
 return dict(steps=len(trace),trace=trace,learning_curve=curve,before=before,after=measure(q,data,method),new_RL=0,new_Critic=0,trainable_parameters=sum(v.numel() for v in variables(q,method)),fixed_sha=body,full_current_history_BPTT=method=='trainable_representation',recorded_action_targets=True)

def export(obj,p,path):
 q,opt=obj;s=dict(actor=copy.deepcopy(q.state_dict()),actor_opt=None if opt is None else copy.deepcopy(opt.state_dict()));b=io.BytesIO();torch.save(dict(architecture=Prior.ARCH,actor=s['actor'],actor_opt=s['actor_opt'],joint_hash=digest(s),policy_hash=q.phash(),single_actor=True,no_runtime_teacher=True,no_compact_banks=True),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
