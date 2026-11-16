"""Four real gradients and nearest-history labels, zero physics/optimizer updates."""
from pathlib import Path
import os,sys,time,json,io
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t585_recovery_conflict_audit';DOC=ROOT/'stage/docs/day_20261116/t585'
sys.path.insert(0,str(ROOT/'stage/experiments/t584_unbounded_recovery_learning'));import t584_runtime as R
L=R.L;read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def used(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def guard():
 p=read(OUT/'protocol.json');assert time.time()<p['all_stop'] and used(OUT)<=p['max_bytes'];assert used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30;assert __import__('shutil').disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in p['inputs'].items():assert sha(f)==h,f

def register():
 assert not OUT.exists();old=ROOT/'stage/runs/t584_unbounded_recovery_learning';assert read(old/'audit.json')['state']=='PASS';assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids'])
 p=read(old/'protocol.json');v=read(old/'summary.json')['result'];assert not v['accepted'] and len(v['check']['lost_initial'])==7 and not v['new_handoff_success_ids']
 inputs={str(f):sha(f) for f in [HERE/'run.py',DOC/'PLAN.md',old/'protocol.json',old/'summary.json',old/'audit.json',old/'SEALED.json',Path(v['initial']['path']),Path(v['proposal']['path'])]+[Path(r['path']) for r in p['keep_rows']+p['recovery_rows']]}
 m=p['base_model']
 while m:inputs[m['path']]=m['sha256'];m=m.get('parent')
 for r in v['baseline_handoff']+v['post_handoff']:inputs[r['path']]=r['sha256']
 cfg=dict(id='T585_OFFLINE_FOUR_GRADIENT_LABEL_AUDIT',inputs=inputs,max_bytes=2**20,all_stop=p['absolute_all_stop'],main_seconds=120,audit_seconds=120,new_native=0,new_optimizer=0,new_gradients=4,
  scope='Fixed initial/proposal teacher+weighted keep gradients; nearest protected feature label disagreement; native first takeover command differences. No training, variable grid, model selection or navigation.',user_authorization='2026-11-16 user: 检查，继续思考下一步')
 assert time.time()+360<cfg['all_stop'] and used(ROOT/'stage/runs')+cfg['max_bytes']+131072<2**30
 OUT.mkdir();save(OUT/'protocol.json',cfg);save(HERE/'protocol.json',cfg);guard();print('T585_REGISTERED_ZERO_NATIVE_ZERO_OPTIMIZER')

def objective_grad(q,groups,weight):
 f=torch.cat([g['f'] for g in groups]);b=torch.cat([g['b'] for g in groups]);t=torch.cat([g['target'] for g in groups]);w=torch.cat([torch.full((len(g['f']),),1/(len(groups)*len(g['f'])),dtype=torch.float64) for g in groups]);z=L.loc(q,f,b)
 loss=(.5*((z-t)/.2).square().mean(-1)*w).sum()*weight
 g=torch.autograd.grad(loss,list(q.continual.parameters()));return float(loss.detach()),torch.cat([v.flatten() for v in g]).detach().numpy()

def calculate():
 guard();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 old=ROOT/'stage/runs/t584_unbounded_recovery_learning';p=read(old/'protocol.json');s=read(old/'summary.json');v=s['result'];keep,teach=L.dataset(p);vectors={};points={}
 for tag,model in [('initial',v['initial']),('proposal',v['proposal'])]:
  guard();q=L.frozen(p,model['path']);q.continual.requires_grad_(True);met=L.metrics(q,(keep,teach));assert met==v['update']['before' if tag=='initial' else 'after']
  tl,tg=objective_grad(q,teach,1.);kl,kg=objective_grad(q,keep,p['keep_weight']);vectors[tag+'_teacher_grad']=tg;vectors[tag+'_weighted_keep_grad']=kg
  nt=float(np.linalg.norm(tg));nk=float(np.linalg.norm(kg));points[tag]=dict(teacher_loss=tl,weighted_keep_loss=kl,teacher_grad_norm=nt,weighted_keep_grad_norm=nk,keep_to_teacher_norm=nk/nt,gradient_cosine=None if nt*nk==0 else float(tg@kg/(nt*nk)),fit=met)
 # Nearest is in the exact frozen184-D Actor features, not physical distance.
 ff=torch.cat([g['f'] for g in keep]);tt=torch.cat([g['target'] for g in keep]);fn=(ff*ff).sum(-1);near=[];pairs=[]
 for ri,g in enumerate(teach):
  ds=[];ix=[]
  for start in range(0,len(g['f']),64):
   guard();z=g['f'][start:start+64];d=((z*z).sum(-1)[:,None]+fn[None]-2*z@ff.T).clamp_min(0);idx=d.argmin(-1);exact=((z-ff[idx])**2).sum(-1).sqrt();ds.append(exact);ix.append(idx)
  d=torch.cat(ds);idx=torch.cat(ix);gap=(g['target']-tt[idx].clamp(-1,1)).abs().max(-1).values;ident=(g['f']==ff[idx]).all(-1);near.append(dict(job=p['recovery_rows'][ri]['job_id'],nearest_feature_distance_quantiles=torch.quantile(d,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),nearest_clipped_target_maxdiff_quantiles=torch.quantile(gap,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),exact_same_features=int(ident.sum()),exact_same_features_inconsistent_targets=int((ident & (gap>1e-6)).sum()),near_NOT_identical_NOT_incompatibility_proof=True))
  for k in torch.argsort(d)[:3].tolist():pairs.append(dict(job=p['recovery_rows'][ri]['job_id'],teacher_suffix_index=k,protected_flat_index=int(idx[k]),feature_distance=float(d[k]),teacher_command=g['target'][k].tolist(),nearest_old_command=tt[idx[k]].clamp(-1,1).tolist()))
 first=[]
 for src,a,b in zip(p['recovery_rows'],v['baseline_handoff'],v['post_handoff']):
  t=R.E.N.S.arrays(src['path']);x=R.E.N.S.arrays(a['path']);y=R.E.N.S.arrays(b['path']);k=int(t['teacher_first_declared']);assert np.array_equal(t['states'][k],x['states'][k]) and np.array_equal(t['states'][k],y['states'][k]);first.append(dict(job=src['job_id'],k=k,teacher=t['actions'][k].tolist(),initial=x['actions'][k].tolist(),proposal=y['actions'][k].tolist(),initial_error=(x['actions'][k]-t['actions'][k]).tolist(),proposal_error=(y['actions'][k]-t['actions'][k]).tolist()))
 result=dict(state='COMPLETE',new_native=0,new_optimizer=0,actual_gradients=4,points=points,nearest_label_comparison=near,closest9_pairs=pairs,actual_first_takeover_commands=first,
  exact_label_incompatibility_proven=any(x['exact_same_features_inconsistent_targets'] for x in near),
  raw_symptom=dict(initial_success39=v['check']['success']+len(v['check']['lost_initial']),proposal_success=v['check']['success'],lost=v['check']['lost_initial'],all3_initial_and_proposal_handoff_failed=all(not r['success'] for r in v['baseline_handoff']+v['post_handoff']),RED_reproduced_from_verified_native_evidence=True),
  notes='Gradient opposition is a local fixed-checkpoint objective measurement, not sole causal proof, no training remedy validated. Feature-nearest labels need not be incompatible, success may admit multiple valid controls.')
 return result,vectors

def main():
 assert not (OUT/'summary.json').exists();r,a=calculate();b=io.BytesIO();np.savez_compressed(b,**a);assert len(b.getvalue())<512*1024
 with (OUT/'gradients.npz').open('xb') as f:f.write(b.getvalue())
 r['gradient_file_sha256']=sha(OUT/'gradients.npz');r['gradient_array_hashes']={k:digest(v) for k,v in a.items()};save(OUT/'summary.json',r);guard();print('T585_MAIN_COMPLETE',r['points']['proposal']['gradient_cosine'],flush=True)
def audit():
 assert not (OUT/'audit.json').exists();r,a=calculate();saved=read(OUT/'summary.json');assert sha(OUT/'gradients.npz')==saved.pop('gradient_file_sha256');hs=saved.pop('gradient_array_hashes');assert saved==r
 with np.load(OUT/'gradients.npz',allow_pickle=False) as z:
  for k,v in a.items():assert np.array_equal(v,z[k]) and digest(v)==hs[k]
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,discarded_recomputed_gradients=4,all_gradient_arrays_and_metrics_exact=True,all_input_model_SHA_unchanged=True));print('T585_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
