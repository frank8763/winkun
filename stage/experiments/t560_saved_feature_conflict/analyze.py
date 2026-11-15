"""Saved finite-support diagnostic: no native, no optimizer, no new policy."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'stage/experiments/t559_smooth_policy_consolidation'));import t559_runtime as R
L,np,torch=R.L,R.np,R.torch
DOC=ROOT/'stage/docs/day_20261115/t560'
def main():
 assert not (DOC/'RESULT.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=R.configuration();R.verify(p,True);data=L.dataset(p);s=R.read(R.OUT/'summary.json');q=L.frozen(p,s['result']['model']['path']);out=[];fs=[];ys=[];task=[]
 with torch.no_grad():
  for i,g in enumerate(data):
   pred=L.locations(q,g);err=(pred.clamp(-1,1).float().double()-g['command'].double()).abs().amax(-1);need=(g['target']-g['base']).abs().amax(-1);critical=need>.1;ee=err.square();total=float(ee.sum());out.append(dict(id=g['id'],n=len(err),critical_n=int(critical.sum()),critical_fraction=float(critical.double().mean()),critical_error_energy_fraction=float(ee[critical].sum()/max(total,1e-30)),bad_n=int((err>.1).sum()),bad_outside_critical_n=int(((err>.1)&~critical).sum()),max_error=float(err.max()),max_required_correction=float(need.max())))
   fs.append(g['f']);ys.append(g['target']);task.extend([i]*len(g['f']))
  f=torch.cat(fs);y=torch.cat(ys);tid=torch.tensor(task);uniq,inv=torch.unique(f,dim=0,return_inverse=True);duplicate_conflicts=0;groups=0
  for i,n in enumerate(torch.bincount(inv)):
   if n>1:
    groups+=1;v=y[inv==i];duplicate_conflicts+=int(float((v.max(0).values-v.min(0).values).abs().max())>1e-9)
  # Nearest other-history neighbor in exact frozen feature metric; avoid all-pairs storage.
  pairs=[];raw_nn=[]
  for start in range(0,len(f),128):
   z=f[start:start+128];d=torch.cdist(z,f,p=2,compute_mode='donot_use_mm_for_euclid_dist');d[tid[start:start+len(z),None]==tid[None]]=float('inf');dist,j=d.min(-1);delta=(y[start:start+len(z)]-y[j]).abs().amax(-1);raw_nn.extend(dist.tolist())
   for k in range(len(z)):
    if float(delta[k])>.2:pairs.append(dict(i=start+k,j=int(j[k]),task=data[int(tid[start+k])]['id'],neighbor=data[int(tid[j[k]])]['id'],distance=float(dist[k]),target_difference=float(delta[k]),secant=float(delta[k]/max(float(dist[k]),1e-15))))
  pairs.sort(key=lambda a:a['distance']);trace=np.load(R.OUT/'fit_trace.npz')['trace'];windows={str(k):dict(mean_loss=float(trace[max(0,k-100):k,0].mean()),mean_gradient_norm=float(trace[max(0,k-100):k,1].mean())) for k in [100,500,1000,1500]}
 result=dict(state='COMPLETE_SAVED_SUPPORT_DIAGNOSTIC',epoch=time.time(),native=0,optimizer=0,total_rows=len(f),unique_features=len(uniq),exact_duplicate_groups=groups,exact_duplicate_conflicts=duplicate_conflicts,nearest_other_history_distance_quantiles=np.quantile(raw_nn,[0,.1,.5,.9,1]).tolist(),nearest_conflicting_pairs=pairs[:12],histories=out,fit_trace_windows=windows,input_sources={str(R.OUT/'proposal.pt.gz'):R.sha(R.OUT/'proposal.pt.gz'),str(R.OUT/'fit_trace.npz'):R.sha(R.OUT/'fit_trace.npz')},limitations='Finite saved features, nearest-pair secants not impossibility or causal evidence; no new training/native, no proof of closed-loop recovery.')
 R.save(DOC/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['histories','input_sources']},ensure_ascii=False))
if __name__=='__main__':main()
