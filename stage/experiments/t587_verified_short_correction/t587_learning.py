"""Qualified short intervention labels, same T584 Actor and optimization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t584_unbounded_recovery_learning'));import t584_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
make,state,load,fixed,feature,loc,frozen,fit,export,metrics=Prior.make,Prior.state,Prior.load,Prior.fixed,Prior.feature,Prior.loc,Prior.frozen,Prior.fit,Prior.export,Prior.metrics

def dataset(p,rows):
 q=Prior.fresh(p);keep=[];teach=[]
 for r in p['keep_rows']:
  assert sha(r['path'])==r['sha256'];a=G.E.N.S.arrays(r['path']);f,b=feature(q,torch.tensor(a['states']));keep.append(dict(f=f,b=b,target=b.clone()))
 for r in rows:
  assert r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']
  a=G.E.N.S.arrays(r['path']);own=a['ownership'].astype(bool);assert own.sum()==p['correction_decisions'] and np.array_equal(a['actions'][own],a['teacher_actions'][own])
  f,b=feature(q,torch.tensor(a['states']));teach.append(dict(f=f[own],b=b[own],target=torch.tensor(a['actions'][own],dtype=torch.float64)))
 assert len(keep)==27 and 1<=len(teach)<=3;return keep,teach
