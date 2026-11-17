"""Prepared zero-learning parent/candidate native trajectory pair. Not auto-launched."""
from pathlib import Path
import sys,os,time,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t610_retention_trajectory_pair';DOC=ROOT/'stage/docs/day_20261117/t610'
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'));import t608_runtime as R
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def setup():
 R.NIGHT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(OUT,'compare');R.verify(plan(),True)
def plan():
 p=read(HERE/'protocol.json');assert p['authorization_approved'];cfg=read(p['actor_configuration']);cfg.update(p);cfg.update(out=str(OUT),child_bytes=p['max_bytes']);return cfg

def metrics(p,rows):
 a,b=[R.E.N.S.arrays(x['path']) for x in rows];assert np.array_equal(a['states'][0],b['states'][0]) and np.array_equal(a['initial_native_integration'],b['initial_native_integration']);n=min(len(a['states']),len(b['states']))
 aa,bb=torch.tensor(a['states']),torch.tensor(b['states']);qa=R.L.frozen(p,p['models'][0]['path']);qb=R.L.frozen(p,p['models'][1]['path'])
 with torch.no_grad():A=qa.sequence(aa).clamp(-1,1).numpy()[:n];B=qb.sequence(bb).clamp(-1,1).numpy()[:n];BA=qb.sequence(aa).clamp(-1,1).numpy()[:n];AB=qa.sequence(bb).clamp(-1,1).numpy()[:n]
 assert np.array_equal(A.astype(np.float32),a['actions'][:n]) and np.array_equal(B.astype(np.float32),b['actions'][:n])
 direct=BA-A;feedback=B-BA;reverse_feedback=AB-A;reverse_direct=B-AB
 assert np.allclose(direct+feedback,B-A,atol=1e-15,rtol=0) and np.allclose(reverse_feedback+reverse_direct,B-A,atol=1e-15,rtol=0)
 xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
 return dict(common_decisions=n,initial120_and_native_integration_exact=True,first_common_xy_gap_over1cm=next((i for i,z in enumerate(xy) if z>.01),None),max_common_xy_gap=float(xy.max()),same_parent_history_action_RMS=float(np.sqrt(np.mean(direct**2))),direct_component_L2=float(np.linalg.norm(direct)),feedback_component_L2=float(np.linalg.norm(feedback)),reverse_direct_L2=float(np.linalg.norm(reverse_direct)),reverse_feedback_L2=float(np.linalg.norm(reverse_feedback)),two_order_algebraic_decomposition_not_causal_certificate=True,first1cm_is_descriptive_not_acceptance_threshold=True)

def main():
 assert OUT.exists() and not (OUT/'launch.json').exists();p=plan();setup();R.E.N.offline();R.E.L.install_stop_handlers();rt=R.Base.Runtime(p);save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),new_Adam=0,new_RL=0,scope='one fixed task two frozen models plus2exact'));rows=[];state='FAILED';error=None
 try:
  rt.pool=R.Native.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  for k,m in enumerate(p['models']):
   vv=rt.request([p['job']],m,'FROZEN_TRAJECTORY_'+str(k),full=True);R.Native.inspect(p,vv);rows.append(vv[0][1]);save(OUT/'PROGRESS.json',dict(rows=rows));rt.request([p['job']],m,'EXACT_FROZEN_'+str(k),reference=[rows[-1]])
  detail=metrics(p,rows);state='COMPLETE';save(OUT/'METRICS.json',detail)
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,rows=rows,new_Adam=0,new_RL=0));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=plan();setup();R.G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=R.G.rows(OUT);reps=R.G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==2 and len(rows)+len(reps)+4==s['commands']==8;by={x['path']:x for x in rows}
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());R.Native.inspect(p,[(a,r)])
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and metrics(p,rows)==read(OUT/'METRICS.json')
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_Adam=0,all_full_arrays_and_native_integrations_exact=True,physics=s['physics'],commands=s['commands'],two_order_decomposition_exact=True))
if __name__=='__main__':globals()[sys.argv[1]]()
