"""Read-only exact compact-basis support on12existing actual trajectories."""
from pathlib import Path
import os,sys,time,io,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t588_actual_support_audit';DOC=ROOT/'stage/docs/day_20261116/t588'
sys.path.insert(0,str(ROOT/'stage/experiments/t587_verified_short_correction'));import t587_runtime as R
L=R.L;read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np
T550=L.Prior.Old.Parent

def used(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def guard():
 p=read(OUT/'protocol.json');assert time.time()<p['all_stop'] and used(OUT)<=p['max_bytes'];assert used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30;assert __import__('shutil').disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in p['inputs'].items():assert sha(f)==h,f

def register():
 assert not OUT.exists();old=ROOT/'stage/runs/t587_verified_short_correction';assert read(old/'audit.json')['state']=='PASS';assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids']);assert not read(old/'summary.json')['result']['qualified_jobs']
 p=read(old/'protocol.json');ref=[next(r for r in p['keep_rows'] if r['job_id']==j['id']) for j in p['handoff_jobs']];full=p['recovery_rows'];fail=p['historical_handoff_baseline'];short=R.G.rows(old);entries=[]
 for kind,rr in [('old_success_record',ref),('actual_failed_handoff',fail),('full_GAP_success',full),('short_GAP_failed',short)]:
  for i,row in enumerate(rr):entries.append(dict(kind=kind,job=row['job_id'],row=row,k=p['handoff_jobs'][i]['student_takeover_first'],teacher_stop=p['handoff_jobs'][i]['student_takeover_first']+20))
 inputs={str(f):sha(f) for f in [HERE/'support_audit.py',DOC/'PLAN.md',old/'protocol.json',old/'summary.json',old/'audit.json',old/'SEALED.json',Path(p['initial']['path'])]+[Path(e['row']['path']) for e in entries]}
 m=p['base_model']
 while m:inputs[m['path']]=m['sha256'];m=m.get('parent')
 cfg=dict(id='T588_ACTUAL_COMPACT_SUPPORT_AUDIT',inputs=inputs,entries=entries,actor_protocol=str(old/'protocol.json'),initial=p['initial'],max_bytes=2**20,all_stop=p['absolute_all_stop'],main_seconds=120,audit_seconds=120,new_native=0,new_optimizer=0,new_gradients=0,scope='Four fixed existing actual trajectory types x3, exact learned RBF support only; no policy edit, intervention, simulator, optimization, selection or implied causal rescue.',user_authorization='2026-11-16 user: 下一步做什么，继续; existing14:00 unchanged')
 assert time.time()+420<cfg['all_stop'] and used(ROOT/'stage/runs')+cfg['max_bytes']+131072<2**30
 OUT.mkdir();save(OUT/'protocol.json',cfg);save(HERE/'protocol.json',cfg);guard();print('T588_REGISTERED_READ_ONLY',flush=True)

def calculate():
 guard();torch.set_num_threads(1);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
 p=read(OUT/'protocol.json');cfg=read(p['actor_protocol']);q=L.frozen(cfg,p['initial']['path']);before=digest(q.state_dict());sets=[('repair',q.repair_centers,q.repair_radii,q.repair_weights),('added',q.added_centers,q.added_radii,q.added_weights)]+[(f'layer{i}',z.centers,z.radii,z.weights) for i,z in enumerate(q.layers)];stats=[];arrays={}
 for index,e in enumerate(p['entries']):
  guard();a=R.E.N.S.arrays(e['row']['path']);f,base=L.feature(q,torch.tensor(a['states']));loc=L.loc(q,f,base);counts=[];parts=[];ratio=[]
  for name,c,r,w in sets:
   assert len(c)>0 and (r>0).all();cc=[];pp=[];dd=[]
   for start in range(0,len(f),16):
    z=f[start:start+16];d=((z[:,None]-c)**2).sum(-1)/r.square();phi=torch.relu(1-d).square();assert torch.equal(phi,T550.Prior.Old.basis(z,c,r));cc.append((phi!=0).sum(-1));pp.append(phi@w);dd.append(d.min(-1).values.sqrt())
   counts.append(torch.cat(cc));parts.append(torch.cat(pp));ratio.append(torch.cat(dd))
  count=torch.stack(counts,1);contrib=torch.stack(parts,1);dist=torch.stack(ratio,1);active=(count!=0).any(-1);effective=contrib.abs().max(-1).values.max(-1).values>1e-12;total=contrib.sum(1)
  stem=f'trace{index}';arrays.update({stem+'_counts':count.numpy(),stem+'_minimum_distance_over_radius':dist.numpy(),stem+'_branch_corrections':contrib.numpy(),stem+'_locations':loc.numpy(),stem+'_native_counts':a['native_counts'].copy()})
  checkpoints=[]
  for key,j in [('start',0),('takeover',e['k']),('4s_handback',e['teacher_stop']),('last',len(f)-1)]:
   if j<len(f):checkpoints.append(dict(name=key,decision=j,any_basis_active=bool(active[j]),active_counts=count[j].tolist(),min_distance_over_radius=float(dist[j].min()),correction=total[j].tolist(),location=loc[j].tolist()))
  mismatch=float((loc-torch.tensor(a['locations'])).abs().max())
  if e['kind']!='old_success_record':assert mismatch<1e-10,mismatch
  stats.append(dict(kind=e['kind'],job=e['job'],decisions=len(f),source_path=e['row']['path'],any_basis_active_fraction=float(active.double().mean()),effective_correction_fraction=float(effective.double().mean()),branch_active_fraction=[float(x.double().mean()) for x in (count!=0).T],native_recorded_location_difference_max=mismatch,source_record_actions_match_current_mean_fraction=float((loc.clamp(-1,1).float()==torch.tensor(a['actions'])).all(-1).double().mean()),checkpoints=checkpoints))
 assert digest(q.state_dict())==before;guard()
 return dict(state='COMPLETE',new_native=0,new_optimizer=0,new_gradients=0,model_unchanged=True,model_sha256=sha(p['initial']['path']),basis_sets=[dict(name=n,centers=len(c),radius_min=float(r.min()),radius_median=float(r.median()),radius_max=float(r.max())) for n,c,r,w in sets],traces=stats,interpretation='Actual support observations only; support loss is not a simulated causal rescue, no proof of sole cause. Old success histories may have different source Actor metadata; current mean-command agreement is explicitly reported. No waypoint/controller or model alteration.'),arrays

def main():
 assert not (OUT/'summary.json').exists();r,a=calculate();b=io.BytesIO();np.savez_compressed(b,**a);assert len(b.getvalue())<768*1024
 with (OUT/'support_arrays.npz').open('xb') as f:f.write(b.getvalue())
 r['arrays_sha256']=sha(OUT/'support_arrays.npz');r['array_hashes']={k:digest(v) for k,v in a.items()};save(OUT/'summary.json',r);guard();print('T588_MAIN_COMPLETE',flush=True)
def audit():
 assert not (OUT/'audit.json').exists();r,a=calculate();s=read(OUT/'summary.json');assert s.pop('arrays_sha256')==sha(OUT/'support_arrays.npz');hs=s.pop('array_hashes');assert s==r
 with np.load(OUT/'support_arrays.npz',allow_pickle=False) as z:
  for k,v in a.items():assert np.array_equal(v,z[k]) and digest(v)==hs[k]
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,new_gradients=0,all12_traces_all_saved_support_arrays_exact=True,all_model_input_SHA_unchanged=True));print('T588_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
