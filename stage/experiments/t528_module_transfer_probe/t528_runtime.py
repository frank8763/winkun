"""Two frozen neural module swaps; diagnosis, never a deployed router."""
from pathlib import Path
import fcntl,gzip,io,json,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as A
sys.path.insert(0,str(ROOT/'stage/experiments/t502_fullfield_probe'));import t502_runtime as F
G,P,E,M,torch,np=A.G,A.P,A.G.E,A.G.M,A.torch,A.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
OUT=ROOT/'stage/runs/t528_module_transfer_probe'
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');assert M.used(OUT)<=root['max_bytes'] and M.used()-M.used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and set(os.sched_getaffinity(0))<={14,15} and not torch.cuda.is_initialized()
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f
F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify

def hybrid(p,mode):
 base=P.checkpoint(p['initial']['path'])['joint_state']['actor'];candidate=P.checkpoint(p['learned']['path'])['joint_state']['actor'];state={}
 for k,v in base.items():
  backbone=k.startswith(('encoder.','core.'));readout=k.startswith(('head.','drive_adapter.','steer_adapter.'))
  if not backbone and not readout:assert torch.equal(v,candidate[k])
  use=(mode=='learned_backbone' and backbone) or (mode=='learned_readout' and readout);state[k]=(candidate if use else base)[k].clone()
 q=A.Actor(p).eval().requires_grad_(False);q.load_state_dict(state);return q

def export(q,path):
 b=io.BytesIO();torch.save(dict(joint_state=dict(actor=q.state_dict()),policy_hash=q.phash(),architecture='T528_frozen_module_swap_actor_only_NOT_resumable_optimizer',no_optimizer=True),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<640*1024;Path(path).write_bytes(blob);return dict(path=str(path),sha256=sha(path),policy_hash=q.phash(),bytes=len(blob))

def evaluate(rt,p,model,mode,reference=None):
 rt.role=mode+('_EXACT' if reference else '_FIRST');jobs=p['probe_jobs'];rt.guard();assert sum(rt.counts)+sum(j['case']['step_cap'] for j in jobs)<=p['max_physics'];rt.commands+=2
 req=[dict(model=model['path'],job=j,seed=None,query=False,full_path=None) for j in jobs];rows=[]
 for i,((a,r),j,request) in enumerate(zip(rt.pool.request(req),jobs,req)):
  assert r['policy_hash']==model['policy_hash'];r.update(job_id=j['id'],role=rt.role,seed=None,request=request,loaded_model_sha256=sha(model['path']),full=False,full_arrays_generated_but_not_saved=True);r['field_detail']=F.detail(p,r,a) if j['id'].startswith('t502_') else None
  if reference:
   old=reference[i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail'];r.update(exact=True,reference_path=old['path'],independent_new_sample=False);rt.replays.append(r);name='replays.jsonl'
  else:
   f=Path(p['out'])/'episodes'/(mode+'_'+str(i)+'.npz');f.parent.mkdir(exist_ok=True);E.N.write_npz(f,{k:a[k] for k in ['actions','native_counts','end_flags']},p['compact_bytes']);r.update(path=str(f),sha256=sha(f));rt.rows.append(r);name='cases.jsonl'
  with (Path(p['out'])/name).open('a') as f:f.write(json.dumps(r)+'\n')
  rows.append(r)
 return rows

def main(phase):
 folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=G.B.Runtime(p);state,error,result='FAILED',None,None
 try:
  rt.pool=F.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);results={}
  for mode in p['modes']:
   model=p['hybrid_models'][mode];rows=evaluate(rt,p,model,mode);evaluate(rt,p,model,mode,rows);results[mode]=dict(model=model,tally=G.R.M.tally(rows),by_task={r['job_id']:{k:r[k] for k in ['success','collision','timeout','remaining_m']} for r in rows})
  result=dict(modes=results,ActorAdam=0,ValueAdam=0,fullfield24=None,old8=None,accepted=False,only_two_tasks_diagnostic_NOT_candidate_promotion=True);state='COMPLETE'
 except BaseException:error=traceback.format_exc();save(folder/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(folder/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit(phase):
 folder=OUT/phase;p=F.R.child_plan(folder);F.R.setup(folder);verify(p,True);s=read(folder/'summary.json');assert s['state']=='COMPLETE' and not (folder/'audit.json').exists();G.stopped(folder);rows=G.rows(folder);rep=G.rows(folder,'replays.jsonl');assert len(rows)==len(rep)==4
 for mode in p['modes']:
  q=hybrid(p,mode);model=p['hybrid_models'][mode];assert sha(model['path'])==model['sha256'] and q.phash()==model['policy_hash'];saved=P.checkpoint(model['path']);assert digest(q.state_dict())==digest(saved['joint_state']['actor'])
  selected=[r for r in rows if r['role']==mode+'_FIRST'];assert G.R.M.tally(selected)==s['result']['modes'][mode]['tally']
  assert {r['job_id']:{k:r[k] for k in ['success','collision','timeout','remaining_m']} for r in selected}==s['result']['modes'][mode]['by_task']
 by={r['path']:r for r in rows}
 for r in rows:
  a=E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and set(a)=={'actions','native_counts','end_flags'} and int(a['native_counts'].sum())==r['physics']
  for k,v in a.items():assert digest(v)==r['array_hashes'][k]
  assert a['end_flags'][-1].tolist()==[int(r['collision']),int(r['success']),int(r['timeout'])] and not r['administrative_censor']
 for r in rep:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['physics']==sum(r['physics'] for r in rows+rep)<=p['max_physics'] and s['commands']==12
 save(folder/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,all_frozen_module_sources_exact=True,all_native_fullarray_hashes_and_integrations_replay_exact=True,compact_only_NOT_full120_reconstructed=True,physics=s['physics'],commands=s['commands']));print('T528_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
