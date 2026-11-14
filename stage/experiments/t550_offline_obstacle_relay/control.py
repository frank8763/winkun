"""Register locked afternoon scope, meaningful zero/native contracts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,json,sys,time,traceback,os
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t550_runtime as R
ROOT,OUT,DOC,L,np,torch,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.np,R.torch,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();old=ROOT/'stage/runs/t549_curvature_gated_transfer';s=read(old/'summary.json');assert s['state']=='COMPLETE' and read(old/'audit.json')['state']=='PASS';R.G.stopped(old);assert not Path('/proc',str(read(old/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T549.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 source=read(old/'protocol.json');p=read(source['actor_configuration']);rows=R.G.rows(old)[1:];assert len(rows)==32;before=s['result']['check'];assert before['old8']['success']==8 and before['field24']['success']==13
 protected=[]
 for r in source['protected20_rows']+[next(r for r in rows if r['job_id']=='t502_field_sector0_offset2')]:
  assert sha(r['path'])==r['sha256'] and r['success'] and not r['collision'];protected.append({k:r[k] for k in ['job_id','path','sha256','success','collision']})
 assert len(protected)==21
 def epoch(hour,minute=0):return datetime(2026,11,14,hour,minute,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
 jobs=p['old8_jobs']+p['field_jobs'];byid={r['job_id']:r for r in rows};failed=[j for j in jobs if not byid[j['id']]['success'] or byid[j['id']]['collision']];failed.sort(key=lambda j:(0 if byid[j['id']]['collision'] else 1,j['sector_offset'],j['start_sector']));assert len(failed)==11
 finaljobs=read(ROOT/'stage/runs/t542_offline_field_relay/final_selected407/protocol.json')['final_jobs'];assert len(finaljobs)==407
 auth=dict(user_reply='允许此有界下午预算，20:30全部停止',date='2026-11-14',additional_physics=50_000_000,additional_commands=1500,additional_main_seconds=24000,prior_actual_physics=7342193,prior_actual_commands=342,day_max_physics=57342193,day_max_commands=1842,all_stop_epoch=epoch(20,30),CPU_total_one_core=True,GPU=False,global_bytes=2**30,free_bytes=40*2**30);save(DOC/'T550_BUDGET_AUTHORIZATION.json',auth)
 p.update(id='T550_OFFLINE_REACTIVE_TEACHING_AND_PROTECTED_TRANSFER',actor_configuration=source['actor_configuration'],initial=s['result']['model'],source_protocol=source['source_protocol'],protected_rows=protected,before_check=before,before_rows=rows,evaluation_jobs=jobs,failure_jobs=failed,teacher_kinds=['APF','GAP'],max_fit_proposals=8,final_jobs=finaljobs,main_seconds=1800,learning_seconds=1200,audit_seconds=600,child_bytes=7*2**20,full_bytes=2**20,compact_bytes=12288,model_bytes=1572864,max_bytes=64*2**20,max_physics=50_000_000,max_commands=1500,max_main_seconds=24000,absolute_learning_stop=epoch(19),absolute_native_stop=epoch(19,30),absolute_all_stop=epoch(20,30),final_reserve_seconds=7200,transport_seconds=600,source_seconds=180,prior_actual_physics=7342193,prior_actual_commands=342,accepted=False,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,scope='Fixed11failedIDs,APFthenGAPeachonce. ActualsafeGOAL+exactthenoneprotectedidentitybasisfit+shadow,max8proposals. Eachcandidate32+32exact,allprevioussuccessIDsandfullparenttrajectoriesexactmustretainwithsafeincrease. Final407onlyifcompleteworstreserve. ZeroRL/Adam;finiteexamplememoryNOTgeneralization. No runtime teacher/map/planner/route/waypoints/taskID, no seed/parameter/windowgrid. Local retention and old contact<=9 strict gate separate; no globalpromotion.')
 p['protected_files']={source['initial']['path']:source['initial']['sha256'],p['initial']['path']:p['initial']['sha256']};p['inputs']=dict(p['protected_files'])
 for row in protected:p['inputs'][row['path']]=row['sha256']
 for f in [old/'protocol.json',old/'summary.json',old/'audit.json',old/'cases.jsonl',ROOT/'stage/docs/long_distance_20261101/release_receipt_T549.json',ROOT/'stage/runs/t542_offline_field_relay/final_selected407/protocol.json',DOC/'T550_PLAN.md',DOC/'T550_REACTIVE_TEACHING_SOURCES.md',DOC/'T550_BUDGET_AUTHORIZATION.json',DOC/'T550_COLD_CACHE.json']:p['inputs'][str(f)]=sha(f)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 physics_bound=4*sum(j['case']['step_cap'] for j in failed)+8*2*sum(j['case']['step_cap'] for j in jobs)+sum(j['case']['step_cap'] for j in finaljobs)+2000
 commands_bound=44+8*64+407+22*4+8*4+4+8
 assert physics_bound<=p['max_physics'] and commands_bound<=p['max_commands'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<=2**30
 OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 save(DOC/'T550_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',worst_all_physics=physics_bound,hard_physics=50_000_000,worst_all_commands=commands_bound,hard_commands=1500,max_main_seconds=24000,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,teacher_full_max22=22*2**20,candidate_full_max8=8*2**20,model_delta8_max=8*p['model_bytes'],compact_reserved=8*2**20,metadata_partial_reserved=14*2**20,all_stop='2026-11-14 20:30 Asia/Shanghai',new_Adam=0,new_RL=0,exact_native_includes_repeats=True,final407_physics=sum(j['case']['step_cap'] for j in finaljobs),no_automatic_stability_or_newdistance=True))
 print('T550_REGISTERED_NOT_LAUNCHED',physics_bound,commands_bound)

def preflight():
 root=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists();R.setup(OUT);R.verify(root,True);p=read(root['actor_configuration']);p.update(root);q0=L.frozen(p,p['initial']['path']);q=L.extend(q0,p,2);row=next(r for r in root['protected_rows'] if r['job_id']=='t502_field_sector0_offset2');a=R.E.N.S.arrays(row['path']);x=torch.tensor(a['states'][:16]);assert torch.equal(q.sequence(x),q0.sequence(x));x2=x.clone();x2[8:]+=1;assert torch.equal(q.sequence(x2)[:8],q.sequence(x)[:8]);assert not any(z.requires_grad for z in q.parameters())
 parameters=np.asarray([.01,15.,.6,1.,np.deg2rad(25.)]);scans=[np.ones(48,np.float32),np.r_[np.ones(20),np.full(8,.15),np.ones(20)].astype(np.float32)]
 for kind in root['teacher_kinds']:
  for scan in scans:
   u=R.T.control(kind,np.zeros(3),np.asarray([3.,0.]),scan,parameters);assert np.isfinite(u).all() and np.abs(u).max()<=1
  for invalid in [np.full(48,np.nan),np.ones(47)]:
   try:R.T.control(kind,np.zeros(3),np.ones(2),invalid,parameters)
   except AssertionError:pass
   else:raise AssertionError('Invalid scan was accepted')
 # Real features verify identity support and a nonzero reachable neural readout.
 features=L.Prior.Old.features(q0,torch.tensor(a['states'][220:224]));distance=((features[:,None]-features)**2).sum(-1);distance.fill_diagonal_(float('inf'));radius=.49*distance.min(-1).values.sqrt();assert torch.equal(L.Prior.Old.basis(features,features,radius),torch.eye(4,dtype=features.dtype));q.layers[-1].centers.copy_(features[:2]);q.layers[-1].radii.copy_(radius[:2]);q.layers[-1].weights.copy_(torch.tensor([[.1,-.1],[-.1,.1]],dtype=torch.float64));model=L.export(q,p,OUT/'discarded_preflight.pt.gz');assert L.frozen(p,model['path']).phash()==q.phash();Path(model['path']).unlink()
 # Worst raw size of 840 centers plus serialization margin is below delta cap.
 assert 840*(184+1+2)*8+65536<p['model_bytes']
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),physics=0,commands=0,new_Adam=0,new_RL=0,teacher_scan_finite_negative_tests=True,initial_function_exact=True,causal_history=True,real_feature_K_identity=True,nonzero_new_layer_checkpoint_reload_exact=True,delta_raw_size_bound=True,new_scientific_claim_NOT_validated=True));print('T550_ZERO_NATIVE_PASS')

def engineering_register():
 root=read(OUT/'protocol.json');assert not (OUT/'native_preflight').exists();job=copy.deepcopy(root['failure_jobs'][0]);job['case']['step_cap']=500;folder=OUT/'native_preflight';folder.mkdir();save(folder/'protocol.json',dict(_root_sha256=sha(OUT/'protocol.json'),kind='engineering',teacher_kind='APF',probe_job=job,initial=root['initial'],protected_rows=root['protected_rows'],before_check=root['before_check'],before_rows=root['before_rows'],max_physics=1000,max_commands=6,main_seconds=300,learning_seconds=120,child_bytes=2*2**20))
 print('T550_ENGINEERING_REGISTERED')
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  save(DOC/('T550_'+sys.argv[1]+'_ERROR.json'),dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));raise
