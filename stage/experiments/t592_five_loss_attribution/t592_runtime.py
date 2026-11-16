"""Five fixed losses, frozen pre/post actors; no optimizer or new teacher."""
from pathlib import Path
import sys,os,time,fcntl,traceback,copy,json,ast
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t592_five_loss_attribution';DOC=ROOT/'stage/docs/day_20261116/t592'
sys.path.insert(0,str(ROOT/'stage/experiments/t591_joint_representation_pair'))
import t591_runtime as R
L,Base,F,G,E,P,torch,np=R.L,R.Base,R.F,R.G,R.E,R.P,R.torch,R.np
read,save,sha,digest=R.read,R.save,R.sha,R.digest
sys.path.insert(0,str(ROOT/'stage/experiments/t579_frozen_progress_attribution'))
import t579_runtime as Describe

def setup(folder):
 R.OUT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(folder)
def verify(p,full=False):return R.verify(p,full)

def describe(p,row,a):
 report=Describe.describe(row,a)
 old=next(x for x in p['reference_rows'] if x['job_id']==row['job_id']);b=E.N.S.arrays(old['path'])
 assert sha(old['path'])==old['sha256']
 for key in ['initial_native_integration','task_goal_xy','scene_sha256']:assert np.array_equal(a[key],b[key]),(row['job_id'],key)
 n=min(len(a['poses']),len(b['poses']));delta=a['poses'][:n]-b['poses'][:n]
 pos=np.linalg.norm(delta[:,:2],axis=1);yaw=np.abs(np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2])))
 hits=np.flatnonzero((pos>=.03)|(yaw>=np.deg2rad(3)));k=int(hits[0]) if len(hits) else None
 report.update(reference_T550_sha256=old['sha256'],first_3cm_or_3deg_decision=k,first_deviation_seconds=None if k is None else float(a['native_counts'][:k].sum()*.002),recovery_candidate_decision=None if k is None else max(0,k-1),max_common_position_difference_m=float(pos.max()),first_command_difference=(a['actions'][0]-b['actions'][0]).tolist(),same_native_initial_scene_goal=True,not_a_verified_teacher_recovery=True)
 return report

def prepare():
 assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
 prev=ROOT/'stage/runs/t591_joint_representation_pair';p=copy.deepcopy(read(prev/'protocol.json'));s=read(prev/'summary.json');launch=read(prev/'launch.json')
 assert read(prev/'audit.json')['state']=='PASS' and s['state']=='COMPLETE'
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T591.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 assert all(not Path('/proc',str(i)).exists() for i in read(prev/'SEALED.json')['scientific_pids']+[launch['pid']])
 lost=set(s['result'][1]['check']['lost_initial']);jobs=[j for j in p['evaluation_jobs'] if j['id'] in lost];refs=[x for x in p['keep_rows'] if x['job_id'] in lost]
 assert len(jobs)==len(refs)==5
 model=dict(s['result'][1]['proposal'],label='after600');initial_path=prev/'initial.pt.gz'
 setup(DOC);q=L.frozen(p,initial_path);initial=dict(path=str(initial_path),sha256=sha(initial_path),policy_hash=q.phash(),label='before600')
 cap=4*sum(j['case']['step_cap'] for j in jobs);assert cap==594796 and cap+s['physics']<=p['max_physics'] and 24+s['commands']<=p['max_commands']
 end=launch['all_stop']-4;native=end-600
 p.update(id='T592_FIVE_FIXED_LOSSES_PRE_POST_ATTRIBUTION',out=str(OUT),jobs=jobs,models=[initial,model],reference_rows=refs,prior_candidate_rows=[x for x in G.rows(prev/'trainable_representation') if x['job_id'] in lost and x['role']=='ORIGINAL_START49'],
  main_seconds=900,learning_seconds=0,audit_seconds=180,transport_seconds=180,source_seconds=90,absolute_learning_stop=time.time(),absolute_native_stop=native,absolute_all_stop=end,
  max_physics=cap,max_commands=24,max_bytes=13*2**20,child_bytes=13*2**20,inputs={},protected_files={},new_Actor_Adam_cap=0,shadow_Adam_cap=0,audit_discarded_Adam_cap=0,preflight_discarded_Adam_cap=0,new_RL=0,
  authorization_approved=True,user_authorization='2026-11-16 user 下一步开始 after completed T591. New bounded frozen attribution,0learning, within existing90minute deadline and unused physical/command ceiling; no extension or new optimizer allocation.',
  scope='Five fixed lost IDs only. Before600 five compact+exact; after600 five full+exact precisely reconstruct T591. Compare each to existing verified T550 success history. No teacher, learning, model selection, promotion or automatic next mechanism.')
 upper=7*p['full_bytes']+5*p['compact_bytes']+3*2**20;glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072
 assert upper<=p['max_bytes'] and glob<2**30 and time.time()+1380<end and time.time()+900<native
 checks=[]
 for m in p['models']:
  assert sha(m['path'])==m['sha256'];z=L.frozen(p,m['path']);assert z.phash()==m['policy_hash'];before=digest(z.state_dict())
  for row in refs:
   assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);x=torch.tensor(a['states'][:48]);z.sample=False;z.reset_noise();h=None;uu=[]
   with torch.no_grad():
    expected=z.sequence(x).clamp(-1,1).float()
    for xx in x:
     u,_,h,_=z.step(xx[None],h);uu.append(u[0])
   err=float((torch.stack(uu)-expected).abs().max());assert err<=2e-6;checks.append(dict(model=m['label'],job=row['job_id'],step_sequence_action_max=err))
  assert digest(z.state_dict())==before
 assert not torch.cuda.is_initialized()
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',checks=checks,models_unchanged=True,own_upper_bytes=upper,global_worst=glob))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',prev/'audit.json',prev/'summary.json',Path(p['actor_configuration'])]+[Path(x['path']) for x in refs]+[Path(j['exact_snapshot']) for j in jobs]:p['inputs'][str(f)]=sha(f)
 for m in p['models']:p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256']
 save(DOC/'AUTHORIZATION.json',dict(user_reply='下一步开始',scope=p['scope'],prior90minute_cap_unchanged=True,prior_approved_physics=6025900,prior_actual_physics=s['physics'],new_physics=cap,combined_max_physics=s['physics']+cap,prior_actual_commands=s['commands'],new_commands=24,combined_max_commands=s['commands']+24,all_stop=end,new_optimizer=0))
 p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json')
 save(DOC/'BUDGET.json',dict(max_physics=cap,max_commands=24,own_bytes=p['max_bytes'],own_upper=upper,global_worst=glob,main_seconds=900,audit_seconds=180,transport_seconds=180,source_seconds=90,admission_seconds=1380,absolute_all_stop=end))
 OUT.mkdir();save(OUT/'protocol.json',p);save(HERE/'protocol.json',p);verify(dict(p,_root_sha256=sha(OUT/'protocol.json')),True)
 print('T592_REGISTERED_ZERO_NATIVE_ZERO_OPTIMIZER',cap,glob,flush=True)

def main():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);status='FAILED';error=None;reports=[]
 try:
  rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  for m in p['models']:
   full=m['label']=='after600';vv=rt.request(p['jobs'],m,'FROZEN_'+m['label'],full=full);rr=[r for a,r in vv]
   for a,r in vv:
    F.M.audit_full_history(p,r,a);reports.append(dict(model=m['label'],**describe(p,r,a)))
    if full:
     prior=next(x for x in p['prior_candidate_rows'] if x['job_id']==r['job_id']);assert r['array_hashes']==prior['array_hashes'] and r['integration_sha256']==prior['integration_sha256']
   rt.request(p['jobs'],m,'EXACT_'+m['label'],full=full,reference=rr)
  save(OUT/'trajectory_diagnosis.json',reports);status='COMPLETE'
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=status,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=reports,new_Actor=0,new_Adam=0,new_RL=0,after_five_are_historical_exact_repeats=True,full407=None,globally_promoted=False));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert status=='COMPLETE',error

def audit():
 p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
 rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==10;by={r['path']:r for r in rows};reports=read(OUT/'trajectory_diagnosis.json');assert reports==s['result']
 for i,r in enumerate(rows):
  assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);assert int(a['native_counts'].sum())==r['physics'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
  if r['full']:
   F.M.audit_full_history(p,r,a);assert dict(model='after600',**describe(p,r,a))==reports[i]
   old=next(x for x in p['prior_candidate_rows'] if x['job_id']==r['job_id']);assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['commands']==24 and s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,all_10_exact_native_replays=True,after5_full_current_history_and_metrics_reconstructed=True,before5_compact_live_metrics_not_independent_full_pose_reconstruction=True,all_model_SHA_unchanged=True,physics=s['physics'],commands=s['commands']));print('T592_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
