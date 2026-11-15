"""Frozen student diagnosis: explicitly research-only despite failed offline gate."""
from pathlib import Path
import sys,json,time,shutil
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t567_runtime as R
ROOT,OUT,DOC,L,torch,np,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.torch,R.np,R.read,R.save,R.sha,R.digest
IDS=['t502_field_sector0_offset2','t502_field_sector1_offset2','t502_field_sector4_offset2']
def build():
 p=read(ROOT/'stage/runs/t566_constrained_joint_step/protocol.json');by={j['id']:j for j in p['evaluation_jobs']};baseline={r['job_id']:r for r in p['keep_rows']};model=read(ROOT/'stage/runs/t562_full_history_representation/summary.json')['result']['model'];p.update(id='T567_RESEARCH_ONLY_FROZEN_STUDENT_THREE_TARGET_DIAGNOSIS',jobs=[by[i] for i in IDS],job_ids=IDS,baseline_rows={i:baseline[i] for i in IDS},model=model,models=dict(student=model),max_physics=2*sum(by[i]['case']['step_cap'] for i in IDS),max_commands=10,main_seconds=1800,max_main_seconds=1800,learning_seconds=0,audit_seconds=600,transport_seconds=300,source_seconds=180,max_bytes=int(5.25*2**20),child_bytes=int(5.25*2**20),full_bytes=928*1024,scope='NewexplicitdiagnosticprotocolfreezeT562researchstudent,3targets+3exactfullhistories;failedofflinegateNOTwaivedforpromotion, noqualificationclaim/noold8/32orfull407/nooptimizer/RL/nohuman/teacher.Originalstartgoalclockgeometryand12064carryactions;serialsingleworker.')
 for key in ['evaluation_jobs','probe_jobs','perturb_jobs','direction_evidence']:p.pop(key,None)
 return p

def preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);p=build();cfg=read(p['actor_configuration']);cfg.update(p);q=L.frozen(cfg,p['model']['path']);assert sha(p['model']['path'])==p['model']['sha256'] and q.phash()==p['model']['policy_hash'];bounds=[]
 for i,row in p['baseline_rows'].items():
  assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);x=torch.tensor(a['states'][:16]);seq=q.sequence(x);future=x.clone();future[8:]+=1;assert torch.equal(q.sequence(future)[:8],seq[:8]);h=None;seen=[];q.reset_noise();q.sample=False
  with torch.no_grad():
   for z in x:u,loc,h,_=q.step(z[None],h);seen.append(loc)
  assert float((torch.cat(seen)-seq).abs().max())<1e-10
  packed=R.E.N.S.packed(a);n=len(a['states']);maxn=642;bound=65536
  for v in packed.values():
   bound+=(maxn*int(v[0].nbytes) if v.ndim and v.shape[0]==n else (maxn+1)*int(v[0].nbytes) if v.ndim and v.shape[0]==n+1 else v.nbytes)
  assert bound<=p['full_bytes'];bounds.append(dict(id=i,uncompressed_schema_upper=bound,max_decisions=642))
 # Original outcome descriptors reproduce safe baseline when fed its real success trace.
 tests=[]
 for j in p['jobs']:
  r=p['baseline_rows'][j['id']];a=R.E.N.S.arrays(r['path']);v=R.diagnose(p,r,a);assert v['success'] and v['first_3cm_or3deg_deviation_decision'] is None;tests.append(v)
 storage=5*p['full_bytes']+512*1024;assert storage<=p['max_bytes'];used=R.used(ROOT/'stage/runs');assert used+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',epoch=time.time(),new_native=0,new_optimizer=0,source={str(f):sha(f) for f in HERE.glob('*.py')},bounds=bounds,storage_bytes=storage,global_worst=used+p['max_bytes']+131072,model_hash=q.phash(),same_sequence_and_serial_step=True,baseline_descriptors=tests,no_promotion_or_offline_gate_qualification=True));print('T567_ZERO_PREFLIGHT_PASS',p['max_physics'],storage)

def prepare():
 assert not OUT.exists();pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER' and all(sha(f)==h for f,h in pref['source'].items());p=build();a=read(DOC/'AUTHORIZATION.json');assert a['scope_approved'];p.update(absolute_all_stop=a['all_stop_epoch'],absolute_native_stop=a['native_stop_epoch'],absolute_learning_stop=a['learning_stop_epoch'])
 prior=[read(ROOT/'stage/docs/day_20261115/t555/FINAL_REPORT.json')['ledger']]
 for n in ['t557_bc_stage_isolation','t559_smooth_policy_consolidation','t561_sine_policy_consolidation','t562_full_history_representation','t563_guarded_tail_fit','t564_local_peak_descent','t565_joint_residual_step','t566_constrained_joint_step']:prior.append(read(ROOT/'stage/runs'/n/'summary.json'))
 phys=sum(x['physics'] for x in prior);cmd=sum(x['commands'] for x in prior);main=sum(x.get('main_seconds',x.get('elapsed_s',0)) for x in prior);assert phys+p['max_physics']<=19_000_000 and cmd+p['max_commands']<=600 and main+p['max_main_seconds']<=12300
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC/'SOURCES.md',ROOT/'stage/docs/day_20261115/t566/FINDINGS_AND_NEXT.md',Path(p['model']['path'])]:p['inputs'][str(f)]=sha(f)
 p['protected_files'][p['model']['path']]=p['model']['sha256'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30 and time.time()+1800+600+300+180+60<p['absolute_all_stop'];OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics=p['max_physics'],commands=10,main_seconds=1800,day_physics_worst=phys+p['max_physics'],day_commands_worst=cmd+10,day_main_worst=main+1800,own_bytes=p['max_bytes'],full_bytes=p['full_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,new_optimizer=0,new_RL=0));print('T567_REGISTERED')
if __name__=='__main__':globals()[sys.argv[1]]()
