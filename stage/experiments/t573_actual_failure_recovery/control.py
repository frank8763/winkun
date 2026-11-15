"""One fixed recovery qualification for the best measured research A candidate."""
from pathlib import Path
import sys,json,time,shutil
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t573_runtime as R
ROOT,OUT,DOC,L,torch,np,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.torch,R.np,R.read,R.save,R.sha,R.digest

def build():
 old=ROOT/'stage/runs/t572_functional_trust_region';p=read(old/'protocol.json');s=read(old/'unconstrained/summary.json');assert s['state']=='COMPLETE' and not s['result']['accepted'];assert s['result']['check']['old8']['success']==8 and s['result']['check']['field24']['success']==14;assert s['result']['check']['missing_reference23']==['t502_field_sector2_offset2'];assert read(old/'unconstrained/audit.json')['state']=='PASS';model=s['result']['model'];job=next(j for j in p['evaluation_jobs'] if j['id']=='t502_field_sector2_offset2');baseline=next(r for r in R.G.rows(old/'unconstrained') if r['job_id']==job['id']);reference=next(r for r in p['keep_rows'] if r['job_id']==job['id']);teacher=read(ROOT/'stage/runs/t568_student_recovery_qualification/protocol.json')
 for k in ['physical_wheelbase_m','nominal_speed_m_s','slow_distance_m','minimum_heading_speed_fraction','curvature_distance_floor_m']:p[k]=teacher[k]
 p.update(id='T573_FROZEN_A_SOLE_LOST_SUCCESS_ACTUAL_RECOVERY_QUALIFICATION',model=model,models=dict(student=model),job=job,baseline_compact_row=baseline,old_success_row=reference,max_physics=4*job['case']['step_cap'],max_commands=12,main_seconds=600,max_main_seconds=600,learning_seconds=0,audit_seconds=120,transport_seconds=240,source_seconds=120,max_bytes=5*2**20,child_bytes=5*2**20,full_bytes=960*1024,inputs={},protected_files={},scope='FrozenunpromotedT572Asolelostsector2off2actualfull+exact;oneprefixbeforefirst3cm/3degdeviationvsT550oldsuccess,thenonelegacygoalfeedbackactual+exactifprefixexists.0optimizer/RL/learning,newrecoveryGOALNOTstudentgain;nostarttime/noise/teachergrid. T572Rmustendfirst.')
 return p

def preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);p=build();cfg=read(p['actor_configuration']);cfg.update(p);q=L.frozen(cfg,p['model']['path']);assert q.phash()==p['model']['policy_hash'];reference=R.E.N.S.arrays(p['old_success_row']['path']);x=torch.tensor(reference['states'][:16]);q.sample=False;q.reset_noise();h=None;locs=[]
 with torch.no_grad():
  for z in x:u,loc,h,_=q.step(z[None],h);locs.append(loc[0]);assert torch.isfinite(u).all()
 error=float((torch.stack(locs)-q.sequence(x)).abs().max());assert error<1e-9
 # Prefix rule known controlled geometric divergence; no new native episode.
 from copy import deepcopy
 fake=deepcopy(reference);fake['poses']=reference['poses'].copy();fake['poses'][4:,0]+=.031;first,meta=R.prefix(p,fake,reference);assert first==3 and meta['first_divergence_decision']==4
 packed=R.E.N.S.packed(reference);n=len(reference['states']);mx=(p['job']['case']['step_cap']+99)//100;upper=65536
 for z in packed.values():upper+=mx*int(z[0].nbytes) if z.ndim and z.shape[0]==n else (mx+1)*int(z[0].nbytes) if z.ndim and z.shape[0]==n+1 else z.nbytes
 upper+=mx*2*4+40;assert upper<=p['full_bytes'];storage=4*p['full_bytes']+768*1024;assert storage<=p['max_bytes'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 params=np.asarray([.03,60.,.6,1.,np.deg2rad(25)]);a,_,_=R.Teacher.feedback(p,reference['poses'][0],reference['task_goal_xy'],params);assert np.isfinite(a).all() and np.abs(a).max()<=1;save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_optimizer=0,new_RL=0,source={str(f):sha(f) for f in HERE.glob('*.py')},actual_A_model_readback_and_sequence_step_error=error,synthetic_prefix_rule_checked=True,teacher_bounds_only_not_recovery_success=True,full_teacher_schema_upper=upper,storage_upper=storage,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072));print('T573_PREFLIGHT_PASS',p['max_physics'],upper)

def prepare():
 assert not OUT.exists();pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER' and all(sha(f)==h for f,h in pref['source'].items());p=build();a=read(DOC/'AUTHORIZATION.json');p.update(absolute_all_stop=a['all_stop_epoch'],absolute_native_stop=a['native_stop_epoch'],absolute_learning_stop=a['learning_stop_epoch']);prior=ROOT/'stage/runs/t572r_primal_feasible_solver';assert read(prior/'bounded/summary.json')['state']=='COMPLETE' and read(prior/'bounded/audit.json')['state']=='PASS' and read(prior/'SEALED.json')['all_scientific_processes_stopped'];assert all(not Path('/proc',str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids']);assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
 b=read(ROOT/'stage/docs/day_20261115/t572r/BUDGET.json')['day_actual_before'];s=read(prior/'bounded/summary.json');phys=b['physics']+s['physics'];cmd=b['commands']+s['commands'];seconds=b['main_seconds']+s['elapsed_s'];assert phys+p['max_physics']<=19_000_000 and cmd+p['max_commands']<=600 and seconds+p['main_seconds']<=12300
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'AUTHORIZATION.json',DOC/'PLAN.md',DOC/'SOURCES.md',DOC/'PREFLIGHT.json',Path(p['model']['path']),Path(p['actor_configuration']),Path(p['old_success_row']['path']),ROOT/'stage/runs/t572_functional_trust_region/unconstrained/cases.jsonl',ROOT/'stage/runs/t572_functional_trust_region/unconstrained/audit.json']:p['inputs'][str(f)]=sha(f)
 p['protected_files']={p['model']['path']:p['model']['sha256'],p['old_success_row']['path']:p['old_success_row']['sha256']};assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30;assert time.time()+600+120+240+120+60<p['absolute_all_stop'] and time.time()+600<p['absolute_native_stop'];OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics=p['max_physics'],commands=12,main_seconds=600,day_actual_before=dict(physics=phys,commands=cmd,main_seconds=seconds),day_worst=dict(physics=phys+p['max_physics'],commands=cmd+12,main_seconds=seconds+600),own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,new_optimizer=0,new_RL=0));print('T573_REGISTERED')
if __name__=='__main__':globals()[sys.argv[1]]()
