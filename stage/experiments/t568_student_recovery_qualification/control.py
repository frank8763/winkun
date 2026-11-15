"""One fixed real-prefix teacher recovery, no student learning after16:30."""
from pathlib import Path
import sys,json,time,shutil,math
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t568_runtime as R
ROOT,OUT,DOC,L,torch,np,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.torch,R.np,R.read,R.save,R.sha,R.digest
OLD=ROOT/'stage/runs/t567r_single_worker_recovery'
def build():
 p=read(OLD/'protocol.json');assert read(OLD/'audit.json')['state']=='PASS';row=R.G.rows(OLD)[0];assert row['job_id']=='t502_field_sector0_offset2' and row['timeout'] and not row['collision'];assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);old=read(ROOT/'stage/runs/t549_curvature_gated_transfer/protocol.json');review=read(ROOT/'stage/docs/day_20261114/T548_SAVED_RECOVERY_REVIEW.json');kmax=1/review['nominal_min_radius_m'];gain=review['measured_curvature_at_saturated_command_q10q50q90'][0]/kmax;d=np.linalg.norm(a['poses'][:,:2]-a['task_goal_xy'],axis=1);angle=np.arctan2(a['task_goal_xy'][1]-a['poses'][:,1],a['task_goal_xy'][0]-a['poses'][:,0])-a['poses'][:,2];angle=np.arctan2(np.sin(angle),np.cos(angle));proxy=2*np.abs(np.sin(angle))/np.maximum(d,.2)*(.2/np.maximum(d,.2))**(2*gain-2);hits=np.flatnonzero(proxy>kmax);assert len(hits);first=int(hits[0])-1;assert first>0 and np.cos(angle[first])>0 and proxy[first]<=kmax;job=p['jobs'][0]
 p.update(id='T568_ONE_STUDENT_PREFIX_REACTIVE_RECOVERY_QUALIFICATION',job=job,baseline_row=row,teacher_first=first,fixed_prefix=dict(rule='LAST_FEASIBLE_BEFORE_FIRST_PROXY_CURVATURE_CROSSING_NO_GRID',decision=first,seconds=float(a['native_counts'][:first].sum()*.002),distance_m=float(d[first]),heading_error_deg=float(np.rad2deg(angle[first])),measured_gain_proxy=float(gain),curvature_limit=float(kmax),source_sha256=row['sha256'],not_reachability_certificate=True),max_physics=2*job['case']['step_cap'],max_commands=6,main_seconds=900,max_main_seconds=900,learning_seconds=0,audit_seconds=300,transport_seconds=300,source_seconds=180,max_bytes=int(3.5*2**20),child_bytes=int(3.5*2**20),full_bytes=960*1024,scope='FixedT562realprefixthenlegacygoalfeedbackteacherONLYqualification;onecase+oneexact,0learning/RL.Originalclock/history/actuatorpreserved;earlycurvatureproxydoesnotguaranteeGOAL. QualifiedsuccessNOTautonomouspolicyimprovement.')
 for k in ['physical_wheelbase_m','nominal_speed_m_s','slow_distance_m','minimum_heading_speed_fraction','curvature_distance_floor_m']:p[k]=old[k]
 return p

def preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);p=build();cfg=read(p['actor_configuration']);cfg.update(p);q=L.frozen(cfg,p['model']['path']);assert q.phash()==p['model']['policy_hash'];a=R.E.N.S.arrays(p['baseline_row']['path']);x=torch.tensor(a['states'][:16]);q.sample=False;q.reset_noise();h=None
 with torch.no_grad():
  for i,z in enumerate(x):u,loc,h,_=q.step(z[None],h);assert np.array_equal(u[0].numpy(),a['actions'][i])
 params=np.array([.03,60.,.6,1.,math.radians(25)]);u,steer,speed=R.Teacher.feedback(p,a['poses'][p['teacher_first']],a['task_goal_xy'],params);assert np.isfinite(u).all() and np.abs(u).max()<=1 and speed>0
 pack=R.E.N.S.packed(a);n=len(a['states']);mx=642;bound=65536
 for v in pack.values():bound+=mx*int(v[0].nbytes) if v.ndim and v.shape[0]==n else (mx+1)*int(v[0].nbytes) if v.ndim and v.shape[0]==n+1 else v.nbytes
 bound+=mx*2*4+5*8;assert bound<=p['full_bytes'];storage=3*p['full_bytes']+512*1024;assert storage<=p['max_bytes'];used=R.used(ROOT/'stage/runs');assert used+p['max_bytes']+131072<2**30;assert 'range(len(self.children))' in (HERE/'t568_runtime.py').read_text()
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',epoch=time.time(),source={str(f):sha(f) for f in HERE.glob('*.py')},new_native=0,new_optimizer=0,fixed_prefix=p['fixed_prefix'],full_uncompressed_upper=bound,storage_bytes=storage,global_worst=used+p['max_bytes']+131072,student_prefix_commands_exact=True,teacher_bounds_test_only_not_native_recovery=True));print('T568_ZERO_PREFLIGHT_PASS',p['max_physics'],bound)

def prepare():
 assert not OUT.exists();pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER' and all(sha(f)==h for f,h in pref['source'].items());p=build();a=read(DOC/'AUTHORIZATION.json');assert a['scope_approved'];p.update(absolute_all_stop=a['all_stop_epoch'],absolute_native_stop=a['native_stop_epoch'],absolute_learning_stop=a['learning_stop_epoch'])
 prior=[read(ROOT/'stage/docs/day_20261115/t555/FINAL_REPORT.json')['ledger']]
 for n in ['t557_bc_stage_isolation','t559_smooth_policy_consolidation','t561_sine_policy_consolidation','t562_full_history_representation','t563_guarded_tail_fit','t564_local_peak_descent','t565_joint_residual_step','t566_constrained_joint_step','t567_frozen_closed_loop_diagnosis','t567r_single_worker_recovery']:prior.append(read(ROOT/'stage/runs'/n/'summary.json'))
 phys=sum(x['physics'] for x in prior);cmd=sum(x['commands'] for x in prior);main=sum(x.get('main_seconds',x.get('elapsed_s',0)) for x in prior);assert phys+p['max_physics']<=19_000_000 and cmd+6<=600 and main+900<=12300
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC/'SOURCES.md',Path(p['model']['path']),Path(p['baseline_row']['path']),ROOT/'stage/docs/day_20261114/T548_SAVED_RECOVERY_REVIEW.json',ROOT/'stage/runs/t549_curvature_gated_transfer/protocol.json']:p['inputs'][str(f)]=sha(f)
 p['protected_files'][p['model']['path']]=p['model']['sha256'];p['protected_files'][p['baseline_row']['path']]=p['baseline_row']['sha256'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30 and time.time()+900+300+300+180+60<p['absolute_all_stop'];OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics=p['max_physics'],commands=6,main_seconds=900,day_physics_worst=phys+p['max_physics'],day_commands_worst=cmd+6,day_main_worst=main+900,own_bytes=p['max_bytes'],full_bytes=p['full_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,new_optimizer=0,new_RL=0));print('T568_REGISTERED')
if __name__=='__main__':globals()[sys.argv[1]]()
