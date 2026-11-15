"""Read-only recovery of saved T574 mixed ownership; zero native/optimizer.

The historical producer and auditor remain unchanged. This local policy-tape
checker preserves all original checks and generalizes only ownership interval.
"""
from pathlib import Path
import sys,os,json,time,copy,shutil,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t574r_readonly_handoff_audit';DOC=ROOT/'stage/docs/day_20261115/t574r';OLD=ROOT/'stage/runs/t574_neural_recovery_handoff'
sys.path.insert(0,str(ROOT/'stage/experiments/t574_neural_recovery_handoff'));import t574_runtime as R
P,E,F,G,L,np,torch=R.P,R.E,R.F,R.G,R.L,R.np,R.torch
read,save,sha,digest=R.read,R.save,R.sha,R.digest

def policy_tape(p,r,a):
 # Ownership semantics are an exact registered interval, including return.
 expected=np.zeros(len(a['states']),np.uint8);start=p['teacher_first'];holds=int(a['teacher_holds']);expected[start:None if holds<0 else start+holds]=1
 assert np.array_equal(a['ownership'],expected),'ownership interval differs from registered full/20decision correction';assert r['teacher'] and not a['stochastic'].any()
 q=L.frozen(p,r['request']['model']);q.rng=None if r['seed'] is None else np.random.default_rng(np.random.SeedSequence(r['seed']));q.reset_noise();h=None
 with torch.no_grad():
  for i,x in enumerate(a['states']):
   q.sample=bool(a['stochastic'][i]);hb=np.zeros(64) if h is None else h[0,0].numpy().copy();command,loc,h,_=q.step(torch.tensor(x[None]),h);assert np.array_equal(hb,a['hidden_before'][i])
   for k,v in q.tape[-1].items():assert np.array_equal(v,a[k][i]),(i,k)
   if not a['ownership'][i]:assert np.array_equal(command[0].numpy(),a['actions'][i])
   else:assert np.array_equal(a['teacher_actions'][i],a['actions'][i])
   assert np.abs(E.N.K.H.rebuild(x,a['poses'][i],a['history_poses'][i],a['task_goal_xy'],a['history_goals'][i])-x).max()<=2e-6
   n=int(a['native_counts'][i]);z=E.N.K.R.GAMMA**n;flags=a['end_flags'][i];d0=np.linalg.norm(a['poses'][i,:2]-a['task_goal_xy']);d1=np.linalg.norm(a['next_poses'][i,:2]-a['task_goal_xy']);reward=-.000002*(1-z)/(1-E.N.K.R.GAMMA)+.03*d0-z*.03*d1*(not flags.any())
   if flags.any():reward+=E.N.K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
   assert abs(reward-a['rewards'][i])<1e-12
 assert np.array_equal(h[0,0].numpy(),a['last_hidden_after'])

def prefix_teacher(p,r,a,baseline):
 k=p['teacher_first'];assert len(a['states'])>k
 for key in ['states','next_states','actions','native_counts','poses','next_poses','hidden_before','history_poses','history_goals','history_times','locations']:assert np.array_equal(a[key][:k],baseline[key][:k]),key
 for key in ['states','poses','hidden_before','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],baseline[key][k]),key
 assert np.array_equal(a['initial_native_integration'],baseline['initial_native_integration']);teacher=L.frozen(p,p['neural_teacher']['path']);teacher.sample=False;teacher.reset_noise();th=None;student=L.frozen(p,p['model']['path']);student.sample=False;student.reset_noise();h=None
 with torch.no_grad():
  for i,x in enumerate(a['states']):
   if i==k:assert np.array_equal(a['teacher_takeover_hidden_before'],np.zeros((1,1,64)) if th is None else th.numpy())
   own,_,h,_=student.step(torch.tensor(x[None]),h);command,_,th,_=teacher.step(torch.tensor(x[None]),th);assert np.array_equal(command[0].numpy(),a['teacher_actions'][i]);assert np.array_equal(a['actions'][i],command[0].numpy() if a['ownership'][i] else own[0].numpy())
 policy_tape(p,r,a);return F.detail(p,r,a)

def expect_reject(fn,label):
 try:fn()
 except AssertionError:return label
 raise AssertionError('Negative regression did not reject '+label)

def prepare():
 assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True);raw=read(OLD/'protocol.json');rows=G.rows(OLD);assert read(OLD/'summary.json')['state']=='FAILED' and len(rows)==2 and len(G.rows(OLD,'replays.jsonl'))==1;assert read(OLD/'SEALED.json')['all_scientific_processes_stopped'];assert not Path('/proc',str(read(OLD/'launch.json')['pid'])).exists()
 used=R.used(ROOT/'stage/runs');assert used+512*1024+131072<2**30;auth=read(DOC/'AUTHORIZATION.json');assert time.time()+300<auth['all_stop_epoch'];p=dict(id='T574R_READ_ONLY_OWNERSHIP_INTERVAL_AUDIT_RECOVERY',old_protocol=str(OLD/'protocol.json'),max_bytes=512*1024,main_seconds=180,absolute_all_stop=auth['all_stop_epoch'],new_native=0,new_optimizer=0,new_RL=0,new_native_commands=0,source={str(f):sha(f) for f in HERE.glob('*.py')},inputs={})
 for f in [OLD/'protocol.json',OLD/'cases.jsonl',OLD/'replays.jsonl',OLD/'summary.json',OLD/'failure.json',OLD/'SEALED.json',Path(raw['model']['path']),Path(raw['neural_teacher']['path']),Path(raw['baseline_row']['path']),ROOT/'stage/experiments/t574_neural_recovery_handoff/t574_runtime.py',ROOT/'stage/experiments/t449_offline_joint_recovery/run.py',DOC/'PLAN.md',DOC/'AUTHORIZATION.json']+[Path(r['path']) for r in rows]:p['inputs'][str(f)]=sha(f)
 OUT.mkdir();save(OUT/'protocol.json',p);save(DOC/'BUDGET.json',dict(new_native=0,new_native_commands=0,new_optimizer=0,new_RL=0,offline_audit_processes=1,main_seconds=180,own_bytes=512*1024,global_worst=used+512*1024+131072,original_physics=109330,original_commands=7,missing_native_exact_replays=1));print('T574R_REGISTERED_ZERO_NATIVE')

def main():
 start=time.time();cfg=read(OUT/'protocol.json');assert not (OUT/'launch.json').exists();assert time.time()+cfg['main_seconds']+90<cfg['absolute_all_stop'];assert all(sha(f)==h for f,h in cfg['inputs'].items()) and all(sha(f)==h for f,h in cfg['source'].items());assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1);torch.use_deterministic_algorithms(True);P.frozen=L.frozen;receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');import subprocess;assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,source_remote=receipt['remote_commit'],new_native=0,new_optimizer=0,new_RL=0));old=read(cfg['old_protocol']);p=read(old['actor_configuration']);p.update(old);rows=G.rows(OLD);reps=G.rows(OLD,'replays.jsonl');arrays=[E.N.S.arrays(r['path']) for r in rows];baseline=E.N.S.arrays(p['baseline_row']['path'])
 # Deterministic original reproduction: exactly the real four-second-return case.
 try:R.recovery_audit(p,rows[1],arrays[1],baseline)
 except AssertionError as ex:
  frames=__import__('traceback').extract_tb(ex.__traceback__);assert frames[-1].name=='audit_full_history' and frames[-1].lineno==360;original_red=True
 else:raise AssertionError('Original handoff failure no longer reproduced')
 details=[]
 for r,a in zip(rows,arrays):
  assert sha(r['path'])==r['sha256'] and digest(a)==r['array_digest'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items());assert int(a['native_counts'].sum())==r['physics'];details.append(prefix_teacher(p,r,a,baseline))
 negatives=[];bad=dict(arrays[1],ownership=arrays[1]['ownership'].copy());bad['ownership'][p['teacher_first']+20]=1;negatives.append(expect_reject(lambda:policy_tape(p,rows[1],bad),'wrong_teacher_owner_after_return'))
 bad=dict(arrays[1],teacher_takeover_hidden_before=arrays[1]['teacher_takeover_hidden_before'].copy());bad['teacher_takeover_hidden_before'].flat[0]+=.001;negatives.append(expect_reject(lambda:prefix_teacher(p,rows[1],bad,baseline),'copied_or_modified_teacher_hidden'))
 bad=dict(arrays[1],rewards=arrays[1]['rewards'].copy());bad['rewards'][0]+=.00001;negatives.append(expect_reject(lambda:policy_tape(p,rows[1],bad),'modified_reward'))
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 s=read(OLD/'summary.json');assert s['physics']==sum(r['physics'] for r in rows+reps)==109330 and s['commands']==len(rows)+len(reps)+4==7;assert details[0]['field_qualified_success'] and int(arrays[0]['teacher_holds'])==-1;assert int(arrays[1]['teacher_holds'])==20;autonomous=len(arrays[1]['states'])-p['teacher_first']-20;assert autonomous>0;assert all(sha(f)==h for f,h in cfg['inputs'].items());assert not torch.cuda.is_initialized()
 result=dict(original_producer_state='FAILED_PRESERVED',read_only_evidence_recovery='PASS',teacher_qualified=details[0]['field_qualified_success'],full_teacher_detail=details[0],full_teacher_native_exact_replay_verified=True,handoff_detail=details[1],handoff_autonomous_decisions=autonomous,handoff_safe_goal=bool(details[1]['field_qualified_success']),handoff_native_exact_replay_executed=False,handoff_missing_native_repeat=1,learning_eligible=False,new_student_model=None,new_student_full_task_autonomy=None,new_native=0,new_optimizer=0,new_RL=0,original_physics=s['physics'],original_commands=s['commands'],not_two_successes_or_independent_samples=True)
 save(OUT/'derived_findings.json',result);save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),scope='read-onlyfullrawstudentteacherhistory/intervalownership/density/reward;notmissingphysicalreplay',original_real_failure_reproduced=original_red,all_original_files_unchanged=True,negative_regressions=negatives,new_native=0,new_optimizer=0,new_RL=0));save(OUT/'summary.json',dict(state='COMPLETE_READ_ONLY_EVIDENCE_RECOVERY',elapsed_s=time.time()-start,physics=0,native_commands=0,result=result));files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='local_manifest.json'];save(OUT/'local_manifest.json',dict(files=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in files]));assert R.used(OUT)<=cfg['max_bytes'];print('T574R_READ_ONLY_FULL_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
