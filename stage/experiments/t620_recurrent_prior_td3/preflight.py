"""True-history gradient and record-path contracts, zero Adam/native steps."""
from pathlib import Path
import sys,time,copy,io,ast,unittest.mock as mock
import t620_learning as L
import t620_runtime as R
ROOT,HERE,DOC=R.ROOT,R.HERE,R.DOC
torch,np=L.torch,L.np

def main():
 root=L.read(HERE/'protocol.json');p=L.read(root['actor_configuration']);p.update(root);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 # Read-only initialization, without the run admission or simulator Runtime.
 R.setup(ROOT/'stage/docs/day_20261117/t620');online=L.read(ROOT/'stage/runs/t608_current_policy_memory/rl_01/summary.json')['result']['training_rows'][:3];data=L.dataset(p,online);obj=L.make(p);clone=L.make(p);before=L.digest(L.state(obj));assert before==L.digest(L.state(clone));assert obj['actor'].phash()==p['student_initial']['policy_hash']
 rng=np.random.default_rng(p['learner_seed']);gen=torch.Generator().manual_seed(p['learner_seed']+1);pack=L.batch(data,p,rng);reference=L.frozen(p,p['student_initial']['path']);start=time.monotonic();loss,detail=L.losses(obj,reference,pack,p,gen);loss.backward();critic_nonzero=sum(int(torch.count_nonzero(v.grad)) for v in obj['critics'].parameters() if v.grad is not None);assert critic_nonzero>0 and bool(torch.isfinite(loss));critic_seconds=time.monotonic()-start
 assert all(v.grad is None for v in L.variables(obj['actor']))
 for q in obj['critics']:q.requires_grad_(False)
 start=time.monotonic();actor,ad=L.losses(obj,reference,pack,p,gen,True);actor.backward();an=float(torch.linalg.vector_norm(torch.cat([v.grad.flatten() for v in L.variables(obj['actor']) if v.grad is not None])));assert an>0 and math_finite(an);actor_seconds=time.monotonic()-start
 for q in obj['critics']:q.requires_grad_(True)
 assert L.digest(L.state(obj))==before and all(v.grad is None for name,v in obj['actor'].named_parameters() if not name.startswith('branch.'))
 # Each critic's independent recurrent sequence agrees with its actual stream.
 x=pack[1][:48,:1]
 errors=[]
 with torch.no_grad():
  for critic in obj['critics']:
   batch=critic.features(x);h=None;stream=[]
   for xx in x:
    z,h=critic.core(critic.encoder(((xx-critic.mean)/critic.std).clamp(-10,10))[None],h);stream.append(z[0])
   err=float((batch-torch.stack(stream)).abs().max());assert err<1e-12;errors.append(err)
 # Store-size bound uses synthetic moment slots only, never optimizer.step.
 bound=copy.deepcopy(L.state(obj))
 for key,params in [('actor_opt',L.variables(obj['actor'])),('critic_opt',list(obj['critics'].parameters()))]:
  for i,v in zip(bound[key]['param_groups'][0]['params'],params):bound[key]['state'][i]=dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v))
 buf=io.BytesIO();torch.save(dict(state=bound,initial=p['student_initial'],state_sha='0'*64,fixed_body_sha='0'*64),buf);joint_upper=int(len(buf.getvalue())*1.001)+32768;assert joint_upper<=p['joint_bytes']
 # Exercise actual Runtime.request result persistence and envelope with a
 # saved native payload whose request is absent at receive time.
 base=R.R.Base;row=L.read(ROOT/'stage/runs/t615_original_start_drift_pair/summary.json')['rows'][0];a=R.R.E.N.S.arrays(row['path']);rt=object.__new__(base.Runtime);rt.p=dict(p,max_physics=p['max_physics'],max_commands=100);rt.counts=[0,0,0];rt.commands=2;rt.rows=[];rt.replays=[];rt.guard=lambda learning=False:None
 class VirtualPool:
  def request(self,reqs):
   assert len(reqs)==1 and reqs[0]['model']==p['student_initial']['path'];return [(a,copy.deepcopy(row))]
 rt.pool=VirtualPool();saves=[];appended=[]
 virtual=Path('/virtual/stage/t620/direct_00')
 with mock.patch.object(base,'ACTIVE',virtual),mock.patch.object(base,'save',side_effect=lambda f,j:saves.append((str(f),j))),mock.patch.object(base,'append',side_effect=lambda n,j:appended.append((n,j))),mock.patch.object(base,'sha',return_value=row['sha256']),mock.patch.object(Path,'exists',return_value=True):
  out=rt.request([row['request']['job']],p['student_initial'],'RECORD_PATH_FIXTURE',full=True)
 assert len(saves)==len(appended)==1 and all(path.startswith(str(virtual)) for path,j in saves);assert out[0][1]['request']['model']==p['student_initial']['path']
 R.N.inspect(p,out)
 # Clipped action-space exploration, not latent-noise magnitude alone.
 with torch.no_grad():mu=obj['actor'].sequence(data[1][0]['x'][:-1]).numpy()
 rg=np.random.default_rng(p['learner_seed']);sample=np.clip(mu[None]+rg.normal(size=(128,*mu.shape))*np.asarray(p['raw_sigma']),-1,1);sd=np.std(sample,axis=0);std=np.median(sd,axis=0);assert bool((std>.005).all())
 # Both unchanged and guide-prefix modes retain their own actual Actor tape.
 R.N.inspect(p,[(a,row)])
 for rec in p['recovery_rows']:
  arrays=R.R.E.N.S.arrays(rec['path']);R.N.inspect(p,[(arrays,rec)])
 guide=p['training_jobs_by_arm']['rollin_00'][0];source=R.R.E.N.S.arrays(guide['guide_source']);k=guide['guide_first'];guide_checks=[]
 for decision in [k-1,k]:
  q=L.frozen(p,p['student_initial']['path']);q.sample=False;q.reset_noise();h=None
  with torch.no_grad():
   for xx in source['states'][:decision]:_,_,h,_=q.step(torch.tensor(xx[None]),h)
  session=object.__new__(R.N.Session);session.rows=[None]*decision;session.q=q;session.hidden=h;session.guide_first=k;session.source=source;session.packet=lambda:source['states'][decision].copy();observed=[]
  def no_native(s,forced=None,owner=0,actor_active=True):
   with torch.no_grad():_,_,s.hidden,_=s.q.step(torch.tensor(s.packet()[None]),s.hidden)
   assert owner==int(decision<k)
   if decision<k:assert np.array_equal(forced,source['actions'][decision])
   else:assert forced is None
   observed.append(True)
  with mock.patch.object(R.N.F.FieldSession,'step',new=no_native):R.N.Session.step(session)
  assert len(observed)==1 and len(q.tape)==decision+1;guide_checks.append(decision)
 # Confirm exact native return shape at the repaired Pool.receive boundary.
 test=object.__new__(R.N.Pool);test.p=dict(p,out='/virtual/stage/t620');test.serial=1;test.pending=[dict(model=p['student_initial']['path'],job=guide)];captured=[];bare={key:value for key,value in row.items() if key!='request'}
 with mock.patch.object(R.N.Previous.Pool,'receive',return_value=(a,bare)),mock.patch.object(R.N.L,'save',side_effect=lambda f,j:captured.append(j)):
  _,annotated=R.N.Pool.receive(test,0)
 assert len(captured)==1 and 'request' not in annotated and annotated['guide_first']==k
 # Packed file bounds for normal fresh episodes and the longest partial.
 schema=R.R.E.N.S.arrays(row['path']);n=len(schema['states']);storage=[]
 for count,cap in [(644,p['fresh_file_bytes']),(840,p['full_bytes'])]:
  big={key:np.zeros((count+(len(v)-n),*v.shape[1:]),dtype=v.dtype) if v.ndim and len(v) in [n,n+1] else v for key,v in schema.items()};big['recorded_guide_first']=np.asarray(k,np.int64);buf=io.BytesIO();np.savez(buf,**R.R.E.N.S.packed(big));upper=int(len(buf.getvalue())*1.001)+32768;assert upper<cap;storage.append(dict(decisions=count,safe_upper=upper,cap=cap))
 assert not torch.cuda.is_initialized();report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',prior_episodes=len(data[0]),three_historical_fixtures_not_new_online=True,Actor_initial_function_exact=True,critic_nonzero_gradient_entries=critic_nonzero,actor_gradient_norm=an,critic_gradient_seconds=critic_seconds,actor_gradient_seconds=actor_seconds,critic_stream_sequence_max_errors=errors,critic_loss=detail,actor_loss=ad,all_model_and_optimizer_parameters_unchanged=before==L.digest(L.state(obj)),joint_storage_safe_upper=joint_upper,joint_cap=p['joint_bytes'],synthetic_moment_slots_storage_only_not_actual_Adam=True,actual_Runtime_request_record_and_envelope_path_checked=True,record_path_does_not_use_root_output=True,existing_three_real_prefix_recoveries_audited=True,actual_guide_Session_step_boundary_checked=guide_checks,native_return_before_request_annotation_checked=True,packed_storage_bounds=storage,median_executed_command_std=std.tolist(),scalar_noise_not_control_success=True,new_native=0,new_optimizer=0,epoch=time.time());L.save(DOC/'PREFLIGHT.json',report);print('T620_REAL_HISTORY_OFFPOLICY_PREFLIGHT_PASS',report)

def math_finite(x):return bool(np.isfinite(x))
if __name__=='__main__':main()
