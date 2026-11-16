"""Concrete new learning proposal; zero-update preparation before budget approval."""
from pathlib import Path
import sys,ast,copy,time,datetime,json
import t596_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np
def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
 old=ROOT/'stage/runs/t591_joint_representation_pair';p=copy.deepcopy(read(old/'protocol.json'));summary=read(old/'summary.json')
 assert read(old/'audit.json')['state']=='PASS' and read(ROOT/'stage/runs/t594_current_goal_recovery/audit.json')['state']=='PASS'
 assert read(ROOT/'stage/runs/t595_lossless_cold_logs/audit.json')['state']=='PASS'
 rec=[r for r in R.G.rows(old/'trainable_representation') if r['role']=='TRAINED_STUDENT_HANDOFF_FULL'];assert len(rec)==3 and all(R.R.safe(r) for r in rec)
 jobs=[next(j for j in p['evaluation_jobs'] if j['id']==i) for i in ['t502_field_sector1_offset1','t502_field_sector0_offset3']]
 keys=['dev_task6_13','t502_field_sector1_offset1','t502_field_sector0_offset3','wide117_8m_dev6_LEFT_17']
 epoch=lambda x:datetime.datetime.fromisoformat('2026-11-16T'+x+':00+08:00').timestamp()
 p.update(id='T596_EXACT_T550_RECURRENT_JOINT_ONPOLICY_PILOT',out=str(OUT),initial=p['base_model'],recovery_rows=rec,training_jobs=[j for j in jobs for _ in range(4)],training_seeds=[[596,g,k] for g in range(2) for k in range(4)],
  protected_field15=[i for i in p['initial_success_ids'] if i.startswith('t502_')],key_replay_ids=keys,
  rl_lr=3e-6,rl_steps=16,target_kl=.01,keep_weight=10.,recovery_weight=.1,
  main_seconds=5400,learning_seconds=3600,audit_seconds=1800,transport_seconds=300,source_seconds=120,
  absolute_learning_stop=epoch('18:50'),absolute_native_stop=epoch('19:15'),absolute_all_stop=epoch('20:00'),
  max_commands=90,max_bytes=18*2**20,child_bytes=18*2**20,model_bytes=2*2**20,inputs={},protected_files={},authorization_approved=False,
  new_Actor_Adam_cap=16,shadow_Adam_cap=16,audit_discarded_Adam_cap=16,preflight_discarded_Adam_cap=4,new_RL=16,new_Critic=0,
  user_authorization='User explicitly requested next recurrent learning step and20:00allstop; new independent physical/command/optimizer allocation pending after concrete preparation.',
  scope='One fixed fresh8batch:4near-goal old-success original starts +4canonical failed offset3 original starts. Train exact T550 initialized encoder/GRU/head/adapters jointly; sparse basis support/weights and normalization frozen.16clipped PG joint auxiliary updates maximum,KL stop. No newCritic/teacher.3initialhandoff+3exact,8fullfresh+8exact,then49compact+4exact+3compacthandoff+3exact. All39initialIDs+15field+3handoff safe and at least1neworiginalfield ID required research retention. No seed/weight/grid/extra rounds/407/unseen/globalpromotion.')
 cap=8*sum(j['case']['step_cap'] for j in jobs)+sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in keys)+4*sum(j['case']['step_cap'] for j in p['handoff_jobs'])
 p['max_physics']=cap;upper=10*p['full_bytes']+55*p['compact_bytes']+p['model_bytes']+2*2**20;glob=R.R.used(ROOT/'stage/runs')+p['max_bytes']+131072
 assert upper<=p['max_bytes'] and glob<2**30 and time.time()+7650<p['absolute_all_stop'] and time.time()+5400<p['absolute_native_stop'] and time.time()+3600<p['absolute_learning_stop']
 R.setup(DOC);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=L.fresh(p);initial=digest(q.state_dict());assert q.phash()==p['base_model']['policy_hash'];data=L.references(p)
 checks=[]
 with torch.no_grad():
  for g in data[:39]:
   assert torch.equal(q.sequence(g['x']),g['target']);checks.append(dict(id=g['id'],decisions=len(g['x']),initial_function_exact=True))
  assert L.losses(q,data,p,False)['old']==0.
 x=data[0]['x'][:48];h=None;q.sample=False;q.reset_noise();steps=[]
 with torch.no_grad():
  seq=q.sequence(x).clamp(-1,1).float()
  for z in x:u,_,h,_=q.step(z[None],h);steps.append(u[0])
 assert torch.equal(torch.stack(steps),seq)
 q.requires_grad_(False)
 for v in L.variables(q):v.requires_grad_(True)
 q.zero_grad(set_to_none=True);start=time.monotonic();q.sequence(data[-1]['x']).square().mean().backward();duration=time.monotonic()-start
 groups={}
 for prefix in ['encoder.','core.','head.','drive_adapter.','steer_adapter.']:
  pars=[v for k,v in q.named_parameters() if k.startswith(prefix)];assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in pars)
  groups[prefix]=dict(parameters=sum(v.numel() for v in pars),nonzero_gradient_values=sum(int(torch.count_nonzero(v.grad)) for v in pars))
 assert digest(q.state_dict())==initial and all(g['nonzero_gradient_values']>0 for g in groups.values()) and not torch.cuda.is_initialized();q.zero_grad(set_to_none=True)
 # True current-policy numerical path checked against existing mathematically
 # identical zero-residual T582 initial rollouts; no new physical experiment.
 prior=R.G.rows(ROOT/'stage/runs/t582_fresh_multitask_pair')[:4];arrays=[R.E.N.S.arrays(r['path']) for r in prior]
 _,dens=L.Credit.peer_batch((q,None),arrays,p)
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_PREPARATION_ONLY',initial_policy_hash=q.phash(),old39_full_history_function_exact=checks,initial_old_loss=0.,real48step_sequence_actions_exact=True,full_history_gradient_seconds=duration,gradient_groups=groups,density_contract_existing_equal_function_data=dens,new_native=0,new_optimizer=0,own_upper=upper,global_worst=glob))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),old/'audit.json',ROOT/'stage/runs/t595_lossless_cold_logs/audit.json']+[Path(x['path']) for x in p['keep_rows']+p['recovery_rows']+prior]:p['inputs'][str(f)]=sha(f)
 for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 for j in p['handoff_jobs']:p['inputs'][j['source_path']]=sha(j['source_path'])
 model=p['base_model']
 while model:p['inputs'][model['path']]=model['sha256'];p['protected_files'][model['path']]=model['sha256'];model=model.get('parent')
 save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=cap,max_commands=90,actual_worst_commands=89,own_bytes=p['max_bytes'],own_upper=upper,global_worst=glob,main_seconds=5400,learning_seconds=3600,audit_seconds=1800,backup_seconds=300,source_seconds=120,total_admission_seconds=7650,learning_stop='18:50',native_stop='19:15',all_stop='20:00',live_Adam_cap=16,shadow_Adam_cap=16,audit_discarded_Adam_cap=16,preflight_discarded_Adam_cap=4,new_scientific_budget='PENDING'))
 print('T596_PREPARED_ZERO_NATIVE_ZERO_OPTIMIZER',cap,glob,groups,flush=True)
def synthetic(p,data):
 q=L.fresh(p);x=data[0]['x'][:16].numpy();out=[]
 for seed in range(4):
  q.sample=True;q.rng=np.random.default_rng(596+seed);q.reset_noise();h=None;loc=[];tape=[]
  with torch.no_grad():
   for z in x:u,b,h,_=q.step(torch.tensor(z[None]),h);loc.append(b[0].numpy().copy());tape.append(q.tape[-1])
  flags=np.zeros((len(x),3),np.uint8);flags[-1,2]=1
  out.append(dict(states=x.copy(),next_states=np.r_[x[1:],x[-1:]],locations=np.asarray(loc),raw_actions=np.asarray([v['raw_actions'] for v in tape]),atom=np.asarray([v['atom'] for v in tape]),logp=np.asarray([v['logp'] for v in tape]),ownership=np.zeros(len(x),np.uint8),stochastic=np.ones(len(x),np.uint8),end_flags=flags,native_counts=np.full(len(x),100,np.int64),rewards=np.linspace(-.001,.001,len(x))*(1+.2*seed)))
 return out+copy.deepcopy(out)
def optimizer_preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json');assert auth['approved'] and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands']
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.references(p);arrays=synthetic(p,data);a,b=L.make(p),L.make(p);before=L.state(a);u=L.fit(a,arrays,data,p,lambda:None,lambda *_:None,steps=2);v=L.fit(b,arrays,data,p,lambda:None,lambda *_:None,steps=2)
 assert u==v and digest(L.state(a))==digest(L.state(b));mature=L.state(a);L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
 file=DOC/'discarded_mature.pt.gz';m=L.export(a,p,file);assert L.frozen(p,file).phash()==a[0].phash();file.unlink();save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',discarded_Adam=2*u['Actor_Adam'],cap=4,synthetic_NOT_navigation=True,all_ActorAdam_shadow_mature_rollback_and_readback_exact=True,bytes=m['bytes']))
 p['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',p)
def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json');assert auth['approved'] and auth['all_stop_epoch']==p['absolute_all_stop'] and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands'] and auth['own_bytes']==p['max_bytes']
 assert read(DOC/'PREFLIGHT_OPTIMIZER.json')['state']=='PASS_AUTHORIZED_ZERO_NATIVE'
 assert time.time()+7650<p['absolute_all_stop'] and time.time()+5400<p['absolute_native_stop'] and time.time()+3600<p['absolute_learning_stop'] and R.R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 p.update(authorization_approved=True,user_authorization=auth['user_reply']);p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);print('T596_REGISTERED_NOT_LAUNCHED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
