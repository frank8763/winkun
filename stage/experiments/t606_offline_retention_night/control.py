"""Zero-native preparation, explicit authorization, and bounded optimizer checks."""
from pathlib import Path
import sys,copy,time,datetime,math,io,ast,fcntl,json
import t606_runtime as R
L,ROOT,HERE,NIGHT,DOC=R.L,R.ROOT,R.HERE,R.NIGHT,R.DOC
read,save,sha,digest,torch,np=L.read,L.save,L.sha,L.digest,L.torch,L.np

def schedule_probe(cfg):
 class Dispatched(Exception):pass
 reports=[]
 bc_bound=2*sum(r['request']['job']['case']['step_cap'] for r in cfg['recovery_rows'])+sum(j['case']['step_cap'] for j in cfg['evaluation_jobs'])+sum(j['case']['step_cap'] for j in cfg['evaluation_jobs'] if j['id'] in cfg['key_replay_ids'])
 rl_bound=bc_bound+8*max(j['case']['step_cap'] for j in cfg['evaluation_jobs'])
 for arm in ['raw','command','rl']:
  for level in range(4):
   pc=dict(cfg,max_commands=71 if arm=='rl' else 63,max_physics=rl_bound if arm=='rl' else bc_bound)
   rt=R.Base.Runtime(pc);rt.last=time.time()+1000
   class Virtual:
    def request(self,requests):rt.counts[0]+=sum(x['job']['case']['step_cap'] for x in requests);raise Dispatched()
   rt.pool=Virtual();calls=0
   stages=[]
   if arm=='rl':stages=[([max(cfg['evaluation_jobs'],key=lambda j:j['case']['step_cap'])]*4,False),([max(cfg['evaluation_jobs'],key=lambda j:j['case']['step_cap'])]*4,True)]
   if level:
    rec=[r['request']['job'] for r in cfg['recovery_rows']];stages += [(rec,False),(rec,True)]
   if level>=2:stages += [([j for j in cfg['evaluation_jobs'] if j['id'] in cfg['sentinel_ids']],False)]
   if level>=3:stages += [([j for j in cfg['evaluation_jobs'] if j['id'] not in cfg['sentinel_ids']],False),([j for j in cfg['evaluation_jobs'] if j['id'] in cfg['key_replay_ids']],True)]
   try:
    for jobs,replay in stages:
     for k in range(0,len(jobs),2):
      batch=jobs[k:k+2]
      try:rt.request(batch,cfg['student_initial'],'VIRTUAL_NO_NATIVE',reference=[dict(path='unused') for _ in batch] if replay else None)
      except Dispatched:calls+=len(batch)
    commands=calls+4 if calls else 0
    assert commands<=pc['max_commands'];reports.append(dict(arm=arm,level=level,commands=commands,virtual_reserved_physics=sum(rt.counts)))
   finally:fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert max(x['commands'] for x in reports if x['arm']=='raw')==63 and max(x['commands'] for x in reports if x['arm']=='rl')==71
 assert max(x['virtual_reserved_physics'] for x in reports if x['arm']=='rl')==rl_bound
 assert max(x['virtual_reserved_physics'] for x in reports if x['arm']=='raw')==bc_bound
 return dict(state='PASS_ACTUAL_REQUEST_VIRTUAL_DISPATCH_ZERO_NATIVE',reports=reports,worker_creation=0,full_conditions_count=len(reports),no_pool_no_native_command_count=0)

def prepare():
 assert not NIGHT.exists() and not (DOC/'PROPOSAL.json').exists()
 cfg=read(ROOT/'stage/runs/t604_reverse_initialization/protocol.json');old=read(ROOT/'stage/experiments/t577_offline_recovery_night/protocol.json')
 epoch=lambda s:datetime.datetime.fromisoformat(s).timestamp()
 common=dict(main_seconds=1200,learning_seconds=600,audit_seconds=300,storage_reserve=4*2**20,max_physics=2879616,max_commands=63)
 root=dict(id='T606_OFFLINE_RETENTION_NIGHT',out=str(NIGHT),actor_configuration=str(ROOT/'stage/runs/t604_reverse_initialization/protocol.json'),
  base_model=cfg['base_model'],student_initial=cfg['student_initial'],teacher_initial_success_ids=cfg['teacher_initial_success_ids'],repair_ids=cfg['repair_ids'],
  arms=['raw','command','rl'],rounds_per_arm=6,fit_steps=32,fit_lr=3e-4,rl_steps=8,rl_lr=1e-5,target_kl=.01,protected_max_change=.002,
  step_factors=[1.,.25,.0625,.015625,.00390625],raw_sigma=cfg['raw_sigma'],max_bytes=60*2**20,model_bytes=384*1024,full_bytes=2*2**20,compact_bytes=12*1024,
  max_physics=120000000,max_commands=4000,max_main_seconds=36000,absolute_learning_stop=epoch('2026-11-17T07:00:00+08:00'),absolute_native_stop=epoch('2026-11-17T07:45:00+08:00'),absolute_all_stop=epoch('2026-11-17T08:30:00+08:00'),
  transport_seconds=720,source_seconds=120,finalization_seconds=900,final_storage_reserve=10*2**20,root_tail_bytes=2*2**20,stress_chunks=256,
  round_template=common,rl_template=dict(max_physics=cfg['max_physics']+8*max(j['case']['step_cap'] for j in cfg['evaluation_jobs']),max_commands=71,storage_reserve=12*2**20,main_seconds=1800,learning_seconds=900),
  final_template=dict(main_seconds=2400,learning_seconds=0,audit_seconds=300,storage_reserve=10*2**20,max_physics=sum(j['case']['step_cap'] for j in old['final_template']['final_jobs']),max_commands=411,final_jobs=old['final_template']['final_jobs']),
  stress_template=dict(main_seconds=900,learning_seconds=0,audit_seconds=120,storage_reserve=512*1024),authorization_approved=False,
  inputs=dict(cfg['inputs']),protected_files=dict(cfg['protected_files']),no_new_runtime_teacher=True,new_Critic=0,live_Adam_cap=432,shadow_Adam_cap=432,audit_historical_Adam_cap=432,preflight_discarded_Adam_cap=12)
 probe=copy.deepcopy(cfg['evaluation_jobs'][0]);probe['case']['step_cap']=5000
 root['engineering_template']=dict(main_seconds=300,learning_seconds=0,audit_seconds=180,storage_reserve=2*2**20,max_physics=40000,max_commands=12,full_bytes=256*1024,probe_job=probe,training_seeds=[[60620261116,7,0,k] for k in range(4)])
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);pc=dict(cfg,**root,arm='command',resume_optimizer=False);data=L.dataset(pc);q=L.make(pc)[0];teacher=L.frozen(pc,root['student_initial']['path']);initial=q.phash()
 with torch.no_grad():
  for g in data:assert torch.equal(q.sequence(g['x']),teacher.sequence(g['x']))
  packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);raw,command=L.protection(q,data,packed);assert raw<1e-12 and command<1e-12
 for mode in ['raw','command']:
  q.zero_grad(set_to_none=True);loss=L.risks(q,data,packed,mode);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in L.variables(q))
 q.zero_grad(set_to_none=True);assert q.phash()==initial
 # Exact array schema extrapolation: decision arrays and n+1 histories.
 a=L.G.E.N.S.arrays(ROOT/'stage/runs/t604_reverse_initialization/episodes/full_0.npz');n=len(a['states']);cap=max(j['case']['step_cap'] for j in cfg['evaluation_jobs']);decisions=math.ceil(cap/100)+1;b={}
 for k,v in a.items():
  shape=list(v.shape)
  if v.ndim and len(v) in [n,n+1]:shape[0]=decisions+(len(v)==n+1)
  b[k]=np.zeros(shape,dtype=v.dtype)
 buf=io.BytesIO();np.savez(buf,**b);uncompressed=len(buf.getvalue());deflate_upper=uncompressed+uncompressed//1000+32768
 assert deflate_upper<root['full_bytes']
 schedule=schedule_probe(pc);save(DOC/'SCHEDULE_PREFLIGHT.json',schedule)
 pref=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all42_initial_histories_exact_T598=True,three_recovery_initial_raw_max=raw,three_recovery_initial_command_max=command,two_objective_gradients_finite=True,
  live_Adam_cap=432,shadow_Adam_cap=432,audit_historical_Adam_cap=432,preflight_discarded_Adam_pending=12,full_schema_decisions=decisions,uncompressed_npz_upper=uncompressed,deflate_safe_upper=deflate_upper,full_file_cap=root['full_bytes'],conditional_storage_not_all_rounds_guaranteed=True,global_worst=R.used(ROOT/'stage/runs')+root['max_bytes']+131072)
 assert pref['global_worst']<2**30
 for f in HERE.glob('*.py'):ast.parse(f.read_text());root['inputs'][str(f)]=sha(f)
 for f in [DOC/'PLAN.md',ROOT/'stage/runs/t604_reverse_initialization/protocol.json',ROOT/'stage/experiments/t577_offline_recovery_night/protocol.json',ROOT/'stage/runs/t605_night_cold_metadata/audit.json']:
  root['inputs'][str(f)]=sha(f)
 root['inputs'][root['student_initial']['path']]=root['student_initial']['sha256']
 save(DOC/'PREFLIGHT.json',pref);root['inputs'][str(DOC/'PREFLIGHT.json')]=sha(DOC/'PREFLIGHT.json');root['inputs'][str(DOC/'SCHEDULE_PREFLIGHT.json')]=sha(DOC/'SCHEDULE_PREFLIGHT.json')
 save(DOC/'PROPOSAL.json',root);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',**{k:root[k] for k in ['max_physics','max_commands','max_main_seconds','max_bytes','live_Adam_cap','shadow_Adam_cap','audit_historical_Adam_cap','preflight_discarded_Adam_cap','absolute_learning_stop','absolute_native_stop','absolute_all_stop']},global_worst=pref['global_worst'],native_engineering_included=True,all_rounds_conditional_full_worst_admission=True))
 print('T606_PREPARED_NO_NATIVE_NO_OPTIMIZER',pref['global_worst'],deflate_upper,flush=True)

def fixture(p):
 source=read(ROOT/'stage/runs/t596_recurrent_joint_online/summary.json')['result']['training_rows'][:4];q=L.make(p)[0];arrays=[]
 for k,row in enumerate(source):
  a={s:v.copy() for s,v in L.G.E.N.S.arrays(row['path']).items()};x=torch.tensor(a['states'])
  with torch.no_grad():loc=q.sequence(x);innovation=np.random.default_rng(6061700+k).normal(size=loc.shape);raw=loc+loc.new_tensor(innovation)*loc.new_tensor(p['raw_sigma']);atom=torch.where(raw>=L.P.L.EDGE,1,torch.where(raw<=-L.P.L.EDGE,-1,0)).to(torch.int8);logp=L.P.L.density(loc,raw,atom,p['raw_sigma'])
  a.update(locations=loc.numpy(),raw_actions=raw.numpy(),atom=atom.numpy(),logp=logp.numpy(),stochastic=np.ones(len(x),np.uint8),ownership=np.zeros(len(x),np.uint8));arrays.append(a)
 return arrays

def optimizer_preflight():
 assert not NIGHT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists();root=read(DOC/'PROPOSAL.json');assert read(DOC/'AUTHORIZATION.json')['approved'];torch.set_num_threads(1);torch.use_deterministic_algorithms(True);cfg=read(root['actor_configuration']);reports=[]
 for arm in root['arms']:
  p=dict(cfg,**root,arm=arm,resume_optimizer=False);data=L.dataset(p);arrays=fixture(p) if arm=='rl' else None;a,b=L.make(p),L.make(p);before=L.state(a)
  u=L.fit(a,data,p,lambda:None,lambda *_:None,arrays,steps=2);v=L.fit(b,data,p,lambda:None,lambda *_:None,arrays,steps=2);assert u==v and digest(L.state(a))==digest(L.state(b));mature=L.state(a);L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
  file=DOC/('discarded_'+arm+'.pt.gz');m=L.export(a,p,file);q=L.frozen(p,file);assert q.phash()==a[0].phash();q.sample=False;q.reset_noise();h=None;x=data[-1]['x'][:64];out=[]
  with torch.no_grad():
   z=q.sequence(x)
   for xx in x:_,ll,h,_=q.step(xx[None],h);out.append(ll[0])
  assert float((z-torch.stack(out)).abs().max())<1e-12;file.unlink();reports.append(dict(arm=arm,actual_discarded_Adam=2*u['attempted_Adam'],applied_steps_per_clone=u['applied_parameter_steps'],updated_sequence_step_exact=True,model_bytes=m['bytes'],synthetic_density_fixture_only=arm=='rl'))
 save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',reports=reports,discarded_Adam=sum(r['actual_discarded_Adam'] for r in reports),maximum12=True,whole_ActorAdam_shadow_mature_rollback_readback=True,synthetic_fixture_not_onpolicy_physical_data=True))
 root['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',root)

def register():
 assert not NIGHT.exists();root=read(DOC/'PROPOSAL.json');a=read(DOC/'AUTHORIZATION.json');assert a['approved'] and a['max_physics']==root['max_physics'] and a['max_commands']==root['max_commands'] and a['max_bytes']==root['max_bytes'] and a['all_stop_epoch']==root['absolute_all_stop']
 assert time.time()+1800<root['absolute_learning_stop'] and time.time()+2400<root['absolute_all_stop'];assert R.used(ROOT/'stage/runs')+root['max_bytes']+131072<2**30
 for f,h in root['inputs'].items():assert sha(f)==h,f
 root['authorization_approved']=True;root['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');NIGHT.mkdir();blob=json.dumps(root,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(NIGHT/'protocol.json').write_text(blob)
 print('T606_REGISTERED_NOT_LAUNCHED',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
