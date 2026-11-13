"""Register the entire finite offline relay; exercise actual seams before launch."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t542_runtime as R
ROOT,OUT,DOC,L,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();DOC.mkdir(exist_ok=True)
 end=datetime(2026,11,14,8,30,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
 assert time.time()+4*3600<end
 basefile=ROOT/'stage/runs/t531_joint_success_distillation/protocol.json';base=read(basefile);initial=base['neural_teacher'];candidate=read(ROOT/'stage/runs/t539_terminal_success_pg/summary.json')['result']['model']
 assert sha(initial['path'])==initial['sha256'] and sha(candidate['path'])==candidate['sha256']
 logs=[ROOT/'stage/runs/t507_joint_success_transfer/pilot/cases.jsonl',ROOT/'stage/runs/t508_frozen_completion/pilot/cases.jsonl'];lookup={r['job_id']:r for f in logs for r in R.G.rows(f.parent)};rows=[lookup[j['id']] for j in base['old8_jobs']+base['field_jobs']];assert len(rows)==32 and all(r['policy_hash']==initial['policy_hash'] for r in rows)
 cfg=dict(base,evaluation_jobs=base['old8_jobs']+base['field_jobs']);adapted=[]
 for row in rows:
  a=R.E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256'];adapted.append(R.envelope(row['request']['job'],row,a,base['field_limit_m']))
 before=R.assess(cfg,adapted);assert before['old8']['success']==8 and len(before['field_success_ids'])==11
 p=dict(id='T542_OFFLINE_FRESH_POLICY_FIELD_RELAY',actor_configuration=str(basefile),source_protocol=base['source_protocol'],source_initial=initial,initial=initial,frozen_candidate=candidate,initial_check=before,initial_rows_files=[str(f) for f in logs],
  arms=['terminal','rtg','readout'],rounds_per_arm=24,round_robin=True,seed=20261113542,inner_steps=4,learning_rate=.001,replay_weight=1.,target_kl=.01,
  raw_sigma=base['raw_sigma'],max_bytes=108*2**20,child_bytes=20*2**20,full_bytes=2*2**20,model_bytes=640*1024,compact_bytes=12*1024,
  max_physics=100_000_000,max_commands=4000,max_main_seconds=40000,max_rounds=72,main_seconds=2400,learning_seconds=1800,audit_seconds=300,manifest_seconds=60,
  absolute_learning_stop=end-90*60,absolute_native_stop=end-60*60,absolute_all_stop=end,stage_cpus=[14,15],GPU=False,
  scope='Frozen T539 complete32+32exact first; three fixed current-policy fresh-data mechanisms,24rounds each round-robin upper bound,4training+4nativeexact per round,4clippedSGD+4shadow,all32+32exact ifnonzero signal. All prior successIDs retained; count gain or >=.05m total remaining reduction at equal count may retain research progress. Not a safety proof or original algorithm. A terminalLOO homogeneous batch creates0PG, no synthetic real training. B original physical-time RTG; C identicalB with only11906readout params trainable. All starts original fixedfield resets, no teacher/map/waypoints/router.',
  authorization='User explicitly asked full overnight offline background relay,confirmed2026-11-14 08:30all. Finite new100M/4000/40000main and60MiBown; no ended-directory restart or continuation ofT539/T540. No Phase inspection,aggregateoneCPU/GPUoff.',
  offline=True,source_seconds=180,transport_seconds=900,final_reserve_seconds=4500,shared_contract_failure_stops_chain=True,
  target_priority=['t502_field_sector1_offset1','t502_field_sector0_offset2'])
 p['regression_configuration']=str(ROOT/'stage/runs/t501_joint_command_repair/protocol.json')
 p['protected_files']={initial['path']:initial['sha256'],candidate['path']:candidate['sha256'],str(basefile):sha(basefile)}
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for f in [ROOT/'stage/experiments/common/native_case_record.py',DOC/'PLAN.md',DOC/'LITERATURE.md',DOC/'STORAGE_LEDGER.json',DOC/'REMOTE_COLD_LEDGER.json',*[Path(f) for f in p['initial_rows_files']],Path(p['regression_configuration']),ROOT/'stage/runs/t539_terminal_success_pg/protocol.json']:
  p['inputs'][str(f)]=sha(f)
 # Capture every already imported Stage scientific dependency, not unrelated work.
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for row in base['keep_rows']:p['inputs'][row['path']]=row['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<=2**30
 OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 # Exactly the old8 plus24field capped worst-case, fresh four use max field cap.
 full32=sum(j['case']['step_cap'] for j in cfg['evaluation_jobs']);roundcap=2*full32+8*max(j['case']['step_cap'] for j in base['field_jobs'])
 save(DOC/'BUDGET.json',dict(all_stop=end,all_stop_BJT='2026-11-14 08:30',learn_stop_BJT='2026-11-14 07:00',native_stop_BJT='2026-11-14 07:30',max_physics=100000000,max_commands=4000,max_main_seconds=40000,max_rounds=72,per_round_max_physics=roundcap,per_round_max_commands=76,per_round_bytes=20*2**20,root_bytes=108*2**20,global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],free_bytes=__import__('shutil').disk_usage(ROOT).free,stage_CPU_total_cores=1,stage_CPU_affinity=[14,15],GPU=False,night_native_probe_included=2000,at_admission_each_full_round_and_final_reserve_required=True,number_rounds_not_guaranteed=72))
 print('T542_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
 p=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists();R.verify(p,True);base=read(p['actor_configuration']);cfg=dict(base,**p);cfg['out']=str(OUT);cfg['arm']='terminal';cfg['kind']='round'
 source=R.E.N.S.arrays(base['keep_rows'][0]['path']);x=source['states'][:8];items=[]
 for i in range(4):
  q=L.make(cfg)[0].eval().requires_grad_(False);q.sample=True;q.rng=R.np.random.default_rng([542,999,i]);h=None
  with R.torch.no_grad():
   for z in x:_,_,h,_=q.step(R.torch.tensor(z[None]),h)
  tape=q.tape;flags=R.np.zeros((8,3),R.np.uint8);flags[-1,1 if i%2==0 else 2]=1
  items.append(dict(states=x.copy(),next_states=R.np.concatenate([x[1:],x[-1:]]),ownership=R.np.zeros(8,R.np.uint8),stochastic=R.np.ones(8,R.np.uint8),end_flags=flags,raw_actions=R.np.asarray([t['raw_actions'] for t in tape]),atom=R.np.asarray([t['atom'] for t in tape]),locations=R.np.asarray([t['locations'] for t in tape]),logp=R.np.asarray([t['logp'] for t in tape]),native_counts=R.np.full(8,100,R.np.uint8),rewards=R.np.arange(8,dtype=float)*(1 if i%2==0 else -1)))
 records=[]
 for arm in p['arms']:
  c=dict(cfg,arm=arm);obj=L.make(c);shadow=L.make(c);before=L.state(obj);a=L.fit(obj,items,c,lambda:None,lambda *_:None);b=L.fit(shadow,items,c,lambda:None,lambda *_:None)
  assert a==b and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,c,OUT/('discarded_'+arm+'.pt.gz'));assert L.A.frozen(c,model['path']).phash()==model['policy_hash'];Path(model['path']).unlink()
  if arm=='readout':assert all(R.torch.equal(v,before['actor'][k]) for k,v in obj[0].state_dict().items() if not k.startswith(('drive_adapter.','steer_adapter.')))
  for o in [obj,shadow]:L.load(o,before);assert digest(L.state(o))==digest(before)
  records.append(dict(arm=arm,discarded_synthetic_steps=2*a['steps'],trainable=a['trainable_parameters'],shadow_serialize_whole_state_rollback_pass=True))
 # Worst-shape compact streams, real native-record envelope regression.
 lookup={r['job_id']:r for f in p['initial_rows_files'] for r in R.G.rows(Path(f).parent)};rows=[lookup[j['id']] for j in base['old8_jobs']+base['field_jobs']]
 for row in rows:
  a=R.E.N.S.arrays(row['path']);raw={k:v for k,v in row.items() if k not in ['job_id','field_detail']};r=R.envelope(row['request']['job'],raw,a,base['field_limit_m']);assert r['job_id']==row['job_id']
 for j in base['old8_jobs']+base['field_jobs']:
  n=(j['case']['step_cap']+99)//100;a=dict(actions=R.np.random.default_rng(0).normal(size=(n,2)).astype('f4'),native_counts=R.np.full(n,100,'u1'),end_flags=R.np.zeros((n,3),'u1'));assert len(R.pack(a))<=p['compact_bytes']
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NEW_NATIVE',pid=os.getpid(),synthetic_clones=records,total_discarded_synthetic_SGD=sum(r['discarded_synthetic_steps'] for r in records),existing32_real_records_envelope_pass=True,all_output_routes_current=True,scientific_training_not_executed=True,child_native_contract_still_required=True));print('T542_ZERO_NATIVE_PREFLIGHT_PASS',flush=True)
def amend():
 assert not (OUT/'launch.json').exists();old=read(OUT/'protocol.json');oldsha=sha(OUT/'protocol.json');rev=OUT/'prelaunch_revision_001'
 assert sha(rev/'protocol.json')==oldsha and read(OUT/'native_preflight/audit.json')['state']=='PASS'
 assert read(DOC/'LARGE_COLD_LEDGER.json')['state']=='COMPLETE_REMOTE_FULL_BINARY_READBACK_THEN_CACHE_EVICTION'
 assert read(DOC/'LARGE_COLD_LEDGER_2.json')['state']=='COMPLETE_REMOTE_FULL_BINARY_READBACK_THEN_CACHE_EVICTION'
 p=copy.deepcopy(old);p.update(rounds_per_arm=24,max_rounds=72,max_bytes=108*2**20,child_bytes=20*2**20,full_bytes=2*2**20)
 p['scope']=p['scope'].replace('12rounds each','24rounds each')
 p['prelaunch_revision']=dict(original_root_sha256=oldsha,original_native_preflight_protocol_sha256=sha(rev/'native_preflight_protocol.json'),original_native_preflight_audit_sha256=sha(OUT/'native_preflight/audit.json'),original_zero_native_preflight_sha256=sha(OUT/'preflight.json'),new_native=0,new_optimizer=0,all_Actor_reward_density_learning_rate_inner_steps_seeds_targets_unmodified=True,changes=['finite24rounds_per_arm72total','108MiBown_within_existing1GiB','2MiBfull_shape_bound_and20MiBchild_reserve','final_field_boundary_qualification','more_complete_final_time_budget_reservation','systemd_shutdown5secondmargin','current_source_hashes_rebound_after_prelaunch_metadata_revision'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for f in [DOC/'PLAN.md',DOC/'LARGE_COLD_LEDGER.json',DOC/'LARGE_COLD_LEDGER_2.json']:p['inputs'][str(f)]=sha(f)
 assert R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+65536<=2**30
 save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
 cp=read(OUT/'native_preflight/protocol.json');assert cp.pop('_root_sha256')==oldsha;assert cp=={k:v for k,v in read(rev/'native_preflight_protocol.json').items() if k!='_root_sha256'}
 save(OUT/'native_preflight/protocol.json',dict(cp,_root_sha256=sha(OUT/'protocol.json')))
 b=read(DOC/'BUDGET.json');b.update(max_rounds=72,rounds_per_arm=24,per_round_bytes=20*2**20,full_file_max=2*2**20,root_bytes=p['max_bytes'],final_bytes=16*2**20,global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],global_limit=2**30)
 base=read(p['actor_configuration']);rows=[r for f in p['initial_rows_files'] for r in R.G.rows(Path(f).parent)];r=max(rows,key=lambda r:r['physics']);a=R.E.N.S.arrays(r['path']);packed=R.E.N.S.packed(a);n=len(a['states'])
 worst=sum(((900+(v.shape[0]==n+1))*int(v.nbytes/v.shape[0]) if v.ndim and v.shape[0] in [n,n+1] else v.nbytes) for v in packed.values())+65536
 assert worst<2*2**20;b['full_900_decision_uncompressed_shape_plus_headers_upper']=worst
 save(DOC/'BUDGET.json',b);save(DOC/'PRELAUNCH_REVISION.json',dict(state='METADATA_AND_BOUND_REVISED_NO_SCIENTIFIC_UPDATE',**p['prelaunch_revision'],new_root_sha256=sha(OUT/'protocol.json'),full900_bound_bytes=worst,native_preflight_rebound_only_root_pointer_not_scientific_parameters=True))
 R.verify(p,True);print('T542_PRELAUNCH_BOUND_REVISION_PASS',b['global_reserved'],worst,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
