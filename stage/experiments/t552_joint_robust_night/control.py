"""Register fixed three-arm night; no simulation in prepare or synthetic tests."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t552_runtime as R
sys.path.insert(0,str(R.ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning
ROOT,OUT,DOC,L,np,torch,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.np,R.torch,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();prior=ROOT/'stage/runs/t550_offline_obstacle_relay';s=read(prior/'protocol.json');r=read(ROOT/'stage/docs/day_20261114/T550_FINAL_REPORT.json');probe=ROOT/'stage/runs/t551_retention_recovery_probe';assert read(probe/'audit.json')['state']=='PASS';assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T551.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';R.G.stopped(probe);assert read(DOC/'COLD_CACHE.json')['state']=='COMPLETE_REMOTE_AND_DUPLICATE_CACHE_EVICTION'
 rows=R.G.rows(probe);restore=rows[:6];neighbor=rows[24:28];pairs=[dict(baseline=rows[22+i//2],neighbor=rows[24+i]) for i in range(4)];keep=s['protected_rows']+[R.G.rows(prior/'attempt_00_gap')[1],R.G.rows(prior/'attempt_01_gap')[1]];assert len(keep)==23 and all(x['success'] and not x['collision'] for x in restore+keep+neighbor)
 ids=read(probe/'protocol.json')['lost6_ids'];by={j['id']:j for j in s['final_jobs']};evaluation=s['old8_jobs']+[by[i] for i in ids]+s['field_jobs'];rr={r['job_id']:r for r in R.G.rows(prior/'final407')};before=R.assess(dict(evaluation_jobs=evaluation),[rr[j['id']] for j in evaluation]);assert len(before['success_ids'])==23 and before['old8']['success']==8 and before['restore6']['success']==0
 auth=read(DOC/'AUTHORIZATION.json');end=auth['all_stop'];dt=lambda h,m:datetime(2026,11,15,h,m,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
 p=dict(id='T552_JOINT_RETENTION_REAL_RL_LOCAL_RESPONSE_FIXED_ARMS',actor_configuration=s['actor_configuration'],source_protocol=s['source_protocol'],frozen_base=r['retained_model'],restore_rows=restore,keep_rows=keep,neighbor_rows=neighbor,pair_rows=pairs,old8_jobs=s['old8_jobs'],field_jobs=s['field_jobs'],evaluation_jobs=evaluation,final_jobs=s['final_jobs'],before_check=before,arms=['supervised','rl','response'],max_rounds_per_arm=4,training_jobs=[by[i] for i in [ids[0],ids[3],'t502_field_sector1_offset2','t502_field_sector4_offset2']],raw_sigma=[.2,.2],gamma_native=.99998,readout_seed=552,readout_amplitude=.2,bc_steps=32,rl_steps=4,bc_lr=.00003,rl_lr=.00003,target_kl=.01,pair_weight=1.,model_bytes=256*1024,full_bytes=2**20,compact_bytes=12288,max_bytes=auth['own_max_bytes'],child_bytes=7*2**20,max_physics=auth['additional_physics'],max_commands=auth['additional_commands'],max_main_seconds=auth['additional_main_seconds'],main_seconds=1800,learning_seconds=1200,audit_seconds=600,final_reserve_seconds=7200,final_bytes=8*2**20,child_reserve_bytes=6*2**20,transport_seconds=900,source_seconds=180,absolute_learning_stop=dt(6,30),absolute_native_stop=dt(7,15),absolute_all_stop=end,arm_rejection_stop=2,scope='Joint3families:6restoreexpertoldsuccess+23immutableT550successlocation+4actualperturbationneighborlocation. Pulseownedactionsneverteacherlabels; causallocationsuseunforcedActor. A32BC+4matchedBCcontrol,B32BC+4freshclippedPPO,Cplusfinitepairedresponse. Fixed5986readout,frozencausalGRUcarryandT550body. Same4jobs4seedsperroundperarm;physicalrepeatnotindependentseedreplicate. Max4round/arm,2rejectionsstoparm;neverrestartendeddir. Eachcandidate38+38exact:retainallprevioussuccessIDsandstrictlyaddsafeIDwithfieldcontactsnonincrease. All275onlyfinal407,notglobalguarantee. NoCritic,map,planner,waypoint,runtimeteacher,router,taskID;notcrosssiteororiginalalgorithm. Completefuturebranchandfinalworstbudgetreservedatadmission. Storageadmissionmayyieldfewerthan12rounds;noinfinitefill.')
 p['gamma_native']=float(R.E.N.K.R.GAMMA);assert abs(p['gamma_native']-.99998)<.001
 for f in [DOC/'PLAN.md',DOC/'SOURCES.md']:assert f.exists(),f
 p['protected_files']={}
 def lock_model(m):
  p['protected_files'][m['path']]=m['sha256']
  if 'parent' in m:lock_model(m['parent'])
 lock_model(p['frozen_base']);p['inputs']=dict(p['protected_files'])
 for row in restore+keep+neighbor+[rows[22],rows[23]]:p['inputs'][row['path']]=row['sha256']
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for f in [DOC/'AUTHORIZATION.json',DOC/'PLAN.md',DOC/'SOURCES.md',DOC/'COLD_CACHE.json',s['actor_configuration'],prior/'protocol.json',probe/'protocol.json',probe/'audit.json',probe/'cases.jsonl',ROOT/'stage/docs/long_distance_20261101/release_receipt_T550.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T551.json']:p['inputs'][str(f)]=sha(f)
 physround=2*(sum(j['case']['step_cap'] for j in p['training_jobs'])+sum(j['case']['step_cap'] for j in evaluation));finalphys=sum(j['case']['step_cap'] for j in p['final_jobs']);assert physround*12+finalphys+2000<=p['max_physics'];assert 12*88+411+8<=p['max_commands'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<=2**30
 OUT.mkdir();cfg=read(p['actor_configuration']);cfg.update(p);p['initial']=L.zero_export(cfg,OUT/'initial_zero.pt.gz');p['protected_files'][p['initial']['path']]=p['initial']['sha256'];p['inputs'][p['initial']['path']]=p['initial']['sha256'];save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',night_max_physics=p['max_physics'],night_max_commands=p['max_commands'],night_max_main_seconds=p['max_main_seconds'],round_worst_physics=physround,round_worst_commands=88,rounds_max=12,final407_worst_physics=finalphys,final407_commands=411,native_engineering_worst_physics=2000,native_engineering_commands=8,total_native_worst_physics=physround*12+finalphys+2000,total_commands=12*88+411+8,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,per_next_round_complete_reserved_bytes=p['child_reserve_bytes'],final_complete_reserved_bytes=p['final_bytes'],not_claiming12full_rounds_worst_storage_simultaneously_fits=True,storage_admission_each_round=True,CPU_total_one_core=True,GPU=False,all_stop=p['absolute_all_stop']));print('T552_REGISTERED',physround,finalphys,flush=True)

def preflight():
 root=read(OUT/'protocol.json');assert not (OUT/'preflight.json').exists();p=read(root['actor_configuration']);p.update(root);p['arm']='response';R.setup(OUT);R.verify(p,True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=L.initial(p);old=L.Old.frozen(p,p['frozen_base']['path']);x=torch.tensor(R.E.N.S.arrays(p['keep_rows'][0]['path'])['states'][:16]);assert torch.equal(q.sequence(x),old.sequence(x));x2=x.clone();x2[8:]+=1;assert torch.equal(q.sequence(x2)[:8],q.sequence(x)[:8]);f,b=L.feature_mean(q,x);assert (q.sequence(x)-b).abs().max()<1e-10
 data,pairs=L.cached_dataset(p,limit=12);obj=L.make(p);shadow=L.make(p);initial=L.state(obj);fixed=L.fixed_hash(obj[0]);events=[];trace=L.bc(obj,data,pairs,p,lambda:None,lambda a,b:events.append([a,b]),steps=2);assert trace==L.bc(shadow,data,pairs,p,lambda:None,lambda *_:None,steps=2);assert digest(L.state(obj))==digest(L.state(shadow));assert any(float(v.abs().max())>0 for v in obj[0].extra.parameters());m=L.export(obj,p,OUT/'synthetic_discarded.pt.gz');assert L.frozen(p,m['path']).phash()==obj[0].phash();mature=L.state(obj)
 # Real input states, fabricated bounded-Gaussian actions for zero-native math
 # contracts only. These arrays are NOT scientific navigation or training data.
 arrays=[]
 for seed in range(4):
  a=R.E.N.S.arrays(p['keep_rows'][0]['path']);a={k:v[:8].copy() if getattr(v,'ndim',0)>0 and len(v)==len(a['states']) else v.copy() for k,v in a.items()};qq=L.frozen(p,m['path']);qq.sample=True;qq.reset_noise();qq.rng=np.random.default_rng(5520+seed);h=None
  with torch.no_grad():
   for z in a['states']:u,loc,h,_=qq.step(torch.tensor(z[None]),h)
  for name in ['locations','logp','raw_actions','atom']:a[name]=np.asarray([t[name] for t in qq.tape])
  a['ownership']=np.zeros(8,np.uint8);a['stochastic']=np.ones(8,np.uint8);a['end_flags']=np.zeros((8,3),np.uint8);a['end_flags'][-1,2]=1;arrays.append(a)
 pp=dict(p,rl_steps=2);u=L.rl_or_control(obj,arrays,data,pairs,pp,lambda:None,lambda *_:None);v=L.rl_or_control(shadow,arrays,data,pairs,pp,lambda:None,lambda *_:None);assert u==v and digest(L.state(obj))==digest(L.state(shadow));assert fixed==L.fixed_hash(obj[0]);L.load(obj,mature);assert digest(L.state(obj))==digest(mature);L.load(obj,initial);assert digest(L.state(obj))==digest(initial)
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE_SYNTHETIC_DISCARDED',pid=os.getpid(),new_native=0,synthetic_discarded_Adam=8,trainable_parameters=5986,frozen_backbone_exact=True,initial_function_exact=True,causal_history=True,real33history_feature_cache=True,physical4pair_response_contract=True,exact_Adam_shadow_and_mature_rollback=True,checkpoint_reload=True,full_worker_NOT_yet_native_validated=True));print('T552_ZERO_NATIVE_PASS',flush=True)

def engineering_register():
 p=read(OUT/'protocol.json');assert read(OUT/'preflight.json')['state']=='PASS_ZERO_NATIVE_SYNTHETIC_DISCARDED';folder=OUT/'native_contract';assert not folder.exists();jobs=copy.deepcopy(p['training_jobs'][:2])
 for j in jobs:j['case']['step_cap']=500
 folder.mkdir();save(folder/'protocol.json',dict(_root_sha256=sha(OUT/'protocol.json'),kind='engineering',initial=p['initial'],probe_jobs=jobs,probe_seeds=[55201,55202],max_physics=2000,max_commands=8,main_seconds=300,learning_seconds=0,child_bytes=2*2**20));print('T552_NATIVE_CONTRACT_REGISTERED')
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:save(DOC/('ERROR_'+sys.argv[1]+'_'+str(int(time.time()))+'.json'),dict(error=traceback.format_exc(),new_launch_NOT_implied=True,epoch=time.time()));raise
