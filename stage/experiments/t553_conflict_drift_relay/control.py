"""Zero-native readiness and immutable deferred relay template registration."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t553_learning as L
ROOT=L.ROOT;DOC=ROOT/'stage/docs/overnight_20261114_relay';PRIOR=ROOT/'stage/runs/t552_joint_robust_night';OUT=ROOT/'stage/runs/t553_conflict_drift_relay';read,save,sha,digest=L.read,L.save,L.sha,L.digest;torch,np=L.torch,L.np

def preflight():
 assert not (DOC/'PREFLIGHT.json').exists();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);base=read(PRIOR/'protocol.json');p=read(base['actor_configuration']);p.update(base);p.update(arm='integral',cagrad_c=.5);trials=[];rng=np.random.default_rng(553)
 for name,g in [('aligned',[[1.,0],[2.,0]]),('opposed',[[1.,0],[-1.,0]]),('zero',[[0.,0],[0.,0]]),('onezero',[[1.,0],[0.,0]])]+[('random'+str(i),rng.normal(size=(2,5)).tolist()) for i in range(12)]:
  g=torch.tensor(g,dtype=torch.float64);d,m=L.cagrad(g);assert torch.isfinite(d).all() and 0<=m['w']<=1 and m['objective']<=m['objective_at_equal']+1e-10;z,_=L.cagrad(g,0.);assert torch.allclose(z,g.sum(0),atol=1e-12,rtol=1e-12);trials.append(dict(name=name,**m))
 obj=L.make(p);shadow=L.make(p);data,pairs=L.cached_dataset(p,limit=12);before=L.state(obj);a=L.bc(obj,data,pairs,p,lambda:None,lambda *_:None,steps=2);b=L.bc(shadow,data,pairs,p,lambda:None,lambda *_:None,steps=2);assert a==b and digest(L.state(obj))==digest(L.state(shadow));mature=L.state(obj);arrays=[]
 for seed in range(4):
  a=L.G.E.N.S.arrays(p['keep_rows'][0]['path']);n=len(a['states']);a={k:v[:8].copy() if getattr(v,'ndim',0)>0 and len(v)==n else v.copy() for k,v in a.items()};q=copy.deepcopy(obj[0]).requires_grad_(False);q.sample=True;q.reset_noise();q.rng=np.random.default_rng(5530+seed);h=None
  with torch.no_grad():
   for x in a['states']:u,loc,h,_=q.step(torch.tensor(x[None]),h)
  for name in ['locations','logp','raw_actions','atom']:a[name]=np.asarray([t[name] for t in q.tape])
  a['ownership']=np.zeros(8,np.uint8);a['stochastic']=np.ones(8,np.uint8);a['end_flags']=np.zeros((8,3),np.uint8);a['end_flags'][-1,2]=1;arrays.append(a)
 pp=dict(p,rl_steps=2);a=L.rl_or_control(obj,arrays,data,pairs,pp,lambda:None,lambda *_:None);b=L.rl_or_control(shadow,arrays,data,pairs,pp,lambda:None,lambda *_:None);assert a==b and digest(L.state(obj))==digest(L.state(shadow));L.load(obj,mature);assert digest(L.state(obj))==digest(mature);L.load(obj,before);assert digest(L.state(obj))==digest(before)
 # Constant nonzero command bias has positive accumulating-prefix energy,
 # unlike zero bias; saturated unchanged executed commands cost zero.
 dt=torch.tensor([.2,.2,.2],dtype=torch.float64);d=torch.ones(3,2,dtype=torch.float64)*.01;cost=((d*dt[:,None]).cumsum(0).square()/dt.cumsum(0)[:,None]);assert torch.all(cost[1:]>cost[:-1]);assert torch.equal(torch.tensor([1.2]).clamp(-1,1)-torch.tensor([1.1]).clamp(-1,1),torch.zeros(1))
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),epoch=time.time(),new_native=0,new_commands=0,synthetic_discarded_Adam=8,cagrad_fixed64_solver_tests=trials,c0_exact_original_sum_gradient=True,live_shadow_mature_reject_exact=True,synthetic_RL_density=True,integral_drift_constant_bias_and_boundary_control_tests=True,actual_new_RL_NOT_started=True,raw_for_synthetic_only_not_physical_samples=True));print('T553_ZERO_NATIVE_PASS',flush=True)

def prepare():
 assert not OUT.exists() and not (HERE/'template.json').exists();assert read(DOC/'PREFLIGHT.json')['state']=='PASS_ZERO_NATIVE';p=read(PRIOR/'protocol.json');epoch=lambda h,m:datetime(2026,11,15,h,m,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();p.update(id='T553_DEFERRED_CAGRAD_AND_PHYSICAL_INTEGRAL_RETENTION',arms=['cagrad','integral'],max_rounds_per_arm=2,cagrad_c=.5,arm_rejection_stop=2,max_bytes=16*2**20,child_reserve_bytes=6*2**20,child_bytes=6*2**20,final_bytes=8*2**20,absolute_learning_stop=epoch(8,15),absolute_native_stop=epoch(9,0),absolute_all_stop=epoch(10,0),authorization='用户：预估时间，如果快要结束了就开启接力的实验，要思考怎么做接力的，到明天十点都行。仅延长最终截止，原60M/2000cmd/28000主管线总额不增加。当前T552科学协议与截止不修改，T553等待其服务全退出/独立审计/全部本地manifest通过。',deferred_prior_run=str(PRIOR),shared_total_physics=60_000_000,shared_total_commands=2000,shared_total_main_seconds=28000,shared_preflight_discarded_Adam=8,scope='Twofixednewmechanisms:2objectiveCAGradstylec.5/golden64,noscipy;secondplusphysicalprefixcumulativeexecutedcommanddrift. SameT5525986readout+frozenT550,32BC+4freshPPO+shadow,max2rounds/arm. InitialkeptT552Actor+freshAdamforallarms;noinheritrejectedproposals. Exactpriorended/audit/manifestthenactualsharedremaining+globalcapacityadmission. NoactiveT552source/protocol/writerchanges. Full38IDnonregressionandstrictgain/replay;final407reserved. Nooriginality/theoreticalguarantee/crosssiteclaim. Full source immutable remote registration bundle before watcher. Derivedruntimeprotocolonlybindspredeclaredpriorretainedmodel/actualremainingbudget;nofreeparameterorgrid.')
 # Future raw output path and depleted budgets are bound only after prior exits.
 p.pop('inputs',None);p.pop('protected_files',None);p.pop('initial',None);inputs={}
 for name in ['PLAN.md','SOURCES.md','AUTHORIZATION.json','PREFLIGHT.json']:inputs[str(DOC/name)]=sha(DOC/name)
 for f in HERE.glob('*.py'):inputs[str(f)]=sha(f)
 for f,h in read(PRIOR/'protocol.json')['inputs'].items():inputs[f]=h
 p['registered_inputs']=inputs;save(HERE/'template.json',p);save(DOC/'BUDGET.json',dict(state='DEFERRED_REGISTERED_NOT_NATIVE_LAUNCHED',shared_unchanged_physics=60_000_000,shared_unchanged_commands=2000,shared_unchanged_main_seconds=28000,new_round_worst_physics=3830996,new_round_commands=88,max_new_rounds=4,new_final407_worst_physics=11307371,new_final407_commands=411,new_all_worst_physics=26631355,new_all_worst_commands=763,actual_remaining_admission_after_T552=True,own_bytes=16*2**20,global_bytes=2**30,free_bytes=40*2**30,all_stop='2026-11-15 10:00 Asia/Shanghai',prior_deadline_unchanged=True,no_new_native_yet=True));print('T553_DEFERRED_TEMPLATE_REGISTERED')
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:save(DOC/('ERROR_'+sys.argv[1]+'_'+str(int(time.time()))+'.json'),dict(error=traceback.format_exc(),new_native_launch=False,epoch=time.time()));raise
