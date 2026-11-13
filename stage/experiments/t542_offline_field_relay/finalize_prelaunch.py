"""Final operational bound lock and real admission-seam negative checks.

No new navigation or optimizer calls. The actual 2k-step engineering probe
remains under its preserved original protocol and source; changing capacities
and a finite upper round count does not relabel it as navigation acceptance.
"""
from pathlib import Path
import copy,json,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t542_runtime as R
import importlib.util
_spec=importlib.util.spec_from_file_location('_t542_night_driver_unique',HERE/'driver.py');D=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(D)
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 assert not (OUT/'launch.json').exists() and not (DOC/'LOCAL_READINESS.json').exists();old=read(OUT/'protocol.json');before_sha=sha(OUT/'protocol.json')
 p=copy.deepcopy(old);p['max_main_seconds']=40000;p['authorization']=p['authorization'].replace('33000main','40000main')
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 p['inputs'][str(DOC/'PLAN.md')]=sha(DOC/'PLAN.md')
 p['prelaunch_revision']['max_main_seconds_revision']=[33000,40000];p['prelaunch_revision']['final_operational_relock_previous_root_sha256']=before_sha
 save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
 child=read(OUT/'native_preflight/protocol.json');child['_root_sha256']=sha(OUT/'protocol.json');save(OUT/'native_preflight/protocol.json',child)
 assert all(p[k]==old[k] for k in ['raw_sigma','inner_steps','learning_rate','replay_weight','target_kl','seed','source_initial','initial','frozen_candidate','max_physics','max_commands','absolute_learning_stop','absolute_native_stop','absolute_all_stop'])
 reg=read(p['regression_configuration']);base=read(p['actor_configuration']);cap=sum(j['case']['step_cap'] for j in reg['old275_jobs']+reg['six_jobs']+reg['wide8_jobs']+base['field_jobs'])
 over=dict(kind='round',main_seconds=2400,max_physics=3871332,max_commands=76);usage=(2000,8,12);now=time.time();used=R.used(OUT);assert D.admission(p,usage,over,now,used,cap)[0]
 negative={}
 trials=[('physical',(p['max_physics']-over['max_physics']-cap+1,8,12),now,used),('commands',(2000,p['max_commands']-76-411+1,12),now,used),('main_seconds',(2000,8,p['max_main_seconds']-2400-3300+1),now,used),('native_time',usage,p['absolute_native_stop']-2400-3300-p['audit_seconds']-p['manifest_seconds']-19,used),('all_time',usage,p['absolute_all_stop']-2400-2*(p['audit_seconds']+p['manifest_seconds'])-3300-1199,used),('learning_time',usage,p['absolute_learning_stop']-p['learning_seconds']+1,used),('bytes',usage,now,p['max_bytes']-p['child_bytes']-16*2**20+1)]
 for key,u,t,b in trials:
  good,checks=D.admission(p,u,over,t,b,cap);assert not good and not checks[key],(key,checks);negative[key]=True
 # Current output path wins over the historical ancestor configuration.
 c=R.plan(OUT/'native_preflight');assert c['out']==str(OUT/'native_preflight') and c['out']!=base['out'];assert c['initial']==p['source_initial']
 assert not R.accepts(p['initial_check'],p['initial_check'])
 lost=copy.deepcopy(p['initial_check']);lost['field_success_ids']=lost['field_success_ids'][1:];lost['distance_sum']-=10.;assert not R.accepts(p['initial_check'],lost)
 # True physical/optimizer records from the already completed preflight.
 a=read(OUT/'native_preflight/audit.json');s=read(OUT/'native_preflight/summary.json');z=read(OUT/'preflight.json');assert a['state']=='PASS' and s['physics']==2000 and s['commands']==8 and z['total_discarded_synthetic_SGD']==24
 assert all(not Path('/proc',str(i)).exists() for g in read(OUT/'native_preflight/worker_generations.json') for i in [g['parent'],*g['workers']])
 R.verify(p,True);budget=read(DOC/'BUDGET.json');budget.update(max_main_seconds=40000,max_rounds=72,number_rounds_not_guaranteed=72,child_bytes=p['child_bytes'],root_bytes=p['max_bytes'],global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']);save(DOC/'BUDGET.json',budget)
 # Baseline-rate estimate is a schedule, not a promise or fabricated execution.
 historic=read(ROOT/'stage/runs/t519_fresh_full_actor_rl/pilot/summary.json');vnow=now;u=(2000+1682386,76,12+1024.);b=used;count=0
 while count<72 and D.admission(p,u,over,vnow,b,cap)[0]:
  count+=1;vnow+=historic['elapsed_s']+60;u=(u[0]+historic['physics'],u[1]+76,u[2]+historic['elapsed_s']);b+=int(2.1*2**20)
 save(DOC/'LOCAL_READINESS.json',dict(state='PASS_NOT_YET_LAUNCHED',protocol_sha256=sha(OUT/'protocol.json'),preflight_discarded_synthetic_SGD=24,native_preflight_physics=2000,native_preflight_commands=8,native_preflight_independent_audit_sha256=sha(OUT/'native_preflight/audit.json'),zero_native_regression_sha256=sha(OUT/'preflight.json'),negative_admission_checks=negative,output_route_regression=True,lost_prior_success_rejected=True,all_scientific_learning_parameters_unchanged_by_bound_revisions=True,new_native=0,new_optimizer=0,illustrative_old_runtime_rounds_before_full_reserve=count,illustrative_main_and_audit_hours=(vnow-now)/3600,not_promised_actual_runtime=True,all_stop=p['absolute_all_stop'],no_network_needed_by_science=True,epoch=time.time()))
 print('T542_FINAL_READINESS_PASS',count,'illustrative_hours',round((vnow-now)/3600,2),'global_reserved',budget['global_reserved'],flush=True)
if __name__=='__main__':main()
