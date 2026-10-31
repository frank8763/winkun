"""Bounded episodic Gaussian parameter PG; matched frozen distribution control."""
from support import *
import json,time,fcntl,shutil,multiprocessing as mp,traceback,copy
class BudgetStop(Exception):pass

def main():
 lock=(OUT/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'status.json').exists(),'No duplicate T207'
 plan=json.loads((HERE/'protocol.json').read_text());assert json.loads((OUT/'contracts.json').read_text())['state']=='PASS'
 for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
 start=time.monotonic();deadline=start+plan['wall_seconds'];wall_deadline=time.time()+plan['wall_seconds'];physics=episodes=updates=0;training_physics=training_episodes=0;generation=-1;pipes=[];workers=[];results=[];tasks=[];means={a:np.zeros(6) for a in plan['arms']};policies={};replays=[];reason=None;witness=None;selection=None;final_rows=[];reload_rows=[];robust=[];invalid=[];exports={};stable=False;finalizing=False;inflight=0
 sigma=np.asarray(plan['sigma']).ravel();rng=np.random.default_rng(plan['seed']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p=Policy().eval().requires_grad_(False);p.load_state_dict(parent)
 with np.load(OUT/'basis.npz') as z:basis=z['basis'].copy()
 with np.load(plan['retention_targets']) as z:anchors=torch.tensor(z['states']);targets=torch.tensor(z['actions']);offsets=z['offsets'].copy()
 (OUT/'heads').mkdir()
 def write(name,v):C.save(OUT/name,v)
 def append(name,v):
  with (OUT/name).open('a') as f:f.write(json.dumps(v)+'\n')
 def used(path):return sum(x.stat().st_size for x in path.rglob('*') if x.is_file())
 def safe(row):return row['complete'] and row['success'] and not row['collision']
 def status(state):write('status.json',{'state':state,'generation':generation,'physics_steps':physics,'episodes':episodes,'training_physics_steps':training_physics,'training_episodes':training_episodes,'parameter_update_attempts':updates,'elapsed_s':time.monotonic()-start,'means':{a:v.tolist() for a,v in means.items()},'witness':witness,'selection':selection,'own_bytes':used(OUT),'all_new_runs_bytes':used(ROOT/'stage/runs')})
 def guard(count=0,steps=0):
  if time.monotonic()+(0 if finalizing else plan['final_reserve_seconds'])>=deadline:raise BudgetStop('WALL_FINAL_RESERVE')
  if episodes+count>plan['max_episodes'] or physics+steps>plan['max_physics']:raise BudgetStop('COUNT_RESERVE')
  ownreserve=plan['pending_bytes']+(0 if finalizing else plan['export_reserve_bytes'])
  if used(OUT)+ownreserve>=plan['final_bytes'] or used(ROOT/'stage/runs')+plan['pending_bytes']>=1024**3 or shutil.disk_usage(ROOT).free<=40*1024**3:raise BudgetStop('STORAGE_RESERVE')
 def receive(pipe):
  if not pipe.poll(max(.1,min(120,deadline-time.monotonic()+2))):raise BudgetStop('WORKER_RESPONSE_DEADLINE')
  reply=pipe.recv();assert 'error' not in reply,reply;return reply
 def make_task(spec):
  guard();pipes[0].send(('make_task',spec));row=receive(pipes[0]);append('tasks.jsonl',{**spec,**row});return {**row,'case':spec['case']}
 def heads_for(coeff,label):
  set_coefficients(p,parent,basis,coeff);path=OUT/'heads'/(label+'.npz');assert not path.exists();R.save_heads(path,p)
  with torch.no_grad():err=(p(anchors)-targets).square().mean(1);penalty=float(sum(err[offsets[j]:offsets[j+1]].mean() for j in range(7))/7)
  append('behavior_parameters.jsonl',{'id':label,'coefficients':np.asarray(coeff).tolist(),'heads':str(path),'sha256':C.sha(path),'old7_balanced_MSE':penalty,'generation':generation})
  return path,penalty
 def job(task,path,label,arm=None,role='evaluation',retain=False):return {'id':label,'task':task['id'],'case':task['case'],'exact_snapshot':task['snapshot'],'heads':str(path),'seed':0,'mode':'deterministic','arm':arm,'round':generation,'role':role,'summary_only':not retain,'initial_pose':task['pose']}
 def batch(jobs):
  nonlocal physics,episodes,training_physics,training_episodes,inflight
  allrows=[]
  for off in range(0,len(jobs),plan['workers']):
   sub=jobs[off:off+plan['workers']];need=sum(j['case']['step_cap'] for j in sub);guard(len(sub),need);inflight=need
   for pipe,j in zip(pipes,sub):pipe.send(('rollout',j))
   rows=[receive(pipe) for pipe in pipes[:len(sub)]];physics+=sum(r['physics_steps'] for r in rows);episodes+=len(rows);inflight=0
   for r in rows:
    if r['role']=='parameter_training':training_physics+=r['physics_steps'];training_episodes+=int(r['complete'])
    append('rollouts.jsonl',r)
   results.extend(rows);allrows.extend(rows);status('RUNNING')
   if not all(r['complete'] for r in rows):raise BudgetStop('WALL_CENSORED_ROLLOUTS')
  return allrows
 def old7(path,label,arm):return batch([job(tasks[i],path,f'{label}_old{i}',arm) for i in [0,1,2,3,4,5,7]])
 try:
  ctx=mp.get_context('spawn')
  for i in range(plan['workers']):
   a,b=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']=str(i+1);w=ctx.Process(target=R.worker,args=(b,i,str(OUT),plan['checkpoint'],wall_deadline));w.start();b.close();pipes.append(a);workers.append(w)
  os.environ['STAGE_PROCESS_CPU']='0';write('workers.json',{'parent':os.getpid(),'workers':[w.pid for w in workers]})
  for case in plan['primary_cases']:
   t=make_task({'id':'original_'+case['id'],'case':case});assert t['valid'];tasks.append(t)
  path,_=heads_for(np.zeros(6),'parent');baseline=batch([job(t,path,f'parent_{i}') for i,t in enumerate(tasks)]);assert all(safe(baseline[i]) for i in [0,1,2,3,4,5,7]) and baseline[6]['collision']
  for generation in range(plan['max_generations']):
   # Reserve the whole sampling generation before spending either arm's data.
   guard(16,16*plan['primary_cases'][6]['step_cap']);eps=rng.normal(size=(plan['pairs_per_generation'],6));np.savez_compressed(OUT/f'epsilon_{generation}.npz',epsilon=eps,mu_learned=means['learned'],mu_frozen=means['frozen'],sigma=sigma);collections={};meta={}
   for arm in plan['arms']:
    jobs=[];descriptions=[]
    for sign in [1.,-1.]:
     for i,e in enumerate(eps):
      coeff=means[arm]+sign*sigma*e;label=f'g{generation}_{arm}_s{sign}_p{i}';path,mse=heads_for(coeff,label);descriptions.append({'coeff':coeff,'mse':mse,'heads':path,'id':label});jobs.append(job(tasks[6],path,label,arm,'parameter_training' if arm=='learned' else 'frozen_exploration_control',True))
    collections[arm]=batch(jobs);meta[arm]=descriptions
   for arm in plan['arms']:
    for index,row in enumerate(collections[arm]):
     if safe(row) and witness is None:witness={'arm':arm,'generation':generation,'sample_index':index,'coefficients':meta[arm][index]['coeff'].tolist(),'heads':str(meta[arm][index]['heads']),'rollout':row,'origin':'sampled_behavior_policy','previous_learned_updates':updates}
   if witness:reason='SAMPLED_ORIGINAL_LEFT_GOAL';break
   arm='learned';values=np.asarray([r['environment_return']-plan['retention_weight']*m['mse'] for r,m in zip(collections[arm],meta[arm])]);K=plan['pairs_per_generation'];g=score_gradient(values[:K],values[K:],eps,sigma);white=sigma*g;norm=float(np.linalg.norm(white));delta=plan['whitened_step_norm']*sigma*white/max(norm,1e-20);proposal=np.clip(means[arm]+delta,-plan['mean_box'],plan['mean_box']);assert .5*np.sum(((proposal-means[arm])/sigma)**2)<=plan['max_mean_KL']+1e-12;updates+=1
   previous=means[arm].copy();accepted=0.;gates=[]
   for factor in plan['interpolation_factors']:
    coeff=previous+factor*(proposal-previous);path,_=heads_for(coeff,f'g{generation}_learned_factor{factor}');gate=old7(path,f'g{generation}_learned_factor{factor}',arm);gates.append({'factor':factor,'passed':all(safe(r) for r in gate)})
    if all(safe(r) for r in gate):means[arm]=coeff;accepted=factor;break
   central=[]
   for arm in plan['arms']:
    path,_=heads_for(means[arm],f'g{generation}_{arm}_central');row=batch([job(tasks[6],path,f'g{generation}_{arm}_central_LEFT',arm,retain=True)])[0];central.append({'arm':arm,'safe_GOAL':safe(row),'remaining_m':row['remaining_m']})
    if safe(row) and witness is None:witness={'arm':arm,'generation':generation,'coefficients':means[arm].tolist(),'heads':str(path),'rollout':row,'origin':'central_mean_policy','previous_learned_updates':updates}
   metric={'generation':generation,'epsilon_sha256':C.sha(OUT/f'epsilon_{generation}.npz'),'regularized_returns':values.tolist(),'raw_environment_returns':{a:[r['environment_return'] for r in rr] for a,rr in collections.items()},'score_gradient':g.tolist(),'whitened_score_norm':norm,'proposed_mean':proposal.tolist(),'accepted_mean':means['learned'].tolist(),'chosen_interpolation':accepted,'old7_gates':gates,'central':central,'parameter_updates':updates};append('generations.jsonl',metric);print(json.dumps(metric),flush=True);status('GENERATION_COMPLETE')
   if witness:reason='CENTRAL_ORIGINAL_LEFT_GOAL';break
  if reason is None:reason='GENERATION_BUDGET_COMPLETE'
 except BudgetStop as e:reason=str(e)
 except BaseException:
  reason='FAILED';write('failure.json',{'state':'FAILED','physics_steps':physics,'unrecorded_inflight_physics_upper_bound':inflight,'traceback':traceback.format_exc()});raise
 finally:
  finalizing=True
  if policies is not None:
   for arm in plan['arms']:
    set_coefficients(p,parent,basis,means[arm]);path=OUT/f'{arm}_final.pt';torch.save({'policy':p.state_dict(),'coefficients':means[arm].tolist(),'parent_sha256':C.sha(plan['checkpoint']),'protocol_sha256':C.sha(HERE/'protocol.json'),'origin':'parameter-space PG mean' if arm=='learned' else 'frozen distribution mean','parameter_update_attempts':updates if arm=='learned' else 0,'automatic_promotion':False},path);exports[arm]={'path':str(path),'sha256':C.sha(path)}
  try:
   if reason!='FAILED' and tasks and time.monotonic()+20<deadline:
    for arm in plan['arms']:
     path,_=heads_for(means[arm],f'{arm}_final_eval');final_rows+=batch([job(t,path,f'{arm}_final_old{i}',arm,retain=i==6) for i,t in enumerate(tasks)])
    if witness:
     gates=old7(witness['heads'],'witness_gate',witness['arm'])
     if all(safe(r) for r in gates):
      set_coefficients(p,parent,basis,np.asarray(witness['coefficients']));file=OUT/'selected_policy.pt';torch.save({'policy':p.state_dict(),'witness':witness,'protocol_sha256':C.sha(HERE/'protocol.json'),'automatic_promotion':False},file);saved=Policy().eval();saved.load_state_dict(torch.load(file,map_location='cpu',weights_only=False)['policy']);assert all(torch.equal(v,saved.state_dict()[k]) for k,v in p.state_dict().items());path=OUT/'heads/selected_reload.npz';R.save_heads(path,saved);selection={'path':str(file),'sha256':C.sha(file),'witness_origin':witness['origin'],'arm':witness['arm']}
      guard(steps=witness['rollout']['physics_steps']);pipes[0].send(('replay',witness['rollout']));check=receive(pipes[0]);physics+=check['physics_steps'];replays.append(check);write('replays.json',replays);assert check['verified']
      reload_rows=batch([job(t,path,f'selected_old{i}',witness['arm'],retain=i==6) for i,t in enumerate(tasks)]);grid=[]
      for i in [6,7]:
       for dx in plan['robustness_xy']:
        for dy in plan['robustness_xy']:
         for deg in plan['robustness_yaw']:
          t=make_task({'id':f'grid_{i}_{len(grid)+len(invalid)}','case':tasks[i]['case'],'near_snapshot':tasks[i]['snapshot'],'delta':[dx,dy,deg]})
          if t['valid']:grid.append(t)
          else:invalid.append(t)
      robust=batch([job(t,path,f'selected_grid_{i}',witness['arm']) for i,t in enumerate(grid)]);stable=len(reload_rows)==8 and all(safe(r) for r in reload_rows) and len(robust)==54 and not invalid and all(safe(r) for r in robust)
  except BudgetStop as e:append('final_incomplete.jsonl',{'reason':str(e),'final_old8_completed':len(final_rows),'saved_reload_completed':len(reload_rows),'robustness_completed':len(robust)})
  except BaseException:
   reason='FAILED';write('failure.json',{'state':'FAILED','stage':'final_evaluation','physics_steps':physics,'traceback':traceback.format_exc()});raise
  finally:
   for pipe in pipes:
    try:pipe.send(None)
    except OSError:pass
   for w in workers:
    w.join(3)
    if w.is_alive():w.terminate();w.join(3)
  if reason!='FAILED':
   for source,h in plan['input_hashes'].items():assert C.sha(source)==h,source
   summary={'state':'COMPLETE','stop_reason':reason,'physics_steps':physics,'episodes':episodes,'training_physics_steps':training_physics,'training_episodes':training_episodes,'parameter_update_attempts':updates,'optimizer_updates_to_deep_backbone':0,'witness':witness,'selection':selection,'mean_coefficients':{a:v.tolist() for a,v in means.items()},'exports':exports,'final_old8':final_rows,'saved_reload':reload_rows,'robustness_completed':len(robust),'robustness_safe_GOAL':sum(safe(r) for r in robust),'stable_5m_gate_passed':stable,'all_workers_stopped':True,'elapsed_s':time.monotonic()-start,'own_bytes':used(OUT),'all_new_runs_bytes':used(ROOT/'stage/runs'),'automatic_promotion':False,'scope':plan['method_contract']};assert physics<=plan['max_physics'] and episodes<=plan['max_episodes'] and used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3;write('summary.json',summary);write('status.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['final_old8','saved_reload']}),flush=True)
if __name__=='__main__':main()
