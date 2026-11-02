"""Finite endpoint occupancy pilot under a newly declared iid behavior policy.

Only the two small value heads are trained. The navigation Actor is frozen.
No CRL theorem or safe first-hit interpretation is asserted for these heads.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t308_iid_native_goal_value'
DOC=ROOT/'stage/docs/overnight_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t305_goal_conditioned_contract'))
from gc_contract import *
sys.path.insert(0,str(ROOT/'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
sys.path.insert(0,str(ROOT/'stage/experiments/t291_hard_guard_ablation'))
from t291_codec import pack_full,unpack_full
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
from t280_worker import LimitedLog
import copy,gzip,hashlib,importlib.util,io,json,os,shutil,time,traceback

NATIVE_KEYS=['states','actions','means','logp','native_counts','next_states','poses','next_poses','end_flags','rewards','environment_rewards','aux_costs']
TYPE={'coordinate':0,'coordinate_absorbed_goal':0,'collision':1,'task_timeout':2,'collector_censored_unknown':3}

def read(path):return json.loads(Path(path).read_text())

def actor(p):
    q=H.policy(read(p['policy_protocol'])).eval().requires_grad_(False)
    assert S.digest(q.state_dict())==p['policy_hash']
    return q

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<=p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def packed(a):
    base=pack_full({k:a[k] for k in NATIVE_KEYS})
    base.update({k:v for k,v in a.items() if k not in NATIVE_KEYS})
    return base

def arrays(path):
    with np.load(path) as z:d={k:z[k].copy() for k in z.files}
    extra={k:v for k,v in d.items() if k not in NATIVE_KEYS and k!='last_next_state'}
    base=unpack_full({k:v for k,v in d.items() if k not in extra})
    return dict(base,**extra)

def generate(job,q,native,create,witness,deadline,counts,index,previous_total):
    env=None;rows=[];hist=[];innovations=[];steps=0;error=None;term=trunc=False;info={};initial=None;integration=None;remaining=None
    try:
        env,obs,_=R.task_env(create,job);obs=C.perturb_anchor(env,obs,job)
        assert not env._collision();scene=C.model_hash(env);goal=env.goal_xy.copy();cap=job['case']['step_cap']
        previous=np.zeros(2,np.float32);pn=0;rng=np.random.default_rng(job['seed'])
        initial=T.L.packet(env,obs,0,cap,previous,0)
        if job.get('expected_initial120_sha256'):assert S.digest(initial)==job['expected_initial120_sha256']
        while not(term or trunc) and steps<cap and time.time()<deadline:
            assert shutil.disk_usage(ROOT).free>=40*1024**3
            x=T.L.packet(env,obs,steps,cap,previous,pn);pose=np.asarray(env._pose()[:3]);meta=witness.get(env)
            assert np.abs(H.rebuild(x,pose,meta[0],goal,meta[1])-x).max()<=2e-6
            with torch.no_grad():
                mean=q.step(torch.tensor(x[None]))[1][0]
                z=rng.normal(size=2)
                if not rows:z[1]+=job['first_steer_innovation_offset']
                action=T.L.AR.sample(mean,z).numpy()
                lp=float(D.terms(mean,torch.tensor(action))[1].sum())
            d0=float(env._goal_state()[1]);before=steps
            try:
                for _ in range(min(100,cap-steps)):
                    if time.time()>=deadline:break
                    obs,_,term,trunc,info=native.step_checked(env,action);steps+=1
                    if term or trunc:break
            except BaseException:error=traceback.format_exc()
            n=steps-before
            if n:
                d1=float(env._goal_state()[1]);discount=R.GAMMA**n
                terminal=term or trunc;event=float(info.get('reach_goal',False))-float(info.get('collision',False))-.5*float(trunc)
                reward=-.000002*(1-discount)/(1-R.GAMMA)+.03*d0-discount*.03*d1*(not terminal)+(R.GAMMA**(n-1)*event if terminal else 0.)
                nxt=T.L.packet(env,obs,steps,cap,action,n)
                rows.append((x,action,mean.numpy(),lp,n,nxt,pose,np.asarray(env._pose()[:3]),[info.get('collision',False),info.get('reach_goal',False),trunc],reward))
                hist.append(meta);innovations.append(z.copy());previous=action.copy();pn=n
            with counts.get_lock():counts[index]=previous_total+steps
            if error or not n:break
        if rows:hist.append(witness.get(env))
        remaining=float(env._goal_state()[1]);integration=R.arrays_hash(R.integration(env))
        assert scene==C.model_hash(env) and np.array_equal(goal,env.goal_xy)
    except BaseException:
        # Administrative/contract interruption evidence is not navigation failure.
        error=traceback.format_exc()
        if env is not None:
            partial={name:np.asarray([r[i] for r in rows]) for i,name in enumerate(NATIVE_KEYS[:10])}
            partial.update(history_poses=np.asarray([h[0] for h in hist]),history_goals=np.asarray([h[1] for h in hist]),
                history_times=np.asarray([h[2] for h in hist]),innovations=np.asarray(innovations),
                interruption_integration=R.integration(env),interruption_pose=np.asarray(env._pose()))
            buf=io.BytesIO();np.savez_compressed(buf,**partial);blob=buf.getvalue();assert len(blob)<=256*1024
            f=OUT/'workers'/str(index)/'interruption.npz';assert not f.exists();f.write_bytes(blob)
            C.save(f.with_suffix('.json'),dict(job=job,physics_steps=steps,completed_decisions=len(rows),sha256=C.sha(f),traceback=error,administrative_censor=True))
        raise
    finally:
        with counts.get_lock():counts[index]=previous_total+steps
        if env is not None:
            witness.events.pop(id(env),None);env.close()
    assert rows,'No completed decision: initialization/deadline failure, no invented episode'
    names=NATIVE_KEYS[:10];a={k:np.asarray([r[i] for r in rows]) for i,k in enumerate(names)}
    a['native_counts']=a['native_counts'].astype(np.uint8);a['end_flags']=a['end_flags'].astype(np.uint8)
    a['environment_rewards']=a['rewards'].copy();a['aux_costs']=T.endpoint_cost(a,cap)
    a.update(history_poses=np.asarray([h[0] for h in hist]),history_goals=np.asarray([h[1] for h in hist]),history_times=np.asarray([h[2] for h in hist]),innovations=np.asarray(innovations))
    row=dict(id=job['id'],family=job['family'],side=job['side'],split=job['split'],arm=job['arm'],seed=job['seed'],
        original_job=job,physics_steps=steps,complete=bool(term or trunc),success=bool(info.get('reach_goal')),
        collision=bool(info.get('collision')),timeout=bool(trunc),remaining_m=remaining,error=error,
        censored=not bool(term or trunc),policy_hash=job['policy_hash'],initial120_sha256=S.digest(initial),
        goal_xy=goal.tolist(),scene_sha256=scene,integration_sha256=integration,array_hashes={k:S.digest(v) for k,v in a.items()},
        iid_rho=0.,forced_first_action_diagnostic=job['arm']!='center',Actor_updates=0)
    return dict(row=row,arrays=a)

def worker(pipe,index,p,counts):
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;total=0;witness=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);q=actor(p)
        spec=importlib.util.spec_from_file_location('t308_history_witness',ROOT/'stage/experiments/t295_hindsight_support/run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=module.Witness(Base)
        pipe.send(dict(ready=True,pid=os.getpid(),policy_hash=S.digest(q.state_dict()),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            job=pipe.recv()
            if job is None:break
            item=generate(job,q,native,create,witness,job['deadline_epoch'],counts,index,total);total+=item['row']['physics_steps'];pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),completed_worker_physics=total);C.save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        if witness is not None:witness.close()
        C.save(folder/'log_accounting.json',dict(bytes_observed=log.total,bytes_written=log.written));pipe.close();log.close()

def make_pool(rows):
    pool=[]
    for r in rows:
        a=arrays(r['path']);laws=[]
        for k in range(len(a['states'])):
            law=future_law(a['native_counts'],a['end_flags'],k,28000,r['complete'])
            kinds=np.array([TYPE[e['type']] for e in law]);points=np.array([a['next_poses'][e['index'],:2] if TYPE[e['type']]==0 else [0.,0.] for e in law])
            laws.append((np.cumsum([e['mass'] for e in law]),kinds,points))
        pool.append((r,a,laws))
    return pool

def batch(pool,rng,n):
    observations=[];actions=[];poses=[];queries=[];types=[]
    for _ in range(n):
        r,a,laws=pool[int(rng.integers(len(pool)))];k=int(rng.integers(len(laws)));cdf,kind,points=laws[k]
        j=min(int(np.searchsorted(cdf,rng.random(),side='right')),len(cdf)-1)
        observations.append(a['states'][k]);actions.append(a['actions'][k]);poses.append(a['poses'][k]);queries.append(points[j]);types.append(kind[j])
    poses,queries,types=np.asarray(poses),np.asarray(queries),np.asarray(types)
    body=query_matrix(poses,queries)/10.;body[:,types!=0]=0.
    equivalent=(types[:,None]==types[None,:]) & ((types[:,None]!=0) | (queries[:,None]==queries[None,:]).all(-1))
    valid=types!=3;keep=np.flatnonzero(valid)
    assert len(keep)>1
    return (torch.tensor(np.asarray(observations)),torch.tensor(np.asarray(actions)),torch.tensor(body,dtype=torch.float32),torch.tensor(types[None,:].repeat(n,0)),torch.tensor(equivalent),torch.tensor(keep))

def scores(model,data,blind=False):
    x,a,b,t,eq,keep=data
    if blind:a=torch.zeros_like(a)
    logits=model.logits(x,torch.zeros(len(x),2),a,b,t)
    z=logits[keep][:,keep];eq=eq[keep][:,keep]
    # Equal coordinate/token queries are all positives, never false negatives.
    nce=(torch.logsumexp(z,1)-torch.logsumexp(z.masked_fill(~eq,-torch.inf),1)).mean()
    loss=nce+.01*torch.logsumexp(z,1).square().mean()
    top=z.max(1,keepdim=True).values;tie=z==top
    accuracy=((tie&eq).sum(1)/tie.sum(1)).mean()
    return loss,nce,accuracy

def fit(p,rows,guard,event,holder):
    train=make_pool([r for r in rows if r['split']=='train' and r['arm']=='center'])
    assert len(train)==8;torch.manual_seed(p['seed']);prototype=Critic();initial=S.digest(prototype.state_dict())
    models={n:copy.deepcopy(prototype) for n in ['action','blind']};opts={n:torch.optim.Adam(q.parameters(),lr=p['head_learning_rate']) for n,q in models.items()}
    holder.update(models=models,optimizers=opts,initial=initial);rng=np.random.default_rng(p['seed']);losses=np.empty((p['head_steps'],2),np.float64)
    for step in range(p['head_steps']):
        guard();data=batch(train,rng,p['batch_size'])
        for col,(name,q) in enumerate(models.items()):
            opt=opts[name];opt.zero_grad();loss,_,_=scores(q,data,name=='blind');assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),5.);event(name,step);opt.step();losses[step,col]=float(loss.detach())
    return models,opts,losses,initial

def kernel_value(a,goal,width):
    law=future_law(a['native_counts'],a['end_flags'],0,28000,True)
    return float(sum(e['mass']*np.exp(-float(np.square(a['next_poses'][e['index'],:2]-goal).sum())/(2*width**2)) for e in law if TYPE[e['type']]==0))

def evaluate(p,rows,models):
    result={};rng=np.random.default_rng(p['evaluation_seed'])
    for side in ['ALL','LEFT','RIGHT']:
        pool=make_pool([r for r in rows if r['split']=='test' and r['arm']=='center' and (side=='ALL' or r['side']==side)])
        values={n:[] for n in models}
        with torch.no_grad():
            for _ in range(p['validation_batches']):
                data=batch(pool,rng,p['batch_size'])
                for name,q in models.items():values[name].append([float(v) for v in scores(q,data,name=='blind')[1:]])
        result[side]={name:dict(NCE=float(np.mean(v,axis=0)[0]),tie_aware_top1=float(np.mean(v,axis=0)[1])) for name,v in values.items()}
    lookup={r['id']:r for r in rows};controls=[]
    with torch.no_grad():
        for center in [r for r in rows if r['split']=='test' and r['arm']=='center']:
            group=[lookup[center['family']+'_'+arm] for arm in ['minus','center','plus']];a=[arrays(r['path']) for r in group];c=a[1]
            assert all(np.array_equal(z['states'][0],c['states'][0]) for z in a)
            length=len(c['states']);indices=[min(length-2,max(0,int(f*length))) for f in [.2,.5,.8]]
            assert len(set(indices))==3 and not c['end_flags'][indices].any(),'Insufficient nonterminal coordinate labels: no metric substitution'
            for k in indices:
                goal=c['next_poses'][k,:2];real=np.array([kernel_value(z,goal,p['kernel_width_m']) for z in a])
                x=torch.tensor(np.stack([z['states'][0] for z in a]));act=torch.tensor(np.stack([z['actions'][0] for z in a]));body=torch.tensor(relative(c['poses'][0],goal),dtype=torch.float32)[None,None].repeat(3,1,1)/10.;types=torch.zeros((3,1),dtype=torch.long)
                predicted={}
                for name,q in models.items():predicted[name]=q.logits(x,torch.zeros(3,2),act if name=='action' else torch.zeros_like(act),body,types)[:,0].numpy()
                controls.append(dict(family=center['family'],side=center['side'],query_endpoint_index=k,query_world=goal.tolist(),real_kernel_occupancy=real.tolist(),predicted={n:v.tolist() for n,v in predicted.items()}))
    control_metrics={}
    for side in ['LEFT','RIGHT']:
        selected=[r for r in controls if r['side']==side];modelscores={}
        for name in models:
            correct=0;pairs=0;choices=[];uniform=[]
            for r in selected:
                y=np.array(r['real_kernel_occupancy']);v=np.array(r['predicted'][name]);ties=v==v.max();choices.append(float(y[ties].mean()));uniform.append(float(y.mean()))
                for i,j in [(0,1),(0,2),(1,2)]:
                    if abs(y[i]-y[j])>p['value_margin']:
                        pairs+=1;correct+=int(np.sign(v[i]-v[j])==np.sign(y[i]-y[j]))
            modelscores[name]=dict(informative_pairs=pairs,correct_pairs=correct,sign_accuracy=correct/pairs if pairs else None,
                chosen_true_occupancy_mean=float(np.mean(choices)),tie_uniform_true_occupancy_mean=float(np.mean(uniform)))
        control_metrics[side]=modelscores
    gate_prediction=result['ALL']['action']['NCE']<=.95*result['ALL']['blind']['NCE'] and all(result[s]['action']['NCE']<=1.02*result[s]['blind']['NCE'] for s in ['LEFT','RIGHT'])
    gate_control=all(control_metrics[s]['action']['informative_pairs']>=6 and control_metrics[s]['action']['sign_accuracy']>=.9 and control_metrics[s]['action']['chosen_true_occupancy_mean']>control_metrics[s]['action']['tie_uniform_true_occupancy_mean'] for s in ['LEFT','RIGHT'])
    return dict(future_validation=result,control_metrics=control_metrics,prediction_gate=bool(gate_prediction),real_control_gate=bool(gate_control),
        combined_gate=bool(gate_prediction and gate_control),controls=controls,Actor_updates=0,
        interpretation='Finite endpoint smoothed occupancy ranking under fixed iid continuation; 4 paired noise streams per heldout reset, not a universal Q/safety certificate. No Actor update is authorized by this run.')
