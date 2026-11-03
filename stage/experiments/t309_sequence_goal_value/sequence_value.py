"""Fixed H=3 contrastive representation and single-action DQC pilot.

No native Actor updates. Observed future actions are not open-loop causal
interventions. DQC expectiles are not calibrated safe reach probabilities.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t309_sequence_goal_value';DOC=ROOT/'stage/docs/distance_20261103'
sys.path.insert(0,str(ROOT/'stage/experiments/t308_iid_native_goal_value'))
import t308_core as B
from t308_core import *
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t309_sequence_goal_value';DOC=ROOT/'stage/docs/distance_20261103'
NATIVE_KEYS=B.NATIVE_KEYS;TYPE=B.TYPE
import importlib.util

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert S.used(OUT)<=p['final_bytes'] and S.used(ROOT/'stage/runs')<=p['global_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']

class SequenceCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_action=torch.nn.Sequential(torch.nn.Linear(128,64),torch.nn.Tanh(),torch.nn.Linear(64,32))
        self.query=torch.nn.Sequential(torch.nn.Linear(6,32),torch.nn.Tanh(),torch.nn.Linear(32,32))
    def logits(self,x,actions,body,types):
        phi=self.state_action(torch.cat([x,torch.zeros(len(x),2),actions],-1))
        psi=self.query(torch.cat([body,torch.nn.functional.one_hot(types,4).to(x.dtype)],-1))
        return (phi[:,None,:]*psi).sum(-1)

def actions_for(a,kind):
    if kind=='blind':return torch.zeros_like(a)
    if kind in ['single','distilled']:
        out=torch.zeros_like(a);out[:,:2]=a[:,:2];return out
    assert kind=='sequence';return a

def logits(q,data,kind):
    x,a,b,t,eq,keep=data
    return q.logits(x,actions_for(a,kind),b,t)

def objective(q,data,kind):
    z=logits(q,data,kind);keep=data[-1];z=z[keep][:,keep];eq=data[-2][keep][:,keep]
    nce=(torch.logsumexp(z,1)-torch.logsumexp(z.masked_fill(~eq,-torch.inf),1)).mean()
    loss=nce+.01*torch.logsumexp(z,1).square().mean()
    tie=z==z.max(1,keepdim=True).values;acc=((tie&eq).sum(1)/tie.sum(1)).mean()
    return loss,nce,acc

def pool(rows):
    result=[]
    for r in rows:
        a=B.arrays(r['path']);n=len(a['states']);assert n>=3,'Too short to establish H3 contract'
        assert not a['end_flags'][:-1].any()
        laws=[]
        for k in range(n-2):
            law=future_law(a['native_counts'],a['end_flags'],k,28000,r['complete'])
            types=np.array([TYPE[e['type']] for e in law]);points=np.array([a['next_poses'][e['index'],:2] if TYPE[e['type']]==0 else [0.,0.] for e in law])
            laws.append((np.cumsum([e['mass'] for e in law]),types,points))
        result.append((r,a,laws))
    return result

def draw_batch(items,rng,n):
    xs=[];acts=[];poses=[];queries=[];types=[]
    for _ in range(n):
        r,a,laws=items[int(rng.integers(len(items)))];k=int(rng.integers(len(laws)));cdf,kind,points=laws[k];j=min(int(np.searchsorted(cdf,rng.random(),side='right')),len(cdf)-1)
        xs.append(a['states'][k]);acts.append(a['actions'][k:k+3].reshape(6));poses.append(a['poses'][k]);queries.append(points[j]);types.append(kind[j])
    poses,queries,types=np.asarray(poses),np.asarray(queries),np.asarray(types);body=query_matrix(poses,queries)/10.;body[:,types!=0]=0
    eq=(types[:,None]==types[None,:])&((types[:,None]!=0)|(queries[:,None]==queries[None,:]).all(-1));keep=np.flatnonzero(types!=3);assert len(keep)>1
    return (torch.tensor(np.asarray(xs)),torch.tensor(np.asarray(acts)),torch.tensor(body,dtype=torch.float32),torch.tensor(np.tile(types,(n,1))),torch.tensor(eq),torch.tensor(keep))

def fit(p,rows,guard,event,holder):
    items=pool([r for r in rows if r['split']=='train' and r['arm']=='center']);assert len(items)==32
    torch.manual_seed(p['seed']);prototype=SequenceCritic();initial=S.digest(prototype.state_dict())
    models={n:copy.deepcopy(prototype) for n in ['sequence','single','blind']};opts={n:torch.optim.Adam(q.parameters(),lr=p['lr']) for n,q in models.items()}
    holder.update(models=models,optimizers=opts,initial=initial);rng=np.random.default_rng(p['seed']);losses=np.empty((p['head_steps'],4),np.float64)
    for k in range(p['head_steps']):
        guard();data=draw_batch(items,rng,p['batch_size'])
        for col,name in enumerate(['sequence','single','blind']):
            opt=opts[name];opt.zero_grad();loss,_,_=objective(models[name],data,name);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(models[name].parameters(),5.,error_if_nonfinite=True);event(name,k);opt.step();losses[k,col]=float(loss.detach())
    teacher=models['sequence'];th=S.digest(teacher.state_dict());teacher.requires_grad_(False)
    student=copy.deepcopy(prototype);student.query.load_state_dict(teacher.query.state_dict());student.query.requires_grad_(False)
    models['distilled']=student;opts['distilled']=torch.optim.Adam(student.state_action.parameters(),lr=p['lr']);rng=np.random.default_rng(p['distillation_seed'])
    for k in range(p['head_steps']):
        guard();data=draw_batch(items,rng,p['batch_size']);keep=data[-1]
        with torch.no_grad():target=logits(teacher,data,'sequence')[keep][:,keep]
        pred=logits(student,data,'distilled')[keep][:,keep];delta=target-pred;weight=torch.where(delta>=0,p['expectile'],1-p['expectile']);loss=(weight*delta.square()).mean();assert torch.isfinite(loss)
        opt=opts['distilled'];opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.state_action.parameters(),5.,error_if_nonfinite=True);event('distilled',k);opt.step();losses[k,3]=float(loss.detach())
    assert S.digest(teacher.state_dict())==th
    assert S.digest(student.query.state_dict())==S.digest(teacher.query.state_dict())
    return models,opts,losses,initial

def compact(a):
    return dict(native_counts=a['native_counts'],end_flags=a['end_flags'],next_poses=a['next_poses'],
        actions=a['actions'],innovations=a['innovations'],environment_rewards=a['environment_rewards'],
        first_state=a['states'][0],first_pose=a['poses'][0])

def control_arrays(r):
    if r['arm']=='center':return compact(B.arrays(r['path']))
    with np.load(r['path']) as z:return {k:z[k].copy() for k in z.files}

def evaluate(p,rows,models):
    validation=[r for r in rows if r['split']=='test' and r['arm']=='center'];assert len(validation)==32
    rng=np.random.default_rng(p['evaluation_seed']);metrics={}
    with torch.no_grad():
        for side in ['ALL','LEFT','RIGHT']:
            items=pool([r for r in validation if side=='ALL' or r['side']==side]);values={n:[] for n in models}
            for _ in range(p['validation_batches']):
                data=draw_batch(items,rng,p['batch_size'])
                for name,q in models.items():values[name].append([float(v) for v in objective(q,data,name)[1:]])
            metrics[side]={n:dict(NCE=float(np.mean(v,axis=0)[0]),tie_aware_top1=float(np.mean(v,axis=0)[1])) for n,v in values.items()}
    by={r['id']:r for r in rows};controls=[]
    with torch.no_grad():
        for r in validation:
            group=[by[r['family']+'_'+arm] for arm in ['minus','center','plus']];a=[control_arrays(item) for item in group];goal=np.array(r['goal_xy']);real=np.array([B.kernel_value(v,goal,p['kernel_width_m']) for v in a]);pred={}
            assert all(np.array_equal(v['first_state'],a[1]['first_state']) for v in a)
            # Canonical batch1 avoids numerical ordering of identical blind rows.
            for name in ['single','blind','distilled']:
                values=[]
                for v in a:
                    x=torch.tensor(v['first_state'][None]);act=torch.zeros(1,6);act[:,:2]=torch.tensor(v['actions'][:1]);body=torch.tensor(relative(v['first_pose'],goal),dtype=torch.float32)[None,None]/10.;t=torch.zeros((1,1),dtype=torch.long)
                    values.append(float(models[name].logits(x,actions_for(act,name),body,t)[0,0]))
                pred[name]=values
            assert np.ptp(pred['blind'])==0.
            controls.append(dict(family=r['family'],side=r['side'],distance_band=r['distance_band'],goal_fixed_before_collection=True,real_kernel_occupancy=real.tolist(),predicted=pred,
                native_outcomes=[dict(arm=z['arm'],safe=bool(z['success'] and not z['collision']),contact=z['collision'],timeout=z['timeout']) for z in group]))
    cm={}
    for side in ['LEFT','RIGHT']:
        cm[side]={}
        for band in p['distance_bands']+['ALL']:
            selected=[r for r in controls if r['side']==side and (band=='ALL' or r['distance_band']==band)];scores={}
            for name in ['single','blind','distilled']:
                correct=0;n=0;choice=[];uniform=[];family_means=[]
                for r in selected:
                    y=np.array(r['real_kernel_occupancy']);v=np.array(r['predicted'][name]);choice.append(float(y[v==v.max()].mean()));uniform.append(float(y.mean()));family_means.append(float(y.max()-y.min()))
                    for i,j in [(0,1),(0,2),(1,2)]:
                        if abs(y[i]-y[j])>p['value_margin']:n+=1;correct+=int(np.sign(v[i]-v[j])==np.sign(y[i]-y[j]))
                scores[name]=dict(informative_pairs=n,correct_pairs=correct,sign_accuracy=correct/n if n else None,
                    chosen_true_mean=float(np.mean(choice)),uniform_true_mean=float(np.mean(uniform)),independent_noise_streams=len(selected),
                    empirical_control_range_per_noise=family_means)
            cm[side][str(band)]=scores
    representation=metrics['ALL']['sequence']['NCE']<=.95*metrics['ALL']['blind']['NCE'] and metrics['ALL']['sequence']['NCE']<=.98*metrics['ALL']['single']['NCE'] and all(metrics[s]['sequence']['NCE']<=1.02*metrics[s]['blind']['NCE'] for s in ['LEFT','RIGHT'])
    control=all(cm[s]['ALL']['distilled']['informative_pairs']>=6 and cm[s]['ALL']['distilled']['sign_accuracy']>=.9 and cm[s]['ALL']['distilled']['chosen_true_mean']>cm[s]['ALL']['distilled']['uniform_true_mean'] for s in ['LEFT','RIGHT'])
    validated=[]
    for side in ['LEFT','RIGHT']:
        for band in p['distance_bands']:
            z=cm[side][str(band)]['distilled']
            if z['informative_pairs']>=6 and z['sign_accuracy']>=.9 and z['chosen_true_mean']>z['uniform_true_mean']:validated.append(dict(side=side,distance_band=band))
    return dict(future_validation=metrics,control_metrics=cm,controls=controls,representation_gate=bool(representation),single_action_control_gate=bool(control),combined_gate=bool(representation and control),validated_goal_groups=validated,
        Actor_updates=0,six_metre_gate_evaluated=False,automatic_next_training=False,
        limitations='Four paired noise streams per fixed validation task; pairs correlated, no independent-pair CI. Actual fixed goal kernels, not data-selected center future queries or calibrated first-hit probabilities. Per-distance missing control information cannot be hidden by ALL averages. Observed chunks not open-loop interventions; DQC expectile is approximate improved-continuation projection.')

def worker(pipe,index,p,counts):
    B.OUT=OUT
    folder=OUT/'workers'/str(index);folder.mkdir(parents=True);log=LimitedLog(folder/'worker.log');sys.stdout=sys.stderr=log;witness=None;total=0
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);native,create=S.native_setup(folder);q=B.actor(p)
        spec=importlib.util.spec_from_file_location('t309_history',ROOT/'stage/experiments/t295_hindsight_support/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=module.Witness(Base);original=R.task_env
        def factory(create,job):
            env,obs,actual=original(create,job);before=R.arrays_hash(R.integration(env));obs=witness.alternate(env,job['active_goal_xy'],witness.get(env));assert before==R.arrays_hash(R.integration(env));return env,obs,actual
        R.task_env=factory
        pipe.send(dict(ready=True,pid=os.getpid(),policy_hash=S.digest(q.state_dict()),CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            j=pipe.recv()
            if j is None:break
            item=B.generate(j,q,native,create,witness,j['deadline_epoch'],counts,index,total);total+=item['row']['physics_steps'];item['row'].update(distance_band=j['distance_band']);pipe.send(item)
    except EOFError:pass
    except BaseException:
        failure=dict(error=traceback.format_exc(),completed_worker_physics=total);C.save(folder/'failure.json',failure)
        try:pipe.send(failure)
        except OSError:pass
    finally:
        if witness is not None:witness.close()
        C.save(folder/'log_accounting.json',dict(bytes_observed=log.total,bytes_written=log.written));pipe.close();log.close()
