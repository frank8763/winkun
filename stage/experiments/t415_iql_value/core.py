"""Finite-horizon, physical-time IQL value pilot; Actor remains frozen."""
from pathlib import Path
import sys,os,json,time,copy,io,gzip,hashlib,traceback,fcntl,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t415_iql_value';DOC=ROOT/'stage/docs/day_20261107'
sys.path.insert(0,str(ROOT/'stage/experiments/t414_dependency_recovery'))
import t414_learning as E
torch,np=E.torch,E.np
read,save,sha,digest=E.read,E.save,E.sha,E.digest

def verify(p):
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert E.N.K.S.used(OUT)<=p['own_bytes'] and E.N.K.S.used(ROOT/'stage/runs')<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30

def data(p):
    base=E.N.science(p['actor_protocol']);actor=E.safe_initial(base,p['actor_protocol']).eval().requires_grad_(False)
    result={'train':[],'validation':[]};checks=[];seen={}
    for source in p['sources']:
        a=E.N.S.arrays(source['path']);n=a['native_counts'].astype(np.int64);flags=a['end_flags'].astype(bool)
        assert a['states'].shape==a['next_states'].shape==(len(n),120)
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert flags[-1].any() and not flags[:-1].any() and np.all((n>0)&(n<=100))
        assert not np.any(flags.sum(1)>1) and np.isfinite(a['actions']).all() and np.abs(a['actions']).max()<=1
        # Existing causal remaining-time field, true80s task deadline. No new
        # critic-only future/pose/goal-truth information enters the feature.
        elapsed=np.r_[0,np.cumsum(n)];expected=(40000-elapsed)/40000
        assert np.max(np.abs(a['states'][:,112]-expected[:-1]))<2e-6
        assert np.max(np.abs(a['next_states'][:,112]-expected[1:]))<2e-6
        goal=np.asarray(a.get('task_goal_xy',source.get('goal_xy')),dtype=np.float64);assert goal.shape==(2,)
        d0=np.linalg.norm(a['poses'][:,:2]-goal,axis=1)
        d1=np.linalg.norm(a['next_poses'][:,:2]-goal,axis=1)
        gamma=E.N.K.R.GAMMA;disc=gamma**n;terminal=flags.any(1)
        reward=-.000002*(1-disc)/(1-gamma)+.03*d0-disc*.03*d1*(~terminal)
        reward+=gamma**(n-1)*(flags[:,1].astype(float)-flags[:,0]-.5*flags[:,2])
        assert np.max(np.abs(reward-a['rewards']))<2e-12
        uid=digest(dict({k:a[k] for k in ['states','next_states','actions','rewards','native_counts','end_flags']},actual_task_goal=goal))
        if uid in seen:
            assert seen[uid]==source['split'],'Train/validation exact trajectory leakage'
            checks.append(dict(source=source,exact_duplicate_skipped=True,uid=uid));continue
        seen[uid]=source['split'];x=torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]]))
        with torch.no_grad():
            normalized=actor.normalized(x);hidden=actor.core(actor.encoder(normalized)[:,None])[0][:,0]
            features=torch.cat([normalized,hidden],1).float()
        returns=np.empty(len(n),np.float64);value=0.
        for i in range(len(n)-1,-1,-1):value=float(reward[i])+float(disc[i])*(not terminal[i])*value;returns[i]=value
        brute=float(np.dot(gamma**elapsed[:-1],reward));assert abs(brute-returns[0])<2e-12
        row=dict(x=features[:-1],nx=features[1:],a=torch.tensor(a['actions']).float(),r=torch.tensor(reward).float()[:,None],
            discount=torch.tensor(disc*(~terminal)).float()[:,None],returns=torch.tensor(returns).float()[:,None],
            first=len(result[source['split']]),success=bool(flags[-1,1]),uid=uid)
        result[source['split']].append(row);checks.append(dict(source=source,uid=uid,decisions=len(n),GOAL=bool(flags[-1,1]),CONTACT=bool(flags[-1,0]),TIMEOUT=bool(flags[-1,2]),G0=returns[0],native_counts_sha=digest(n),features_sha=digest(features)))
    for split in result:
        rows=result[split];assert rows
        result[split]={'episodes':rows,**{k:torch.cat([r[k] for r in rows]) for k in ['x','nx','a','r','discount','returns']}}
    assert actor.phash()==p['frozen_actor_policy_hash']
    return result,checks

class Values(torch.nn.Module):
    def __init__(self,p):
        super().__init__()
        def net(n):return torch.nn.Sequential(torch.nn.Linear(n,128),torch.nn.LayerNorm(128),torch.nn.ReLU(),torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Linear(128,1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['seed']);self.q1=net(138);self.q2=net(138);self.v=net(136)
    def qs(self,x,a):
        z=torch.cat([x,a],1);return self.q1(z),self.q2(z)

def state(model,target,qo,vo):return dict(model=copy.deepcopy(model.state_dict()),target=copy.deepcopy(target.state_dict()),Q_Adam=copy.deepcopy(qo.state_dict()),V_Adam=copy.deepcopy(vo.state_dict()))
def metrics(model,target,d,p):
    with torch.no_grad():
        q=torch.minimum(*target.qs(d['x'],d['a']));v=model.v(d['x']);bellman=d['r']+d['discount']*model.v(d['nx'])
        w=torch.exp(p['inverse_temperature']*(q-v)).clamp_max(p['weight_cap'])
        predictions=[];off=0
        for episode in d['episodes']:
            predictions.append(dict(uid=episode['uid'],GOAL=episode['success'],Q0=float(q[off]),MC_G0=float(episode['returns'][0])));off+=len(episode['x'])
        return dict(decisions=len(q),behavior_MC_MSE=float((q-d['returns']).square().mean()),bellman_MSE=float(((model.qs(d['x'],d['a'])[0]-bellman).square()+(model.qs(d['x'],d['a'])[1]-bellman).square()).mean()/2),
            weights=dict(min=float(w.min()),max=float(w.max()),mean=float(w.mean()),std=float(w.std(unbiased=False)),ESS_fraction=float(w.sum().square()/(len(w)*w.square().sum())),cap_fraction=float((w>=p['weight_cap']).float().mean())),episodes=predictions,
            limitation='Q is IQL implicit-policy value; behavior MC and in-dataset Bellman errors are diagnostics, not action-ranking, causal gain or deployment safety certificates.')

def fit(p,d,guard,event):
    model=Values(p);target=copy.deepcopy(model).requires_grad_(False)
    qo=torch.optim.Adam([*model.q1.parameters(),*model.q2.parameters()],lr=p['lr']);vo=torch.optim.Adam(model.v.parameters(),lr=p['lr'])
    rng=np.random.default_rng(p['seed']);train=d['train'];trace=[];initial={k:metrics(model,target,v,p) for k,v in d.items()}
    for k in range(p['steps']):
        guard();ids=torch.tensor(rng.integers(len(train['x']),size=p['batch_size']));x,a=train['x'][ids],train['a'][ids]
        with torch.no_grad():qmin=torch.minimum(*target.qs(x,a))
        delta=qmin-model.v(x);weight=torch.where(delta>=0,p['expectile'],1-p['expectile']);vloss=(weight*delta.square()).mean()
        assert torch.isfinite(vloss);vo.zero_grad(set_to_none=True);vloss.backward();torch.nn.utils.clip_grad_norm_(model.v.parameters(),10.,error_if_nonfinite=True)
        event('V','BEFORE',k);vo.step();event('V','AFTER',k)
        with torch.no_grad():bellman=train['r'][ids]+train['discount'][ids]*model.v(train['nx'][ids])
        q1,q2=model.qs(x,a);qloss=(q1-bellman).square().mean()+(q2-bellman).square().mean()
        assert torch.isfinite(qloss);qo.zero_grad(set_to_none=True);qloss.backward();torch.nn.utils.clip_grad_norm_([*model.q1.parameters(),*model.q2.parameters()],10.,error_if_nonfinite=True)
        event('Q','BEFORE',k);qo.step();event('Q','AFTER',k)
        with torch.no_grad():
            for name in ['q1','q2']:
                for dest,src in zip(getattr(target,name).parameters(),getattr(model,name).parameters()):dest.lerp_(src,p['target_tau'])
        if k%10==0 or k==p['steps']-1:trace.append([k,float(qloss.detach()),float(vloss.detach())])
    final={k:metrics(model,target,v,p) for k,v in d.items()};saved=state(model,target,qo,vo)
    return dict(initial=initial,final=final,trace=trace,state_hashes={k:digest(v) for k,v in saved.items()},Q_updates=p['steps'],V_updates=p['steps'],Actor_updates=0),saved

def run():
    p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    counts=dict(Q=0,V=0);save(OUT/'main_started.json',dict(pid=os.getpid(),epoch=start));result=None;error=None
    def guard():
        assert time.time()<min(start+p['main_seconds']-10,p['learning_stop_absolute'])
        if (counts['Q']+counts['V'])%200==0:
            verify(p);save(OUT/'status.json',dict(state='IQL_VALUE_FIT',pid=os.getpid(),epoch=time.time(),updates=counts,Actor_updates=0,native=0))
    def event(head,where,k):
        with (OUT/'optimizer_calls.jsonl').open('a') as stream:stream.write(json.dumps([head,where,k])+'\n')
        if where=='AFTER':counts[head]+=1
    try:
        d,checks=data(p);save(OUT/'data_contract.json',dict(state='PASS',sources=checks,frozen_actor_hash=p['frozen_actor_policy_hash'],train_decisions=len(d['train']['x']),validation_decisions=len(d['validation']['x']),no_unexecuted_teacher_queries_as_transitions=True,no_native_replays_added=True))
        result,saved=fit(p,d,guard,event);buf=io.BytesIO();torch.save(saved,buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<p['model_bytes'];(OUT/'values.pt.gz').write_bytes(blob)
        save(OUT/'metrics.json',result);result=dict(result,model_sha256=sha(OUT/'values.pt.gz'))
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,updates=counts,native=0,Actor_updates=0))
    finally:
        summary=dict(state='COMPLETE' if error is None else 'FAILED',pid=os.getpid(),elapsed_s=time.time()-start,native=0,Actor_updates=0,updates=counts,result=result,error=error,not_six_metre_acceptance=True,seven_eight_entered=False)
        save(OUT/'summary.json',summary);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None;print('T415_VALUE_COMPLETE_NO_ACTOR',flush=True)

def audit():
    p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.time();d,checks=data(p);assert read(OUT/'data_contract.json')['sources']==checks;events=[]
    def guard():assert time.time()<start+p['audit_seconds']-5
    result,saved=fit(p,d,guard,lambda h,w,k:events.append([h,w,k]))
    assert result==read(OUT/'metrics.json');assert events==[json.loads(line) for line in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
    stored=torch.load(io.BytesIO(gzip.decompress((OUT/'values.pt.gz').read_bytes())),weights_only=False,map_location='cpu')
    assert {k:digest(v) for k,v in stored.items()}==result['state_hashes']=={k:digest(v) for k,v in saved.items()}
    verify(p);save(OUT/'audit.json',dict(state='PASS',elapsed_s=time.time()-start,new_native=0,new_Actor=0,historical_discarded_Q_Adam=p['steps'],historical_discarded_V_Adam=p['steps'],all_values_target_Adam_metrics_and_event_order_exact=True,all_consumed_input_SHA_and_physical_reward_time_contract=True,no_six_stability_claim=True))
    print('T415_AUDIT_PASS_NO_NATIVE_NO_ACTOR',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
