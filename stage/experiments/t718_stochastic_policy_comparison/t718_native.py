"""Frozen neural recovery after exact learner prefix; diagnostics only."""
from pathlib import Path
import importlib.util,sys,os,gzip,json,time,fcntl
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t718_recovery_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
N=C.M.N;Base=N.N;F=Base.F;A=C.A;np,torch=A.np,A.torch
NORMAL=Base.C.E.payload

class Session(Base.C.Session):
    def __init__(self,job,*args,**kw):
        super().__init__(job,*args,**kw);self.first=job['recovery_first'];self.source=Base.C.E.N.S.arrays(job['source_path']);assert C.sha(job['source_path'])==job['source_sha256'];self.prefix_actions=[];self.suffix_sample=bool(self.q.sample)
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0 and actor_active
        k=len(self.rows);prefix=k<self.first;self.q.sample=self.suffix_sample and not prefix
        action=self.source['actions'][k] if prefix else np.zeros(2,np.float32);self.prefix_actions.append(action.copy())
        return F.FieldSession.step(self,action if prefix else None,owner=int(prefix),actor_active=True)

def payload(session,tape=None):
    a=NORMAL(session,tape);a.update(source_prefix_actions=np.asarray(session.prefix_actions[:len(a['states'])],np.float32),recovery_first=np.asarray(session.first,np.int64));return a

def worker(pipe,index,p,counts):
    Base.C.ACTIVE=p;real=Base.G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;Base.G.T.Session=Session;Base.G.B.native_setup=lambda folder,pc:Base.C.C.consistent_reset.setup(F,folder,pc,Base.C.C.CANONICAL_SETUP)
    Base.P.frozen=A.frozen;Base.C.E.payload=payload;Base.G.T.worker(pipe,index,p,counts)

def inspect(p,a,row):
    job=row['request']['job'];k=int(a['recovery_first']);assert k==job['recovery_first']>=0 and len(a['states'])>k
    assert C.sha(job['source_path'])==job['source_sha256'];b=Base.C.E.N.S.arrays(job['source_path'])
    for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][:k],b[key][:k]),('prefix',key)
    for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],b[key][k]),('handoff',key)
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
    assert not a['stochastic'][:k].any() and (a['stochastic'][k:].all() if row['seed'] is not None else not a['stochastic'][k:].any())
    assert a['ownership'][:k].all() and not a['ownership'][k:].any()
    assert np.array_equal(a['source_prefix_actions'][:k],b['actions'][:k]) and not a['source_prefix_actions'][k:].any()
    proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32);assert np.array_equal(a['actions'][k:],proposals[k:])
    assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    Base.P.frozen=A.frozen
    F.M.audit_full_history(p,dict(row,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
    assert all(A.digest(v)==row['array_hashes'][key] for key,v in a.items())
    assert int(a['native_counts'].sum())==row['physics']<=job['case']['step_cap']


def install():
    N.BASE_INSPECT=inspect
def collect(name):
    p=N.plan(name);folder=Path(p['out']);N.verify(p,force=True);N.setup(p);N.T.R.E.L.install_stop_handlers();m=N.model(p,p['model_name']);assert C.sha(m['path'])==m['sha256']
    lock=(C.ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rt=N.S.NativeRuntime(p);error=None;fresh=[]
    C.save(folder/'science_started.json',dict(pid=os.getpid(),epoch=time.time()))
    try:
        N.T.pool(rt);v=rt.request(p['replicated_jobs'],m,'stochastic',True,seeds=p['common_seeds']);fresh=[r for a,r in v]
        rt.request(p['replicated_jobs'],m,'exact_stochastic',True,fresh,seeds=p['common_seeds'])
    except BaseException:
        import traceback
        error=traceback.format_exc();C.save(folder/'failure.json',dict(error=error,raw_preserved=True,no_retry=True))
    finally:
        N.T.close(rt);N.verify(p,force=True)
        C.save(folder/'summary.json',dict(state='FAILED' if error else 'COMPLETE',error=error,pid=os.getpid(),model=m,physics=sum(rt.counts),commands=rt.commands+7,total=len(p['replicated_jobs']),safe=sum(N.T.Prior.safe(r) for r in fresh),collision=sum(r['collision'] for r in fresh),timeout=sum(not r['success'] and not r['collision'] for r in fresh),elapsed_seconds=time.time()-rt.start,new_learning=0,conditional_stochastic_recovery_only=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit(name):
    p=N.plan(name);folder=Path(p['out']);N.verify(p,force=True);N.setup(p);summary=C.read(folder/'summary.json');assert summary['state']=='COMPLETE' and not Path('/proc',str(summary['pid'])).exists()
    for g in C.read(folder/'worker_generations.json'):assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows};assert len(by)==len(rows)==108
    for r in rows:
        N.verify(p);assert r['full'] and C.sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==summary['model']['sha256'] and r['policy_hash']==summary['model']['policy_hash'];inspect(p,Base.C.E.N.S.arrays(r['path']),r)
        if r.get('exact'):
            previous=by[r['reference_path']];assert r['array_hashes']==previous['array_hashes'] and r['integration_sha256']==previous['integration_sha256']
    fresh=[r for r in rows if not r.get('exact')];assert len(fresh)==54
    assert [(r['job_id'],r['seed']) for r in fresh]==[(j['id'],s) for j,s in zip(p['replicated_jobs'],p['common_seeds'])]
    assert sum(r['physics'] for r in rows)==summary['physics']<=p['max_physics'];assert len(rows)+14==summary['commands']<=p['max_commands'];assert sum(N.T.Prior.safe(r) for r in fresh)==summary['safe']
    N.verify(p,force=True);C.save(folder/'audit.json',dict(state='PASS',rows=108,fresh=54,full_prefix_native_owncarry_density_reward_integral_and_sourceSHA=True,exact_repeats=True,seed_schedule_exact=True,new_native=0,new_learning=0))

def install():
    N.BASE_INSPECT=inspect;N.evaluate=collect;N.audit=audit;C.M.worker=worker
