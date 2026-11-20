"""Fixed first-centimetre student occupancy, no takeover-time search."""
from pathlib import Path
import ast,copy,os,sys,time,types,json
import t668_runtime as S
from status_io import save
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,N,read,sha=S.T,S.N,S.read,S.sha


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    src=ROOT/'stage/runs/t666_failure_full_history';gate=ROOT/'stage/runs/t667_causal_response_gate'
    p=copy.deepcopy(read(src/'protocol.json'));p.pop('registration_epoch',None)
    for folder in [src,gate]:
        assert read(folder/'FINAL_REPORT.json')['all_scientific_processes_stopped']
        assert read(folder/'audit.json')['state']=='PASS'
        assert all(not Path('/proc',str(pid)).exists() for pid in read(folder/'SEALED.json')['scientific_pids'])
        assert not Path('/proc',str(read(folder/'DRIVER_STARTED.json')['pid'])).exists()
    assert read(gate/'RESULTS.json')['state']=='SCIENTIFIC_GATE_FAILED'
    g=read(gate/'RESULTS.json');original=read(src/'summary.json')['rows'];jobs=[]
    for pair in p['pairs']:
        departure=next(x for x in g['pairs'] if x['id']==pair['id'])['first_cm_index'];assert departure is not None and departure>0
        row=next(r for r in original if r['role']==pair['id']+'_candidate')
        assert sha(row['path'])==row['sha256']
        a=T.R.E.N.S.arrays(row['path']);assert departure<len(a['states']) and not a['end_flags'][:departure].any()
        jobs.append(dict(pair['job'],source_path=row['path'],source_sha256=row['sha256'],recovery_first=departure))
    now=time.time();auth=read(DOC.parent/'AUTHORIZATION.json')
    p.update(id='T668_FOUR_ACTUAL_DEPARTURE_T501_RECOVERY',authorization_approved=False,out=str(OUT),arm='actual_departure_recovery',
        max_physics=186672,max_commands=14,max_bytes=64*2**20,child_bytes=64*2**20,main_seconds=1200,audit_seconds=600,
        absolute_learning_stop=min(now+1200,auth['absolute_learning_stop']),absolute_native_stop=min(now+1200,auth['absolute_native_stop']),
        absolute_all_stop=min(now+3000,auth['absolute_all_stop']),recovery_jobs=jobs,raw_sigma=[.2,.2],new_Actor=0,new_Value=0,new_optimizer=0)
    assert 2*sum(j['case']['step_cap'] for j in jobs)==p['max_physics']
    T.torch.set_num_threads(1);os.sched_setaffinity(0,{14});S.setup(p)
    assert T.R.N.Pool.__init__.__globals__['worker'] is N.worker
    q=N.L.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash']
    # Exercise the actual source-prefix Session.step dispatch without native.
    original_step=N.F.FieldSession.step;captured=[]
    def capture(self,forced=None,owner=0,actor_active=True):
        captured.append(dict(forced=forced,owner=owner,actor_active=actor_active));return None
    try:
        N.F.FieldSession.step=capture
        for job in jobs:
            a=T.R.E.N.S.arrays(job['source_path']);obj=N.Session.__new__(N.Session)
            obj.first=job['recovery_first'];obj.source=a;obj.prefix_actions=[];obj.q=types.SimpleNamespace(sample=True)
            obj.rows=[None]*(obj.first-1);obj.step();assert captured[-1]['owner']==1 and captured[-1]['actor_active']
            assert S.np.array_equal(captured[-1]['forced'],a['actions'][obj.first-1])
            obj.rows=[None]*obj.first;obj.step();assert captured[-1]['owner']==0 and captured[-1]['forced'] is None and not obj.q.sample
    finally:N.F.FieldSession.step=original_step
    res=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<10*2**30
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all_source_science_exited=True,
        four_first_cm_indices_fixed=[j['recovery_first'] for j in jobs],actual_forced_prefix_to_own_teacher_step_dispatch_checked=True,
        canonical_worker_installed=True,frozen_T501_loader_exact=True,remaining_original_horizon_not_reset=True,
        max_all8_physics=186672,max_commands=14,new_native=0,new_optimizer=0,resources=res)
    save(DOC/'PREFLIGHT.json',pre)
    dependencies={Path(f) for f in p['inputs']}
    for f,h in p['inputs'].items():assert sha(f)==h,f
    dependencies|=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',gate/'RESULTS.json',gate/'audit.json',DOC.parent/'AUTHORIZATION.json'}
    for job in jobs:dependencies.add(Path(job['source_path']))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):dependencies.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(dependencies)}
    save(HERE/'protocol.json',p);save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','new_optimizer']})
    print(json.dumps(pre))


if __name__=='__main__':main()
