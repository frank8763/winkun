"""Guarded actual Session.step/worker-entry fixtures without native stepping."""
from pathlib import Path
import ast,copy,importlib.util,json,os,sys,time,types
import t684_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,N,M,torch,np=S.T,S.N,S.M,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def entry_fixture(p,row):
    a=T.R.E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256']
    job=dict(row['request']['job'],temporal_tau=2.);q=N.frozen(p,p['initial']['path']);N.configure(q,2.)
    q.sample=True;q.rng=np.random.default_rng([68420261121,0,0]);q.reset_noise();before=digest(q.state_dict())
    # The exact native Session.step runs on a recorded state until its guard
    # aborts before native.step_checked. This is an interface fixture, never a
    # fresh environment rollout or evidence for newly reconstructed physics.
    calls=[0];session=N.Session.__new__(N.Session)
    session.q=q;session.rows=[];session.pn=0;session.steps=0;session.cap=90000
    session.hidden=torch.zeros((1,1,64),dtype=torch.float64);session.goal=a['task_goal_xy'].copy()
    session.packet=lambda:a['states'][0].copy();session.done=lambda:False
    session.env=types.SimpleNamespace(_pose=lambda:a['poses'][0].copy(),_goal_state=lambda:(None,float(np.linalg.norm(a['poses'][0,:2]-a['task_goal_xy']))))
    session.w=types.SimpleNamespace(get=lambda unused:(a['history_poses'][0],a['history_goals'][0],a['history_times'][0]))
    def forbidden(*args):calls[0]+=1;raise AssertionError('Native step reached during zero-step fixture')
    session.native=types.SimpleNamespace(step_checked=forbidden);session.physics=forbidden
    class Stop(Exception):pass
    def guard(physical=False):
        if physical:raise Stop()
    session.guard=guard
    for previous in [None,100]:
        if previous is not None:session.rows=[(None,None,previous)];session.pn=previous
        try:session.step();raise AssertionError('Missing physical pre-step refusal')
        except Stop:pass
        assert q.temporal_count==previous
    assert q.tape[0]['temporal_rho']==0 and float(q.tape[1]['temporal_rho'])==np.exp(-.1)
    assert calls[0]==0 and digest(q.state_dict())==before
    # Invoke actual worker entry with only the loop substituted by a binding
    # witness, so startup/reset closures and the canonical factory stay unrun.
    original_worker=N.G.T.worker;original_affinity=os.sched_setaffinity;original_session=N.G.T.Session
    original_setup=N.G.B.native_setup;original_loader=N.P.frozen;oldactive=N.B.ACTIVE
    marker=[]
    def witness(pipe,index,pp,counts):
        assert N.G.T.Session is N.Session and N.P.frozen is N.frozen and N.B.ACTIVE is pp
        marker.append(True)
    try:
        N.G.T.worker=witness;N.worker(None,0,p,None)
    finally:
        N.G.T.worker=original_worker;N.G.T.Session=original_session;N.G.B.native_setup=original_setup
        N.P.frozen=original_loader;N.B.ACTIVE=oldactive;os.sched_setaffinity=original_affinity
    assert marker==[True]
    return dict(actual_Session_step_guarded_before_native=True,actual_previous_hold_clock_routed=True,
        actual_worker_entry_binding_with_loop_witness=True,no_native_loop_or_new_scene_claim=True,physics_calls=calls[0])

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    preceding=read(HERE/'protocol.json')['preflight_seconds'] if (HERE/'protocol.json').exists() else 0
    last=ROOT/'stage/runs/t683_temporal_density_contract';report=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert report['state']=='COMPLETE' and report['gate']=='TEMPORAL_LATENT_CONTRACT_PASS' and report['audit']['state']=='PASS'
    ids=seal['scientific_pids']+[report['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    source=ROOT/'stage/runs/t679_episode_parameter_exploration';prior=read(source/'protocol.json');p=copy.deepcopy(prior);p.pop('registration_epoch',None)
    for key in ['calibration_rows','validation_rows','screen_jobs','key_jobs','training_jobs','field_jobs','coverage_jobs','paired_jobs','pending_completion',
                'scope_reference','sentinel_jobs','reconstruction_jobs','precheck_jobs','replacement_jobs']:
        p.pop(key,None)
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p.update(id='T684_MATCHED_TEMPORAL_NOISE_COLLECTION',authorization_approved=False,out=str(OUT),arm='temporal_noise_collection',
        max_physics=16*sum(j['case']['step_cap'] for j in prior['collection_jobs']),max_commands=134,max_bytes=320*2**20,child_bytes=320*2**20,
        main_seconds=3000,audit_seconds=300,absolute_learning_stop=min(now+3000,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+3000,auth['absolute_native_stop']),absolute_all_stop=min(now+3600,auth['absolute_all_stop']),
        finalization_seconds=60,new_Actor=0,new_Value=0,new_optimizer=0)
    assert p['max_physics']==4969728 and 64*p['full_bytes']+64*p['compact_bytes']+16*2**20<p['max_bytes']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);S.setup(p)
    hist=prior['calibration_rows'];base=N.BASE_FROZEN(p,p['initial']['path']);basehash=digest(base.state_dict());count=0
    for row in hist:
        a=T.R.E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256'];n=min(64,len(a['states']));states=a['states'][:n];counts=a['native_counts'][:n]
        for tau in [0.,2.]:
            seed=[68420261121,0,0];q=N.frozen(p,p['initial']['path']);N.configure(q,tau);q.sample=True;q.rng=np.random.default_rng(seed);q.reset_noise()
            hidden=None
            for i,x in enumerate(states):
                q.temporal_count=None if i==0 else int(counts[i-1])
                with torch.no_grad():action,loc,hidden,unused=q.step(torch.tensor(x[None]),hidden)
            tape=copy.deepcopy(q.tape);reference=N.Contract.trajectory(base,states,counts,tau,seed)
            for key,ref in [('raw_actions','raw'),('locations','mu'),('innovations','epsilon'),('noise_latent','noise'),('logp','latent_logp'),('conditional_clip_logp','conditional_clip_logp')]:
                assert np.array_equal(np.stack([v[key] for v in tape]),reference[ref]),key
            assert q.phash()==N.behavior(p,tau) and digest(q.state_dict())==basehash
            # Reset after use must reproduce both RNG/latent state from h0.
            q.reset_noise();q.rng=np.random.default_rng(seed);q.temporal_count=None
            with torch.no_grad():q.step(torch.tensor(states[0])[None],None)
            assert all(np.array_equal(v,q.tape[0][k]) for k,v in tape[0].items())
            count+=1
    fixture=entry_fixture(p,hist[0]);S.setup(p)
    iidjobs,seeds=S.schedule(p,0.);arjobs,arseeds=S.schedule(p,2.)
    assert seeds==arseeds and len(seeds)==32 and len(set(map(tuple,seeds)))==32
    assert all(dict((k,v) for k,v in a.items() if k!='temporal_tau')==dict((k,v) for k,v in b.items() if k!='temporal_tau') for a,b in zip(iidjobs,arjobs))
    resources=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',real_frozen_loader_no_parameter_change=True,all8external64prefixes_both_tau_match_T683=count==16,
        noise_and_GRU_reset_without_leakage=True,behavior_hash_distinguishes_tau=True,paired32_innovation_streams=True,
        canonical_pool_worker_binding=True,clock_entry_fixture=fixture,native_scene_first_action_contract_checked_at_registered_run=True,
        no_new_environment_data_claim=True,new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T684_PRIMARY_REVIEW.md'}
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json','RESULTS.json']}
    deps|={Path(v['path']) for v in hist}|{source/'protocol.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+preceding
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
