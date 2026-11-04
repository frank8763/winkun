"""Finite independent FAILED-run evidence review, never restarts navigation."""
from canonical import CanonicalActor,numerical_worker
from t338_core import *
import multiprocessing as mp,signal,base64
CODE=Path(__file__).resolve().parent;REVIEW=ROOT/'stage/runs/t338_failure_review'

def prepare():
    assert not REVIEW.exists()
    old=read(OUT/'protocol.json');files=set(old['input_hashes'])
    files|={str(f) for f in CODE.glob('*.py')}
    files|={str(f) for f in OUT.rglob('*') if f.is_file()}
    files|={str(DOC/n) for n in ['T338_DRIVER_ERROR.json','T338_driver.log','T338_audit.log','T338_backup.log']}
    p=dict(scope='FAILED_T338_READONLY_PARTIAL_EVIDENCE_AND_UNTRAINED_NUMERIC_PROTOTYPE',wall_seconds=300,
        final_bytes=128*1024,global_bytes=1<<30,min_free_bytes=40*1024**3,
        geometry_commands=0,physics_steps=0,optimizer_calls=0,training_restarts=0,
        input_hashes={f:K.C.sha(f) for f in sorted(files)})
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes']
    REVIEW.mkdir();save(CODE/'protocol.json',p);save(REVIEW/'protocol.json',p)
    print('Registered read-only failed-run review:0native/0Adam/128KiB/300s')

def main():
    start=time.time();p=read(CODE/'protocol.json');assert not (REVIEW/'summary.json').exists()
    assert (CODE/'protocol.json').read_bytes()==(REVIEW/'protocol.json').read_bytes()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Review300s')));signal.alarm(295)
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    original=read(OUT/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='FAILED' and s['all_workers_stopped']
    pids=[read(OUT/'launch.json')['pid'],read(OUT/'launch.json')['orchestrator_pid'],s['pid'],*read(OUT/'workers.json')['workers'],read(DOC/'T338_WRAPPER.json')['pid']]
    assert all(not Path('/proc/'+str(pid)).exists() for pid in pids)
    cases=[json.loads(v) for v in (OUT/'cases.jsonl').read_text().splitlines()]
    reps=[json.loads(v) for v in (OUT/'replays.jsonl').read_text().splitlines()]
    assert len(cases)+len(reps)+1==s['commands']==288
    assert sum(v['physics_steps'] for v in [*cases,*reps])==s['physics_steps']==4178976
    assert not list(OUT.glob('workers/*/partial.npz')) and not (OUT/'optimizer_calls.jsonl').exists()
    assert s['updates']==dict(live_actor=0,shadow_actor=0)
    reference=arrays(original['reference']);training=[]
    for row in cases:
        assert K.C.sha(row['path'])==row['sha256'];a=arrays(row['path'])
        assert row['complete'] and not row.get('error')
        for key,v in a.items():
            if key in row['array_hashes']:assert digest(v)==row['array_hashes'][key]
        if row['full_arrays_saved']:assert {k:digest(v) for k,v in a.items()}==row['array_hashes']
        flags=a['end_flags'];assert not flags[:-1].any() and flags[-1].any()
        assert bool(flags[-1,0])==row['collision'] and bool(flags[-1,1])==row['success'] and bool(flags[-1,2])==row['timeout']
        assert int(a['native_counts'].astype(np.int64).sum())==row.get('suffix_steps',row['physics_steps'])
        if row['role']!='training':continue
        assert row['prefix_steps']+row['suffix_steps']==row['physics_steps']
        assert np.array_equal(a['states'][0],reference['states'][row['prefix']])
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        assert np.array_equal(a['initial_native_integration'],arrays(original['prefix_native_references'][str(row['prefix'])])['initial_native_integration'])
        assert K.R.arrays_hash(a['final_native_integration'])==row['integration_sha256']
        for k,x in enumerate(a['states']):
            assert np.max(np.abs(K.H.rebuild(x,a['poses'][k],a['history_poses'][k],row['goal_xy'],a['history_goals'][k])-x))<=2e-6
            d0=K.H.fields(a['poses'][k],row['goal_xy'])[1];d1=K.H.fields(a['next_poses'][k],row['goal_xy'])[1]
            n=int(a['native_counts'][k]);c,g,t=map(bool,a['end_flags'][k]);z=K.R.GAMMA**n
            r=-.000002*(1-z)/(1-K.R.GAMMA)+.03*d0-z*.03*d1*(not(c or g or t))+(K.R.GAMMA**(n-1)*(float(g)-float(c)-.5*float(t)) if c or g or t else 0.)
            assert abs(r-a['rewards'][k])<1e-12 and r==a['environment_rewards'][k]
        assert np.array_equal(np.random.default_rng(row['seed']).normal(size=(len(a['states']),2)),a['innovations'])
        assert len([r for r in reps if r['id']==row['id'] and r['full_arrays_native_exact']])==1
        training.append((row,a))
    assert len(training)==4 and sum(len(a['states']) for _,a in training)==640
    q=Actor(original,'tanh').requires_grad_(False);canonical=CanonicalActor(original,'tanh').requires_grad_(False)
    diagnostics=[]
    for row,a in training:
        x=torch.tensor(a['states']);z=torch.tensor(a['innovations'])
        with torch.no_grad():batch=q.draw(x,z);fixed=canonical.draw(x,z)
        error={}
        for key,v in batch.items():
            assert np.array_equal(fixed[key].numpy(),a[key]),(row['id'],key)
            error[key]=dict(max_abs=float(np.max(np.abs(v.numpy().astype(np.float64)-a[key]))),unequal_values=int(np.sum(v.numpy()!=a[key])))
        diagnostics.append(dict(id=row['id'],original_batch_errors=error,canonical_draw_all_fields_exact=True))
    model=s['models']['tanh'];assert K.C.sha(model['path'])==model['sha256']
    checkpoint=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),weights_only=False,map_location='cpu')
    initial={k:v for k,v in q.state_dict().items() if not k.startswith('fixed.')}
    assert digest(initial)==digest(checkpoint['state']['actor'])==model['hashes']['actor']
    assert checkpoint['state']['optimizer']['state']=={}
    # New numerical subprocess exercises BOTH distributions one row at a time,
    # while the parent receives a whole saved trajectory. No native setup.
    childchecks=[];x=torch.tensor(training[0][1]['states']);z=torch.tensor(training[0][1]['innovations'])
    for arm in original['arms']:
        qa=CanonicalActor(original,arm).requires_grad_(False)
        with torch.no_grad():expected=qa.draw(x,z)
        ctx=mp.get_context('spawn');parent,child=ctx.Pipe();os.environ['STAGE_PROCESS_CPU']='1'
        proc=ctx.Process(target=numerical_worker,args=(child,original,arm));proc.start();child.close();os.environ['STAGE_PROCESS_CPU']='0'
        parent.send(dict(policy_np=K.S.state_np(qa),x=x.numpy(),z=z.numpy()));assert parent.poll(90)
        actual=parent.recv();proc.join(10);assert proc.exitcode==0 and not proc.is_alive()
        assert all(np.array_equal(v.numpy(),actual[k]) for k,v in expected.items())
        with torch.no_grad():loc,sd=qa.stats(x);lp=qa.logp(loc,sd,expected['raw_actions'],expected['atom'])
        assert torch.equal(lp,expected['logp'])
        childchecks.append(dict(arm=arm,numerical_process_pid=proc.pid,all_fields_and_learner_logp_exact=True,native_added=0,optimizer_calls=0))
    bundle={}
    for n in ['T338_DRIVER_ERROR.json','T338_driver.log','T338_audit.log','T338_backup.log']:
        f=DOC/n;bundle[n]=dict(sha256=K.C.sha(f),bytes_base64=base64.b64encode(f.read_bytes()).decode())
    save(REVIEW/'original_closure_logs_bundle.json',bundle)
    result=dict(state='PASS_PARTIAL_EVIDENCE_ONLY_ORIGINAL_RUN_REMAINS_FAILED',scope=p['scope'],
        original_commands=288,original_physics=4178976,original_training_unique=4,original_training_replays=4,
        original_all_replays=6,original_live_Adam=0,original_shadow_Adam=0,original_preflight_discarded_Adam=4,
        retained_initial_Actor_and_empty_Adam_exact=True,all_original_pids_absent=True,full_training_arrays_reward_history_density_exact_canonical=True,
        failure='Preflight compared batch16 against batch16; runtime worker batch1 vs learner batch160 was not covered. Original strict exactness check correctly stopped before first Adam.',
        diagnostics=diagnostics,untrained_canonical_prototype_checks=childchecks,
        no_CAPG_native_or_scientific_comparison_completed=True,new_geometry=0,new_physics=0,new_optimizer_calls=0,
        no_training_restart=True,original_summary_sha256=K.C.sha(OUT/'summary.json'),elapsed_s=time.time()-start)
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h,f
    save(REVIEW/'summary.json',result);save(DOC/'T338_PARTIAL_AUDIT.json',result)
    assert K.S.used(REVIEW)<=p['final_bytes'] and K.S.used(ROOT/'stage/runs')<=p['global_bytes']
    signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k not in ['diagnostics']},ensure_ascii=False))

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if REVIEW.exists():save(REVIEW/'failure.json',dict(error=traceback.format_exc(),native_added=0,optimizer_calls=0))
        raise
