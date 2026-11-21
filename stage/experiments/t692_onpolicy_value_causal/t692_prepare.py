"""Real learner and Session/worker entry preflight, zero native/optimizer."""
from pathlib import Path
import ast,copy,json,os,resource,sys,time,unittest.mock as mock
import t692_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
L,N,T,M,torch,np=S.L,S.N,S.T,S.M,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def entry(p):
    objects=[L.make(p,arm) for arm in ['aware','blind']]
    assert digest(L.state(objects[0]))==digest(L.state(objects[1]))
    initial=[digest(L.state(o)) for o in objects]
    points=torch.tensor(np.random.default_rng(692).normal(size=(8,186)))
    assert torch.equal(objects[0][0](points),objects[1][0](points)) and not objects[0][0](points).any()
    changed=points.clone();changed[:,184:]+=1
    assert torch.equal(objects[1][0](points),objects[1][0](changed))
    for a,b in zip(objects[0][0].parameters(),objects[1][0].parameters()):assert a.data_ptr()!=b.data_ptr()
    # Synthetic slots enumerate all1024 values; multiplication2 exactly gives
    # sum over executed decisions /512, with no inverse realized length.
    errors=torch.tensor([.2,.4,.6,.8],dtype=torch.float64);masked=torch.zeros(1024,dtype=torch.float64);masked[:4]=errors
    assert float((masked*2).mean())==float(errors.sum()/512)
    before=digest(L.state(objects[0]));pack=(points,torch.ones(8,dtype=torch.float64),torch.ones(8,dtype=torch.float64))
    loss=L.loss(objects[0][0],pack);loss.backward();assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in objects[0][0].parameters())
    assert digest(L.state(objects[0]))==before
    saved=L.state(objects[0]);L.load(objects[1],saved);assert digest(L.state(objects[1]))==before
    assert all(not o[1].state for o in objects)
    for n in [0,1,64,128]:S.count_commands([None]*n,n+6)
    try:S.count_commands([],3);raise AssertionError('Old no-native3 command bug not rejected')
    except AssertionError as err:
        if str(err)=='Old no-native3 command bug not rejected':raise
    return dict(identical_initial_values_and_state=True,independent_model_storage=True,blind_action_mask_exact=True,
        fullfixedslot_mask_expectation_identity=True,finite_backward_no_Adam_state_change=True,no_native6_and_nonempty_management_regression=True)

def native_entry(p):
    checks=[]
    for job in p['branch_jobs']:
        source=T.R.E.N.S.arrays(job['source_path']);assert len(source['states'])>20
        for delta in p['action_deltas']:
            q=M.L.frozen(p,p['initial']['path']);q.rng=np.random.default_rng([69220261121,0,0]);q.reset_noise()
            session=N.Session.__new__(N.Session);session.q=q;session.source=source;session.branch=20;session.rows=[]
            session.branch_command=np.clip(source['locations'][20]+np.asarray(delta),-1,1).astype(np.float32)
            witness=[]
            def fake_step(s,forced,owner,actor_active):
                witness.append(dict(forced=None if forced is None else forced.copy(),owner=owner,sample=s.q.sample,actor_active=actor_active));return 'NO_NATIVE'
            with mock.patch.object(N.F.FieldSession,'step',fake_step):
                for index in [0,19,20,21]:
                    session.rows=[None]*index;session.packet=lambda:source['states'][index].copy()
                    assert N.Session.step(session)=='NO_NATIVE'
            assert witness[0]['owner']==witness[1]['owner']==witness[2]['owner']==1 and witness[3]['owner']==0
            assert not any(x['sample'] for x in witness[:3]) and witness[3]['sample']
            assert np.array_equal(witness[0]['forced'],source['actions'][0]) and np.array_equal(witness[2]['forced'],session.branch_command) and witness[3]['forced'] is None
            # Real Actor.step on full external prefix reconstructs own carry;
            # stochastic draw starts at suffix, no prefix RNG consumption.
            h=None;q.rng=np.random.default_rng([69220261121,0,0]);q.reset_noise()
            with torch.no_grad():
                for i in range(22):q.sample=i>20;unused,mu,h,extra=q.step(torch.tensor(source['states'][i:i+1]),h)
            assert not np.asarray([x['innovations'] for x in q.tape[:21]]).any()
            assert np.array_equal(q.tape[21]['innovations'],np.random.default_rng([69220261121,0,0]).normal(size=(1,2))[0])
            with torch.no_grad():full=q.sequence(torch.tensor(source['states'][:22]))
            assert np.max(np.abs(full.numpy()-np.asarray([x['locations'] for x in q.tape])))<1e-10
            checks.append(dict(task=job['id'],delta=delta,prefix_source_sha256=job['source_sha256'],single_quantized_command=session.branch_command.tolist()))
    captured=[]
    def witness_worker(pipe,index,pc,counts):
        assert N.G.T.Session is N.Session and N.P.frozen is N.L.frozen
        assert callable(N.G.B.native_setup);captured.append(index)
    original_affinity=os.sched_setaffinity
    try:
        with mock.patch.object(N.G.T,'worker',witness_worker):N.worker(None,0,p,None)
    finally:os.sched_setaffinity=original_affinity
    assert captured==[0]
    return dict(actual_Session_step_boundary_cases=checks,owncarry_sequence_step_and_suffix_RNG_reset_exact=True,
        real_worker_bindings_Session_loader_canonical_factory=True,physical_Session_constructor_pending_formal_first_action=True,
        synthetic_super_step_only_NOT_actual_scene_or_native_execution=True)

def main():
    start=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists();DOC.mkdir(exist_ok=True)
    last=ROOT/'stage/runs/t691r_command_audit_recovery';rep=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert rep['state']=='COMPLETE' and all(r['audit']['state']=='PASS' for r in rep['children'])
    assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids']+[rep['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']])
    old=read(ROOT/'stage/runs/t690_score_weighted_baseline/protocol.json');p=copy.deepcopy(old);p.pop('inputs',None);p.pop('registration_epoch',None)
    a=read(DOC.parent/'AUTHORIZATION.json');now=time.time();groups=L.B.dataset(p);by={j['id']:j for j in p['evaluation_jobs']}
    ids=['dev_task6_00','dev_task6_02','t502_field_sector0_offset1','t502_field_sector7_offset2'];jobs=[]
    for key in ids:
        row=groups[key][6 if key.startswith('t502_') else 3];jobs.append(dict(by[key],source_path=row['path'],source_sha256=row['sha256'],branch_decision=20))
    p.update(id='T692_SAME_BEHAVIOR_MC_VALUE_AND_CAUSAL_BRANCH',authorization_approved=False,out=str(OUT),new_Actor=0,new_Value=1000,new_optimizer=1000,
        branch_jobs=jobs,action_deltas=[[.05,0.],[-.05,0.],[0.,.05],[0.,-.05]],value_seed=69220261121,batch_seed=69220261121,
        max_physics=4655264,max_commands=134,main_seconds=5700,audit_seconds=600,learning_seconds=900,native_reserve_seconds=4400,
        max_bytes=512*2**20,child_bytes=512*2**20,global_bytes=10*2**30,full_bytes=4*2**20,fresh_bytes=4*2**20,model_bytes=2*2**20,
        absolute_learning_stop=min(now+1200,a['absolute_learning_stop']),absolute_native_stop=min(now+6500,a['absolute_native_stop']),
        absolute_all_stop=min(now+7200,a['absolute_all_stop']),finalization_seconds=60,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'))
    assert sum(j['case']['step_cap'] for j in jobs)*32==p['max_physics']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);S.setup(p)
    def forbidden(*args,**kwargs):raise AssertionError('T692 preflight forbids optimizer or native steps')
    torch.optim.Adam.step=forbidden
    if 'mujoco' in sys.modules:sys.modules['mujoco'].mj_step=forbidden
    contract=entry(p);boundary=native_entry(p);last_guard=[0.]
    def guard(phase='',key=''):
        assert time.time()<start+240
        if time.time()-last_guard[0]>5:S.Wait.Raw.resources();last_guard[0]=time.time()
    dataset=L.data(p,guard);objects={arm:L.make(p,arm) for arm in ['aware','blind']}
    assert digest(L.state(objects['aware']))==digest(L.state(objects['blind']))
    pack=L.batch(dataset,np.random.default_rng(p['batch_seed']));value=L.loss(objects['aware'][0],pack);value.backward()
    assert bool(torch.isfinite(value)) and all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in objects['aware'][0].parameters())
    predictions=L.branch_predictions(p,objects,dataset)
    for key,rows in predictions.items():assert len({tuple(row['command']) for row in rows.values()})==4
    initial_metrics={arm:L.metrics(obj,dataset,True) for arm,obj in objects.items()};assert initial_metrics['aware']==initial_metrics['blind']
    # Conditional regression gate fixtures are definitions, not fitted science.
    metric=dict(fixed_decision_loss=1.,initial_decision_MSE=1.,old384_fixed_loss=1.,field_fixed_loss=1.)
    assert not L.regression_gate(dict(aware=dict(heldout=metric),blind=dict(heldout=metric)))['passed']
    better={k:.8*v for k,v in metric.items()};assert L.regression_gate(dict(aware=dict(heldout=better),blind=dict(heldout=metric)))['passed']
    resources=S.Wait.Raw.resources();peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<a['memory_bytes_cap']
    assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',learner_contract=contract,native_entry_contract=boundary,
        all1724_real_sources_features_locations_density_quantization_and_split=True,training_episodes=1293,heldout_episodes=431,
        feature_hash=dataset['manifest']['feature_sha256'],initial_model_metrics_exactly_equal=True,real_full_data_backward_finite=True,
        branch_prediction4commands_distinct_all4prefixes=True,heldout_gate_identity_and_positive_fixture=True,
        peak_rss_bytes=peak,resources=resources,new_native=0,new_optimizer=0,elapsed_seconds=time.time()-start)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in old['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T692_ONPOLICY_VALUE_PRIMARY_REVIEW.md',DOC.parent/'NEXT_T692_ONPOLICY_VALUE_CAUSAL_SPEC.md',DOC.parent/'t691r/FINDINGS.md'}
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','nominal/audit.json','induced/audit.json']}
    deps.add(DOC/'PRE_REGISTRATION_NOTES.json')
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+read(DOC/'PRE_REGISTRATION_NOTES.json')['charged_preflight_upper_seconds']
    save(HERE/'protocol.json',p);save(DOC/'BUDGET.json',{k:p[k] for k in ['new_Actor','new_Value','new_optimizer','max_physics','max_commands','main_seconds','audit_seconds','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre),flush=True)

if __name__=='__main__':main()
