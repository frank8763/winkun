"""Bind the real new entrance, with native and optimizer stepping prohibited."""
from pathlib import Path
import ast,copy,hashlib,inspect,json,math,os,resource,sys,time,traceback,unittest.mock as mock
import t693_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.ROOTOUT,S.ROOTDOC
L,N,T,Est,torch,np=S.L,S.N,S.T,S.Est,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save


def rejects(call):
    try:call()
    except AssertionError:return
    raise AssertionError('Corrupted contract was not rejected')


def native_contract(p):
    jobs,seeds=S.schedule(p);witnesses=[]
    for j,seed in zip(jobs,seeds):
        q=L.M.L.frozen(p,p['initial']['path']);q.rng=np.random.default_rng(seed);q.reset_noise()
        source=T.R.E.N.S.arrays(j['source_path']);k=j['branch_decision']
        session=N.Session.__new__(N.Session);session.q=q;session.source=source;session.branch=k
        session.branch_command=np.asarray(j['branch_command'],np.float32);session.branch_latent=np.asarray(j['branch_latent'])
        session.decision_integrations=[];session.env=object();seen=[]
        def fake_step(s,forced,owner,actor_active):
            seen.append(dict(command=None if forced is None else forced.copy(),owner=owner,sample=s.q.sample,active=actor_active))
            return 'NO_PHYSICAL_STEP'
        with mock.patch.object(N.F.FieldSession,'step',fake_step),mock.patch.object(N.E.N.K.R,'integration',lambda env:np.arange(203,dtype=float)):
            for i in [0,k,k+1]:
                session.rows=[None]*i;session.steps=int(source['native_counts'][:i].sum());session.packet=lambda i=i:source['states'][min(i,len(source['states'])-1)].copy()
                assert session.step()=='NO_PHYSICAL_STEP'
        assert seen[0]['owner']==seen[1]['owner']==1 and seen[2]['owner']==0
        assert not seen[0]['sample'] and not seen[1]['sample'] and seen[2]['sample']
        assert np.array_equal(seen[1]['command'],session.branch_command) and seen[2]['command'] is None
        assert len(session.decision_integrations)==3
        witnesses.append(dict(task=j['id'],slot=k,method=j['method'],pair=j['pair'],sign=j['sign']))
    # One complete actual saved trajectory goes through the real own-carry/
    # density/reward/history auditor, then deliberately damaged carry must fail.
    groups=L.B.dataset(p);row=groups[Est.IDS[0]][0];a=T.R.E.N.S.arrays(row['path'])
    N.F.M.audit_full_history(p,row,a)
    damaged=dict(a,hidden_before=a['hidden_before'].copy());damaged['hidden_before'][0,0]+=.1
    rejects(lambda:N.F.M.audit_full_history(p,row,damaged))
    # Native-prefix assertions are exercised with controlled array witnesses.
    # These are NOT actual new scene/physics results or a Q validity test.
    job=jobs[0];source=T.R.E.N.S.arrays(job['source_path']);k=job['branch_decision'];n=min(k+2,len(source['states']))
    a={key:value.copy() for key,value in source.items()}
    for key,value in list(a.items()):
        if value.ndim and value.shape[0]==len(source['states']):a[key]=value[:n].copy()
        elif value.ndim and value.shape[0]==len(source['states'])+1:a[key]=value[:n+1].copy()
    a.update(ownership=(np.arange(n)<=k).astype(np.uint8),stochastic=(np.arange(n)>k).astype(np.uint8),
             branch_latent=np.asarray(job['branch_latent']),decision_native_integration=np.tile(a['final_native_integration'],(n,1)))
    a['actions'][k]=job['branch_command'];a['innovations'][:k+1]=0
    if k+1<n:
        a['states'][k+1,117:119]=a['actions'][k];a['next_states'][k,117:119]=a['actions'][k]
    a['end_flags'][:]=0;a['end_flags'][-1,2]=1
    assert N.inspect_prefix(a,source,job)==k
    for field in ['ownership','actions','states','native_counts','branch_latent','decision_native_integration']:
        broken={name:value.copy() for name,value in a.items()}
        if field=='ownership':broken[field][k]=0
        elif field=='actions':broken[field][k,0]+=.1
        elif field=='states':broken[field][0,0]+=.1
        elif field=='native_counts':broken[field][0]=int(broken[field][0])-1
        elif field=='branch_latent':broken[field][0]+=.1
        else:broken[field][-1,0]+=.1
        rejects(lambda broken=broken:N.inspect_prefix(broken,source,job))
    # Real recurrent step consumes no prefix innovations; the suffix RNG starts
    # with its first draw at k+1 and is reproducible from the locked namespace.
    q=L.M.L.frozen(p,p['initial']['path']);q.rng=np.random.default_rng(seeds[0]);q.reset_noise();h=None
    with torch.no_grad():
        for i in range(k+2):
            q.sample=i>k;_,_,h,_=q.step(torch.tensor(source['states'][min(i,len(source['states'])-1):min(i,len(source['states'])-1)+1]),h)
    assert not np.asarray([r['innovations'] for r in q.tape[:k+1]]).any()
    assert np.array_equal(q.tape[k+1]['innovations'],np.random.default_rng(seeds[0]).normal(size=(1,2))[0])
    captured=[];original=os.sched_setaffinity
    def fake_worker(pipe,index,pc,counts):
        assert N.G.T.Session is N.Session and N.P.frozen is N.L.frozen
        assert callable(N.G.B.native_setup);captured.append(index)
    try:
        with mock.patch.object(N.G.T,'worker',fake_worker):N.worker(None,0,p,None)
    finally:os.sched_setaffinity=original
    assert captured==[0]
    # Restore ordinary worker setup after the controlled binding witness.
    S.setup(dict(p,arm='mvd'))
    for n in [0,1,55,144,462]:S.count_commands([None]*n,n+6)
    rejects(lambda:S.count_commands([],3))
    return dict(actual_Session_step_boundaries=witnesses,real_saved_fullhistory_auditor_and_corrupt_carry_rejection=True,
                prefix_ownership_command_state_count_latent_integration_corruption_rejected=True,
                full_owncarry_and_suffix_RNG_start=True,canonical_worker_loader_binding=True,
                six_management_commands_in_actual_audit_paths=True,
                physical_Session_constructor_and_new_scene_pending_formal_first_action=True,
                controlled_superstep_and_arrays_NOT_new_native_results=True)


def learner_contract(p,guard):
    jobs,seeds=S.schedule(p);fake=[]
    for index,j in enumerate(jobs):
        # Contrasting synthetic labels exercise both real gradient entrances.
        # They are never registered, used for actual learning, or stored as raw.
        y=(1 if j['sign']==0 else -1)
        fake.append(dict(job_id=j['id'],request=dict(job=j),complete=True,success=y==1,collision=y==-1,
                         timeout=False,field_boundary=False,path=f'SYNTHETIC_NO_NATIVE_{index}'))
    fakecredits=S.credits(p,fake);virtual=HERE/'NOT_A_RUN_PREFLIGHT_ONLY'
    p=dict(p,collection_out=str(virtual),collection_credit_sha256='VIRTUAL_NOT_SCIENCE')
    original_read,original_sha=L.read,L.sha
    def mocked_read(f):
        if Path(f)==virtual/'audit.json':return dict(state='PASS')
        if Path(f)==virtual/'CREDITS.json':return fakecredits
        return original_read(f)
    def mocked_sha(f):return 'VIRTUAL_NOT_SCIENCE' if Path(f)==virtual/'CREDITS.json' else original_sha(f)
    results={}
    with mock.patch.object(L,'read',mocked_read),mock.patch.object(L,'sha',mocked_sha):
        for arm in ['mvd','score']:
            pp=dict(p,arm=arm);q,opt=L.make(pp);before=L.optimizer_state(q,opt);guard()
            sources,panel,keep=L.data(pp,q,guard)
            detail,arrays=L.direction(pp,q,opt,sources,panel,keep,guard)
            assert arrays is not None and detail['RL_gradient_norm']>0
            assert digest(before)==digest(L.optimizer_state(q,opt)) and not opt.state
            shadow,other=L.make(pp);L.load(shadow,other,before)
            assert digest(L.optimizer_state(shadow,other))==digest(before)
            assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(q.parameters(),shadow.parameters()))
            zeros=[dict(x,coefficient=np.zeros(2)) for x in sources]
            zero,rejected=L.direction(pp,q,opt,zeros,panel,keep,guard)
            assert rejected is None and zero['reason']=='ZERO_OR_NONFINITE_RL_GRADIENT' and zero['Actor_Adam']==0
            results[arm]=dict(gradient_error=detail['gradient_error'],numeric=detail['numeric'],synthetic_credit_only=True,
                             fullstate_and_Adam_restore=True,zero_RL_gradient_rejected_before_Adam=True,
                             real387_reference_and68_KL_path=True)
    return results


def storage(p):
    # Conservative ZIP/NumPy headers plus deflate growth, using every actual
    # source field's dtype/trailing shape at the task's complete native cap.
    bounds=[]
    for e in p['fixed_records']:
        if not e['active']:continue
        a=T.R.E.N.S.arrays(e['source']);n=len(a['states']);limit=math.ceil(e['step_cap']/100);raw=0
        for key,v in a.items():
            if v.ndim and v.shape[0] in [n,n+1]:
                raw+=(limit+int(v.shape[0]==n+1))*math.prod(v.shape[1:])*v.dtype.itemsize
            else:raw+=v.nbytes
        raw+=limit*203*8+16
        upper=int(raw*1.02)+(len(a)+2)*1024
        assert upper<p['full_bytes'],(e['task'],upper,p['full_bytes'])
        bounds.append(dict(task=e['task'],source=e['source_sha256'],raw_zip_upper=upper))
    collection=144*p['full_bytes']+3*p['full_bytes']+16*2**20+p['model_bytes']
    candidate=55*p['full_bytes']+407*p['compact_bytes']+3*p['full_bytes']+3*p['model_bytes']+24*2**20
    assert collection<p['collection_bytes'] and candidate<p['child_bytes']
    assert p['collection_bytes']+2*p['child_bytes']+p['final_storage_reserve']<p['max_bytes']
    used=T.R.used(ROOT/'stage/runs');assert used+p['max_bytes']<8*2**30
    return dict(full_schema_complete_caps=bounds,collection_upper=collection,candidate_upper=candidate,
                global_used=used,global_with_root_worst=used+p['max_bytes'],no_raw_deletion_needed=True)


def main():
    started=time.time();os.sched_setaffinity(0,{14});assert not OUT.exists()
    assert not (DOC/'PREFLIGHT.json').exists(),'Do not repeat completed entry preflight'
    old=read(ROOT/'stage/runs/t690_score_weighted_baseline/protocol.json');p=copy.deepcopy(old)
    p.pop('inputs',None);p.pop('registration_epoch',None)
    pre=read(DOC/'PREPARATION.json');assert pre['state']=='PREPARATION_CONTRACT_PASS_NOT_LAUNCH_PREFLIGHT' and pre['support']['passed']
    a=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p.update(id='T693_MVD_VS_PAIRED_SCORE_LOCAL8_ONE_STEP',authorization_approved=False,out=str(OUT),arms=['collection','mvd','score'],arm='root',
        fixed_records=pre['records'],Actor_steps=1,new_Value=0,new_optimizer=2,preflight_optimizer=0,
        collection_main_seconds=3000,collection_audit_seconds=300,collection_max_physics=5392224,collection_max_commands=150,
        collection_bytes=1280*2**20,child_bytes=384*2**20,max_bytes=2304*2**20,global_bytes=10*2**30,
        final_storage_reserve=16*2**20,night_main_seconds=14400,audit_total_seconds=1500,night_max_physics=32642866,night_max_commands=1086,
        child_main_seconds=5700,child_audit_seconds=600,learning_seconds=1200,full_evaluation_reserve_seconds=3500,
        absolute_all_stop=min(now+16200,a['absolute_all_stop']),campaign_learning_stop=a['absolute_learning_stop'],campaign_native_stop=a['absolute_native_stop'],
        absolute_learning_stop=min(now+16200,a['absolute_learning_stop']),absolute_native_stop=min(now+15900,a['absolute_native_stop']),
        finalization_seconds=60,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),learner_device='cpu',fresh_bytes=4*2**20)
    hashes={p['initial']['policy_hash']}
    for prefix in ['t665_remaining_verification','t671_joint_outcome_feasibility','t678_reference_target_pilot','t690_score_weighted_baseline','t691_induced_state_rehearsal','t691r_command_audit_recovery']:
        for f in (ROOT/'stage/runs'/prefix).rglob('LEARNING.json'):
            d=read(f)
            if d.get('proposal'):hashes.add(d['proposal']['policy_hash'])
    p['consumed_candidate_hashes']=sorted(hashes)
    Wait=S.Wait;Wait.Raw.resources()
    last=ROOT/'stage/runs/t692_onpolicy_value_causal';seal=read(last/'SEALED.json');report=read(last/'FINAL_REPORT.json')
    assert all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids']+[report['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']])
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer/native stepping during preflight')
    torch.optim.Adam.step=forbidden
    import mujoco
    mujoco.mj_step=forbidden
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);S.setup(dict(p,arm='mvd'))
    last_guard=[0.]
    def guard():
        assert time.time()<started+600
        if time.time()-last_guard[0]>5:Wait.Raw.resources();last_guard[0]=time.time()
    native=native_contract(p);learn=learner_contract(p,guard);byteproof=storage(p)
    jobs,seeds=S.schedule(p)
    assert sum(j['case']['step_cap'] for j in jobs)*2==p['collection_max_physics']
    assert len(jobs)*2+6==p['collection_max_commands']
    assert p['collection_max_physics']/2500+300<3000 and p['child_max_physics']/2500+120<5700
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<a['memory_bytes_cap']
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',native_entry=native,learner_entry=learn,storage=byteproof,peak_rss_bytes=peak,
                new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,preparation_not_repeated=True,elapsed_seconds=time.time()-started)
    save(DOC/'PREFLIGHT.json',result)
    deps={Path(f) for f in old['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREPARATION.json',DOC/'PREFLIGHT.json',DOC.parent/'T693_METHOD_SELECTION_PRIMARY_REVIEW.md',DOC.parent/'t692/FINDINGS.md'}
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json']}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    prior_failure=read(DOC/'PRE_REGISTRATION_NOTES.json')['charged_upper_seconds'] if (DOC/'PRE_REGISTRATION_NOTES.json').exists() else 0
    p['preflight_seconds']=time.time()-started+10+prior_failure
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['night_max_physics','night_max_commands','night_main_seconds','audit_total_seconds','new_optimizer','max_bytes','absolute_all_stop','preflight_seconds']})
    print(json.dumps({k:v for k,v in result.items() if k not in ['native_entry','storage']}),flush=True)


if __name__=='__main__':main()
