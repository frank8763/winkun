"""Bounded zero-update data, reset and real loader/auditor admission."""
import copy, gzip, io, json, math, os, resource, sys, time, traceback
from t699_common import *
import t699_actor as A

def main():
    started=time.time(); bound=read(DOC/'PREPARATION_BOUND.json')
    assert time.time()<bound['absolute_stop'] and not (DOC/'PREFLIGHT.json').exists()
    os.sched_setaffinity(0,{15}); resources()
    save(DOC/'PREPARATION_STARTED.json',dict(pid=os.getpid(),epoch=started))
    A.torch.set_num_threads(1); A.torch.use_deterministic_algorithms(True)
    import mujoco
    def forbidden(*a,**kw): raise AssertionError('No native integration/optimizer in data preflight')
    mujoco.mj_step=forbidden; A.torch.optim.Adam.step=forbidden
    def guard():
        assert time.time()<bound['absolute_stop']-15
        resources()
    p=copy.deepcopy(read(ROOT/'stage/runs/t696_existing_success_compatibility/protocol.json'))
    for k in ['inputs','registration_epoch','input_stats']:p.pop(k,None)
    parent=A.Parent.frozen(p,p['initial']['path'])
    reference=read(WEEKEND/'t696/RETENTION_DECISION.json')
    a_model=read(ROOT/'stage/runs/t696_existing_success_compatibility/fit/summary.json')['model']
    assert sha(a_model['path'])==a_model['sha256']
    X=[];Y=[];MU=[];COMMAND=[];GROUP=[];records=[]
    rows=p['protected403_rows']+p['novel_rows']
    for index,row in enumerate(rows):
        guard();assert sha(row['path'])==row['sha256']
        arrays=A.Existing.L.T.R.E.N.S.arrays(row['path'])
        features,mu=A.Existing.trajectory(parent,arrays,row['id'],arrays['actions'] if index<403 else None,guard)
        target=A.torch.tensor(arrays['actions'],dtype=A.torch.float64)
        X.append(features); Y.append(A.torch.zeros_like(mu) if index<403 else target-mu)
        MU.append(mu); COMMAND.append(target); GROUP.append(A.torch.full((len(mu),),index,dtype=A.torch.int64))
        records.append(dict(id=row['id'],source_path=row['path'],sha256=row['sha256'],length=len(mu),old=index<403,feature_hash=A.digest(features)))
        if index%40==0:print(json.dumps(dict(stage='owncarry_features',completed=index,total=410)),flush=True)
    x,y,mu,command,groups=map(A.torch.cat,[X,Y,MU,COMMAND,GROUP])
    assert len(x[groups<403])==81956 and len(x[groups>=403])==1737
    saved=A.Existing.Base.P.checkpoint(a_model['path'])['joint_state']['actor']
    assert A.torch.equal(x[groups>=403],saved['new_centers'])
    assert A.torch.equal(y[groups>=403],saved['new_weights'])
    with (OUT/'DATA.pt.gz').open('xb') as f:
        b=io.BytesIO();A.torch.save(dict(x=x,y=y,mu=mu,commands=command,groups=groups,records=records),b);f.write(gzip.compress(b.getvalue(),mtime=0))
    save(OUT/'DATA_MANIFEST.json',dict(records=records,feature_hash=A.digest(x),label_hash=A.digest(y),data_sha256=sha(OUT/'DATA.pt.gz'),old_decisions=81956,new_decisions=1737))
    jobs={j['id']:j for j in p['evaluation_jobs']}
    six=['t502_field_sector'+str(s)+'_offset1' for s in [0,1,2,5,7]]+['t502_field_sector0_offset2']
    parents=sorted(set(six)|{r['id'] for r in p['novel_rows']});assert len(parents)==13
    proposals=[]
    for key in parents:
        for axis,amount in [('x',-.023),('x',.023),('y',-.023),('y',.023),('yaw',-.7),('yaw',.7)]:
            j=copy.deepcopy(jobs[key]);j.pop('initial_contract',None)
            j['id']='T699_new_'+key+'_'+axis+('minus' if amount<0 else 'plus')
            j['source_reset_id']=j['id'];j['expected_initial120_sha256']=None;j['neighborhood_parent_id']=key
            if 'fullfield_start_pose' in j:
                col={'x':0,'y':1,'yaw':2}[axis]
                j['fullfield_start_pose'][col]+=math.radians(amount) if axis=='yaw' else amount
            else:
                j[{'x':'delta_x_m','y':'delta_y_m','yaw':'delta_yaw_deg'}[axis]]+=amount
            proposals.append(j)
    assert len(proposals)==78 and len({j['id'] for j in proposals})==78
    # Real canonical zero-step admission for every new initial state, before any fit/native choice.
    import t698_native as N
    C,G,F=N.C,N.G,N.F
    C.ACTIVE=p
    valid=[];invalid=[];init=[]
    folder=OUT/'preflight_initials';folder.mkdir()
    module,native,create,w,_=C.C.consistent_reset.setup(F,folder,p,C.C.CANONICAL_SETUP)
    original=copy.deepcopy(module.ACTIVE)
    try:
        for job in proposals:
            guard();module.ACTIVE=copy.deepcopy(original)
            if 'circle_shift_x_m' in job:module.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m']
            current=None
            try:
                current=C.Session(job,native,create,w,guard,forbidden,None)
                assert not current.env._collision() and not current.done()
                pose=list(current.env._pose()[:3]);packet=current.packet()
                assert not current.hidden.any() and current.steps==0 and current.env.data.time==0
                if 'fullfield_start_pose' in job:assert A.np.max(A.np.abs(A.np.array(pose)-job['fullfield_start_pose']))<1e-10
                valid.append(job);init.append(dict(id=job['id'],pose=pose,packet_hash=A.digest(packet),native_hash=A.digest(current.initial),scene=current.scene))
            except AssertionError as error:
                invalid.append(dict(id=job['id'],error=str(error),not_navigation_failure=True))
            finally:
                if current is not None:current.close()
    finally:w.close()
    assert len(valid)+len(invalid)==78
    # Same CPU precision/real loader. Zero new output must retain real trajectory exactly.
    q=A.attach(p,A.make_head(108,2026112301));sample=p['protected403_rows'][0]
    arr=A.Existing.L.T.R.E.N.S.arrays(sample['path']);A.Existing.trajectory(q,arr,sample['id'],arr['actions'],guard)
    blob=io.BytesIO();joint=dict(actor=q.state_dict());A.torch.save(dict(joint_state=joint,architecture=A.ARCH,policy_hash=q.phash()),blob)
    fd=os.memfd_create('T699_actual_loader_zero_head',flags=os.MFD_CLOEXEC)
    try:
        os.write(fd,gzip.compress(blob.getvalue(),mtime=0));loaded=A.frozen(p,f'/proc/self/fd/{fd}')
        assert A.digest(loaded.state_dict())==A.digest(q.state_dict())
        A.Existing.trajectory(loaded,arr,sample['id'],arr['actions'],guard)
    finally:os.close(fd)
    p.update(id='T699_COMMON_PARENT_MATCHED_INCREMENTAL_VALUE',out=str(OUT),authorization_approved=False,
        existing_A=a_model,new_jobs=valid,new_invalid_initials=invalid,new_initial_contracts=init,
        stage_cpus=list(range(8,16)),worker_count=7,learner_device='cpu',fit_threads=8,
        replay_widths=[108,434,1736],replay_lrs=[.001,.0003],replay_steps=384,replay_batch_per_group=1024,
        replay_seed=2026112301,second_seed=2026112302,
        absolute_learning_stop=1795390200,absolute_native_stop=1795392900,absolute_all_stop=1795395600,
        campaign_learning_stop=1795390200,campaign_native_stop=1795392900,
        root_seconds=26000,fit_main_seconds=2400,fit_audit_seconds=1500,
        known_main_seconds=2800,known_audit_seconds=360,new_main_seconds=650,new_audit_seconds=180,
        finalization_seconds=60,final_storage_reserve=32*2**20,
        max_bytes=int(4.5*2**30),global_bytes=10*2**30,child_bytes=512*2**20,model_bytes=8*2**20,
        admission_physics_per_second=6000,fit_bytes=320*2**20,
        final_models=['plain','replay_1736','replay_108','replay_434','replay_1736_seed2'],
        transfer_models=['historical_parent','historical_A','plain','replay_1736','replay_108','replay_434','replay_1736_seed2'],
        max_Actor_live=3841,max_Actor_shadow=3841,max_Actor_independent=1537,
        max_Value=0,max_RL=0,all_model_baselines_evaluated_even_if_command_mismatch=True,
        physical_equality_to_source_required_for_baselines=False)
    known_physics=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['screen_jobs'])
    new_exact=valid[:7];p['new_exact_jobs']=new_exact
    new_physics=sum(j['case']['step_cap'] for j in valid+new_exact)
    p['max_physics']=5*known_physics+7*new_physics
    p['max_commands']=5*(500+14)+7*(len(valid)+len(new_exact)+14)
    assert known_physics/6000+120<p['known_main_seconds']
    assert new_physics/6000+120<p['new_main_seconds']
    assert 2400+1500+60<7200 and 2800+360+60<7200
    assert used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']<8*2**30
    ledger=read(WEEKEND/'LEDGER.json');auth=read(WEEKEND/'AUTHORIZATION.json')
    assert ledger['Actor_live']+3841<=4096 and ledger['Actor_shadow']+3841<=4096 and ledger['Actor_independent']+1537<=4096
    assert ledger['physics']+p['max_physics']<auth['max_new_physics'] and ledger['commands']+p['max_commands']<auth['max_new_commands']
    assert ledger['transport_seconds']+600<auth['max_transport_seconds']
    assert time.time()+p['fit_main_seconds']+p['fit_audit_seconds']+5*(p['known_main_seconds']+p['known_audit_seconds'])+7*(p['new_main_seconds']+p['new_audit_seconds'])+180<1795392900
    result=dict(state='PASS',old_ids=403,new_source_ids=7,old_decisions=81956,new_decisions=1737,
        cached_features_exact_historical_centers=True,real_RAM_loader_and_zero_head_commands=True,
        proposed_new_initials=78,valid_new_initials=len(valid),invalid_initials=invalid,
        new_native=0,new_optimizer=0,new_Actor=0,elapsed_seconds=time.time()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    save(DOC/'PREFLIGHT.json',result)
    inherited=read(ROOT/'stage/runs/t696_existing_success_compatibility/protocol.json')['inputs']
    paths={Path(f) for f in inherited}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PRIMARY_REVIEW.md',DOC/'PREFLIGHT.json',DOC/'AUTHORIZATION.json',OUT/'DATA.pt.gz',OUT/'DATA_MANIFEST.json',Path(a_model['path'])}
    p['inputs']={str(f):sha(f) for f in sorted(paths)}
    p['input_stats']={f:[Path(f).stat().st_size,Path(f).stat().st_mtime_ns] for f in p['inputs']}
    (HERE/'protocol.json').write_text(json.dumps(p,separators=(',',':'))+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    start=time.time()
    try:main()
    except BaseException:
        save(DOC/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),elapsed_seconds=time.time()-start,no_updates=True));raise
    finally:
        l=read(WEEKEND/'LEDGER.json');elapsed=time.time()-start;l['science_seconds']+=elapsed
        j=next(j for j in l['jobs'] if j['id']=='T699_PREPARATION_RECOVERY');j.update(state='PASS' if (DOC/'PREFLIGHT.json').exists() else 'FAILED',science_seconds=elapsed)
        save(WEEKEND/'LEDGER.json',l)
