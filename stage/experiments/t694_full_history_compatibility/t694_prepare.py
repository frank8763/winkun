"""Final actual loader/worker/audit contract, no coefficient commit or native."""
from pathlib import Path
import ast,copy,inspect,json,os,resource,sys,time,unittest.mock as mock
import t694_runtime as S
ROOT,HERE,DOC=S.ROOT,S.HERE,S.ROOTDOC
A,N,T,torch,np=S.A,S.N,S.T,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save


def rejected(call):
    try:call()
    except AssertionError:return
    raise AssertionError('Corrupted contract not rejected')


def main():
    started=time.time();os.sched_setaffinity(0,{14});assert not S.ROOTOUT.exists()
    assert not (DOC/'PREFLIGHT.json').exists(),'Do not repeat completed preflight'
    pre=read(DOC/'PREPARATION.json');assert pre['state']=='GEOMETRY_AND_OWNCARRY_PREPARATION_PASS_NOT_COMPLETE_PREFLIGHT'
    old=read(ROOT/'stage/runs/t691_induced_state_rehearsal/protocol.json');p=copy.deepcopy(old);p.pop('inputs',None);p.pop('registration_epoch',None)
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p.update(id='T694_FULL384_PLUS6_COMPACT_READOUT_FINITE_COMPATIBILITY',authorization_approved=False,out=str(S.ROOTOUT),arm='root',
        protected_rows=[r for r in read(p['reference_contract'])['reference_rows'] if r['origin']=='T501_original_success'],
        support_fraction=.49,minimum_feature_separation=1e-10,maximum_centers=1405,model_bytes=8*2**20,
        new_Actor=1,new_Value=0,new_optimizer=0,preflight_optimizer=0,learning_seconds=1200,
        fit_main_seconds=1200,fit_audit_seconds=900,child_main_seconds=5700,child_audit_seconds=600,
        night_main_seconds=6900,audit_total_seconds=1500,night_max_physics=13625321,night_max_commands=468,
        fit_bytes=64*2**20,child_bytes=384*2**20,max_bytes=512*2**20,global_bytes=10*2**30,final_storage_reserve=16*2**20,
        full_evaluation_reserve_seconds=3500,root_seconds=10200,
        absolute_all_stop=min(now+10200,auth['absolute_all_stop']),absolute_learning_stop=min(now+10200,auth['absolute_learning_stop']),
        absolute_native_stop=min(now+9900,auth['absolute_native_stop']),campaign_learning_stop=auth['absolute_learning_stop'],campaign_native_stop=auth['absolute_native_stop'],
        finalization_seconds=60,weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),learner_device='cpu',fresh_bytes=4*2**20)
    assert len(p['protected_rows'])==384 and len(p['novel_rows'])==6 and len(p['screen_jobs'])==55
    S.Wait.Raw.resources();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);S.setup(p)
    def forbidden(*a,**kw):raise AssertionError('No optimizer/native or fitted coefficient commit in preflight')
    torch.optim.Adam.step=forbidden;A.fit=forbidden
    import mujoco
    mujoco.mj_step=forbidden
    # Actual expanded model shape, serialization and loader through anonymous
    # RAM fd. No new learned coefficient is assigned; no model file persists.
    q0=A.ORIGINAL(p,p['initial']['path']);before=digest(q0.state_dict());q=A.Actor(p,1405)
    missing,extra=q.load_state_dict(q0.state_dict(),strict=False)
    assert set(missing)=={'compat_centers','compat_radii','compat_weights'} and not extra
    q.compat_centers.fill_(1e6);assert not q.compat_weights.any()
    blob,joint=A.serialized(q,p);fd=os.memfd_create('T694_zero_readout_loader_only',flags=os.MFD_CLOEXEC)
    try:
        os.write(fd,blob);restored=A.frozen(p,f'/proc/self/fd/{fd}')
        assert restored.phash()==q.phash() and digest(restored.state_dict())==digest(q.state_dict())
        row=p['protected_rows'][0];a=T.R.E.N.S.arrays(row['source_path'])
        _,a0=A.trajectory(q0,a,row['task_id'],a['actions']);_,a1=A.trajectory(restored,a,row['task_id'],a['actions'])
        assert torch.equal(a0,a1)
    finally:os.close(fd)
    assert digest(q0.state_dict())==before and not q.compat_weights.any()
    assert digest(joint['actor_opt'])==digest(A.Base.P.checkpoint(p['initial']['path'])['joint_state']['actor_opt'])
    # Unit basis fixture: exact cardinal and outside-support values, including
    # deliberate negative verification. It is not a fitted navigation model.
    c=torch.zeros(2,184,dtype=torch.float64);c[1,0]=2;r=torch.tensor([.49,.49],dtype=torch.float64)
    assert torch.equal(A.basis(c,c,r),torch.eye(2,dtype=torch.float64))
    assert not A.basis(torch.full((2,184),10.,dtype=torch.float64),c,r).any()
    assert A.same_float32(np.array([.5],np.float32),np.array([.5],np.float32))
    assert not A.same_float32(np.array([0.],np.float32),np.array([-0.],np.float32))
    # Real standard native auditor/own-carry executes against saved data; no
    # Session is physically stepped. Corrupt provenance must be rejected.
    N.P.frozen=A.frozen
    rr=dict(job_id=row['task_id'],request=dict(model=p['initial']['path']),seed=None,teacher=False,
            physics=int(a['native_counts'].sum()),array_hashes={k:digest(v) for k,v in a.items()})
    N.inspect(p,a,rr);N.audit_source_hashes(p,rr)
    broken=copy.deepcopy(rr);broken['array_hashes']['actions']='0'*64
    rejected(lambda:N.audit_source_hashes(p,broken))
    bad=dict(a,ownership=a['ownership'].copy());bad['ownership'][0]=1
    rejected(lambda:N.inspect(p,bad,rr))
    seen=[];original=os.sched_setaffinity
    def worker_witness(pipe,index,pc,counts):
        assert N.G.T.Session is N.C.Session and N.P.frozen is A.frozen
        assert callable(N.G.B.native_setup);seen.append(index)
    try:
        with mock.patch.object(N.G.T,'worker',worker_witness):N.worker(None,0,p,None)
    finally:os.sched_setaffinity=original
    assert seen==[0]
    # Existing canonical Session guards zero steps/GRU, scene/goal and available
    # initial contract before first physical step in formal execution.
    text=inspect.getsource(N.C.Session.__init__)
    assert 'self.steps==0' in text and 'self.hidden.any()' in text and 'self.scene==expected' in text
    assert T.R.N.Pool.__init__.__globals__['worker'] is N.worker
    assert len(p['screen_jobs'])+len(p['screen_jobs'])+(407-len(p['screen_jobs']))+6==p['child_max_commands']==468
    assert p['child_max_physics']/2500+120<5700
    # Complete persistent bound: models, full55, compact407, three partial
    # worker payloads and bounded metadata. No deletion is used for admission.
    fit_upper=3*p['model_bytes']+24*2**20
    evaluation_upper=55*p['full_bytes']+407*p['compact_bytes']+3*p['full_bytes']+24*2**20
    assert fit_upper<p['fit_bytes'] and evaluation_upper<p['child_bytes']
    assert p['fit_bytes']+p['child_bytes']+p['final_storage_reserve']<p['max_bytes']
    assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<auth['archive_trigger_bytes']
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<auth['memory_bytes_cap']
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_NO_COEFFICIENT_COMMIT',
        zero_model_real_RAM_serialization_and_actual_loader_batch1_exact=True,
        standard_fullhistory_native_auditor_and_source_physical_hashes=True,corrupt_hash_and_ownership_rejected=True,
        worker_canonical_factory_owncarry_loader_bindings=True,real_new_scene_pending_formal_first_action=True,
        old_optimizer_provenance_unchanged_NOT_new_head_training_optimizer=True,
        byte_bound=dict(fit_upper=fit_upper,evaluation_upper=evaluation_upper),
        new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,peak_rss_bytes=peak,elapsed_seconds=time.time()-started)
    save(DOC/'PREFLIGHT.json',result)
    deps={Path(f) for f in old['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREPARATION.json',DOC/'PREFLIGHT.json',
        DOC.parent/'EXACT_COMMAND_RETENTION_PRIMARY_REVIEW.md',DOC.parent/'EXACT_ZERO_COORDINATE_PREPARATION.json',
        ROOT/'stage/docs/day_20261114/T546_FINDINGS.md',ROOT/'stage/docs/day_20261114/T547_FINDINGS_AND_NEXT.md',
        ROOT/'stage/docs/day_20261114/T550_REGRESSION_FINDINGS_AND_NEXT.md'}
    deps|={Path(r['source_path']) for r in p['protected_rows']}|{Path(r['path']) for r in p['novel_rows']}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    p['preflight_seconds']=pre['elapsed_seconds']+20+time.time()-started+10
    (HERE/'protocol.json').write_text(json.dumps(p,separators=(',',':'))+'\n')
    assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['night_main_seconds','audit_total_seconds','night_max_physics','night_max_commands','max_bytes','root_seconds','preflight_seconds']})
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
