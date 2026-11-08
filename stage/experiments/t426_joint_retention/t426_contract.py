"""Two fresh-process one-step contracts for the new T426 learner only.

Uses authenticated historical full-state fixtures, explicitly not a claim of
the new starting Actor's61 native successes. No native rollout, model export,
retained training or original T422 history repair occurs in these probes.
"""
from pathlib import Path
import copy,hashlib,json,os,platform,sys,time
import t426_learning as L


def _runtime(torch,np):
    return dict(python=platform.python_version(),torch=str(torch.__version__),
        numpy=str(np.__version__),default_dtype=str(torch.get_default_dtype()),
        intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),
        cpu_autocast=torch.is_autocast_enabled('cpu'),grad_enabled=torch.is_grad_enabled(),
        OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
        MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
        OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'))


def fixture_sources(p):
    """Return existing immutable source rows; importing has no read effects."""
    root=Path(__file__).resolve().parents[3]
    old=json.loads((root/'stage/runs/t425_offline_day_chain/protocol.json').read_text())
    old_ids={job['id'] for job in p['old8_jobs']}
    six_ids={job['id'] for job in p['main_jobs']}
    old_entry=next(entry for entry in old['anchors'] if entry['row']['job_id'] in old_ids)
    six_entry=next(entry for entry in old['anchors'] if entry['row']['job_id'] in six_ids)
    targets=json.loads((root/'stage/runs/t425_offline_day_chain/data/targets.json').read_text())
    targets=[entry for entry in targets if entry['first']==0][:11]
    assert len(targets)==11
    return [old_entry,six_entry],targets


def _write(C,path,value):
    assert path.parent==C.OUT and path.name in ['clone_A.json','clone_B.json','clone_comparison.json']
    assert not path.exists(),('Do not repeat completed clone receipt',path)
    C.save(path,value)


def probe(label):
    import t426_core as C
    assert label in ['A','B']
    E,torch,np=C.E,C.torch,C.np
    p=C.read(C.OUT/'protocol.json');C.verify(p)
    assert not (C.OUT/('clone_'+label+'.json')).exists()
    deadline=time.time()+180;started=time.time()
    E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<deadline-5
    def forbidden(*args,**kwargs):raise AssertionError('T426 preflight forbids native/model exports')
    E.L.Session=forbidden;E.Pool=forbidden;E.export=forbidden;E.N.write_npz=forbidden
    assert C.sha(p['student_start_model'])==p['student_start_sha256']
    q=C.frozen(p);initial_policy_hash=q.phash()
    assert initial_policy_hash==p['expected_policy_hash']
    opt=L.make_optimizer(q,p)
    assert not opt.state
    names={id(value):name for name,value in q.named_parameters()}
    ordered=[names[id(value)] for group in opt.param_groups for value in group['params']]
    assert ordered==[names[id(value)] for value in E.params(q)] and len(set(ordered))==len(ordered)
    mapping=[dict(index=k,name=names[id(value)],shape=list(value.shape),dtype=str(value.dtype))
             for k,value in enumerate(E.params(q))]
    anchor_entries,target_entries=fixture_sources(p)
    anchors=[];sources=[]
    for entry in anchor_entries:
        assert C.sha(entry['path'])==entry['sha256']
        arrays=E.N.S.arrays(entry['path']);assert C.digest(arrays)==entry['row']['array_digest']
        anchors.append((arrays,entry['row']))
        sources.append(dict(kind='historical_anchor_not_currentA_native',path=entry['path'],
            sha256=entry['sha256'],id=entry['row']['job_id'],policy_hash=entry['row']['policy_hash']))
    targets=[]
    for entry in target_entries:
        assert C.sha(entry['path'])==entry['sha256']
        arrays=E.N.S.arrays(entry['path']);assert C.digest(arrays)==entry['row']['array_digest']
        targets.append(dict(arrays=arrays,first=0,id=entry['id']))
        sources.append(dict(kind='historical_actual_safe_t0_recovery',path=entry['path'],
            sha256=entry['sha256'],id=entry['id'],first=0))
    pp=copy.deepcopy(p);pp['preflight_clone_probe']=True
    groups=L.groups(q,pp,anchors,targets)
    initial=E.L.statehash(q,opt);frozen_before=E.original_hash(q)
    runtime=_runtime(torch,np);events=[];calls=0
    original_step=torch.optim.Adam.step
    def step(optimizer,*args,**kwargs):
        nonlocal calls
        guard();assert calls==0,'Exactly one discarded Adam permitted'
        result=original_step(optimizer,*args,**kwargs);calls+=1;return result
    torch.optim.Adam.step=step
    def event(where,k):
        guard();assert where in ['BEFORE','AFTER'] and k==0
        gradient={name:value.grad for name,value in q.named_parameters() if name in ordered}
        assert all(value is not None for value in gradient.values())
        events.append(dict(where=where,k=k,hashes=E.L.statehash(q,opt),
            gradient_sha256=C.digest(gradient),gradient_dtypes={name:str(v.dtype) for name,v in gradient.items()}))
    fit=L.stepfit(q,opt,groups,pp,0,1,guard,event)
    after=E.L.statehash(q,opt)
    assert calls==1 and len(events)==2 and initial['learned']!=after['learned']
    assert E.original_hash(q)==frozen_before
    contract=dict(protocol_sha256=C.sha(C.OUT/'protocol.json'),
        source_files={str(f):C.sha(f) for f in [C.HERE/'t426_contract.py',C.HERE/'t426_learning.py']},
        starting_model_sha256=p['student_start_sha256'],starting_policy_hash=initial_policy_hash,
        fixture_sources=sources,fixtures_not_currentA_success_claim=True,
        ordered_parameter_mapping=mapping,optimizer_defaults=opt.defaults,
        param_groups=opt.state_dict()['param_groups'],runtime_before=runtime,
        runtime_after=_runtime(torch,np),before_hashes=initial,after_hashes=after,
        clipped_gradients_and_states=events,fit=fit,
        frozen_original_before=frozen_before,frozen_original_after=E.original_hash(q),
        native=0,new_navigation_training=0,retained_Actor=0,discarded_fresh_Adam=1,
        no_model_export=True,old_T422_training_audit_not_repaired=True)
    _write(C,C.OUT/('clone_'+label+'.json'),dict(state='ONE_DISCARDED_CLONE_COMPLETE',
        label=label,pid=os.getpid(),elapsed_s=time.time()-started,contract=contract))
    print('T426_CLONE_WRITTEN',label,after,flush=True)


def compare():
    import t426_core as C
    a=C.read(C.OUT/'clone_A.json');b=C.read(C.OUT/'clone_B.json')
    assert a['state']==b['state']=='ONE_DISCARDED_CLONE_COMPLETE'
    assert a['label']=='A' and b['label']=='B' and a['pid']!=b['pid']
    assert a['contract']==b['contract'],'New independent-process numerical contract differs'
    contract=a['contract'];assert contract['before_hashes']!=contract['after_hashes']
    assert contract['native']==contract['new_navigation_training']==contract['retained_Actor']==0
    _write(C,C.OUT/'clone_comparison.json',dict(state='PASS_NEW_ONE_STEP_CROSS_PROCESS_CONTRACT',
        processes_distinct=True,full_numeric_contract_exact=True,
        Actor_Adam_parameter_order_gradients_runtime_exact=True,discarded_fresh_Adam=2,
        native=0,new_navigation_training=0,retained_Actor=0,
        historical_fixtures_not_currentA_native_successes=True,
        old_T422_audit_remains_FAIL=True,
        clone_A_sha256=C.sha(C.OUT/'clone_A.json'),clone_B_sha256=C.sha(C.OUT/'clone_B.json')))
    print('T426_NEW_CLONE_CONTRACT_PASS',flush=True)


if __name__=='__main__':
    assert sys.argv[1:] in [['probe','A'],['probe','B'],['compare']]
    if sys.argv[1]=='compare':compare()
    else:probe(sys.argv[2])
