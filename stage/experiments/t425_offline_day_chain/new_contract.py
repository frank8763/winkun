"""Fresh-process one-step T425 optimizer contract; never old-history repair.

The registered runner invokes `probe A`, `probe B` in separate Python
processes, then `compare`. Each probe discards exactly one fresh-Adam step.
Only the three designated preflight JSON receipts may be written. Importing
this module performs no simulation, optimization, model export or file write.
"""
from pathlib import Path
import sys,os,time,json,hashlib,platform,copy

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t425_offline_day_chain'
import t425_learning as LEARN
E,torch,np=LEARN.E,LEARN.torch,LEARN.np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path=Path(path)
    assert path.parent==OUT
    assert path.name in ['preflight_clone_A.json','preflight_clone_B.json',
                         'preflight_clone_comparison.json']
    assert not path.exists(),('Do not overwrite a preflight clone receipt',path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def _runtime():
    return dict(python=platform.python_version(),torch=str(torch.__version__),
                numpy=str(np.__version__),default_dtype=str(torch.get_default_dtype()),
                default_device=str(torch.get_default_device()),
                intraop_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                grad_enabled=torch.is_grad_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cpu_autocast_enabled=torch.is_autocast_enabled('cpu'),
                OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
                MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
                OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'))


def probe(label):
    assert label in ['A','B']
    destination=OUT/('preflight_clone_'+label+'.json')
    assert not destination.exists()
    start=time.time();deadline=start+300
    p=_read(OUT/'protocol.json')
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for path,digest in p['inputs'].items():assert _sha(path)==digest,path
    E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def forbidden(*args,**kwargs):
        raise AssertionError('New preflight contract forbids native simulation and model export')
    E.L.Session=forbidden;E.Pool=forbidden;E.export=forbidden;E.N.write_npz=forbidden
    def guard():assert time.time()<deadline-5
    assert _sha(p['student_start_model'])==p['student_start_sha256']
    q=E.safe_initial(E.N.science(p),p).eval()
    expected=p.get('expected_policy_hash',p.get('student_policy_hash'))
    if expected is not None:assert q.phash()==expected
    opt=LEARN.make_optimizer(q,p)
    assert opt.defaults['foreach'] is False and opt.defaults['fused'] is False
    assert not opt.state
    variables=E.params(q)
    names={id(value):name for name,value in q.named_parameters()}
    ordered=[names[id(value)] for group in opt.param_groups for value in group['params']]
    assert ordered==[names[id(value)] for value in variables]
    assert len(ordered)==len(set(ordered))
    assert sum(value.numel() for value in variables)==10722
    mapping=[dict(index=index,name=names[id(value)],shape=list(value.shape),
                  dtype=str(value.dtype),requires_grad=value.requires_grad)
             for index,value in enumerate(variables)]
    anchors=[]
    anchor_sources=[]
    for entry in p['anchors']:
        guard();assert _sha(entry['path'])==entry['sha256']
        arrays=E.N.S.arrays(entry['path']);row=entry['row']
        assert E.digest(arrays)==row['array_digest']
        assert row['policy_hash']==q.phash()
        anchors.append((arrays,row))
        anchor_sources.append(dict(path=entry['path'],sha256=entry['sha256'],
                                   job_id=row['job_id'],array_digest=row['array_digest']))
    fixture_root=ROOT/'stage/runs/t422_feedback_recovery_night'
    fixture_summary=_read(fixture_root/'search/summary.json')
    assert fixture_summary['state']=='COMPLETE'
    assert fixture_summary['result']['valid_recovery']
    assert _read(fixture_root/'search/audit.json')['state']=='PASS'
    fixture=Path(fixture_summary['result']['path'])
    assert _sha(fixture)==fixture_summary['result']['sha256']
    arrays=E.N.S.arrays(fixture)
    targets=[dict(arrays=arrays,first=17,id='PREFLIGHT_FIXED_T422_VERIFIED_RECOVERY')]
    groups=LEARN.make_groups(q,p,anchors,targets)
    frozen_before=E.original_hash(q)
    before=E.L.statehash(q,opt)
    runtime_before=_runtime()
    events=[];calls=0
    original_step=torch.optim.Adam.step
    def limited_step(optimizer,*args,**kwargs):
        nonlocal calls
        guard();assert calls==0,'Only one discarded clone Adam step allowed'
        result=original_step(optimizer,*args,**kwargs);calls+=1;return result
    torch.optim.Adam.step=limited_step
    def event(where,index):
        assert index==0 and where in ['BEFORE','AFTER']
        gradient={name:value.grad for name,value in q.named_parameters()
                  if name in ordered}
        assert all(value is not None for value in gradient.values())
        events.append(dict(where=where,index=index,hashes=E.L.statehash(q,opt),
                           clipped_parameter_gradient_sha256=E.digest(gradient),
                           gradient_dtypes={name:str(value.dtype)
                                            for name,value in gradient.items()}))
    pp=copy.deepcopy(p);pp['preflight_clone_probe']=True
    fit=LEARN.fit_a(q,opt,groups,pp,guard,event,steps=1)
    assert calls==1 and len(events)==2
    assert [item['where'] for item in events]==['BEFORE','AFTER']
    after=E.L.statehash(q,opt)
    frozen_after=E.original_hash(q)
    assert before!=after and before['learned']!=after['learned']
    assert frozen_before==frozen_after
    state=opt.state_dict()
    tensor_metadata={str(index):{key:dict(dtype=str(value.dtype),shape=list(value.shape))
                     for key,value in slot.items() if isinstance(value,torch.Tensor)}
                     for index,slot in state['state'].items()}
    contract=dict(protocol_sha256=_sha(OUT/'protocol.json'),
                  source_files={str(file):_sha(file) for file in [HERE/'new_contract.py',HERE/'t425_learning.py']},
                  model_sha256=p['student_start_sha256'],initial_policy_hash=expected,
                  fixture=dict(path=str(fixture),sha256=_sha(fixture),first=17,
                               existing_verified_recovery_not_new_navigation=True),
                  anchors=anchor_sources,ordered_Adam_parameter_mapping=mapping,
                  optimizer_defaults=opt.defaults,param_groups=state['param_groups'],
                  optimizer_state_tensor_metadata=tensor_metadata,
                  runtime_before=runtime_before,runtime_after=_runtime(),
                  before_hashes=before,after_hashes=after,events=events,fit=fit,
                  frozen_original_before=frozen_before,frozen_original_after=frozen_after,
                  native=0,discarded_fresh_Adam=1,new_navigation_training=0,
                  new_retained_policy=0,no_model_export=True,
                  old_T422_audit_not_repaired=True)
    _write(destination,dict(state='FRESH_CLONE_ONE_STEP_COMPLETE',label=label,
                            pid=os.getpid(),elapsed_s=time.time()-start,contract=contract))
    print('T425_NEW_CLONE_CONTRACT_WRITTEN',label,after,flush=True)


def compare():
    a=_read(OUT/'preflight_clone_A.json');b=_read(OUT/'preflight_clone_B.json')
    assert a['state']==b['state']=='FRESH_CLONE_ONE_STEP_COMPLETE'
    assert a['label']=='A' and b['label']=='B' and a['pid']!=b['pid']
    assert a['contract']==b['contract'],'Independent fresh-process contract differs'
    contract=a['contract']
    assert contract['before_hashes']!=contract['after_hashes']
    assert contract['before_hashes']['learned']!=contract['after_hashes']['learned']
    assert contract['frozen_original_before']==contract['frozen_original_after']
    assert contract['native']==contract['new_navigation_training']==contract['new_retained_policy']==0
    assert contract['discarded_fresh_Adam']==1
    _write(OUT/'preflight_clone_comparison.json',dict(
        state='PASS_NEW_FRESH_PROCESS_ONE_STEP_CONTRACT_ONLY',native=0,
        discarded_fresh_clone_Adam=2,new_navigation_training=0,new_retained_policy=0,
        processes_are_distinct=True,full_numeric_contract_exact=True,
        complete_Actor_Adam_before_and_after_exact=True,
        every_clipped_gradient_SHA_parameter_order_options_and_runtime_exact=True,
        actual_parameter_change_verified=True,frozen_original_state_unchanged=True,
        clone_A_sha256=_sha(OUT/'preflight_clone_A.json'),
        clone_B_sha256=_sha(OUT/'preflight_clone_B.json'),
        original_T422_audit_remains_FAIL=True,
        scope='A new independently reproducible preflight step; not old T422 history repair or proof that future full training will reconstruct.'))
    print('T425_NEW_FRESH_PROCESS_CLONE_CONTRACT_PASS',flush=True)


if __name__=='__main__':
    assert sys.argv[1:] in [['probe','A'],['probe','B'],['compare']]
    if sys.argv[1]=='compare':compare()
    else:probe(sys.argv[2])
