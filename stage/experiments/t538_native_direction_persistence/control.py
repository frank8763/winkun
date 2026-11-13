"""Persistence repair only; new 5k replacement allowance must be explicit."""
from pathlib import Path
import copy,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t538_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha
AUTH=DOC/'T538_RECOVERY_AUTHORIZATION.json'
def storage_check():
    assert not OUT.exists();base=read(ROOT/'stage/runs/t537_native_parameter_direction/protocol.json');sizes=[]
    for window in base['windows']:
        with R.np.load(window['reference_suffix_path']) as z:
            a={k.removeprefix('restored__'):z[k] for k in z.files if k.startswith('restored__')}
        # No compression is used. ZIP/NPY byte lengths depend only on field
        # names/shapes/dtypes, not numerical values or cross-branch similarity.
        blob=R.BranchStorage.encode([a]*10);n=len(blob);assert n<160*1024;sizes.append(n)
        import io
        with R.np.load(io.BytesIO(blob)) as z:
            for index in range(10):assert all(R.np.array_equal(v,R.BranchStorage.branch(z,index)[k]) for k,v in a.items())
        # Byte completeness: reject deliberately truncated storage.
        try:
            with R.np.load(io.BytesIO(blob[:-20])) as z:list(z.files)
        except BaseException:pass
        else:raise AssertionError('Truncated NPZ was accepted')
    worst=max(sizes);total=5*worst+320*1024;assert total<2**20
    f=DOC/'T538_STORAGE_CHECK.json';assert not f.exists()
    save(f,dict(state='PASS_EXISTING_ARRAY_STORAGE_ONLY',pid=os.getpid(),new_native=0,new_optimizer=0,measured_window_upper_bytes=sizes,max_window_bytes=worst,raw_and_atomic_partial_plus_metadata_worst_bytes=total,own_budget=2**20,all_10branch_arrays_roundtrip=True,truncated_archive_rejected=True,per_branch_persistence_before_case_log=True,native_measurement_NOT_rerun=True,source_sha256={str(f):sha(f) for f in HERE.glob('*.py')}));print('T538_STORAGE_CONTRACT_PASS_NO_NATIVE',flush=True)
def prepare():
    assert not OUT.exists();auth=read(AUTH)
    assert auth['user_confirmed'] and auth['extra_replacement_physics']==5000 and auth['combined_T537_T538_physics_max']==25000
    prior=ROOT/'stage/runs/t537_native_parameter_direction';base=read(prior/'protocol.json');old=read(prior/'summary.json')
    assert old['state']=='FAILED' and old['physics']==5000 and read(prior/'audit.json')['scope']=='METADATA_ONLY_NOT_FULL_RAW_ARRAY_VERIFICATION'
    assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids'])
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T537.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert time.time()+1380<base['absolute_all_stop'] and auth['absolute_all_stop']==base['absolute_all_stop']
    check=read(DOC/'T538_STORAGE_CHECK.json');assert check['state']=='PASS_EXISTING_ARRAY_STORAGE_ONLY'
    for f,h in check['source_sha256'].items():assert sha(f)==h,f
    p=copy.deepcopy(base);p.update(id='T538_NATIVE_DIRECTION_PERSISTENCE_REPAIR',window_bytes=check['max_window_bytes'],failed_previous_cases=str(prior/'cases.jsonl'),authorization=auth,storage_mode='uncompressed26fields_stacked10branches_atomic_save_after_each_branch_before_metadata',scope='Exact same T537 scientific experiment after explicit additional5k allowance; prior5k firstwindow all hashes must match,all40branches repeated20k with fullarrays saved. TotalT537+T538<=25k. Allparameters/scales/windows/criteria unchanged;no newActor/algorithm/seed/grid or automatic learning.')
    for f in [*HERE.glob('*.py'),DOC/'T538_PLAN.md',DOC/'T538_STORAGE_CHECK.json',AUTH,prior/'summary.json',prior/'audit.json',prior/'cases.jsonl']:p['inputs'][str(f)]=sha(f)
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+65536<2**30
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
    save(DOC/'T538_BUDGET.json',dict(new_physics=20000,prior_T537_physics=5000,combined_T537_T538_physics_max=25000,additional_replacement_authorization=5000,all_T534_T536_T537_T538=136900,commands=45,own_bytes=2**20,window_bytes=p['window_bytes'],complete_raw_atomic_partial_metadata_bound=check['raw_and_atomic_partial_plus_metadata_worst_bytes'],global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+2**20,main_seconds=480,audit_seconds=240,backup_seconds=240,source_seconds=120,Actor_Adam=0,Value_Adam=0,all_stop=p['absolute_all_stop']))
    print('T538_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists()
    # Numerical fixtures already passed T537 at identical scientific settings.
    # Recompute the exact locked policy direction and all source model hashes.
    R.direction(p)
    prior=read(ROOT/'stage/runs/t537_native_parameter_direction/preflight.json');assert prior['state']=='PASS'
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_optimizer=0,storage_contract_sha256=sha(DOC/'T538_STORAGE_CHECK.json'),identical_scientific_first_action_preflight=prior['conditional_hidden_and_first_command_bounds'],direction_and_inputs_verified=True));print('T538_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
