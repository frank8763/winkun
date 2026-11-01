from t278_native import ROOT,C,S,np,torch,policy
from t278_guard import Budget,room
from pathlib import Path
import ast,builtins,dis,fcntl,importlib.util,json,shutil,types
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t278_fixed_goal_starts'


def main():
    assert not (OUT/'preflight.json').exists();p=json.loads((HERE/'protocol.json').read_text())
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    metadata_upper=2*(HERE/'protocol.json').stat().st_size+4*8192+8*8192+128*1024
    assert metadata_upper<=p['metadata_bytes']
    spec=importlib.util.spec_from_file_location('t278_entry',HERE/'run_probe.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    missing=set()
    def check(code):
        for i in dis.get_instructions(code):
            if i.opname=='LOAD_GLOBAL' and i.argval not in vars(mod) and i.argval not in vars(builtins):missing.add(i.argval)
        for co in code.co_consts:
            if isinstance(co,types.CodeType):check(co)
    check(mod.main.__code__);assert not missing,sorted(missing)
    assert room(1024,4096,5120,3072,3072,1024) and not room(1024,4096,5120,3073,3072,1024)
    assert not room(1025,4096,5120,0,3072,1024)
    b=Budget.__new__(Budget);b.outside=1024;b.p=dict(final_bytes=4096,global_bytes=5120,payload_bytes=3072,metadata_bytes=1024);b.payload_bytes=0
    b.charge_payload(3072)
    try:b.charge_payload(1)
    except AssertionError:pass
    else:raise AssertionError('Pending write beyond budget was accepted')
    assert b.payload_bytes==3072
    assert 'logical' not in Budget.hold.__code__.co_names and 'logical' not in Budget.running.__code__.co_names
    test=OUT/'preflight_mutex.lock'
    with test.open('a+b') as a,test.open('a+b') as b:
        fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise AssertionError('Exclusive owner probe not rejected')
        fcntl.flock(a,fcntl.LOCK_UN)
    q=policy(p).eval().requires_grad_(False);assert S.digest(q.state_dict())==p['policy_hash'] and C.sha(p['checkpoint'])==p['checkpoint_sha256']
    for job in p['jobs']:
        with np.load(job['reference_initial_path']) as z:
            x=z['states'][0];assert x.shape==(120,) and x[112]==1
            with torch.no_grad():action=q.step(torch.tensor(x[None]))[0][0].numpy()
            assert action.shape==(2,) and np.isfinite(action).all()
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',physics_steps=0,optimizer_calls=0,native_environments_created=0,
        safe_T255_model_sha_verified=True,no_recursive_scan_in_hold_or_native_step_budget_guard=True,
        exact_budget_boundary_pending_write_and_external_reservation_rejection=True,exclusive_flock_conflict_test=True,
        all_nested_main_globals_resolve=True,source_files_compile=True,all4_reset_cases_locked_no_replacement=True,
        full_payload_and_metadata_reserved=p['payload_bytes']+p['metadata_bytes'],metadata_upper_bytes=metadata_upper,protocol_sha256=C.sha(HERE/'protocol.json'),
        actual_native_reset_validity_and_replay_not_yet_executed=True))
    print(json.dumps(dict(state='PASS',physics_steps=0)),flush=True)


if __name__=='__main__':main()
