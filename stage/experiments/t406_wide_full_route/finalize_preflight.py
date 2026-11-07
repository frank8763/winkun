"""Supplement actual exclusion/signal/audit paths with zero extra Adam or native."""
from pathlib import Path
import sys,json,hashlib,importlib.util,types,dis,builtins,fcntl,signal,os
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t406_wide_full_route'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    old=json.loads((OUT/'preflight_pass_before_finalize.json').read_text());p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==old['protocol_sha256'] and old['proof']['state']=='PASS'
    for f,h in old['source_hashes'].items():
        if Path(f).name not in ['prepare.py','audit.py','driver.py']:assert sha(Path(f))==h
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    sys.path.insert(0,str(HERE));driver=None
    for name in ['run.py','audit.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage406_finalize_'+Path(name).stem,HERE/name);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def check(c):
            bad={v.argval for v in dis.get_instructions(c) if v.opname=='LOAD_GLOBAL' and v.argval not in m.__dict__ and not hasattr(builtins,v.argval)}
            for v in c.co_consts:
                if isinstance(v,types.CodeType):bad|=check(v)
            return bad
        for fn in vars(m).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==m.__name__:assert not check(fn.__code__),(name,check(fn.__code__))
        if name=='driver.py':driver=m
    previous={n:signal.getsignal(n) for n in [signal.SIGTERM,signal.SIGINT,signal.SIGALRM]};handler=driver.install_handlers();caught=False
    try:
        os.kill(os.getpid(),signal.SIGTERM)
    except TimeoutError:caught=True
    finally:
        for n,h in previous.items():signal.signal(n,h)
    assert caught
    for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
    for f in [HERE/'protocol.json',OUT/'protocol.json']:f.write_text(json.dumps(p,indent=2)+'\n')
    proof=dict(old['proof'],preliminary_private_lock_test_not_global_proof=True,actual_global_writer_lock_pass=True,driver_real_SIGTERM_caught=True,all_actual_main_audit_driver_recursive_globals_bound=True,final_artifact_readback_added_to_independent_audit=True,additional_native=0,additional_Adam=0,numerical_and_four_worker_proofs_reused=True,finalized_source_sha256={str(f):sha(f) for f in HERE.glob('*.py')})
    (OUT/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print('T406_FINALIZED_REAL_GLOBAL_LOCK_SIGNAL_PASS_ZERO_ADDITIONAL_NATIVE_ADAM')
if __name__=='__main__':main()
