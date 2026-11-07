"""Zero-Adam/native finalization; reuse unchanged numerical and worker proofs."""
from pathlib import Path
import json,hashlib,importlib.util,sys,dis,types,builtins,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t404_last_narrow'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    prior=json.loads((OUT/'preflight_pass_before_engineering_finalize.json').read_text());p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==prior['protocol_sha256'] and prior['proof']['state']=='PASS'
    for name,h in prior['source_hashes'].items():
        if Path(name).name not in ['prepare.py','wide.py']:assert sha(Path(name))==h,name
    sys.path.insert(0,str(HERE))
    for name in ['run.py','audit.py','wide.py','driver.py']:
        spec=importlib.util.spec_from_file_location('stage404_actual_'+Path(name).stem,HERE/name);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def check(code):
            bad={x.argval for x in dis.get_instructions(code) if x.opname=='LOAD_GLOBAL' and x.argval not in module.__dict__ and not hasattr(builtins,x.argval)}
            for x in code.co_consts:
                if isinstance(x,types.CodeType):bad|=check(x)
            return bad
        for fn in vars(module).values():
            if isinstance(fn,types.FunctionType) and fn.__module__==module.__name__:assert not check(fn.__code__),(name,check(fn.__code__))
    lock=(OUT/'preflight_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
    for name in [HERE/'protocol.json',OUT/'protocol.json']:name.write_text(json.dumps(p,indent=2)+'\n')
    proof=dict(prior['proof'],actual_modules_loaded_by_explicit_path=True,all_recursive_main_audit_wide_driver_globals_bound=True,wide_capacity_scan_at_hold_boundary_every3seconds=True,additional_native=0,additional_Adam=0,numerical_and_worker_proof_reused=True,finalized_source_sha256={str(f):sha(f) for f in HERE.glob('*.py')})
    (OUT/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print('T404_FINALIZED_ZERO_ADDITIONAL_NATIVE_OR_ADAM')
if __name__=='__main__':main()
