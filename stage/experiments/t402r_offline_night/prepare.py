"""Zero-native entry repair; reuse exact already passed numerical/worker checks."""
from night import *
import ast,dis,builtins,types,importlib.util

def register():
    previous=ROOT/'stage/runs/t402_offline_night';old=read(previous/'protocol.json');review=read(DOC/'T402_ENTRY_FAILURE_REVIEW.json');assert review['state']=='PASS_ZERO_NATIVE_ZERO_LIVE_ENTRY_FAILURE_REVIEW'
    assert not OUT.exists();launch=read(previous/'launch.json');p=copy.deepcopy(old)
    p.update(id='T402R_ORIGINAL_BUDGET_ZERO_NATIVE_ENTRY_RECOVERY',learning_stop_epoch=min(old['learning_stop_epoch'],launch['epoch']+old['learning_seconds']),native_stop_epoch=min(old['native_stop_epoch'],launch['epoch']+old['main_seconds']),prior_entry_native=0,prior_entry_optimizer=0,original_preflight_commands=8,original_preflight_Adam=12)
    p['inputs']={k:v for k,v in p['inputs'].items() if '/stage/experiments/t402_offline_night/' not in k}
    for f in [*HERE.glob('*.py'),previous/'preflight.json',previous/'final_report.json',DOC/'T402_ENTRY_FAILURE_REVIEW.json',DOC/'ENTRY_RECOVERY.md']:p['inputs'][str(f)]=sha(f)
    used=N.K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'blocks').mkdir();(OUT/'probes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'RECOVERY_BUDGET.json',dict(state='ORIGINAL_COMBINED_BUDGET_ADMISSION_PASS',original_native=0,original_liveAdam=0,original_preflight_Adam=12,new_preflight_Adam=0,max_physics=p['max_physics'],max_commands=p['max_commands'],own=p['final_bytes'],global_before=used,full_reserved=used+p['final_bytes'],learning_stop=p['learning_stop_epoch'],native_stop=p['native_stop_epoch'],all_stop=p['all_stop_epoch'],extension_seconds=0))
    print('T402R_REGISTERED_ORIGINAL_BUDGET_NOT_LAUNCHED')

def preflight():
    N.offline();p=load_plan();base=verify(p);previous=ROOT/'stage/runs/t402_offline_night';passed=read(previous/'preflight.json');assert passed['state']=='PASS' and passed['discarded_Adam']==12
    # Pure namespace/import repair: compare complete scientific code AST.
    oldroot=ROOT/'stage/experiments/t402_offline_night'
    for name in ['night.py','run.py','audit.py']:
        old=(oldroot/name).read_text().replace('t402_offline_night','t402r_offline_night').replace('T402','T402R')
        new=(HERE/name).read_text()
        if name=='night.py':new=new.replace('contextlib,resource,fcntl','contextlib,resource')
        assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new)),name
    spec=importlib.util.spec_from_file_location('stage402r_real_main_entry',HERE/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def check(code):
        for op in dis.get_instructions(code):
            if op.opname=='LOAD_GLOBAL':assert op.argval in module.__dict__ or hasattr(builtins,op.argval),('Unbound actual main global',op.argval)
        for value in code.co_consts:
            if isinstance(value,types.CodeType):check(value)
    check(module.main.__code__)
    # Execute the actual failing lock operation before any optimizer or physics.
    with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
    save(OUT/'preflight.json',dict(state='PASS',native=0,new_discarded_Adam=0,reused_original_discarded_Adam=12,reused_four_cpu_native_worker_READY_cancel_exit=True,scientific_AST_unchanged=True,all_recursive_actual_main_globals_bound=True,actual_writer_lock_PASS=True,original_pass_sha256=sha(previous/'preflight.json')))
    save(DOC/'RECOVERY_READINESS.json',read(OUT/'preflight.json'));print('T402R_ACTUAL_MAIN_GLOBALS_LOCK_PASS_REUSE12CLONES_ZERO_NATIVE')

if __name__=='__main__':globals()[sys.argv[1]]()
