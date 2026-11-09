"""Recover only the stopped T437 audit's Python import seam, no new training.

The immutable audit is executed with one explicit, source-pinned producer
module load replacing its ambiguous `import run`. Original outputs and error
are never overwritten. Native calls remain forbidden by the original audit.
"""
from pathlib import Path
import sys,json,time,hashlib,types,traceback,os
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'stage/runs/t437_joint_closed_loop_rl'
OUT=ROOT/'stage/runs/t438_t437_audit_recovery'
AUDIT=ROOT/'stage/experiments/t437_joint_closed_loop_rl/audit.py'
PRODUCER=AUDIT.parent/'run.py'
DOC=ROOT/'stage/docs/review_20261109'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def inventory():return {str(f.relative_to(ROOT)):sha(f) for f in OLD.rglob('*') if f.is_file()}
def prepare():
    assert not OUT.exists()
    p=read(OLD/'protocol.json');s=read(OLD/'fresh_rl/summary.json');err=read(OLD/'fresh_rl/audit_error.json')
    assert s['state']=='COMPLETE' and s['all_workers_stopped']
    assert "AttributeError: module 'run' has no attribute 'stage_source'" in err['error']
    # The traceback failed before dataset construction or the first historical fit.
    assert 'learning_audit' in err['error'] and 'reconstruct_pair' not in err['error']
    for file,key in [(OLD/'launch.json','pid'),(OLD/'driver_started.json','pid'),(OLD/'fresh_rl/main_started.json','pid')]:
        assert not Path('/proc',str(read(file)[key])).exists()
    assert all(not Path('/proc',str(pid)).exists() for pid in read(OLD/'fresh_rl/workers.json')['workers'])
    assert not (OLD/'fresh_rl/audit.json').exists()
    assert sha(AUDIT)==p['inputs'][str(AUDIT)] and sha(PRODUCER)==p['inputs'][str(PRODUCER)]
    assert time.time()+1800+600<p['deadline_all']
    OUT.mkdir()
    save(OUT/'protocol.json',dict(id='T438_T437_AUDIT_IMPORT_RECOVERY_ONLY',original_protocol_sha256=sha(OLD/'protocol.json'),
        original_audit_sha256=sha(AUDIT),original_producer_sha256=sha(PRODUCER),wrapper_sha256=sha(Path(__file__)),
        original_error_sha256=sha(OLD/'fresh_rl/audit_error.json'),before_inventory=inventory(),
        max_discarded_historical_Adam=128,already_executed_historical_Adam=0,max_native=0,new_training=0,
        max_seconds=1800,all_stop=p['deadline_all'],own_bytes=2**20,
        scientific_protocol_and_Actor_Adam_unchanged=True,original_errors_preserved=True,
        authorization='User check request plus existing bounded independent T437 audit; repair import only, no new simulation or training.'))
    print('T438_AUDIT_ONLY_REGISTERED',flush=True)
def main():
    p=read(OUT/'protocol.json');assert sha(Path(__file__))==p['wrapper_sha256']
    assert inventory()==p['before_inventory'] and sha(AUDIT)==p['original_audit_sha256'] and sha(PRODUCER)==p['original_producer_sha256']
    assert not (OUT/'started.json').exists()
    start=time.time();save(OUT/'started.json',dict(pid=os.getpid(),epoch=start))
    sys.path.insert(0,str(AUDIT.parent))
    import t437_core as C
    source=AUDIT.read_text();needle='    import run as producer\n';assert source.count(needle)==1
    replacement="    producer_spec=importlib.util.spec_from_file_location('stage438_exact_t437_producer', ROOT/'stage/experiments/t437_joint_closed_loop_rl/run.py')\n    producer=importlib.util.module_from_spec(producer_spec);producer_spec.loader.exec_module(producer)\n    assert hasattr(producer,'stage_source') and Path(producer.__file__).resolve()==ROOT/'stage/experiments/t437_joint_closed_loop_rl/run.py'\n"
    source=source.replace(needle,replacement)
    assert source.count("C.save(out/'audit.json',")==1
    source=source.replace("C.save(out/'audit.json',","C.save(REC/'audit.json',")
    assert source.count('C.E.N.manifest(out)')==1
    source=source.replace('C.E.N.manifest(out)','C.E.N.manifest(REC)')
    module=types.ModuleType('stage438_pinned437_audit');module.__file__=str(AUDIT);module.__dict__['REC']=OUT
    exec(compile(source,str(AUDIT),'exec'),module.__dict__)
    original=C.torch.optim.Adam.step;count=[0]
    def measured(opt,*a,**k):
        assert time.time()<min(start+p['max_seconds'],p['all_stop'])-5
        assert count[0]<p['max_discarded_historical_Adam']
        value=original(opt,*a,**k);count[0]+=1
        with (OUT/'historical_Adam_calls.jsonl').open('a') as f:f.write(json.dumps(dict(index=count[0],new_training=False))+'\n')
        return value
    C.torch.optim.Adam.step=measured
    state='FAILED';error=None
    try:
        module.main('fresh_rl')
        assert read(OUT/'audit.json')['state']=='PASS' and count[0]==128
        assert inventory()==p['before_inventory']
        state='COMPLETE'
    except BaseException:error=traceback.format_exc()
    finally:
        save(OUT/'summary.json',dict(state=state,error=error,elapsed_s=time.time()-start,
            native=0,new_training=0,discarded_historical_Adam=count[0],
            original_run_all_bytes_unchanged=inventory()==p['before_inventory'],
            original_import_error_preserved=True,original_scientific_protocol_not_modified=True))
    assert state=='COMPLETE',error
    print('T438_INDEPENDENT_AUDIT_RECOVERED_PASS_128_HISTORICAL_ADAM',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
