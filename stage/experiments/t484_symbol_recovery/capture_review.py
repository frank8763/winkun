"""Re-run original completed-capture audit with one exact symbol adapter.

No original source/protocol/result file is modified. The audit output goes to a
new review artifact, and the entire sealed original run is SHA-checked unchanged.
"""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/review_20261111';OLD=ROOT/'stage/runs/t481_fresh16_rehearsal'
sys.path.insert(0,str(ROOT/'stage/experiments/t481_fresh16_rehearsal'))
import t481_runtime as A

def snapshot():return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OLD.rglob('*') if f.is_file()}
def main():
    report=A.read(DOC/'T481_FINAL_REPORT.json')
    assert report['state']=='BOUNDED_QUEUE_STOPPED' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for f in ['launch.json','driver_started.json']:
        assert not Path('/proc',str(A.read(OLD/f)['pid'])).exists()
    assert A.read(OLD/'capture/summary.json')['state']=='COMPLETE'
    assert not (OLD/'capture/audit.json').exists() and not (DOC/'T484_CAPTURE_AUDIT.json').exists()
    before=snapshot();original=A.save
    assert not hasattr(A.L,'vtrace')
    A.L.vtrace=A.L.B.vtrace
    def redirected(path,value):
        assert Path(path)==OLD/'capture/audit.json','Unexpected write to immutable original run'
        original(DOC/'T484_CAPTURE_AUDIT.json',dict(value,interface_only_exact_symbol_reexport=True,original_run_unchanged=True))
    A.save=redirected
    A.audit('capture')
    assert snapshot()==before
    original(DOC/'T484_CAPTURE_REVIEW_PROVENANCE.json',dict(state='PASS_ORIGINAL_CAPTURE_AUDIT_WITH_EXACT_SYMBOL_ADAPTER',native=0,optimizer=0,epoch=time.time(),original_all_files_unchanged=True,original_files=len(before),symbol_original_source=str(ROOT/'stage/experiments/t480_old_task_rehearsal/t480_learning.py'),original_failure_preserved=True,completed16_plus16_not_recollected=True))
    print('T484_EXISTING_CAPTURE_AUDIT_PASS_NO_NATIVE_NO_ADAM',flush=True)
if __name__=='__main__':main()
