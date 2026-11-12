"""Sealed T498 children and root, RAM archives and full binary member SHA."""
from pathlib import Path
import importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t498_early_error_repair'
spec=importlib.util.spec_from_file_location('_stage498_archive_helper',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def stopped():
    s=json.loads((OUT/'SEALED.json').read_text());assert s['all_scientific_processes_stopped'] and s['finalizer_writes_only_docs_after_this_point']
    assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
def main():
    A.OLD.install();T=A.OLD.T
    if sys.argv[1]=='source_sync':
        import sync_source;sync_source.main();return
    assert sys.argv[1]=='T498';stopped();A.stopped=stopped
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');deadline=time.time()+750;receipts=[]
    for child in ['repair']:
        folder=OUT/child
        if not folder.exists():continue
        assert json.loads((folder/'audit.json').read_text())['state']=='PASS'
        files=sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix not in ['.partial','.lock','.pyc'])
        receipts.append(A.archive('T498_'+child,files,release,deadline,T))
    # The live supervisor's own stdout is not scientific data and may change
    # during finalization; never archive a still-open finalizer log as sealed.
    files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.partial','.lock','.pyc'])
    receipts.append(A.archive('T498_root',files,release,deadline,T));stopped()
    T.B.write(ROOT/'stage/docs/long_distance_20261101/release_receipt_T498.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T498',url=release['html_url'],members=sum(r['members'] for r in receipts),child_receipts=[r['group'] for r in receipts],bytes=sum(r['bytes'] for r in receipts),excluded_active_finalizer_log='driver.log (not a scientific trajectory/optimizer record)',verified_epoch=time.time()))
    print('T498_FULL_RAW_BACKUP_PASS',flush=True)
if __name__=='__main__':main()
