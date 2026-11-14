"""Sealed raw evidence, RAM-only remote archive, complete member readback."""
from pathlib import Path
import importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t552_joint_robust_night'
spec=importlib.util.spec_from_file_location('_t532_archive_backend',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def stopped():
    s=json.loads((OUT/'SEALED.json').read_text());assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
def main():
    A.OLD.install();T=A.OLD.T
    if sys.argv[1]=='source_sync':
        import sync_source;sync_source.main();return
    assert sys.argv[1]=='T552';stopped();A.stopped=stopped
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T552.json';assert not receipt.exists()
    assert (ROOT/'stage/docs/overnight_20261114/FINAL_REPORT.json').exists()
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name!='driver.log' and f.suffix not in ['.partial','.lock','.pyc'])
    deadline=min(time.time()+850,json.loads((OUT/'protocol.json').read_text())['absolute_all_stop']-120)
    r=A.archive('T552',files,release,deadline,T);stopped()
    T.B.write(receipt,dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T552',url=release['html_url'],members=r['members'],bytes=r['bytes'],child_receipts=[r['group']],excluded_active_finalizer_log='driver.log',verified_epoch=time.time()))
    print('T552_FULL_RAW_BACKUP_PASS',flush=True)
if __name__=='__main__':main()
