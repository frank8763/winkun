"""One RAM-only full raw export after every scientific process has exited."""
from pathlib import Path
import importlib.util
import json
import signal
import time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t634_execution_mean_probe'
spec=importlib.util.spec_from_file_location('_t634_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def read(p):return json.loads(Path(p).read_text())


def stopped():
    s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(i)).exists() for i in s['scientific_pids'])


def main():
    stopped();p=read(OUT/'protocol.json')
    deadline=min(time.time()+p['transport_seconds']-5,p['absolute_all_stop']-90)
    assert deadline>time.time()+60
    def timeout(*args):raise TimeoutError('T634 bounded export')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
    A.OLD.install();T=A.OLD.T
    import urllib.request
    T.B.OPENER=urllib.request.build_opener();A.stopped=stopped
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');receipts=[];total=0
    for folder in sorted(f for f in OUT.iterdir() if f.is_dir()):
        files=sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix not in ['.lock','.pyc','.partial'])
        if files:
            r=A.archive('T634_'+folder.name,files,release,deadline,T)
            receipts.append(r);total+=r['bytes'];assert total<=p['new_remote_raw_bytes_cap']
    files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.pyc','.partial'])
    r=A.archive('T634_root',files,release,deadline,T)
    receipts.append(r);total+=r['bytes'];assert total<=p['new_remote_raw_bytes_cap']
    T.B.write(ROOT/'stage/docs/long_distance_20261101/release_receipt_T634.json',
              dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T634',
                   child_receipts=[r['group'] for r in receipts],bytes=total,
                   files_deleted=0,verified_epoch=time.time()))
    signal.alarm(0);print('T634_FULL_RAW_REMOTE_VERIFIED')


if __name__=='__main__':main()
