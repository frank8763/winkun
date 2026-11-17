"""RAM-only archive after every science writer is stopped; no training imports."""
from pathlib import Path
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t629_offline_night_relay'
spec=importlib.util.spec_from_file_location('_t606_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def stopped():
    j=json.loads((OUT/'SEALED.json').read_text())
    assert j['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(i)).exists() for i in j['scientific_pids'])


def main():
    A.OLD.install();T=A.OLD.T
    import urllib.request
    T.B.OPENER=urllib.request.build_opener()
    if sys.argv[1]=='source_sync':
        import t629_sync_source
        t629_sync_source.main();return
    assert sys.argv[1]=='T629';stopped();A.stopped=stopped
    original=T.B.write
    def write(path,value):
        if Path(path).name=='source_manifest_T629.json':
            value=dict(value,scope='Ended Stage T629 local offline finite relay; stratified fresh PG, qualified actual-state recovery, frozen407 and neighborhood profiles; no global promotion')
        return original(path,value)
    T.B.write=write
    p=json.loads((OUT/'protocol.json').read_text())
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    done=[];total=0
    for child in sorted(f for f in OUT.iterdir() if f.is_dir()):
        files=sorted(f for f in child.rglob('*') if f.is_file() and f.suffix not in ['.lock','.pyc','.partial'])
        if not files:continue
        assert sum(f.stat().st_size for f in files)<80*2**20
        deadline=min(time.time()+p['transport_seconds']-10,p['absolute_all_stop']-60)
        r=A.archive('T629_'+child.name,files,release,deadline,T);done.append(r);total+=r['bytes'];assert total<=p['night_remote_backup_bytes_cap']
    files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.pyc','.partial'])
    r=A.archive('T629_root',files,release,min(time.time()+p['transport_seconds']-10,p['absolute_all_stop']-60),T);done.append(r);total+=r['bytes'];assert total<=p['night_remote_backup_bytes_cap']
    T.B.write(ROOT/'stage/docs/long_distance_20261101/release_receipt_T629.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T629',child_receipts=[r['group'] for r in done],bytes=total,verified_epoch=time.time()))
    print('T629_FULL_RAW_BACKUP_PASS',flush=True)


if __name__=='__main__':main()
