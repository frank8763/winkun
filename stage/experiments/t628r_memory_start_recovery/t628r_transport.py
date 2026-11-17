"""RAM-only archive after every science writer is stopped; no training imports."""
from pathlib import Path
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t628r_memory_start_recovery'
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
        import t628r_sync_source
        t628r_sync_source.main();return
    assert sys.argv[1]=='T628R';stopped();A.stopped=stopped
    original=T.B.write
    def write(path,value):
        if Path(path).name=='source_manifest_T628R.json':
            value=dict(value,scope='Ended Stage T628R four-task fresh-policy recollection, fixed .2 versus .1-to-.05 noise schedule; two rounds per arm maximum, conditional49 development, no Critic or global promotion')
        return original(path,value)
    T.B.write=write
    p=json.loads((OUT/'protocol.json').read_text())
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.pyc','.partial'])
    streamed=p.get('remote_preflight_bytes',0)+sum(json.loads(f.read_text())['bytes'] for pattern in ['batch_*_receipt.json','*.remote.json'] for f in OUT.rglob(pattern))
    assert streamed+sum(f.stat().st_size for f in files)+131072<=p['new_remote_raw_bytes_cap']
    deadline=min(time.time()+p['transport_seconds']-10,p['absolute_all_stop']-30)
    r=A.archive('T628R',files,release,deadline,T)
    stopped();assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    print('T628R_FULL_RAW_BACKUP_PASS',flush=True)


if __name__=='__main__':main()
