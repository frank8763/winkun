"""RAM-only archive after every science writer is stopped; no training imports."""
from pathlib import Path
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t623_common_continuation_branches'
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
        import t623_sync_source
        t623_sync_source.main();return
    assert sys.argv[1]=='T623';stopped();A.stopped=stopped
    original=T.B.write
    def write(path,value):
        if Path(path).name=='source_manifest_T623.json':
            value=dict(value,scope='Ended Stage T623 action-aware versus action-blind recorded behavior-return prediction; frozen Actor; zero native or Actor updates; no policy-Q certification')
        return original(path,value)
    T.B.write=write
    p=json.loads((OUT/'protocol.json').read_text())
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    files=sorted(f for f in OUT.rglob('*') if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.pyc','.partial'])
    deadline=min(time.time()+p['transport_seconds']-10,p['absolute_all_stop']-30)
    r=A.archive('T623',files,release,deadline,T)
    stopped();assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    print('T623_FULL_RAW_BACKUP_PASS',flush=True)


if __name__=='__main__':main()
