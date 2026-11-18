"""One bounded post-science RAM export; preserve all local original bytes."""
from pathlib import Path
import importlib.util
import json
import signal
import time
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/day_20261118/t630'
OUT=ROOT/'stage/runs/t630_early_departure_recovery'
spec=importlib.util.spec_from_file_location('_t630_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def read(p):return json.loads(Path(p).read_text())


def stopped():
    for root in [OUT,ROOT/'stage/runs/t629_offline_night_relay']:
        seal=read(root/'SEALED.json')
        assert seal['all_scientific_processes_stopped']
        assert all(not Path('/proc',str(i)).exists() for i in seal['scientific_pids'])


def main():
    p=read(OUT/'protocol.json');stopped()
    deadline=min(time.time()+p['transport_seconds']-5,p['absolute_all_stop']-120)
    assert deadline>time.time()+60
    def timeout(*args):raise TimeoutError('T630 one bounded post-science export')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
    A.OLD.install();T=A.OLD.T
    import urllib.request
    T.B.OPENER=urllib.request.build_opener();A.stopped=stopped
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11')
    for tag,root,cap in [('T630',OUT,32*2**20),
                         ('T629',ROOT/'stage/runs/t629_offline_night_relay',192*2**20)]:
        done=[];total=0
        for child in sorted(f for f in root.iterdir() if f.is_dir()):
            files=sorted(f for f in child.rglob('*') if f.is_file() and f.suffix not in ['.lock','.pyc','.partial'])
            if not files:continue
            r=A.archive(tag+'_'+child.name,files,release,deadline,T)
            done.append(r);total+=r['bytes'];assert total<=cap
        files=sorted(f for f in root.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.pyc','.partial'])
        r=A.archive(tag+'_root',files,release,deadline,T)
        done.append(r);total+=r['bytes'];assert total<=cap
        T.B.write(ROOT/f'stage/docs/long_distance_20261101/release_receipt_{tag}.json',
                  dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group=tag,
                       child_receipts=[r['group'] for r in done],bytes=total,
                       files_deleted=0,verified_epoch=time.time()))
    signal.alarm(0)
    print('T630_AND_T629_FULL_RAW_REMOTE_VERIFIED')


if __name__=='__main__':main()
