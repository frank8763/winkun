"""Bounded RAM exports of each sealed child directory, excluding old cache."""
from pathlib import Path
import importlib.util,json,signal,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t646_offline_night'
spec=importlib.util.spec_from_file_location('_t646_export',ROOT/'stage/experiments/t493_wide8_night/transport.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def read(f):return json.loads(Path(f).read_text())
def stopped():
    s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(v)).exists() for v in s['scientific_pids'])


def main():
    stopped();p=read(OUT/'protocol.json');deadline=min(time.time()+p['backup_seconds']-5,p['absolute_all_stop']-p['source_seconds']-30)
    def timeout(*args):raise TimeoutError('T646 bounded backup')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(max(1,int(deadline-time.time())))
    A.OLD.install();t=A.OLD.T
    import urllib.request
    t.B.OPENER=urllib.request.build_opener();A.stopped=stopped
    release=t.B.api('GET','releases/tags/stage-eight-only-2026-11-11');receipts=[]
    for name in p['arms']+['profile','coverage']:
        folder=OUT/name
        if not folder.exists():continue
        for child in sorted(v for v in folder.iterdir() if v.is_dir()):
            files=sorted(f for f in child.rglob('*') if f.is_file() and f.suffix not in ['.pyc','.lock','.partial'])
            # Arrays are small per occurrence, but partition large directories
            # below the existing96MiB archive limit, no temporary disk tar.
            groups=[];current=[];size=0
            for f in files:
                if current and size+f.stat().st_size>32*2**20:groups.append(current);current=[];size=0
                current.append(f);size+=f.stat().st_size
            if current:groups.append(current)
            for k,group in enumerate(groups):receipts.append(A.archive(f'T646_{name}_{child.name}_{k}',group,release,deadline,t))
        files=sorted(f for f in folder.iterdir() if f.is_file() and f.suffix not in ['.pyc','.lock','.partial'])
        if files:receipts.append(A.archive(f'T646_{name}_root',files,release,deadline,t))
    files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='driver.log')
    receipts.append(A.archive('T646_root',files,release,deadline,t))
    t.B.write(ROOT/'stage/docs/long_distance_20261101/release_receipt_T646.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T646',child_receipts=[r['group'] for r in receipts],bytes=sum(r['bytes'] for r in receipts),offline_input_originals_already_archived=True,files_deleted=0))
    signal.alarm(0);print('T646_FULL_RAW_REMOTE_VERIFIED')


if __name__=='__main__':main()
