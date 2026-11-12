"""RAM-only binary archives of ended T494 and sealed T495; full member SHA."""
from pathlib import Path
import importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_stage495_locked_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def stopped():
    for name in ['t494_timeout_cause','t495_drive_head_cause']:
        folder=ROOT/'stage/runs'/name
        assert json.loads((folder/'audit.json').read_text())['state']=='PASS'
        s=json.loads((folder/'summary.json').read_text());assert s['state']=='COMPLETE'
        ids=[s['pid'],json.loads((folder/'audit.json').read_text())['pid']]
        for g in json.loads((folder/'worker_generations.json').read_text()):ids.extend([g['parent'],*g['workers']])
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    assert json.loads((ROOT/'stage/runs/t495_drive_head_cause/SEALED.json').read_text())['all_scientific_processes_stopped']
def main():
    A.OLD.install();T=A.OLD.T
    if sys.argv[1]=='source_sync':
        import sync_source;sync_source.main();return
    assert sys.argv[1]=='T494_T495';stopped();A.stopped=stopped
    release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');deadline=time.time()+550
    for group,name in [('T494','t494_timeout_cause'),('T495','t495_drive_head_cause')]:
        folder=ROOT/'stage/runs'/name
        files=sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix not in ['.partial','.lock','.pyc'])
        A.archive(group,files,release,deadline,T)
    stopped();print('T494_T495_FULL_REMOTE_BINARY_AND_MEMBER_SHA_PASS',flush=True)
if __name__=='__main__':main()
