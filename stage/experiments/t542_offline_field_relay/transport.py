"""RAM-only uploads of sealed night evidence; never a science prerequisite."""
from pathlib import Path
import importlib.util,json,sys,time
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t542_offline_field_relay'
sp=importlib.util.spec_from_file_location('_t542_archive_backend',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
def stopped():
 s=json.loads((OUT/'SEALED.json').read_text());assert s['all_scientific_processes_stopped'];assert all(not Path('/proc',str(i)).exists() for i in s['scientific_pids'])
def main():
 A.OLD.install();T=A.OLD.T
 if sys.argv[1]=='source_sync':
  import sync_source;sync_source.main();return
 assert sys.argv[1]=='T542';stopped();A.stopped=stopped;deadline=min(time.time()+850,json.loads((OUT/'protocol.json').read_text())['absolute_all_stop']-240);release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');done=[]
 for folder in sorted(f for f in OUT.iterdir() if f.is_dir()):
  files=sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix not in ['.lock','.partial','.pyc'])
  if files:done.append(A.archive('T542_'+folder.name,files,release,deadline,T))
 files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.partial','.pyc']);done.append(A.archive('T542_root',files,release,deadline,T));stopped()
 T.B.write(ROOT/'stage/docs/long_distance_20261101/release_receipt_T542.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T542',url=release['html_url'],members=sum(r['members'] for r in done),bytes=sum(r['bytes'] for r in done),child_receipts=[r['group'] for r in done],driver_log_excluded=True,verified_epoch=time.time()));print('T542_FULL_RAW_BACKUP_PASS',flush=True)
if __name__=='__main__':main()
