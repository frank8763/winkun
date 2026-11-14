"""Morning readback-only archive recovery, not a navigation budget extension.

Ended native/learning protocols remain untouched. At most900 seconds for
RAM binary upload and every-member SHA readback. Keep local evidence even if
network fails. Original historical REMOTE_PENDING report stays unchanged.
"""
from pathlib import Path
import importlib.util,json,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t542_offline_field_relay';DOC=ROOT/'stage/docs/overnight_20261113'
sp=importlib.util.spec_from_file_location('_t542_morning_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
def stopped():
 s=json.loads((OUT/'SEALED.json').read_text());assert s['all_scientific_processes_stopped'];ids=s['scientific_pids']+[json.loads((OUT/'launch.json').read_text())['pid']];assert all(not Path('/proc',str(i)).exists() for i in ids)
def main():
 receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T542.json';assert not receipt.exists();review=json.loads((DOC/'MORNING_RAW_REVIEW.json').read_text());assert review['state']=='PASS_PERSISTED_EVIDENCE_AND_PARTIAL_ACCOUNTING_ONLY';stopped();A.stopped=stopped;A.OLD.install();T=A.OLD.T
 start=time.time();deadline=start+850;signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Morning backup900second limit')));signal.alarm(900)
 T.B.write(DOC/'MORNING_BACKUP_REGISTRATION.json',dict(new_native=0,new_optimizer=0,archive_only_seconds=900,night_budget_not_extended=True,failed_final_original_untouched=True,epoch=start,scope='All persisted original raw including censored partial,not recovery of59missing observation rows'))
 try:
  release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');done=[]
  for folder in sorted(f for f in OUT.iterdir() if f.is_dir()):
   files=sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix not in ['.lock','.partial','.pyc'])
   if files:done.append(A.archive('T542_'+folder.name,files,release,deadline,T))
  files=sorted(f for f in OUT.iterdir() if f.is_file() and f.name!='driver.log' and f.suffix not in ['.lock','.partial','.pyc']);done.append(A.archive('T542_root',files,release,deadline,T));stopped()
  r=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T542',url=release['html_url'],members=sum(r['members'] for r in done),bytes=sum(r['bytes'] for r in done),child_receipts=[r['group'] for r in done],driver_log_excluded=True,includes_failed_final_partial_evidence=True,not_a_full_failed_final_audit=True,missing59physics_observation_rows_NOT_recovered=True,verified_epoch=time.time());T.B.write(receipt,r);T.B.write(DOC/'MORNING_BACKUP_RESULT.json',dict(state=r['state'],members=r['members'],bytes=r['bytes'],url=r['url'],seconds=time.time()-start));print('T542_MORNING_FULL_RAW_BACKUP_VERIFIED',r['members'],r['bytes'],flush=True)
 except BaseException:
  T.B.write(DOC/'MORNING_BACKUP_PENDING.json',dict(error=traceback.format_exc(),all_local_raw_retained=True,new_native=0,new_optimizer=0));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
