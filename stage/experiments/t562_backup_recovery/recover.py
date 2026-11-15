"""Post-deadline transport-only recovery: no simulator/optimizer or raw edits."""
from pathlib import Path
import importlib.util,time,json
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('_t562_sealed_transport',ROOT/'stage/experiments/t562_full_history_representation/transport.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def main():
 M.stopped();M.A.OLD.install();T=M.A.OLD.T;M.A.stopped=M.stopped
 out=M.OUT;receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T562.json'
 if not receipt.exists():
  release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');files=sorted(f for f in out.rglob('*') if f.is_file() and f.name!='driver.log' and f.suffix not in ['.partial','.lock','.pyc']);M.A.archive('T562',files,release,time.time()+300,T)
 r=json.loads(receipt.read_text());assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';manifest=json.loads((receipt.parent/'source_manifest_T562.json').read_text());assert all(T.B.sha((ROOT/x['path']).read_bytes())==x['sha256'] for x in manifest['files']);M.stopped()
 T.B.write(ROOT/'stage/docs/day_20261115/t562/BACKUP_RECOVERY.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',epoch=time.time(),new_native=0,new_Adam=0,transport_only_after_scientific_deadline=True,original_offline_reports_preserved=True,receipt=str(receipt),members=r['members'],bytes=r['bytes'],archive_sha256=r['sha256']))
 print('T562_BACKUP_RECOVERY_PASS',r['members'],flush=True)
if __name__=='__main__':main()
