"""Verify one cold encoded historical checkpoint remotely before cache eviction.

No original tensor/optimizer/history bytes are discarded: the current encoding
is archived and fully read back as a binary member, independently of its older
original-byte backup. The current learning models are never selected.
"""
from pathlib import Path
import importlib.util,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/overnight_20261113';OUT=ROOT/'stage/runs/t542_offline_field_relay'
sp=importlib.util.spec_from_file_location('_t542_large_reserve_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
def main(second=False):
 assert not (OUT/'launch.json').exists();assert json.loads((OUT/'native_preflight/audit.json').read_text())['state']=='PASS'
 ledger_path=DOC/('LARGE_COLD_LEDGER_2.json' if second else 'LARGE_COLD_LEDGER.json');tag='T542_COLD_BINARY_2' if second else 'T542_COLD_BINARY';f=ROOT/('stage/runs/t165_resume/gamma099/checkpoint_latest.pt.gz.lxz.t383.rxz' if second else 'stage/runs/t165_resume/gamma2s/checkpoint_latest.pt.gz.lxz.t383.rxz');assert not ledger_path.exists() and f.exists() and f.stat().st_size==(27483284 if second else 27630916)
 def stopped():
  assert not (OUT/'launch.json').exists();s=json.loads((OUT/'native_preflight/summary.json').read_text());assert not Path('/proc',str(s['pid'])).exists()
  g=json.loads((OUT/'native_preflight/worker_generations.json').read_text());assert all(not Path('/proc',str(i)).exists() for x in g for i in [x['parent'],*x['workers']])
 stopped();A.stopped=stopped;A.OLD.install();T=A.OLD.T;release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');h=T.B.sha(f.read_bytes());size=f.stat().st_size
 rec=A.archive(tag,[f],release,time.time()+600,T);assert rec['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 ledger=dict(state='JOURNALED_BEFORE_COLD_CACHE_EVICTION',path=str(f.relative_to(ROOT)),bytes=size,sha256=h,receipt='stage/docs/long_distance_20261101/release_receipt_'+tag+'.json',manifest='stage/docs/long_distance_20261101/source_manifest_'+tag+'.json',prior_original_model_and_all_recovery_ledgers_unchanged=True,current_original_encoding_preserved_remotely=True,restoration_script='stage/experiments/t542_offline_field_relay/reserve_large.py restore',epoch=time.time());T.B.write(ledger_path,ledger)
 assert T.B.sha(f.read_bytes())==h;f.unlink();ledger.update(state='COMPLETE_REMOTE_FULL_BINARY_READBACK_THEN_CACHE_EVICTION',released_bytes=size);T.B.write(ledger_path,ledger);print('T542_LARGE_COLD_REMOTE_VERIFIED_CACHE_RELEASE',size,h,flush=True)
def restore(second=False):
 import io,tarfile,hashlib,shutil
 d=json.loads((DOC/('LARGE_COLD_LEDGER_2.json' if second else 'LARGE_COLD_LEDGER.json')).read_text());r=json.loads((ROOT/d['receipt']).read_text());f=ROOT/d['path'];assert not f.exists() and r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';A.OLD.install();T=A.OLD.T;blocks=[]
 for p in sorted(r['parts'],key=lambda p:p['order']):
  a=T.B.api('GET','releases/assets/'+str(p['asset_id']));b=T.readback(a,p['bytes']);assert T.B.sha(b)==p['sha256'];blocks.append(b)
 b=b''.join(blocks);assert T.B.sha(b)==r['sha256']
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:raw=t.extractfile(d['path']).read()
 assert len(raw)==d['bytes'] and T.B.sha(raw)==d['sha256'];used=sum(x.stat().st_size for x in (ROOT/'stage/runs').rglob('*') if x.is_file());assert used+len(raw)+65536<=2**30 and shutil.disk_usage(ROOT).free-len(raw)>=40*2**30
 with f.open('xb') as z:z.write(raw);z.flush();os.fsync(z.fileno())
 print('RESTORED_EXACT_COLD_ENCODING',str(f),len(raw),flush=True)
if __name__=='__main__':main(True) if len(sys.argv)>1 and sys.argv[1]=='second' else restore(len(sys.argv)>2 and sys.argv[2]=='second') if len(sys.argv)>1 else main()
