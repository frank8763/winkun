"""Evict only complete remote-readback rejected T542 model cache entries.

Selected rtg06/07, T507 and data/failure/optimizer logs are never touched.
All original Actor/optimizer bytes remain in original per-child Release assets.
"""
from pathlib import Path
import importlib.util,json,os,sys,time,hashlib,io,tarfile,shutil
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261114';LEDGER=DOC/'T543_COLD_MODELS.json'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,r):Path(f).write_text(json.dumps(r,indent=2)+'\n')
def main():
 assert not LEDGER.exists();parent=ROOT/'stage/runs/t542_offline_field_relay';sealed=read(parent/'SEALED.json');assert all(not Path('/proc',str(i)).exists() for i in sealed['scientific_pids']+[read(parent/'launch.json')['pid']]);assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T542.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 records=[]
 for f in sorted(parent.glob('*/proposal.pt.gz')):
  if f.parent.name in ['rtg_06','rtg_07']:continue
  s=read(f.parent/'summary.json');assert s['state']=='COMPLETE' and not s['result']['accepted'];assert read(f.parent/'audit.json')['state']=='PASS'
  folder=ROOT/'stage/docs/long_distance_20261101';receipt=folder/('release_receipt_T542_'+f.parent.name+'.json');manifest=folder/('source_manifest_T542_'+f.parent.name+'.json');r=read(receipt);assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and r['manifest_sha256']==sha(manifest)
  item=next(v for v in read(manifest)['files'] if v['path']==str(f.relative_to(ROOT)));assert item['sha256']==sha(f) and item['bytes']==f.stat().st_size
  records.append(dict(original_path=item['path'],sha256=item['sha256'],bytes=item['bytes'],receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),manifest=str(manifest.relative_to(ROOT))))
 total=sum(x['bytes'] for x in records);assert total>=9_500_000
 journal=dict(state='JOURNALED_REMOTE_VERIFIED_REJECTED_MODEL_CACHE',records=records,completed=[],original_remote_full_model_and_optimizer_preserved=True,current_models_and_all_trajectories_unchanged=True,restoration=str(Path(__file__).relative_to(ROOT))+' restore original_path',new_native=0,new_optimizer=0,epoch=time.time());save(LEDGER,journal)
 for x in records:
  f=ROOT/x['original_path'];assert sha(f)==x['sha256'];f.unlink();journal['completed'].append(x['original_path']);save(LEDGER,journal)
 journal.update(state='COMPLETE_ALL_ORIGINAL_MODELS_REMOTE_RESTORABLE',released_bytes=total);save(LEDGER,journal);print('T543_VERIFIED_REJECTED_MODEL_CACHE_RELEASE',len(records),total,flush=True)
def original(path):
 path=str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else str(path);x=next(x for x in read(LEDGER)['records'] if x['original_path']==path);assert sha(ROOT/x['receipt'])==x['receipt_sha256'];r=read(ROOT/x['receipt'])
 sp=importlib.util.spec_from_file_location('_t543_remote_model_fetch',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A);A.OLD.install();T=A.OLD.T;blocks=[]
 for part in sorted(r['parts'],key=lambda x:x['order']):
  a=T.B.api('GET','releases/assets/'+str(part['asset_id']));b=T.readback(a,part['bytes']);assert T.B.sha(b)==part['sha256'];blocks.append(b)
 b=b''.join(blocks);assert T.B.sha(b)==r['sha256']
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:raw=t.extractfile(path).read()
 assert len(raw)==x['bytes'] and T.B.sha(raw)==x['sha256'];return raw
def restore(path):
 f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();b=original(f);used=sum(x.stat().st_size for x in (ROOT/'stage/runs').rglob('*') if x.is_file());assert used+len(b)+65536<=2**30 and shutil.disk_usage(ROOT).free-len(b)>=40*2**30
 with f.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 print('RESTORED_EXACT_MODEL',str(f),len(b),flush=True)
if __name__=='__main__':restore(sys.argv[2]) if len(sys.argv)>1 else main()
