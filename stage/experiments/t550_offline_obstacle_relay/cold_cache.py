"""Verified remote-backed rejected training cache, explicit reversible index."""
from pathlib import Path
import importlib.util,io,json,os,tarfile,time,hashlib,sys,shutil
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261114';LEDGER=DOC/'T550_COLD_CACHE.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,a):Path(p).write_text(json.dumps(a,indent=2)+'\n')
def evict():
 assert not LEDGER.exists();parent=ROOT/'stage/runs/t542_offline_field_relay';sealed=read(parent/'SEALED.json');assert all(not Path('/proc',str(i)).exists() for i in sealed['scientific_pids']+[read(parent/'launch.json')['pid']]);assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T542.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 locked=read(ROOT/'stage/runs/t549_curvature_gated_transfer/protocol.json')['inputs'];records=[]
 for d in sorted(parent.iterdir()):
  if not d.is_dir() or d.name in ['rtg_06','rtg_07'] or not (d/'summary.json').exists() or not (d/'audit.json').exists():continue
  s=read(d/'summary.json')
  if s['state']!='COMPLETE' or s['result'].get('accepted') is not False:continue
  assert read(d/'audit.json')['state']=='PASS';receipt=ROOT/('stage/docs/long_distance_20261101/release_receipt_T542_'+d.name+'.json');manifest=ROOT/('stage/docs/long_distance_20261101/source_manifest_T542_'+d.name+'.json');r=read(receipt);assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and r['manifest_sha256']==sha(manifest);items={x['path']:x for x in read(manifest)['files']}
  for f in sorted((d/'episodes').glob('full_*.npz')):
   assert str(f.resolve()) not in locked;key=str(f.relative_to(ROOT));item=items[key];assert item['sha256']==sha(f) and item['bytes']==f.stat().st_size
   records.append(dict(path=key,sha256=item['sha256'],bytes=item['bytes'],receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),manifest=str(manifest.relative_to(ROOT))))
 assert sum(x['bytes'] for x in records)>=38*2**20
 j=dict(state='JOURNALED_BEFORE_CACHE_REMOVAL',records=records,completed=[],current_inputs_models_and_failures_metadata_unchanged=True,original_full_training_arrays_remain_in_verified_remote_archive=True,restore='stage/experiments/t550_offline_obstacle_relay/cold_cache.py restore PATH',epoch=time.time());save(LEDGER,j)
 for x in records:
  f=ROOT/x['path'];assert sha(f)==x['sha256'];f.unlink();j['completed'].append(x['path']);save(LEDGER,j)
 j.update(state='COMPLETE_REMOTE_SHA_VERIFIED_RESTORABLE_CACHE',released_bytes=sum(x['bytes'] for x in records));save(LEDGER,j);print('COLD_CACHE_RELEASE',j['released_bytes'])
def restore(path):
 f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();x=next(x for x in read(LEDGER)['records'] if ROOT/x['path']==f);assert sha(ROOT/x['receipt'])==x['receipt_sha256'];r=read(ROOT/x['receipt']);spec=importlib.util.spec_from_file_location('_t550_fetch',ROOT/'stage/experiments/t493_wide8_night/transport.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);A.OLD.install();T=A.OLD.T;blocks=[]
 for part in sorted(r['parts'],key=lambda x:x['order']):
  a=T.B.api('GET','releases/assets/'+str(part['asset_id']));b=T.readback(a,part['bytes']);assert T.B.sha(b)==part['sha256'];blocks.append(b)
 blob=b''.join(blocks);assert T.B.sha(blob)==r['sha256']
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:raw=tar.extractfile(x['path']).read()
 assert len(raw)==x['bytes'] and T.B.sha(raw)==x['sha256'];used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+len(raw)+131072<=2**30 and shutil.disk_usage(ROOT).free-len(raw)>=40*2**30
 with f.open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
 print('RESTORED_EXACT_BYTES',x['path'],x['sha256'])
if __name__=='__main__':restore(sys.argv[2]) if len(sys.argv)>1 else evict()
