"""Evict only verified remote-backed, already losslessly encoded cold checkpoints.

Current/best/navigation evidence remains local. Original gzip bytes are retained
in the existing full-readback Release; this explicit index supports RAM fetch
or capacity-guarded restoration. No training data, failures or current models
are deleted. This is cache eviction, not research evidence destruction.
"""
from pathlib import Path
import importlib.util,io,json,os,shutil,sys,tarfile,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/overnight_20261113'
sys.path.insert(0,str(Path(__file__).parent));import t542_storage as S
C=S.C;LEDGER=DOC/'REMOTE_COLD_LEDGER.json'
def extract(data,path,h):
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:b=t.extractfile(path).read()
 assert C.sha(b)==h;return b
def evict():
 assert not LEDGER.exists();local=C.read(S.LEDGER);assert local['independent_persisted_decode_pass'];records=[];total=0
 for x in local['records']:
  f=ROOT/x['encoded_path'];r=C.read(ROOT/x['prior_receipt']);m=(ROOT/x['prior_manifest']).read_bytes()
  assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and r['manifest_sha256']==C.sha(m)
  item=next(v for v in json.loads(m)['files'] if v['path']==x['original_path']);assert item['sha256']==x['original_sha256'] and item['bytes']==x['original_bytes']
  assert C.sha(f.read_bytes())==x['encoded_sha256'] and C.sha(S.load_original(x['original_path']))==x['original_sha256']
  records.append(dict(x,prior_receipt_sha256=C.sha((ROOT/x['prior_receipt']).read_bytes()),remote_parts=r['parts']));total+=f.stat().st_size
  if total>=46*2**20:break
 assert total>=40*2**20
 # Journal every path before removal; immutable original remote receipt needed.
 ledger=dict(state='JOURNALED',records=records,completed=[],original_remote_bytes_preserved=True,local_encodings_evicted=True,restoration_script=str(Path(__file__).relative_to(ROOT)),epoch=time.time());C.write(LEDGER,ledger)
 for x in records:
  f=ROOT/x['encoded_path'];assert C.sha(f.read_bytes())==x['encoded_sha256'];f.unlink();ledger['completed'].append(x['original_path']);C.write(LEDGER,ledger)
 ledger.update(state='COMPLETE_VERIFIED_REMOTE_COLD_CACHE',released_bytes=total,global_logical_bytes=C.used(),all_current_inputs_unchanged=True);C.write(LEDGER,ledger);print('VERIFIED_REMOTE_COLD_CACHE',len(records),total,C.used(),flush=True)
def fetch_original(path):
 path=str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else str(path);x=next(v for v in C.read(LEDGER)['records'] if v['original_path']==path)
 assert C.sha((ROOT/x['prior_receipt']).read_bytes())==x['prior_receipt_sha256'];r=C.read(ROOT/x['prior_receipt']);assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 sp=importlib.util.spec_from_file_location('_t542_fetch_archive',ROOT/'stage/experiments/t493_wide8_night/transport.py');T=importlib.util.module_from_spec(sp);sp.loader.exec_module(T);T.OLD.install();api=T.OLD.T;blocks=[]
 for p in sorted(r['parts'],key=lambda p:p['order']):
  a=api.B.api('GET','releases/assets/'+str(p['asset_id']));b=api.readback(a,p['bytes']);assert C.sha(b)==p['sha256'];blocks.append(b)
 blob=b''.join(blocks);assert C.sha(blob)==r['sha256'];return extract(blob,path,x['original_sha256'])
def restore(path):
 f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();b=fetch_original(f)
 assert C.used()+len(b)+65536<=2**30 and shutil.disk_usage(ROOT).free-len(b)>=40*2**30
 with f.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
 assert C.sha(f.read_bytes())==C.sha(b);print('RESTORED_ORIGINAL_BYTES',str(f),len(b),flush=True)
if __name__=='__main__':evict() if len(sys.argv)==1 else restore(sys.argv[1])
