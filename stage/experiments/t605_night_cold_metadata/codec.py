"""Byte-exact cold log storage, only fully remotely verified ended Stage groups.

Original payload remains locally as XZ and remotely in the original archive.
Read historical logs through load_original/restore before old audit tools.
"""
from pathlib import Path
import json,hashlib,lzma,os,sys,time,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/overnight_20261116/t605';OUT=ROOT/'stage/runs/t605_night_cold_metadata'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,j):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');os.replace(tmp,p)
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def prepare():
 assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True);records={}
 protected=set()
 for folder in ['t604_reverse_initialization','t603_qualified_state_aggregation','t598_independent_recurrent_recovery','t596_recurrent_joint_online']:
  p=read(ROOT/'stage/runs'/folder/'protocol.json');protected.update(p['inputs']);protected.update(p['protected_files'])
 patterns=['T*']
 archive=ROOT/'stage/docs/long_distance_20261101'
 for pattern in patterns:
  for manifest in sorted(archive.glob('source_manifest_'+pattern+'.json')):
   group=manifest.stem.removeprefix('source_manifest_');receipt=archive/('release_receipt_'+group+'.json')
   if not receipt.exists():continue
   r=read(receipt)
   if r.get('state')!='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED':continue
   for entry in read(manifest)['files']:
    f=ROOT/entry['path']
    import re
    mt=re.search(r'/t(\d+)_',str(f));number=int(mt[1]) if mt else 9999
    if number>594 or f.name not in ['protocol.json','main.log','cases.jsonl','round_updates.jsonl','optimizer_calls.jsonl','summary.json'] or not f.is_file() or f.stat().st_size<200000 or str(f) in protected:continue
    b=f.read_bytes()
    if sha(b)!=entry['sha256']:continue
    z=lzma.compress(b,preset=1);assert lzma.decompress(z)==b
    if len(z)>=len(b):continue
    records[str(f.relative_to(ROOT))]=dict(path=str(f.relative_to(ROOT)),encoded=str(f.relative_to(ROOT))+'.t605.xz',raw_bytes=len(b),raw_sha256=sha(b),encoded_bytes=len(z),encoded_sha256=sha(z),manifest=str(manifest.relative_to(ROOT)),receipt=str(receipt.relative_to(ROOT)))
 rows=[];saving=0
 for r in sorted(records.values(),key=lambda x:x['raw_bytes']-x['encoded_bytes'],reverse=True):
  rows.append(r);saving+=r['raw_bytes']-r['encoded_bytes']
  if saving>=64*2**20:break
 assert saving>48*2**20 and len(rows)<=180
 plan=dict(state='PREPARED_NOT_MUTATED',records=rows,expected_payload_release=saving,global_before=used(),global_limit=2**30,metadata_cap=512*1024,wall_cap=180,source_sha256=sha(Path(__file__).read_bytes()),new_native=0,new_optimizer=0,user_scope='Prior explicit Stage cleanup authorization; all raw local bytes reversible and original full remote memberSHA receipt verified. No unique evidence discarded.')
 assert used()+512*1024<2**30;save(DOC/'PLAN.json',plan);print('T605_ZERO_MUTATION_PREFLIGHT',len(rows),saving,flush=True)
def run():
 assert not OUT.exists();p=read(DOC/'PLAN.json');assert sha(Path(__file__).read_bytes())==p['source_sha256'];start=time.time();lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir();save(OUT/'launch.json',dict(epoch=start,pid=os.getpid(),native=0,optimizer=0));done=[]
 for r in p['records']:
  assert time.time()-start<p['wall_cap'];f=ROOT/r['path'];e=ROOT/r['encoded'];assert not e.exists();b=f.read_bytes();assert len(b)==r['raw_bytes'] and sha(b)==r['raw_sha256']
  assert read(ROOT/r['receipt'])['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
  match=next(x for x in read(ROOT/r['manifest'])['files'] if x['path']==r['path']);assert match['sha256']==r['raw_sha256']
  z=lzma.compress(b,preset=1);assert len(z)==r['encoded_bytes'] and sha(z)==r['encoded_sha256'] and lzma.decompress(z)==b
  assert used()+len(z)+p['metadata_cap']<2**30
  with e.open('xb') as h:h.write(z);h.flush();os.fsync(h.fileno())
  assert sha(lzma.decompress(e.read_bytes()))==r['raw_sha256'];done.append(r);save(DOC/'LEDGER.json',dict(state='ENCODING',records=done,all_original_raw_locally_recoverable=True));f.unlink()
 for r in done:
  z=(ROOT/r['encoded']).read_bytes();assert sha(z)==r['encoded_sha256'];b=lzma.decompress(z);assert sha(b)==r['raw_sha256'] and len(b)==r['raw_bytes'];assert not (ROOT/r['path']).exists()
 result=dict(state='COMPLETE',records=done,payload_released=sum(r['raw_bytes']-r['encoded_bytes'] for r in done),global_after=used(),all_local_original_bytes_independently_decoded=True,all_original_remote_SHA_receipts_verified=True,new_native=0,new_optimizer=0,elapsed_s=time.time()-start,pid=os.getpid())
 save(DOC/'LEDGER.json',result);save(OUT/'summary.json',{k:v for k,v in result.items() if k!='records'});save(OUT/'audit.json',dict(state='PASS',decoded_files=len(done),all_raw_SHA_verified=True,raw_original_remote_and_local_XZ_preserved=True,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print('T605_COMPLETE',result['payload_released'],result['global_after'],flush=True)
def load_original(path):
 f=Path(path);f=f if f.is_absolute() else ROOT/f
 if f.exists():return f.read_bytes()
 rel=str(f.relative_to(ROOT));r=next(x for x in read(DOC/'LEDGER.json')['records'] if x['path']==rel);z=(ROOT/r['encoded']).read_bytes();assert sha(z)==r['encoded_sha256'];b=lzma.decompress(z);assert sha(b)==r['raw_sha256'];return b
def restore(path):
 f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();b=load_original(f);assert used()+len(b)+131072<2**30
 with f.open('xb') as h:h.write(b);h.flush();os.fsync(h.fileno())
if __name__=='__main__':
 if sys.argv[1]=='restore':restore(sys.argv[2])
 else:globals()[sys.argv[1]]()
