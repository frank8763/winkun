"""Byte-exact cold log storage, only fully remotely verified ended Stage groups.

Original payload remains locally as XZ and remotely in the original archive.
Read historical logs through load_original/restore before old audit tools.
"""
from pathlib import Path
import json,hashlib,lzma,os,sys,time,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/day_20261116/t595';OUT=ROOT/'stage/runs/t595_lossless_cold_logs'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,j):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(j,indent=2)+'\n');os.replace(tmp,p)
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def prepare():
 assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True);records={}
 protected=set()
 for folder in ['t591_joint_representation_pair','t592_five_loss_attribution','t594_current_goal_recovery']:
  p=read(ROOT/'stage/runs'/folder/'protocol.json');protected.update(p['inputs']);protected.update(p['protected_files'])
 patterns=['T552','T463','T479','T501_joint_*','T499_regression_*']
 archive=ROOT/'stage/docs/long_distance_20261101'
 for pattern in patterns:
  for manifest in sorted(archive.glob('source_manifest_'+pattern+'.json')):
   group=manifest.stem.removeprefix('source_manifest_');receipt=archive/('release_receipt_'+group+'.json')
   if not receipt.exists():continue
   r=read(receipt);assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
   for entry in read(manifest)['files']:
    f=ROOT/entry['path']
    if f.suffix!='.jsonl' or not f.is_file() or f.stat().st_size<150000 or str(f) in protected:continue
    b=f.read_bytes()
    if sha(b)!=entry['sha256']:continue
    z=lzma.compress(b,preset=3);assert lzma.decompress(z)==b
    if len(z)>=len(b):continue
    records[str(f.relative_to(ROOT))]=dict(path=str(f.relative_to(ROOT)),encoded=str(f.relative_to(ROOT))+'.t595.xz',raw_bytes=len(b),raw_sha256=sha(b),encoded_bytes=len(z),encoded_sha256=sha(z),manifest=str(manifest.relative_to(ROOT)),receipt=str(receipt.relative_to(ROOT)))
 rows=list(records.values());saving=sum(r['raw_bytes']-r['encoded_bytes'] for r in rows);assert saving>7*2**20
 plan=dict(state='PREPARED_NOT_MUTATED',records=rows,expected_payload_release=saving,global_before=used(),global_limit=2**30,metadata_cap=256*1024,wall_cap=300,source_sha256=sha(Path(__file__).read_bytes()),new_native=0,new_optimizer=0,user_scope='Prior explicit Stage cleanup authorization; all raw local bytes reversible and original full remote memberSHA receipt verified. No unique evidence discarded.')
 assert used()+256*1024<2**30;save(DOC/'PLAN.json',plan);print('T595_ZERO_MUTATION_PREFLIGHT',len(rows),saving,flush=True)
def run():
 assert not OUT.exists();p=read(DOC/'PLAN.json');assert sha(Path(__file__).read_bytes())==p['source_sha256'];start=time.time();lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir();save(OUT/'launch.json',dict(epoch=start,pid=os.getpid(),native=0,optimizer=0));done=[]
 for r in p['records']:
  assert time.time()-start<p['wall_cap'];f=ROOT/r['path'];e=ROOT/r['encoded'];assert not e.exists();b=f.read_bytes();assert len(b)==r['raw_bytes'] and sha(b)==r['raw_sha256']
  assert read(ROOT/r['receipt'])['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
  match=next(x for x in read(ROOT/r['manifest'])['files'] if x['path']==r['path']);assert match['sha256']==r['raw_sha256']
  z=lzma.compress(b,preset=3);assert len(z)==r['encoded_bytes'] and sha(z)==r['encoded_sha256'] and lzma.decompress(z)==b
  assert used()+len(z)+p['metadata_cap']<2**30
  with e.open('xb') as h:h.write(z);h.flush();os.fsync(h.fileno())
  assert sha(lzma.decompress(e.read_bytes()))==r['raw_sha256'];done.append(r);save(DOC/'LEDGER.json',dict(state='ENCODING',records=done,all_original_raw_locally_recoverable=True));f.unlink()
 for r in done:
  z=(ROOT/r['encoded']).read_bytes();assert sha(z)==r['encoded_sha256'];b=lzma.decompress(z);assert sha(b)==r['raw_sha256'] and len(b)==r['raw_bytes'];assert not (ROOT/r['path']).exists()
 result=dict(state='COMPLETE',records=done,payload_released=sum(r['raw_bytes']-r['encoded_bytes'] for r in done),global_after=used(),all_local_original_bytes_independently_decoded=True,all_original_remote_SHA_receipts_verified=True,new_native=0,new_optimizer=0,elapsed_s=time.time()-start,pid=os.getpid())
 save(DOC/'LEDGER.json',result);save(OUT/'summary.json',{k:v for k,v in result.items() if k!='records'});save(OUT/'audit.json',dict(state='PASS',decoded_files=len(done),all_raw_SHA_verified=True,raw_original_remote_and_local_XZ_preserved=True,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print('T595_COMPLETE',result['payload_released'],result['global_after'],flush=True)
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
