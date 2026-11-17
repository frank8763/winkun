"""Read every existing remote member in RAM before any explicitly approved eviction."""
from pathlib import Path
import json,sys,hashlib,io,tarfile,os,time,fcntl,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/overnight_20261117'
sys.path.insert(0,str(ROOT/'stage/experiments/t628_fresh_noise_curriculum'))
import t628_stream as S

def read(f):return json.loads(Path(f).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def download(receipt,T):
 parts=[]
 for part in receipt['parts']:
  asset=T.B.api('GET',f"releases/assets/{part['asset_id']}");blob=T.readback(asset,part['bytes']);assert len(blob)==part['bytes'] and sha(blob)==part['sha256'];parts.append(blob)
 blob=b''.join(parts);assert sha(blob)==receipt['sha256'];return blob

def run():
 approval=read(DOC/'AUTHORIZATION.json');assert approval['state']=='APPROVED' and approval['cold_cache_eviction_approved'];assert not (DOC/'COLD_EVICTION_LEDGER.jsonl').exists();old=ROOT/'stage/runs/t628s_actual_memory_recovery';assert read(old/'audit.json')['state']=='PASS' and read(old/'SEALED.json')['all_scientific_processes_stopped'];assert all(not Path('/proc',str(pid)).exists() for pid in read(old/'SEALED.json')['scientific_pids']);plan=read(DOC/'COLD_STORAGE_CANDIDATES.json');new=read(HERE/'protocol.json');protected=set(new['inputs'])|set(new['protected_files']);T=S.client();proof=[];started=time.time();lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  for group in plan['groups']:
   assert time.time()-started<1800;receipt=read(group['receipt']);manifest=read(group['manifest']);assert sha(Path(group['manifest']).read_bytes())==receipt['manifest_sha256'];data=download(receipt,T);T.B.verify_tar(data,manifest['files']);rows={r['path']:r for r in manifest['files']}
   with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
    for name in group['selected_paths']:
     f=ROOT/name;r=rows[name];assert str(f) not in protected and f.suffix=='.npz';blob=tar.extractfile(name).read();assert len(blob)==r['bytes'] and sha(blob)==r['sha256']==sha(f.read_bytes());proof.append(dict(r,receipt=group['receipt'],manifest=group['manifest']))
  assert len(proof)==plan['count'] and sum(r['bytes'] for r in proof)==plan['bytes'];save(DOC/'COLD_REMOTE_ALL_VERIFIED_BEFORE_EVICTION.json',dict(state='EVERY_REMOTE_MEMBER_READBACK_AND_LOCAL_SHA_PASS',files=proof,epoch=time.time()))
  for r in proof:assert sha((ROOT/r['path']).read_bytes())==r['sha256']
  with (DOC/'COLD_EVICTION_LEDGER.jsonl').open('xb') as ledger:
   for r in proof:
    f=ROOT/r['path'];assert sha(f.read_bytes())==r['sha256'];marker=Path(str(f)+'.t629-remote.json');entry=dict(**r,offline_available=False,restore='stage/experiments/t629_offline_night_relay/t629_cold_storage.py restore '+r['path']);save(marker,entry);f.unlink();ledger.write((json.dumps(entry,separators=(',',':'))+'\n').encode());ledger.flush();os.fsync(ledger.fileno())
  save(DOC/'COLD_STORAGE_COMPLETE.json',dict(state='COMPLETE',bytes=sum(r['bytes'] for r in proof),files=len(proof),no_model_or_video_deleted=True,native=0,optimizer=0,epoch=time.time()))
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def restore(name,verify_only=False):
 f=Path(name);f=f if f.is_absolute() else ROOT/f;row=read(str(f)+'.t629-remote.json');receipt=read(row['receipt']);manifest=read(row['manifest']);assert sha(Path(row['manifest']).read_bytes())==receipt['manifest_sha256'];data=download(receipt,S.client())
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:blob=tar.extractfile(row['path']).read()
 assert sha(blob)==row['sha256'] and len(blob)==row['bytes']
 if not verify_only:
  assert not f.exists() and sum(x.stat().st_size for x in (ROOT/'stage/runs').rglob('*') if x.is_file())+len(blob)+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
  with f.open('xb') as out:out.write(blob);out.flush();os.fsync(out.fileno())
 print(json.dumps(dict(state='EXACT_REMOTE_VERIFY_ONLY' if verify_only else 'EXACT_BYTES_RESTORED',path=str(f),sha256=row['sha256'])))
if __name__=='__main__':
 if sys.argv[1]=='restore':restore(sys.argv[2],len(sys.argv)>3 and sys.argv[3]=='--verify-only')
 else:run()
