"""Three byte-identical ended batches: retain local canonical and remote originals."""
from pathlib import Path
import sys,os,json,hashlib,io,gzip,tarfile,time,shutil,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t403_cold_exact_reserve';DOC=ROOT/'stage/docs/review_20261107'
PAIRS=[('t200_residual_placement/latent/behavior_batch_0.npz','t200_residual_placement/native/behavior_batch_0.npz'),('t196_left_exploration/parameter_noise/behavior_batch_0.npz','t196_left_exploration/action_noise/behavior_batch_0.npz'),('t197_timeout_progress/no_timeout_distance/behavior_batch_0.npz','t197_timeout_progress/timeout_distance/behavior_batch_0.npz')]
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def used():return sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
def transport():
 sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as T
 T.install();return T
def prepare():
 assert not OUT.exists();OUT.mkdir();DOC.mkdir(exist_ok=True)
 records=[];files=[]
 for canonical,duplicate in PAIRS:
  a,b=[ROOT/'stage/runs'/x for x in [canonical,duplicate]];data=a.read_bytes();assert data==b.read_bytes()
  files.extend([a,b]);records.append(dict(original=str(b.relative_to(ROOT)),canonical=str(a.relative_to(ROOT)),bytes=len(data),sha256=sha(data),encoding='exact-byte-alias-v1'))
 manifest=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for f in files]
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w') as tar:
  for f,row in zip(files,manifest):
   info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mtime=0;tar.addfile(info,io.BytesIO(f.read_bytes()))
 data=gzip.compress(buf.getvalue(),compresslevel=3,mtime=0);T=transport();rel=T.B.api('GET','releases/tags/stage-distance-2026-11-01');name='stage-t403-originals-'+sha(data)[:12]+'.tar.gz'
 asset=json.loads(T.req('POST',f"https://uploads.github.com/repos/{T.A.SLUG}/releases/{rel['id']}/assets?name="+name,data,{'Content-Type':'application/octet-stream'}));asset=T.B.api('GET',f"releases/assets/{asset['id']}")
 assert asset['size']==len(data) and asset['digest']=='sha256:'+sha(data);returned=T.readback(asset,len(data));assert returned==data;T.B.verify_tar(returned,manifest)
 save(DOC/'T403_ORIGINAL_MANIFEST.json',dict(files=manifest))
 save(DOC/'T403_ORIGINAL_RECEIPT.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',asset_url=asset['browser_download_url'],sha256=sha(data),members=len(manifest),temporary_disk_bytes=0))
 p=dict(id='T403_FIXED3_DUPLICATES_ONLY',records=records,source_sha256=sha(Path(__file__).read_bytes()),max_seconds=120,native=0,Adam=0,unique_bytes_removed=0,expected_duplicate_bytes=sum(x['bytes'] for x in records),resolver='stage/experiments/t290_exact_duplicate_store/store.py resolve or restore')
 save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);print('T403_REGISTERED_ORIGINALS_REMOTE_VERIFIED',p['expected_duplicate_bytes'],flush=True)
def run():
 p=read(OUT/'protocol.json');assert not (OUT/'summary.json').exists() and sha(Path(__file__).read_bytes())==p['source_sha256'];assert read(DOC/'T403_ORIGINAL_RECEIPT.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);before=used();start=time.time();done=[]
 for row in p['records']:
  assert time.time()-start<120 and used()<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
  original=ROOT/row['original'];canonical=ROOT/row['canonical'];b=original.read_bytes();assert b==canonical.read_bytes() and sha(b)==row['sha256'];ref=Path(str(original)+'.ref.json');assert not ref.exists()
  save(ref,{k:row[k] for k in ['canonical','bytes','sha256','encoding']});save(OUT/'ledger.json',dict(completed=done,current=row['original'],recoverable_from=row['canonical']))
  original.unlink();assert sha(canonical.read_bytes())==row['sha256'];done.append(row['original'])
 save(OUT/'ledger.json',dict(state='ALL_ORIGINAL_BYTES_REMAIN_LOCAL_AND_REMOTE',completed=done,current=None))
 save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),all_workers_stopped=True,native=0,optimizer=0,unique_bytes_removed=0,files=len(done),duplicate_payload_released=p['expected_duplicate_bytes'],before=before,after=used(),elapsed_s=time.time()-start));print('T403_COMPLETE',p['expected_duplicate_bytes'])
def audit():
 p=read(OUT/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
 sys.path.insert(0,str(ROOT/'stage/experiments/t290_exact_duplicate_store'));import store
 for row in p['records']:
  original=ROOT/row['original'];assert not original.exists();assert store.resolve(original)==ROOT/row['canonical'];assert sha(store.resolve(original).read_bytes())==row['sha256']
 save(OUT/'audit.json',dict(state='PASS',all_original_bytes_exact=True,remote_originals_verified=True,native=0,optimizer=0,global_bytes=used()));save(DOC/'T403_AUDIT.json',read(OUT/'audit.json'));print('T403_INDEPENDENT_BYTE_AUDIT_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()
