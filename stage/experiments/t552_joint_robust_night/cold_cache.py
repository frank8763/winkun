"""Only archived failed-teacher cache and exact duplicate histories are evicted.

Fetch full Stage archive in RAM, verify every member, journal restoration before
removing any file. Current models and all successful training inputs stay local.
"""
from pathlib import Path
import importlib.util,io,json,os,time,tarfile,hashlib,sys,shutil
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/overnight_20261114';LEDGER=DOC/'COLD_CACHE.json'
def read(f):return json.loads(Path(f).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(f,v):Path(f).write_text(json.dumps(v,indent=2)+'\n')
def backend():
 spec=importlib.util.spec_from_file_location('_t552_cache_backend',ROOT/'stage/experiments/t493_wide8_night/transport.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.OLD.install();return a.OLD.T

def fetch(group,T):
 receipt=ROOT/('stage/docs/long_distance_20261101/release_receipt_'+group+'.json');manifest=ROOT/('stage/docs/long_distance_20261101/source_manifest_'+group+'.json');r=read(receipt);assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');assets=release.get('assets',[]);prefix='stage-'+group.lower()+'-'
 hits=[a for a in assets if a['name'].startswith(prefix) and a['name'].endswith('.bin')]
 if not hits:
  for page in range(1,31):
   aa=T.B.api('GET','releases/'+str(release['id'])+'/assets?per_page=100&page='+str(page));hits.extend(a for a in aa if a['name'].startswith(prefix) and a['name'].endswith('.bin'))
   if len(aa)<100:break
 groups={}
 for a in hits:groups.setdefault(a['name'].rsplit('-',1)[0],[]).append(a)
 candidates=[v for v in groups.values() if sum(x['size'] for x in v)==r['bytes']];assert len(candidates)==1,(group,'archive_identity_ambiguous');parts=[];blobs=[]
 for a in sorted(candidates[0],key=lambda x:x['name']):
  b=T.readback(a,a['size']);assert a.get('digest')=='sha256:'+sha(b);blobs.append(b);parts.append(dict(asset_id=a['id'],bytes=len(b),sha256=sha(b),asset_name=a['name']))
 blob=b''.join(blobs);T.B.verify_tar(blob,read(manifest)['files']);return blob,dict(group=group,parts=parts,archive_sha256=sha(blob),receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt.read_bytes()),manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest.read_bytes()))

def evict():
 assert not LEDGER.exists();folder=ROOT/'stage/runs/t550_offline_obstacle_relay';assert not Path('/proc',str(read(folder/'launch.json')['pid'])).exists();assert all(not Path('/proc',str(i)).exists() for i in read(folder/'SEALED.json')['scientific_pids']);T=backend();blob,archive=fetch('T550',T);members={x['path']:x for x in read(ROOT/archive['manifest'])['files']};records=[]
 for child in sorted(folder.glob('attempt*')):
  s=read(child/'summary.json')
  if not s['result']['teacher_qualified']:
   row=json.loads((child/'cases.jsonl').read_text().splitlines()[0]);f=Path(row['path']);item=members[str(f.relative_to(ROOT))];assert sha(f.read_bytes())==item['sha256'] and f.stat().st_size==item['bytes'];records.append(dict(path=item['path'],sha256=item['sha256'],bytes=item['bytes'],group='T550'))
 assert len(records)==20
 # Six byte-identical failed histories can be restored from the retained twin.
 old=ROOT/'stage/runs/t551_retention_recovery_probe';assert read(old/'audit.json')['state']=='PASS';assert not Path('/proc',str(read(old/'launch.json')['pid'])).exists();m={x['path']:x for x in read(ROOT/'stage/docs/long_distance_20261101/source_manifest_T551.json')['files']};assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T551.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 for i in range(6):
  f=old/'episodes'/('full_'+str(16+i)+'.npz');canonical=old/'episodes'/('full_'+str(8+i)+'.npz');raw=f.read_bytes();assert raw==canonical.read_bytes();item=m[str(f.relative_to(ROOT))];assert sha(raw)==item['sha256'];records.append(dict(path=item['path'],sha256=item['sha256'],bytes=len(raw),canonical=str(canonical.relative_to(ROOT)),group='T551'))
 total=sum(x['bytes'] for x in records);used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file() and not f.is_symlink());assert used-total+40*2**20+131072<=2**30,(used,total)
 ledger=dict(state='JOURNALED_BEFORE_EVICTION',archives={'T550':archive},records=records,completed=[],all_original_bytes_remote_verified=True,restoration_script='stage/experiments/t552_joint_robust_night/cold_cache.py restore PATH',new_native=0,new_optimizer=0,epoch=time.time());save(LEDGER,ledger)
 for x in records:
  f=ROOT/x['path'];assert sha(f.read_bytes())==x['sha256'];f.unlink();ledger['completed'].append(x['path']);save(LEDGER,ledger)
 ledger.update(state='COMPLETE_REMOTE_AND_DUPLICATE_CACHE_EVICTION',released_bytes=total);save(LEDGER,ledger);print('T552_VERIFIED_CACHE_RELEASE',total)

def restore(path):
 f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();l=read(LEDGER);x=next(x for x in l['records'] if ROOT/x['path']==f)
 if x.get('canonical'):raw=(ROOT/x['canonical']).read_bytes()
 else:
  a=l['archives'][x['group']];T=backend();blocks=[]
  for p in a['parts']:
   obj=T.B.api('GET','releases/assets/'+str(p['asset_id']));b=T.readback(obj,p['bytes']);assert sha(b)==p['sha256'];blocks.append(b)
  blob=b''.join(blocks);assert sha(blob)==a['archive_sha256']
  with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:raw=t.extractfile(x['path']).read()
 assert len(raw)==x['bytes'] and sha(raw)==x['sha256'];used=sum(q.stat().st_size for q in (ROOT/'stage/runs').rglob('*') if q.is_file() and not q.is_symlink());assert used+len(raw)+131072<=2**30 and shutil.disk_usage(ROOT).free-len(raw)>=40*2**30
 with f.open('xb') as h:h.write(raw);h.flush();os.fsync(h.fileno())
 print('RESTORED_EXACT_BYTES',x['path'])
if __name__=='__main__':restore(sys.argv[2]) if len(sys.argv)>1 else evict()
