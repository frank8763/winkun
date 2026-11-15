"""Fresh remote verification and reversible eviction of ended evidence caches only."""
from pathlib import Path
import importlib.util,json,tarfile,io,hashlib,time,sys,concurrent.futures
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/overnight_20261115';LEDGER=DOC/'COLD_CACHE.json'
def read(p):return json.loads(Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def backend():
 s=importlib.util.spec_from_file_location('_night_old_cold',ROOT/'stage/experiments/t552_joint_robust_night/cold_cache.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b);return b

def fetch(group,T,B):
 receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{group}.json';r=read(receipt);manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json';assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 declared=r.get('parts') or ([dict(asset_id=r['asset_id'],bytes=r['bytes'],sha256=r['sha256'])] if 'asset_id' in r else None)
 if not declared:return B.fetch(group,T)
 parts=[];blocks=[]
 for x in sorted(declared,key=lambda x:x.get('order',0)):
  asset=T.B.api('GET','releases/assets/'+str(x['asset_id']));raw=T.readback(asset,x['bytes']);assert sha(raw)==x['sha256'];blocks.append(raw);parts.append(dict(asset_id=x['asset_id'],bytes=len(raw),sha256=sha(raw),asset_name=asset['name']))
 blob=b''.join(blocks);T.B.verify_tar(blob,read(manifest)['files']);return blob,dict(group=group,parts=parts,archive_sha256=sha(blob),receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt.read_bytes()),manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha(manifest.read_bytes()))

def evict():
 B=backend();T=B.backend();records=[];archives={};groups=[];config=read(ROOT/'stage/runs/t575_complete_neural_recovery/protocol.json');protected={x['path'] for x in config['keep_rows']}|{config['repair_row']['path']}|{x['path'] for x in read(ROOT/'stage/runs/t575_complete_neural_recovery/protocol.json')['memory_rows']};protected|={x['path'] for x in [__import__('json').loads(z) for z in (ROOT/'stage/runs/t575_complete_neural_recovery/memory_rows.jsonl').read_text().splitlines()]}
 for n in range(400,544):
  f=ROOT/f'stage/docs/long_distance_20261101/source_manifest_T{n}.json'
  if f.exists() and (ROOT/f'stage/docs/long_distance_20261101/release_receipt_T{n}.json').exists():
   members=read(f)['files'];eligible=[x for x in members if x['path'].endswith('.npz') and (ROOT/x['path']).exists() and str(ROOT/x['path']) not in protected and n!=543]
   if n in [415,416]:eligible +=[x for x in members if x['path'].endswith('/values.pt.gz') and (ROOT/x['path']).exists() and str(ROOT/x['path']) not in protected]
   if n==486:eligible +=[x for x in members if x['path'].endswith('/proposal_B.pt.gz') and (ROOT/x['path']).exists() and str(ROOT/x['path']) not in protected]
   if eligible:groups.append((f'T{n}',eligible))
 for n in [544,546,547,548,549,550,568,573]:
  f=ROOT/f'stage/docs/long_distance_20261101/source_manifest_T{n}.json';eligible=[x for x in read(f)['files'] if '/episodes/full_' in x['path'] and x['path'].endswith('.npz') and (ROOT/x['path']).exists() and str(ROOT/x['path']) not in protected]
  if eligible:groups.append((f'T{n}',eligible))
 # No models in current loader chain, T54323 reference histories, T574 success,
 # T575 own-memory, T576 proposals or reset snapshots are eligible.
 for group,selector in [('T567R','/episodes/full_'),('T571','/memory/')]:
  f=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json';eligible=[x for x in read(f)['files'] if selector in x['path'] and x['path'].endswith('.npz') and (ROOT/x['path']).exists() and str(ROOT/x['path']) not in protected]
  if eligible:groups.append((group,eligible))
 # Each group is verified completely before its own reversible journaled eviction.
 ledger=read(LEDGER) if LEDGER.exists() else dict(state='JOURNALED_PROGRESSIVE_VERIFICATION',records=[],archives={},completed=[],all_original_bytes_fresh_remote_verified=True,released_bytes=0,restore_script='stage/experiments/t577_offline_recovery_night/cold_cache.py restore PATH',epoch=time.time());save(LEDGER,ledger)
 def download(item):
  group,eligible=item;last=None
  for attempt in range(4):
   try:return group,eligible,fetch(group,T,B)
   except Exception as ex:last=str(ex);time.sleep(1)
  raise RuntimeError((group,last))
 groups.sort(key=lambda x:sum(v['bytes'] for v in x[1]),reverse=True)
 executor=concurrent.futures.ThreadPoolExecutor(max_workers=3);futures=[executor.submit(download,x) for x in groups]
 for future in concurrent.futures.as_completed(futures):
  group,eligible,(blob,receipt)=future.result()
  verified=[]
  with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
   for x in eligible:
    f=ROOT/x['path'];raw=f.read_bytes();assert len(raw)==x['bytes'] and sha(raw)==x['sha256'] and t.extractfile(x['path']).read()==raw;verified.append(dict(x,group=group))
  ledger['archives'][group]=receipt;ledger['records']+=verified;save(LEDGER,ledger)
  for x in verified:
   f=ROOT/x['path'];assert sha(f.read_bytes())==x['sha256'];f.unlink();ledger['completed'].append(x['path']);ledger['released_bytes']+=x['bytes'];save(LEDGER,ledger)
  print('VERIFIED_COLD_GROUP',group,ledger['released_bytes'],flush=True)
 executor.shutdown(wait=True)
 assert ledger['released_bytes']>84*2**20,ledger['released_bytes'];ledger['state']='COMPLETE_REMOTE_RESTORABLE';save(LEDGER,ledger);print('T577_COLD_RELEASE',ledger['released_bytes'],flush=True)

def restore(path):
 B=backend();B.LEDGER=LEDGER;B.restore(path)
if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
