from pathlib import Path
import argparse,hashlib,json,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args()
m=json.loads((a.root/'stage/manifests/archive_manifest.json').read_text());bad=[]
for r in m['files']:
 f=a.root/r['archive_path']
 if not f.is_file():bad.append([r['archive_path'],'MISSING']);continue
 h=hashlib.sha256()
 with f.open('rb') as src:
  for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b)
 if f.stat().st_size!=r['bytes'] or h.hexdigest()!=r['sha256']:bad.append([r['archive_path'],'HASH_MISMATCH'])
print(json.dumps({'checked':len(m['files']),'failures':bad},indent=2));sys.exit(bool(bad))
