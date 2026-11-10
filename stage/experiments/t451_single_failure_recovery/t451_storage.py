"""T451 explicit byte-identical references; no links or unique-byte deletion."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t451_single_failure_recovery'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def used(folder=None):return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())
def candidates(protected,required):
    remote={};doc=ROOT/'stage/docs/long_distance_20261101'
    for f in sorted(doc.glob('release_receipt_*.json')):
        d=read(f)
        if d.get('state')!='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' or 'group' not in d:continue
        mf=doc/f"source_manifest_{d['group']}.json"
        if not mf.exists() or sha(mf)!=d['manifest_sha256']:continue
        for r in read(mf)['files']:remote[r['path']]=(r,f,mf)
    groups={}
    for f in (ROOT/'stage/runs').rglob('*'):
        if not f.is_file() or not (f.suffix=='.npz' or f.name.endswith('.pt.gz')) or any(t in str(f) for t in ['t449_','t450_','t451_']):continue
        rel=str(f.relative_to(ROOT))
        if rel not in remote:continue
        r,receipt,mf=remote[rel]
        if f.stat().st_size!=r['bytes'] or sha(f)!=r['sha256']:continue
        groups.setdefault((r['bytes'],r['sha256']),[]).append((f,receipt,mf))
    rows=[]
    for (size,h),items in groups.items():
        if len(items)<2:continue
        canonical=next((x for x in items if str(x[0]) in protected),sorted(items,key=lambda x:str(x[0]))[0]);c=canonical[0]
        for f,receipt,mf in items:
            if f==c or str(f) in protected or any(t in str(f) for t in ['t145_','t146_','t159_','t194_','t255_']):continue
            row=dict(original=str(f.relative_to(ROOT)),canonical=str(c.relative_to(ROOT)),bytes=size,sha256=h,receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt),manifest=str(mf.relative_to(ROOT)),manifest_sha256=sha(mf),canonical_receipt=str(canonical[1].relative_to(ROOT)),canonical_receipt_sha256=sha(canonical[1]),canonical_manifest=str(canonical[2].relative_to(ROOT)),canonical_manifest_sha256=sha(canonical[2]))
            row['reference_bytes']=0
            for _ in range(4):row['reference_bytes']=len((json.dumps(row,ensure_ascii=False,indent=2)+'\n').encode())
            assert row['reference_bytes']==len((json.dumps(row,ensure_ascii=False,indent=2)+'\n').encode());rows.append(row)
    rows.sort(key=lambda x:(-x['bytes'],x['original']));selected=[];total=0
    for row in rows:
        selected.append(row);total+=row['bytes']-row['reference_bytes']
        if total>=required:break
    assert total>=required and len(selected)<=1000,(total,required,len(selected))
    assert not {r['canonical'] for r in selected}&{r['original'] for r in selected}
    return selected,total
def verify_remote(r):
    for prefix,path in [('',r['original']),('canonical_',r['canonical'])]:
        receipt=ROOT/r[prefix+'receipt'];mf=ROOT/r[prefix+'manifest'];assert sha(receipt)==r[prefix+'receipt_sha256'] and sha(mf)==r[prefix+'manifest_sha256']
        d=read(receipt);assert d['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and d['manifest_sha256']==r[prefix+'manifest_sha256']
        m=next(x for x in read(mf)['files'] if x['path']==path);assert m['bytes']==r['bytes'] and m['sha256']==r['sha256']
def encode(p):
    assert not (OUT/'storage_summary.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    before=used();completed=[];start=time.time()
    for r in p['byte_references']:
        assert time.time()-start<180;verify_remote(r);original=ROOT/r['original'];canonical=ROOT/r['canonical'];side=Path(str(original)+'.stage-ref.json')
        assert str(original) not in p['inputs'] and str(original) not in p['constructor_reads'] and not side.exists()
        b=original.read_bytes();assert b==canonical.read_bytes() and len(b)==r['bytes'] and sha(original)==r['sha256']
        save(side,r);assert read(side)==r and sha(canonical)==r['sha256'];original.unlink();completed.append(r)
        save(OUT/'storage_ledger.json',dict(state='REFERENCING',completed=completed,unique_bytes_deleted=0,original_bytes_local_and_remote_preserved=True))
    save(OUT/'storage_summary.json',dict(state='COMPLETE',pid=os.getpid(),files=len(completed),before=before,after=used(),released_payload_bytes=before-used(),elapsed_s=time.time()-start,unique_bytes_deleted=0,no_links=True))
    assert used()-used(OUT)+p['max_bytes']<2**30
def resolve(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f
    if f.exists():return f
    r=read(str(f)+'.stage-ref.json');assert str(f.relative_to(ROOT))==r['original'];c=ROOT/r['canonical'];assert c.stat().st_size==r['bytes'] and sha(c)==r['sha256'];return c
def audit(p):
    s=read(OUT/'storage_summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'storage_audit.json').exists()
    for r in p['byte_references']:
        verify_remote(r);assert not (ROOT/r['original']).exists() and sha(resolve(r['original']))==r['sha256'] and resolve(r['original']).stat().st_size==r['bytes']
    save(OUT/'storage_audit.json',dict(state='PASS',files=len(p['byte_references']),original_bytes_local_canonical_and_remote_exact=True,unique_bytes_deleted=0,new_native=0,optimizer=0))
