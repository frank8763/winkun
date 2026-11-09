"""Reversible encoding of explicitly archived, unused Stage text evidence.

No model/trajectory is encoded. Original remote member bytes remain available;
local gzip decodes to those exact bytes. This module has no execution on import.
"""
from pathlib import Path
import gzip,hashlib,json,os,shutil,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t443_sampling_control'
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def used(folder=None):return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())
def save(f,v):Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def candidates(protected,required):
    doc=ROOT/'stage/docs/long_distance_20261101';remote={}
    for path in sorted(doc.glob('release_receipt_*.json')):
        receipt=read(path)
        if receipt.get('state')!='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED':continue
        if 'group'not in receipt:continue
        manifest=doc/f"source_manifest_{receipt['group']}.json"
        if not manifest.exists():continue
        raw=manifest.read_bytes();assert sha_bytes(raw)==receipt['manifest_sha256']
        for r in json.loads(raw).get('files',[]):remote[r['path']]=(r,path,manifest)
    rows=[]
    for f in (ROOT/'stage/runs').rglob('*'):
        if not f.is_file() or f.suffix not in ['.json','.jsonl','.log','.txt'] or f.stat().st_size<10000:continue
        if f.name=='protocol.json' or any(x in f.name.lower() for x in ['case','replay','optim','failure','error','audit','manifest','ledger']):continue
        if f.parts[-1].endswith('.cold.gz') or any(x in f.parts for x in ['t441_joint_actor_critic','t442_value_warmup','t443_sampling_control']):continue
        relative=str(f.relative_to(ROOT))
        if str(f)in protected or relative in protected or f.with_name(f.name+'.cold.gz').exists():continue
        if relative not in remote:continue
        r,receipt,manifest=remote[relative];raw=f.read_bytes()
        if len(raw)!=r['bytes'] or sha_bytes(raw)!=r['sha256']:continue
        blob=gzip.compress(raw,compresslevel=9,mtime=0);saving=len(raw)-len(blob)
        if saving<=0:continue
        rows.append(dict(original=relative,encoded=relative+'.cold.gz',original_bytes=len(raw),original_sha256=sha_bytes(raw),
            encoded_bytes=len(blob),encoded_sha256=sha_bytes(blob),saving=saving,
            receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha_bytes(receipt.read_bytes()),
            manifest=str(manifest.relative_to(ROOT)),manifest_sha256=sha_bytes(manifest.read_bytes())))
    rows.sort(key=lambda r:(-r['saving'],r['original']))
    selected=[];total=0
    for row in rows:
        selected.append(row);total+=row['saving']
        if total>=required:break
    assert total>=required and len(selected)<=150,('Insufficient verified unused metadata',total,required)
    return selected,total


def verify_remote(r):
    receipt=ROOT/r['receipt'];manifest=ROOT/r['manifest']
    assert sha_bytes(receipt.read_bytes())==r['receipt_sha256']
    assert sha_bytes(manifest.read_bytes())==r['manifest_sha256']
    rr=read(receipt);assert rr['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert rr['manifest_sha256']==r['manifest_sha256']
    members=read(manifest)['files'];m=next(x for x in members if x['path']==r['original'])
    assert m['sha256']==r['original_sha256'] and m['bytes']==r['original_bytes']


def encode(p):
    assert not (OUT/'storage_summary.json').exists(),'Do not repeat ended encoding'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        if args and Path(os.fsdecode(args[0])).name.startswith('python'):
            assert not any(str(ROOT/'stage/experiments') in os.fsdecode(a) for a in args[1:]),('Another Stage Python process',proc.name)
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    import subprocess
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    before=used();start=time.time();completed=[]
    for r in p['cold_storage']:
        assert time.time()-start<180
        original=ROOT/r['original'];encoded=ROOT/r['encoded']
        assert str(original)not in p['inputs'] and str(original)not in p['constructor_reads']
        verify_remote(r);raw=original.read_bytes();assert sha_bytes(raw)==r['original_sha256']
        assert not encoded.exists()
        blob=gzip.compress(raw,compresslevel=9,mtime=0)
        assert sha_bytes(blob)==r['encoded_sha256'] and len(blob)==r['encoded_bytes']
        assert used()+len(blob)+131072<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
        with encoded.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
        decoded=gzip.decompress(encoded.read_bytes());assert decoded==raw and sha_bytes(decoded)==r['original_sha256']
        assert original.read_bytes()==raw
        original.unlink()  # only after verified local decode and prior remote member evidence
        completed.append(r)
        save(OUT/'storage_ledger.json',dict(state='ENCODING',completed=completed,original_bytes_locally_recoverable=True,
            original_bytes_remote_fully_verified=True,new_gzip_remote_copy_NOT_claimed=True,native=0,optimizer=0))
    save(OUT/'storage_summary.json',dict(state='COMPLETE',pid=os.getpid(),files=len(completed),
        released_payload_bytes=sum(r['saving'] for r in completed),before=before,after=used(),elapsed_s=time.time()-start,
        original_remote_and_local_bytes_preserved=True,new_gzip_remote_copy_NOT_claimed=True,native=0,optimizer=0))
    assert used()-used(OUT)+p['max_bytes']<=2**30


def load_original(relative):
    p=read(OUT/'protocol.json');r=next(x for x in p['cold_storage'] if x['original']==str(relative))
    original=ROOT/r['original']
    b=original.read_bytes() if original.exists() else gzip.decompress((ROOT/r['encoded']).read_bytes())
    assert len(b)==r['original_bytes'] and sha_bytes(b)==r['original_sha256'];return b


def audit(p):
    s=read(OUT/'storage_summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    assert not (OUT/'storage_audit.json').exists()
    for r in p['cold_storage']:
        verify_remote(r);assert not (ROOT/r['original']).exists()
        assert sha_bytes((ROOT/r['encoded']).read_bytes())==r['encoded_sha256']
        assert sha_bytes(load_original(r['original']))==r['original_sha256']
    save(OUT/'storage_audit.json',dict(state='PASS',files=len(p['cold_storage']),original_bytes_exact=True,
        remote_original_every_member_preverified=True,new_native=0,optimizer=0))
