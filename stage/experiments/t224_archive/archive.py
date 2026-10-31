"""RAM-only Stage Release archive, fully verified readback and lossless local compaction.

No training. Cold-storage only four nonselected failed T174 diagnostic critics;
all original bytes remain locally in gzip and remotely in verified original tar.
"""
from pathlib import Path
import os,io,json,hashlib,subprocess,urllib.request,urllib.error,urllib.parse,tarfile,gzip,sys,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/results_20261031_archive';DOC.mkdir(exist_ok=True)
OUT=ROOT/'stage/runs/t224_archive';OUT.mkdir(exist_ok=True)
GH=ROOT.parent/'outputs/stage_migration/tools/gh';SLUG='frank8763/winkun';TAG='stage-left-breakthrough-2026-10-31'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def token():return subprocess.check_output([str(GH),'auth','token','--hostname','github.com'],stderr=subprocess.DEVNULL,text=True).strip()
def req(method,url,payload=None,headers=None):
    headers={'Authorization':'Bearer '+token(),'Accept':'application/vnd.github+json','User-Agent':'Stage-byte-verified-archive',**(headers or {})}
    data=json.dumps(payload).encode() if isinstance(payload,dict) else payload
    if isinstance(payload,dict):headers['Content-Type']='application/json'
    with OPENER.open(urllib.request.Request(url,data=data,headers=headers,method=method),timeout=90) as r:return r.read()
def api(method,path,payload=None):return json.loads(req(method,'https://api.github.com/repos/'+SLUG+'/'+path,payload))
def verify_tar(data,manifest):
    entries={r['path']:r for r in manifest};seen=set()
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in entries and member.name not in seen
            b=archive.extractfile(member).read();r=entries[member.name];assert len(b)==r['bytes'] and sha(b)==r['sha256'],member.name;seen.add(member.name)
    assert seen==set(entries)
def publish():
    receipt=DOC/'release_receipt.json';assert not receipt.exists(),'Inspect successful receipt; no duplicate'
    names=[p.name for p in (ROOT/'stage/runs').iterdir() if p.is_dir() and p.name[:4].startswith('t2') and p.name[1:4].isdigit() and 208<=int(p.name[1:4])<=223]
    files=sorted(f for n in names for f in (ROOT/'stage/runs'/n).rglob('*') if f.is_file() and f.suffix not in ('.lock','.partial','.pyc'))
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in files]
    assert sum(r['bytes'] for r in manifest)<100*1024**2
    write(DOC/'source_manifest.json',dict(scope='Stage T208–T223 original failures/successes/models/trajectories, no Phase/chat/credentials',files=manifest))
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w:gz',compresslevel=3) as archive:
        for p,r in zip(files,manifest):assert sha(p.read_bytes())==r['sha256'];archive.add(p,arcname=r['path'],recursive=False)
    data=output.getvalue();verify_tar(data,manifest);name='stage-left-breakthrough-'+sha(data)[:12]+'.tar.gz'
    print(json.dumps(dict(state='RAM_ARCHIVE_VERIFIED',members=len(manifest),bytes=len(data))),flush=True)
    try:release=api('GET','releases/tags/'+TAG)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        release=api('POST','releases',dict(tag_name=TAG,target_commitish='a4fd763f7d6e7e3c85e8c21d5bd284db319133ce',name='Stage original LEFT breakthrough and failed stability gates',draft=False,
            body='Stage only. T221 fixed original old8 8/8 through own-exploration supervised consolidation (not pure RL); T222 LEFT25/27 RIGHT9/27, stable5m gate failed. T223 source gate failed zero training. Includes original raw evidence/models/failures T208–T223. Full archive remote readback and every-member hashes verified by receipt. No Phase, chats, credentials, or generalized-success claim.'))
    assets=api('GET',f"releases/{release['id']}/assets?per_page=100")
    def upload(name,data):
        asset=next((r for r in assets if r['name']==name),None)
        if asset is None:asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),data,{'Content-Type':'application/octet-stream'}))
        current=api('GET',f"releases/assets/{asset['id']}");assert current['size']==len(data) and current['digest']=='sha256:'+sha(data)
        return current
    existing=[a for a in assets if a['name'].startswith('stage-left-breakthrough-') and a['name'].endswith('.tar.gz')]
    asset=api('GET',f"releases/assets/{existing[0]['id']}") if len(existing)==1 else upload(name,data)
    remote=req('GET',f"https://api.github.com/repos/{SLUG}/releases/assets/{asset['id']}",headers={'Accept':'application/octet-stream'});assert len(remote)==asset['size'] and 'sha256:'+sha(remote)==asset['digest'];verify_tar(remote,manifest);data=remote
    index=(DOC/'source_manifest.json').read_bytes();idx=upload('source_manifest.json',index)
    write(receipt,dict(state='PUBLISHED_FULL_REMOTE_READBACK_VERIFIED',url=release['html_url'],asset_id=asset['id'],asset_url=asset['browser_download_url'],archive_sha256=sha(data),archive_bytes=len(data),members=len(manifest),index_asset_id=idx['id'],full_remote_binary_and_all_member_hashes_verified=True,temporary_disk_archive_bytes=0,files_deleted=0))
    print(json.dumps(dict(state='ARCHIVE_DONE',url=release['html_url'],members=len(manifest))),flush=True)
def compact():
    receipt=DOC/'cold_storage_receipt.json';assert not receipt.exists(),'Already compacted'
    old=json.loads((ROOT/'stage/docs/results_20261030_q_audit/release_receipt.json').read_text());asset=old['assets'][0]
    print('VERIFYING_HISTORICAL_RELEASE_FULL_READBACK',flush=True)
    data=req('GET',f"https://api.github.com/repos/{SLUG}/releases/assets/{asset['asset_id']}",headers={'Accept':'application/octet-stream'});assert len(data)==asset['bytes'] and sha(data)==asset['sha256']
    proc=subprocess.run(['zstd','-d','-q','-c'],input=data,capture_output=True,check=True);manifest=json.loads((ROOT/'stage/manifests/q_audit_20261030_files.json').read_text());expected={r['path']:r for r in manifest['files']};target={f'stage/runs/t174_critic_alignment/{arm}/critic_alignment.pt' for arm in ('uniform','terminal_balanced')} | {f'stage/runs/t175_paired_advantage/{arm}/critic_alignment.pt' for arm in ('mc_only','mc_plus_paired')};found={}
    with tarfile.open(fileobj=io.BytesIO(proc.stdout),mode='r:') as archive:
        for member in archive:
            if member.name in expected:
                b=archive.extractfile(member).read();r=expected[member.name];assert len(b)==r['bytes'] and sha(b)==r['sha256']
                if member.name in target:found[member.name]=b
    assert set(found)==target
    # These four models failed their gate, are not the parent used by T175, and
    # are never resumed. Preserve byte-exact full payloads locally compressed.
    records=[];compressed={}
    for name,b in found.items():
        p=ROOT/name;assert p.is_file() and p.read_bytes()==b
        z=gzip.compress(b,compresslevel=6,mtime=0);assert gzip.decompress(z)==b
        assert len(z)<len(b)
        dest=p.with_suffix('.pt.gz');assert not dest.exists();compressed[name]=z
        records.append(dict(original_path=name,original_bytes=len(b),original_sha256=sha(b),compressed_path=str(dest.relative_to(ROOT)),compressed_bytes=len(z),compressed_sha256=sha(z),remote_original_asset=asset['url'],reason='Nonselected failed Stage diagnostic critic; full original payload retained losslessly in local gzip and independently verified remote original archive. T174 selected terminal_balanced_mc and all current retained policies untouched.'))
    write(receipt,dict(state='VERIFIED_BEFORE_LOSSLESS_RELOCATION',records=records,full_remote_original_archive_verified=True,restore='archive.py restore PATH; budget checked before restoring original path, no training restart'))
    for r in records:
        p=ROOT/r['original_path'];assert sha(p.read_bytes())==r['original_sha256'];p.unlink()
        dest=ROOT/r['compressed_path'];dest.write_bytes(compressed[r['original_path']]);assert sha(gzip.decompress(dest.read_bytes()))==r['original_sha256']
    final=json.loads(receipt.read_text());final.update(state='LOSSLESS_LOCAL_COMPACTION_COMPLETE',logical_bytes_freed=sum(r['original_bytes']-r['compressed_bytes'] for r in records),original_payloads_preserved_locally_and_remotely=True);write(receipt,final)
    assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())<1024**3
    print(json.dumps(final),flush=True)
def restore(name):
    records=[]
    for ledger in ('cold_storage_receipt.json','closed_resume_storage_receipt.json','reverse_curriculum_storage_receipt.json','smdp_storage_receipt.json','overnight_reserve_storage_receipt.json'):
        if (DOC/ledger).exists():records+=json.loads((DOC/ledger).read_text())['records']
    r=next(r for r in records if r['original_path']==name);dest=ROOT/name
    assert not dest.exists();z=ROOT/r['compressed_path'];assert sha(z.read_bytes())==r['compressed_sha256'];b=gzip.decompress(z.read_bytes());assert sha(b)==r['original_sha256']
    size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert size+len(b)<1024**3,'Restore reserve would exceed1GiB; no write'
    dest.write_bytes(b);assert sha(dest.read_bytes())==r['original_sha256'];print('RESTORED_VERIFIED',name)
if __name__=='__main__':
    if sys.argv[1]=='publish':publish()
    elif sys.argv[1]=='compact':compact()
    elif sys.argv[1]=='restore':restore(sys.argv[2])
    else:raise ValueError('publish/compact/restore')
