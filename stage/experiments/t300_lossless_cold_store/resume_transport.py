"""Continue only the original frozen backup manifest, within original600s.

No navigation restart, reselection or budget extension. Originals untouched until
all ten packages are fully read back. Failed starter uploads are not evidence.
"""
from compact import *
import urllib.request

def main():
    assert not LEDGER.exists();manifest_path=DOC/'source_manifest_T300.json';deadline=manifest_path.stat().st_mtime+600
    assert time.time()<deadline;lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);no_native_process();before=used()
    manifest=json.loads(manifest_path.read_text())['files'];assert len(manifest)==229 and sum(r['bytes'] for r in manifest)==28021734
    prior=json.loads((DOC/'T300_REMOTE_PARTS.json').read_text())['parts'];assert len(prior)==2;compressed={};records=[]
    for r in manifest:
        f=ROOT/r['path'];b=f.read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'];z=gzip.compress(b,compresslevel=6,mtime=0);dest=Path(str(f)+'.gz');assert not dest.exists();compressed[r['path']]=z;records.append(dict(original_path=r['path'],original_bytes=len(b),original_sha256=sha(b),encoded_path=str(dest.relative_to(ROOT)),encoded_bytes=len(z),encoded_sha256=sha(z)))
    chunks=[];current=[];size=0
    for r in manifest:
        if current and size+r['bytes']>4*1024**2:chunks.append(current);current=[];size=0
        current.append(r);size+=r['bytes']
    if current:chunks.append(current)
    assert len(chunks)==10;release=api('GET','releases/tags/stage-distance-2026-11-01');assets=[]
    for page in range(1,12):
        batch=api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");assets+=batch
        if len(batch)<100:break
    parts=[];failures=[]
    for index,entries in enumerate(chunks):
        assert time.time()<deadline
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w') as tar:
            for r in entries:
                b=(ROOT/r['path']).read_bytes();assert sha(b)==r['sha256'];info=tarfile.TarInfo(r['path']);info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
        data=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);verify_tar(data,entries)
        if index<len(prior):
            asset=api('GET',f"releases/assets/{prior[index]['asset_id']}");assert prior[index]['sha256']==sha(data)
        else:
            base=f'stage-cold-logs-t300-part{index:02d}-'+sha(data)[:12]+'.tar.gz';asset=next((a for a in assets if a['name']==base and a['state']=='uploaded' and a.get('digest')=='sha256:'+sha(data)),None)
            if asset is None:
                for attempt in range(2):
                    assert time.time()<deadline-5;name=base.replace('.tar.gz',f'-transport{attempt+1}.tar.gz');assert not any(a['name']==name for a in assets)
                    try:
                        headers={'Authorization':'Bearer '+__import__('archive').token(),'Accept':'application/vnd.github+json','User-Agent':'Stage-evidence-backup','Content-Type':'application/octet-stream'}
                        request=urllib.request.Request(f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),data=data,headers=headers,method='POST')
                        with __import__('archive').OPENER.open(request,timeout=min(45,max(1,deadline-time.time()-5))) as response:asset=json.loads(response.read())
                        break
                    except (OSError,urllib.error.URLError) as error:
                        failures.append(dict(part=index,attempt=attempt+1,error=repr(error)));write(DOC/'T300_TRANSPORT_ERRORS.json',dict(original_failure='urllib write timeout after two verified parts,0localencodings',continued_with_original_manifest_deadline=deadline,failures=failures));asset=None
                assert asset is not None,'Bounded transport failed; originals still untouched'
        asset=api('GET',f"releases/assets/{asset['id']}");assert asset['state']=='uploaded' and asset['size']==len(data) and asset['digest']=='sha256:'+sha(data)
        remote=readback(asset,len(data));assert remote==data;verify_tar(remote,entries);parts.append(dict(asset_url=asset['browser_download_url'],asset_id=asset['id'],bytes=len(data),sha256=sha(data),members=len(entries)));write(DOC/'T300_REMOTE_PARTS.json',dict(state='PARTIAL_REMOTE_READBACK',parts=parts,local_files_changed=0,original_manifest_deadline=deadline));print(json.dumps(dict(state='PART_FULL_READBACK_VERIFIED',part=index,parts=10)),flush=True)
    no_native_process();assert time.time()<deadline
    receipt=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T300',parts=parts,bytes=sum(r['bytes'] for r in parts),members=len(manifest),manifest_sha256=sha(manifest_path.read_bytes()),temporary_disk_archive_bytes=0,physics_steps=0,optimizer_updates=0,verified_epoch=time.time());write(DOC/'release_receipt_T300.json',receipt)
    ledger=dict(state='REMOTE_VERIFIED_PENDING_LOCAL_ENCODING',records=records,global_bytes_before=before,full_remote_original_binary_and_all_members_verified=True,physics_steps=0,optimizer_steps=0,original_manifest_deadline=deadline,transport_failures_preserved=True,restore='gzip decode in RAM; verify original SHA; reserve full original_bytes underglobal1GiB before exact path restore; ended jobs not resumed.',unique_bytes_deleted=0)
    write(LEDGER,ledger)
    for r in records:
        assert time.time()<deadline;src=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];assert sha(src.read_bytes())==r['original_sha256'] and used()+r['encoded_bytes']<1024**3;dest.write_bytes(compressed[r['original_path']]);assert sha(gzip.decompress(dest.read_bytes()))==r['original_sha256'];src.unlink()
    after=used();ledger.update(state='LOSSLESS_LOG_ENCODING_COMPLETE',global_bytes_after=after,logical_bytes_released=before-after,all_original_bytes_retained_locally_and_remotely=True,completed_epoch=time.time());write(LEDGER,ledger);print(json.dumps(dict(state='LOSSLESS_ENCODING_COMPLETE',released=before-after)),flush=True)

if __name__=='__main__':main()
