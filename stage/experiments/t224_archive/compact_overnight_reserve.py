"""Keep every original byte of three ended historical checkpoints in local gzip."""
from archive import *
import http.client
def main():
    receipt=DOC/'overnight_reserve_storage_receipt.json';assert not receipt.exists(),'No repeat compaction'
    names=['stage/runs/t165_resume/gamma099/checkpoint_latest.pt','stage/runs/t165_resume/gamma2s/checkpoint_latest.pt','stage/runs/t169_value_seed/learn/checkpoint_latest.pt']
    manifest=json.loads((ROOT/'stage/manifests/progress_20261030_files.json').read_text());rows={r['path']:r for r in manifest['files']};originals={};packed={}
    for name in names:
        p=ROOT/name;s=json.loads((p.parent/'status.json').read_text());assert s['state']=='COMPLETED';b=p.read_bytes();r=rows[name];assert len(b)==r['bytes'] and sha(b)==r['sha256'];z=gzip.compress(b,compresslevel=6,mtime=0);assert gzip.decompress(z)==b and len(z)<len(b);originals[name]=b;packed[name]=z
    print(json.dumps(dict(state='LOCAL_FULL_PAYLOAD_ENCODING_VERIFIED',saved_MiB=sum(len(originals[k])-len(packed[k]) for k in names)/1024**2)),flush=True)
    old=json.loads((ROOT/'stage/docs/results_20261030/release_receipt.json').read_text());a=old['assets'][0]
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):return None
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    request=urllib.request.Request(f"https://api.github.com/repos/{SLUG}/releases/assets/{a['asset_id']}",headers={'Authorization':'Bearer '+token(),'Accept':'application/octet-stream','User-Agent':'Stage-verified-range-readback'})
    try:
        with opener.open(request,timeout=30) as response:remote=response.read()
    except urllib.error.HTTPError as error:
        assert error.code in (301,302,303,307,308);url=error.headers['Location'];assert urllib.parse.urlparse(url).hostname in ('release-assets.githubusercontent.com','objects.githubusercontent.com')
        parts=[]
        for begin in range(0,a['bytes'],1024**2):
            end=min(a['bytes'],begin+1024**2)-1;piece=None
            for attempt in range(5):
                try:
                    r=urllib.request.Request(url,headers={'Range':f'bytes={begin}-{end}','User-Agent':'Stage-verified-range-readback'})
                    with OPENER.open(r,timeout=45) as response:
                        assert response.status==206 and response.headers['Content-Range'].startswith(f'bytes {begin}-{end}/');piece=response.read()
                    assert len(piece)==end-begin+1;break
                except (TimeoutError,OSError,http.client.HTTPException):
                    if attempt==4:raise
            parts.append(piece)
            if begin%(16*1024**2)==0:print(json.dumps(dict(remote_verified_transport_bytes=sum(map(len,parts)))),flush=True)
        remote=b''.join(parts)
    assert len(remote)==a['bytes'] and sha(remote)==a['sha256']
    raw=subprocess.run(['zstd','-d','-q','-c'],input=remote,capture_output=True,check=True).stdout;found=set()
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
        for member in archive:
            if member.name in originals:assert archive.extractfile(member).read()==originals[member.name];found.add(member.name)
    assert found==set(names)
    records=[]
    for name in names:
        dest=(ROOT/name).with_suffix('.pt.gz');assert not dest.exists();records.append(dict(original_path=name,original_bytes=len(originals[name]),original_sha256=sha(originals[name]),compressed_path=str(dest.relative_to(ROOT)),compressed_bytes=len(packed[name]),compressed_sha256=sha(packed[name]),remote_original_asset=a['url']))
    result=dict(state='VERIFIED_BEFORE_LOSSLESS_ENCODING',records=records,full_remote_original_archive_verified=True,all_original_checkpoint_members_byte_exact_verified=True,reason='EndedT165andT169fullActorCriticreplayoptimizerpayloadslosslesslyretainedlocallyandremotely; notselectedcurrentparents; originalT164pausedcheckpointandT145T146Phaseuntouched.',restore='archive.py restore PATH after1GiBbudget/originalSHAguard')
    write(receipt,result)
    for r in records:
        p=ROOT/r['original_path'];assert p.read_bytes()==originals[r['original_path']];p.unlink();dest=ROOT/r['compressed_path'];dest.write_bytes(packed[r['original_path']]);assert sha(gzip.decompress(dest.read_bytes()))==r['original_sha256']
    result.update(state='LOSSLESS_LOCAL_COMPACTION_COMPLETE',logical_bytes_freed=sum(r['original_bytes']-r['compressed_bytes'] for r in records),original_payloads_preserved_locally_and_remotely=True);write(receipt,result);print(json.dumps(dict(state=result['state'],saved_MiB=result['logical_bytes_freed']/1024**2)),flush=True)
if __name__=='__main__':main()
