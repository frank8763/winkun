"""Transport-only small binary segments, full RAM reassembly/member audit.
No scientific restart, deletion or native/optimizer activity. Restore by ordered
concatenation of the segment asset bytes, then extract gzip tar normally.
"""
from pathlib import Path
import sys,io,json,gzip,tarfile,time,signal,urllib.parse
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/long_distance_20261101';DEST=ROOT/'stage/docs/distance_20261102'
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
from backup import api,req,readback,require_stopped,used,sha,verify_tar,write,SLUG

def main():
    receipt=DOC/'release_receipt_T302.json';assert not receipt.exists();out=ROOT/'stage/runs/t302_balanced_bridge_learning';require_stopped(out);manifest_path=DOC/'source_manifest_T302.json';manifest=json.loads(manifest_path.read_text())['files'];start=time.time();deadline=start+600
    def alarm(signum,frame):raise TimeoutError('T302 transport-only600second budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600);before=used();buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for r in manifest:
            b=(ROOT/r['path']).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'];info=tarfile.TarInfo(r['path']);info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
    data=gzip.compress(buf.getvalue(),compresslevel=3,mtime=0);assert len(data)<96*1024**2;verify_tar(data,manifest)
    release=api('GET','releases/tags/stage-distance-2026-11-01');assets=[]
    for page in range(1,12):
        a=api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");assets+=a
        if len(a)<100:break
    size=512*1024;chunks=[data[k:k+size] for k in range(0,len(data),size)];parts=[];remote=[];failures=[]
    for k,part in enumerate(chunks):
        assert time.time()<deadline;name=f'stage-t302-raw-{sha(data)[:12]}-segment{k:02d}of{len(chunks):02d}-{sha(part)[:12]}.bin';asset=next((a for a in assets if a['name']==name and a['state']=='uploaded' and a.get('digest')=='sha256:'+sha(part)),None)
        if asset is None:
            for attempt in range(2):
                candidate=name if attempt==0 and not any(a['name']==name for a in assets) else name.replace('.bin',f'-retry{attempt+1}.bin')
                try:
                    assert time.time()<deadline-5
                    asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(candidate),part,{'Content-Type':'application/octet-stream'}));break
                except (OSError,urllib.error.URLError) as error:
                    asset=None;failures.append(dict(segment=k,attempt=attempt+1,error=repr(error)));write(DEST/'T302_CHUNK_TRANSPORT_ERRORS.json',dict(failures=failures,scientific_runs_restarted=False))
            assert asset is not None,'Incomplete segments retained, no backup-complete claim'
        asset=api('GET',f"releases/assets/{asset['id']}");assert asset['state']=='uploaded' and asset['size']==len(part) and asset['digest']=='sha256:'+sha(part);b=readback(asset,len(part));assert b==part;remote.append(b);parts.append(dict(order=k,asset_id=asset['id'],asset_url=asset['browser_download_url'],bytes=len(part),sha256=sha(part)))
        write(DEST/'T302_CHUNK_BACKUP_PROGRESS.json',dict(state='PARTIAL_SEGMENT_READBACK',archive_sha256=sha(data),segments_total=len(chunks),parts=parts,original_files_deleted=0));print(json.dumps(dict(segment=k+1,total=len(chunks),readback_verified=True)),flush=True)
    assembled=b''.join(remote);assert assembled==data;verify_tar(assembled,manifest)
    for r in manifest:
        b=(ROOT/r['path']).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
    require_stopped(out);after=used();assert after==before and after<1024**3
    write(receipt,dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T302',url=release['html_url'],transport='ordered_binary_segments_of_one_gzip_tar',parts=parts,bytes=len(data),sha256=sha(data),members=len(manifest),manifest_sha256=sha(manifest_path.read_bytes()),restore='Download parts in order, concatenate exact segment bytes, verify archive sha256, then extract gzip tar and verify manifest. Segments individually are binary pieces, not standalone tar archives.',temporary_disk_archive_bytes=0,files_deleted=0,physics_steps=0,optimizer_updates=0,newruns_logical_bytes_before=before,newruns_logical_bytes_after=after,verified_epoch=time.time(),original_whole_upload_failure_preserved=True));signal.alarm(0);print(json.dumps(dict(state='BACKUP_VERIFIED',segments=len(chunks),members=len(manifest))),flush=True)

if __name__=='__main__':main()
