"""Completed-run RAM archive in small fully read-back binary segments.

Transport only. No native/optimizer, compaction, local archive or deletion.
"""
from pathlib import Path
import argparse, gzip, io, json, signal, tarfile, time, urllib.parse, urllib.error
from backup import ROOT,DOC,GROUPS,api,req,readback,require_stopped,used,sha,verify_tar,write,SLUG


def main(group):
    receipt=DOC/f'release_receipt_{group}.json';assert not receipt.exists()
    directories=[ROOT/'stage/runs'/name for name in GROUPS[group]]
    for directory in directories:require_stopped(directory)
    start=time.time();deadline=start+600
    def alarm(*_):raise TimeoutError('Completed-evidence transport600second budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(600);before=used()
    files=sorted(p for directory in directories for p in directory.rglob('*') if p.is_file() and p.suffix not in ('.lock','.partial','.pyc'))
    manifest=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for f in files]
    manifest_path=DOC/f'source_manifest_{group}.json'
    m=dict(scope=f'Stage {group} completed original evidence, all failures and models',files=manifest,active_runs_included=False)
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==m
    else:write(manifest_path,m)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for f,r in zip(files,manifest):
            b=f.read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
            info=tarfile.TarInfo(r['path']);info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
    data=gzip.compress(buf.getvalue(),compresslevel=3,mtime=0);assert len(data)<96*1024**2;verify_tar(data,manifest)
    release=api('GET','releases/tags/stage-distance-2026-11-01');assets=[]
    for page in range(1,12):
        batch=api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");assets+=batch
        if len(batch)<100:break
    chunks=[data[k:k+512*1024] for k in range(0,len(data),512*1024)];parts=[];remote=[];failures=[]
    progress=DOC/f'{group}_small_parts_progress.json'
    for k,part in enumerate(chunks):
        assert time.time()<deadline
        name=f'stage-{group.lower()}-raw-{sha(data)[:12]}-segment{k:02d}of{len(chunks):02d}-{sha(part)[:12]}.bin'
        asset=next((a for a in assets if a['name']==name and a['state']=='uploaded' and a.get('digest')=='sha256:'+sha(part)),None)
        if asset is None:
            for attempt in range(2):
                candidate=name if attempt==0 and not any(a['name']==name for a in assets) else name.replace('.bin',f'-retry{attempt+1}.bin')
                try:
                    assert time.time()<deadline-5
                    asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(candidate),part,{'Content-Type':'application/octet-stream'}));break
                except (OSError,urllib.error.URLError) as error:
                    asset=None;failures.append(dict(segment=k,attempt=attempt+1,error=repr(error)))
                    write(DOC/f'{group}_small_parts_errors.json',dict(failures=failures,scientific_runs_restarted=False))
            assert asset is not None,'Partial remote segments retained; backup is not complete'
        asset=api('GET',f"releases/assets/{asset['id']}")
        assert asset['state']=='uploaded' and asset['size']==len(part) and asset['digest']=='sha256:'+sha(part)
        b=readback(asset,len(part));assert b==part;remote.append(b)
        parts.append(dict(order=k,asset_id=asset['id'],asset_url=asset['browser_download_url'],bytes=len(part),sha256=sha(part)))
        write(progress,dict(state='PARTIAL_SEGMENT_READBACK',archive_sha256=sha(data),segments_total=len(chunks),parts=parts,files_deleted=0))
        print(json.dumps(dict(segment=k+1,total=len(chunks),readback_verified=True)),flush=True)
    assembled=b''.join(remote);assert assembled==data;verify_tar(assembled,manifest)
    for f,r in zip(files,manifest):assert f.stat().st_size==r['bytes'] and sha(f.read_bytes())==r['sha256']
    for directory in directories:require_stopped(directory)
    after=used();assert after==before and after<1024**3
    write(receipt,dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group=group,url=release['html_url'],
        transport='ordered_binary_segments_of_one_gzip_tar',parts=parts,bytes=len(data),sha256=sha(data),members=len(manifest),manifest_sha256=sha(manifest_path.read_bytes()),
        restore='Download parts in order; concatenate exact bytes; verify archive SHA; extract gzip tar and verify each manifest member. Individual .bin files are not standalone tar archives.',
        temporary_disk_archive_bytes=0,files_deleted=0,physics_steps=0,optimizer_updates=0,newruns_logical_bytes_before=before,newruns_logical_bytes_after=after,verified_epoch=time.time()))
    write(progress,dict(state='ALL_SEGMENTS_AND_MEMBERS_VERIFIED',archive_sha256=sha(data),segments_total=len(chunks),parts=parts,files_deleted=0))
    signal.alarm(0);print(json.dumps(dict(state='BACKUP_VERIFIED',group=group,segments=len(chunks),members=len(manifest))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('group',choices=GROUPS);main(parser.parse_args().group)
