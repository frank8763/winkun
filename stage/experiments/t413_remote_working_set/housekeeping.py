"""Necessary prelaunch evidence backup and rebuildable Stage cache cleanup.

No navigation, optimizer, installation, or large disk archive. Unlike a training
admission guard, this storage-only operation does not require 40GiB free: its
archive lives in RAM and only a small manifest/receipt is written locally.
"""
from pathlib import Path
import hashlib,io,json,os,sys,tarfile,time
import cold_store as C
ROOT=C.ROOT;DOC=C.DOC

def backup_prelaunch():
    receipt=DOC/'release_receipt_T412_PRELAUNCH.json'
    assert not receipt.exists(),'Already verified; do not upload again'
    directory=ROOT/'stage/runs/t412_closed_expert_likelihood'
    assert not (directory/'launch.json').exists()
    assert not Path('/proc/500991').exists()
    C.check_protected(C.read(C.OUT/'protocol.json'))
    files=sorted(f for f in directory.rglob('*') if f.is_file() and
        f.suffix not in ('.lock','.partial','.pyc') and f.stat().st_size)
    manifest=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=C.sha(f)) for f in files]
    assert sum(r['bytes'] for r in manifest)<2*2**20
    mf=DOC/'source_manifest_T412_PRELAUNCH.json'
    C.save(mf,dict(scope='Necessary T412 registered/preflight evidence only; navigation never launched',files=manifest,
        navigation_steps=0,synthetic_discarded_preflight_Adam=8,active_runs_included=False))
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=3) as archive:
        for f,row in zip(files,manifest):
            assert C.sha(f)==row['sha256'];archive.add(f,arcname=row['path'],recursive=False)
    data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest();T=C.transport()
    T.B.verify_tar(data,manifest)
    release=T.B.api('GET','releases/tags/stage-distance-2026-11-01')
    name='stage-necessary-t412-prelaunch-'+digest[:12]+'.tar.gz'
    # Search all pages; exact asset names and digests prevent duplicate uploads.
    asset=None
    for page in range(1,30):
        rows=T.B.api('GET',f'releases/{release["id"]}/assets?per_page=100&page={page}')
        asset=next((a for a in rows if a['name']==name),None)
        if asset or len(rows)<100:break
    if asset is None:
        asset=json.loads(T.req('POST',f'https://uploads.github.com/repos/{T.A.SLUG}/releases/{release["id"]}/assets?name='+name,data,{'Content-Type':'application/octet-stream'}))
    asset=T.B.api('GET',f'releases/assets/{asset["id"]}')
    assert asset['size']==len(data) and asset['digest']=='sha256:'+digest
    remote=T.readback(asset,len(data));assert remote==data;T.B.verify_tar(remote,manifest)
    for f,row in zip(files,manifest):assert C.sha(f)==row['sha256']
    assert not (directory/'launch.json').exists()
    C.save(receipt,dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',
        group='T412_PRELAUNCH',url=release['html_url'],asset_url=asset['browser_download_url'],asset_id=asset['id'],
        bytes=len(data),sha256=digest,members=len(manifest),manifest_sha256=C.sha(mf),
        new_historical_model_uploads=0,temporary_disk_archive_bytes=0,navigation_steps=0,new_optimizer_updates=0,
        preflight_not_navigation_training=True,verified_epoch=time.time()))
    print(json.dumps(dict(state='NECESSARY_PRELAUNCH_BACKUP_VERIFIED',members=len(manifest),bytes=len(data))))

def caches():
    assert (C.OUT/'summary.json').is_file()
    keep=C.read(C.OUT/'protocol.json')['protected_hashes'];rows=[]
    for f in sorted((ROOT/'stage/experiments').rglob('*.pyc')):
        if '__pycache__' not in f.parts or str(f.resolve()) in keep:continue
        source=f.parent.parent/(f.name.split('.')[0]+'.py')
        if not source.is_file() or f.stat().st_nlink!=1:continue
        rows.append(dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,source=str(source.relative_to(ROOT))))
    C.save(DOC/'DISPOSABLE_CACHE_CLEANUP.json',dict(state='PLANNED_REBUILDABLE_PYC_ONLY',files=rows))
    for row in rows:(ROOT/row['path']).unlink()
    C.check_protected(C.read(C.OUT/'protocol.json'))
    C.save(DOC/'DISPOSABLE_CACHE_CLEANUP.json',dict(state='COMPLETE',files=rows,bytes_deleted=sum(r['bytes'] for r in rows),
        uploaded=False,reason='Compiled Python cache regenerated from retained source',navigation_steps=0,Adam=0))
    print('CACHE_DELETED',len(rows),sum(r['bytes'] for r in rows))

if __name__=='__main__':
    assert sys.argv[1] in ('backup_prelaunch','caches');globals()[sys.argv[1]]()
