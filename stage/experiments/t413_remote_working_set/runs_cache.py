"""Evict inactive plain run payloads covered by an existing full backup proof.

No uploads. A matching prior manifest/member proof and fresh immutable asset
digests are required. Encoded caches, human data, videos, current inputs and all
small research reports remain. Restore downloads only the required run archive.
"""
from pathlib import Path
import fcntl,hashlib,io,json,os,shutil,sys,tarfile,time
import cold_store as C
ROOT=C.ROOT;DOC=C.DOC;INDEX=ROOT/'stage/manifests/run_payloads_remote_20261107.json'
ARCHIVES=ROOT/'stage/docs/long_distance_20261101'

def candidates():
    keep=C.read(C.OUT/'protocol.json')['protected_hashes'];records={};groups={}
    for mf in sorted(ARCHIVES.glob('source_manifest_*.json')):
        rf=mf.with_name(mf.name.replace('source_manifest_','release_receipt_'))
        if not rf.is_file():continue
        receipt=C.read(rf)
        if receipt.get('state')!='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' or receipt.get('manifest_sha256')!=C.sha(mf):continue
        if receipt.get('asset_id'):
            assets=[dict(asset_id=receipt['asset_id'],bytes=receipt['bytes'],sha256=receipt['sha256'],asset_url=receipt['asset_url'])]
        elif receipt.get('transport')=='ordered_binary_segments_of_one_gzip_tar':
            assets=sorted(receipt['parts'],key=lambda x:x['order'])
            if [x['order'] for x in assets]!=list(range(len(assets))):continue
        else:continue
        group=receipt['group'];groups[group]=dict(receipt_path=str(rf.relative_to(ROOT)),receipt_sha256=C.sha(rf),
            manifest_path=str(mf.relative_to(ROOT)),manifest_sha256=C.sha(mf),assets=assets,
            archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'])
        for row in C.read(mf).get('files',[]):
            f=ROOT/row['path'];parts=Path(row['path']).parts
            if len(parts)<4 or parts[:2]!=('stage','runs'):continue
            # Preserve actual human sources and compact historical teacher/recovery
            # demonstrations; do not infer that they are disposable training noise.
            if any(w in parts[2].lower() for w in ['human','manual','demo','correction','handoff','teleop','video']):continue
            if f.suffix not in ('.npz','.pt','.gz') or str(f.resolve()) in keep or not f.is_file() or f.is_symlink():continue
            st=f.stat()
            if st.st_nlink!=1 or st.st_size!=row['bytes'] or C.sha(f)!=row['sha256']:continue
            records[row['path']]=dict(row,group=group,inode=st.st_ino,allocated_bytes=st.st_blocks*512,mode=st.st_mode,mtime_ns=st.st_mtime_ns)
    return records,{k:v for k,v in groups.items() if k in {r['group'] for r in records.values()}}

def evict():
    assert C.read(C.OUT/'summary.json')['state']=='COMPLETE'
    assert not INDEX.exists(),'Inspect existing ledger; never repeat completed cleanup'
    C.check_protected(C.read(C.OUT/'protocol.json'));records,groups=candidates();assert records
    T=C.transport();release=T.B.api('GET','releases/tags/stage-distance-2026-11-01');remote={}
    for page in range(1,40):
        rows=T.B.api('GET',f'releases/{release["id"]}/assets?per_page=100&page={page}')
        remote.update({r['id']:r for r in rows})
        if len(rows)<100:break
    else:raise RuntimeError('Asset pagination incomplete; no deletion')
    for group in groups.values():
        for part in group['assets']:
            asset=remote[part['asset_id']]
            assert asset['state']=='uploaded' and asset['size']==part['bytes'] and asset['digest']=='sha256:'+part['sha256']
    index=dict(schema=1,state='VERIFIED_RUN_CACHE_EVICTION',verification='Existing full binary readback/memberSHA receipt + unchanged receipt/manifest hashes + fresh GitHub asset digest/size + current local SHA',
        groups=groups,shards={},new_uploads=0,Phase_untouched=True)
    before=shutil.disk_usage(ROOT).free;lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        C.check_protected(C.read(C.OUT/'protocol.json'))
        shardroot=ROOT/'stage/manifests/remote_run_cache_20261107'
        for group in groups:
            shard=shardroot/(group+'.json');rows=[r for r in records.values() if r['group']==group]
            C.save(shard,dict(group=group,records=rows))
            index['shards'][group]=dict(path=str(shard.relative_to(ROOT)),sha256=C.sha(shard),files=len(rows))
        C.save(INDEX,index)  # All verified restore mappings durable before unlink.
        for row in records.values():
            f=ROOT/row['path'];assert f.stat().st_ino==row['inode'] and f.stat().st_nlink==1 and C.sha(f)==row['sha256']
            f.unlink()
        C.check_protected(C.read(C.OUT/'protocol.json'));index['state']='REMOTE_ONLY_VERIFIED';C.save(INDEX,index)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    report=dict(state='COMPLETE',files=len(records),groups=len(groups),payload_bytes=sum(r['bytes'] for r in records.values()),
        allocated_bytes=sum(r['allocated_bytes'] for r in records.values()),free_before=before,free_after=shutil.disk_usage(ROOT).free,
        current_inputs_unchanged=True,new_uploads=0,native=0,Adam=0)
    C.save(DOC/'RUN_CACHE_CLEANUP.json',report);print(json.dumps(report))

def restore(path,verify_only=False):
    p=Path(path);p=p if p.is_absolute() else ROOT/p;relative=str(p.resolve().relative_to(ROOT))
    idx=C.read(INDEX);row=None
    for info in idx['shards'].values():
        shard=ROOT/info['path'];assert C.sha(shard)==info['sha256']
        row=next((r for r in C.read(shard)['records'] if r['path']==relative),None)
        if row is not None:break
    if row is None:raise KeyError(relative)
    group=idx['groups'][row['group']]
    assert C.sha(ROOT/group['receipt_path'])==group['receipt_sha256'] and C.sha(ROOT/group['manifest_path'])==group['manifest_sha256']
    T=C.transport();parts=[]
    for part in group['assets']:
        asset=T.B.api('GET',f'releases/assets/{part["asset_id"]}')
        assert asset['size']==part['bytes'] and asset['digest']=='sha256:'+part['sha256']
        parts.append(C.asset_bytes(T,dict(asset,bytes=part['bytes'],sha256=part['sha256'])))
    data=b''.join(parts);assert len(data)==group['archive_bytes'] and hashlib.sha256(data).hexdigest()==group['archive_sha256']
    # Existing complete manifest is verified again before returning one member.
    T.B.verify_tar(data,C.read(ROOT/group['manifest_path'])['files'])
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:body=archive.extractfile(row['path']).read()
    assert len(body)==row['bytes'] and hashlib.sha256(body).hexdigest()==row['sha256']
    if verify_only:return dict(state='RESTORE_BYTES_VERIFIED_IN_RAM',path=str(p),bytes=len(body),sha256=row['sha256'])
    assert not p.exists() and shutil.disk_usage(ROOT).free-len(body)>=40*2**30
    assert C.used(ROOT/'stage/runs')+len(body)<=2**30
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body);assert C.sha(p)==row['sha256']
    os.chmod(p,row['mode']);os.utime(p,ns=(row['mtime_ns'],row['mtime_ns']))
    return dict(state='RESTORED_REQUESTED_FILE_ONLY',path=str(p),bytes=len(body))

if __name__=='__main__':
    if sys.argv[1]=='evict':evict()
    elif sys.argv[1]=='prepare':
        r,g=candidates();print(json.dumps(dict(files=len(r),bytes=sum(x['bytes'] for x in r.values()),groups=len(g))))
    elif sys.argv[1]=='restore':print(json.dumps(restore(sys.argv[2],len(sys.argv)>3 and sys.argv[3]=='--verify-only')))
    else:raise ValueError('prepare/evict/restore')
