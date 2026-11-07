"""Evict inactive historical checkpoints only after full existing-asset proof.

No new historical checkpoints are uploaded. Every byte already exists in the
2026-10-30 Release; the exact compressed asset and each extracted member are
read back before any local alias is unlinked. Current scientific inputs stay.
"""
from pathlib import Path
import os,sys,json,time,hashlib,io,tarfile,subprocess,threading,shutil,fcntl,gc,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t413_remote_working_set';DOC=ROOT/'stage/docs/storage_20261107'
INDEX=ROOT/'stage/manifests/remote_only_20261107.json'
def read(f):return json.loads(Path(f).read_text())
def save(f,v):
    f=Path(f);f.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(f)+'.partial');
    with tmp.open('w') as stream:
        stream.write(json.dumps(v,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(f)
    handle=os.open(str(f.parent),os.O_RDONLY);os.fsync(handle);os.close(handle)
def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as stream:
        for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def used(f):return sum(p.stat().st_size for p in Path(f).rglob('*') if p.is_file())
def transport():
    sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as T
    T.install();return T
def protected():
    p=read(ROOT/'stage/runs/t412_closed_expert_likelihood/protocol.json');base=read(p['source_protocol']);paths={}
    for mapping in [p['inputs'],base['input_hashes']]:
        for f,h in mapping.items():
            q=Path(f);q=q if q.is_absolute() else ROOT/q;assert sha(q)==h,f;paths[str(q.resolve())]=h
    for name in ['t194_drive_sensitivity','t255_overnight','t353_temporal_exploration','t404_last_narrow','t406_wide_full_route','t410_day_offline_chain','t411_closed_teacher_recovery','t412_closed_expert_likelihood']:
        for f in (ROOT/'stage/runs'/name).rglob('*'):
            if f.is_file():paths[str(f.resolve())]=sha(f)
    return paths
def check_protected(p):
    for f,h in p['protected_hashes'].items():assert Path(f).is_file() and sha(f)==h,('Protected current input changed',f)
def prepare():
    assert not OUT.exists();assert not (ROOT/'stage/runs/t412_closed_expert_likelihood/launch.json').exists();assert not Path('/proc/500991').exists()
    keep=protected();inodes={(Path(f).stat().st_dev,Path(f).stat().st_ino) for f in keep};manifest=read(ROOT/'stage/manifests/archive_manifest.json');groups={}
    for row in manifest['files']:
        f=ROOT/row['archive_path']
        if row['storage']!='release' or f.suffix!='.pt' or not f.is_file():continue
        if not row['archive_path'].startswith(('stage/legacy/','stage/current/')):continue
        if any(word in (row['archive_path']+' '+row['original_path']).lower() for word in ['t145','t146','stage_t159']):continue
        s=f.stat();key=(s.st_dev,s.st_ino)
        if key in inodes:continue
        assert s.st_size==row['bytes'] and sha(f)==row['sha256'],f
        aliases={str(f.resolve())};old=Path(row['original_path'])
        if old.is_file() and (old.stat().st_dev,old.stat().st_ino)==key:aliases.add(str(old.resolve()))
        for alias in aliases:assert '/winkun/stage/' in alias or '/mkmini_stage_' in alias or '/outputs/stage_t' in alias,alias
        g=groups.setdefault(key,dict(rows=[],aliases=set(),bytes=s.st_size,sha256=row['sha256'],device=s.st_dev,inode=s.st_ino,links=s.st_nlink,allocated_bytes=s.st_blocks*512,mode=s.st_mode,mtime_ns=s.st_mtime_ns))
        assert row['sha256']==g['sha256'];g['rows'].append(row);g['aliases']|=aliases
    records=[]
    for g in groups.values():
        if len(g['aliases'])!=g['links'] or set(g['aliases'])&set(keep):continue
        g['aliases']=sorted(g['aliases']);g['id']=len(records);records.append(g)
    assert records
    assets=read(ROOT/'stage/manifests/remote_backup_verification.json')['assets'];assets={n:v for n,v in assets.items() if n.endswith('.tar.zst')}
    p=dict(id='T413_GITHUB_FIRST_COLD_CHECKPOINTS',authorization='2026-11-07 user explicitly uploads necessary records and deletes local garbage; keep only working dependencies, Phaseisolated',records=records,assets=assets,protected_hashes=keep,source_sha256=sha(__file__),max_seconds=3600,unique_bytes_removed_without_remote_copy=0,local_model_eviction_bytes=sum(r['bytes'] for r in records),physical_blocks_candidate=sum(r['allocated_bytes'] for r in records),native=0,Adam=0,scope='Only inactive .pt files from originalStage migration manifest; excludeT145/T146,T159 and allcurrent runtime/scientificinputs; retainallcode, summaries, human data andHDvideo',remote_tag='stage-archive-2026-10-30',source_manifest_sha256=sha(ROOT/'stage/manifests/archive_manifest.json'))
    OUT.mkdir();DOC.mkdir(parents=True,exist_ok=True);save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'PLAN.json',dict(state='READY_FOR_SOURCE_COMMIT_SYNC_ONLY',files=len(records),local_bytes=p['local_model_eviction_bytes'],physical_bytes=p['physical_blocks_candidate'],protected_files=len(keep),new_historical_uploads=0));print('COLD_PREPARED',len(records),p['local_model_eviction_bytes'],flush=True)
def asset_bytes(T,asset):
    fields={'Authorization':'Bearer '+T.A.token(),'Accept':'application/octet-stream','User-Agent':'Stage-necessary-cold-restore'}
    code,headers,data=T.response('GET',f'https://api.github.com/repos/{T.A.SLUG}/releases/assets/{asset["id"]}',headers=fields)
    if code==200:
        assert len(data)==asset['bytes'] and hashlib.sha256(data).hexdigest()==asset['sha256'];return data
    assert code==302;url=headers['location'];assert T.urllib.parse.urlparse(url).hostname in ['release-assets.githubusercontent.com','objects.githubusercontent.com']
    out=io.BytesIO();h=hashlib.sha256();size=asset['bytes'];step=8*2**20
    for begin in range(0,size,step):
        end=min(size-1,begin+step-1)
        for attempt in range(3):
            try:code,head,part=T.response('GET',url,headers={'Range':f'bytes={begin}-{end}','User-Agent':'Stage-byte-proof'})
            except T.urllib.error.URLError:
                if attempt==2:raise
                continue
            if code==206 and head.get('content-range','').startswith(f'bytes {begin}-{end}/') and len(part)==end-begin+1:break
        else:raise RuntimeError(('Remote byte range did not verify',asset['name'],begin,code))
        h.update(part);out.write(part);print('READBACK_CHUNK',asset['name'],end+1,size,flush=True)
    assert h.hexdigest()==asset['sha256'];return out.getvalue()
def members(data,wanted=None):
    proc=subprocess.Popen(['zstd','-dc'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);errors=[]
    def feed():
        try:proc.stdin.write(data);proc.stdin.close()
        except BaseException as e:errors.append(repr(e))
    thread=threading.Thread(target=feed);thread.start();proof={};recovered=None
    try:
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for member in archive:
                name=member.name.removeprefix('./');assert not Path(name).is_absolute() and '..' not in Path(name).parts
                if member.isfile():
                    stream=archive.extractfile(member);h=hashlib.sha256();n=0;chunks=[] if name==wanted else None
                    for part in iter(lambda:stream.read(8*2**20),b''):
                        h.update(part);n+=len(part)
                        if chunks is not None:chunks.append(part)
                    assert n==member.size;proof[name]=dict(bytes=n,sha256=h.hexdigest())
                    if chunks is not None:recovered=b''.join(chunks)
                elif member.islnk():
                    target=member.linkname.removeprefix('./');assert target in proof;proof[name]=proof[target].copy()
        thread.join();assert proc.wait(timeout=30)==0 and not errors
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        thread.join(timeout=2)
    return proof,recovered
def run():
    p=read(OUT/'protocol.json');assert p['source_sha256']==sha(__file__) and (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes();assert not (OUT/'summary.json').exists()
    for f in [Path(__file__),HERE/'protocol.json']:assert subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)==f.read_bytes()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')['root_tree']==tree
    check_protected(p);lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();before=shutil.disk_usage(ROOT).free;T=transport();completed=[];receipts=[]
    index=dict(schema=1,state='COLD_EVICTION_IN_PROGRESS',records=completed,assets=receipts,archive_source_sha256=p['source_manifest_sha256'],local_code_preserved=True,Phase_untouched=True)
    for name,stored in sorted(p['assets'].items()):
        assert time.time()-start<p['max_seconds'];asset=T.B.api('GET',f'releases/assets/{stored["id"]}');assert asset['name']==name and asset['size']==stored['bytes'] and asset['digest']=='sha256:'+stored['sha256'];asset=dict(asset,name=name,bytes=stored['bytes'],sha256=stored['sha256'])
        print('FULL_ASSET_READBACK_BEGIN',name,asset['bytes'],flush=True);data=asset_bytes(T,asset);assert len(data)==asset['bytes'] and hashlib.sha256(data).hexdigest()==asset['sha256'];proof,_=members(data)
        receipt=dict(asset_name=name,asset_id=asset['id'],asset_url=asset['browser_download_url'],asset_bytes=asset['bytes'],asset_sha256=asset['sha256'],state='FULL_REMOTE_ARCHIVE_AND_MEMBER_SHA_VERIFIED',members=len(proof));receipts.append(receipt);save(OUT/('readback_'+name+'.json'),receipt);check_protected(p)
        sha_index={(v['bytes'],v['sha256']):member for member,v in proof.items()}
        for record in p['records']:
            if any(r['id']==record['id'] for r in completed):continue
            member=sha_index.get((record['bytes'],record['sha256']))
            if member is None:continue
            for alias in record['aliases']:
                f=Path(alias);s=f.stat();assert s.st_size==record['bytes'] and sha(f)==record['sha256'] and s.st_ino==record['inode'];assert str(f.resolve()) not in p['protected_hashes']
            remote=dict(record,state='REMOTE_ONLY',asset=receipt.copy(),member=member,remote_member_sha256=record['sha256']);completed.append(remote);save(INDEX,index)  # Durable restore mapping BEFORE unlink.
            for alias in record['aliases']:Path(alias).unlink()
            assert all(not Path(alias).exists() for alias in record['aliases']);save(OUT/'eviction_ledger.json',dict(state='REMOTE_VERIFIED_LOCAL_EVICTION',completed_ids=[r['id'] for r in completed],bytes=sum(r['bytes'] for r in completed)));print('EVICTED',record['id'],record['bytes'],len(record['aliases']),flush=True)
        del data;gc.collect();save(INDEX,index)
        if len(completed)==len(p['records']):break
    assert len(completed)==len(p['records']);check_protected(p);index['state']='REMOTE_ONLY_VERIFIED';save(INDEX,index)
    summary=dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,files=len(completed),local_aliases_removed=sum(len(r['aliases']) for r in completed),model_bytes_evicted=sum(r['bytes'] for r in completed),allocated_blocks_evicted=sum(r['allocated_bytes'] for r in completed),free_before=before,free_after=shutil.disk_usage(ROOT).free,new_model_uploads=0,all_remote_assets_and_members_readback_verified=True,protected_inputs_unchanged=True,native=0,Adam=0,Phase_untouched=True);save(OUT/'summary.json',summary);save(DOC/'CLEANUP_REPORT.json',summary);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print('COLD_EVICTION_COMPLETE',summary['model_bytes_evicted'],flush=True)
def restore(path,verify_only=False):
    idx=read(INDEX);requested=Path(path);requested=requested if requested.is_absolute() else ROOT/requested;requested=str(requested.resolve());record=next(r for r in idx['records'] if requested in r['aliases'] or any(str((ROOT/x['archive_path']).resolve())==requested for x in r['rows']));a=record['asset'];T=transport();asset=T.B.api('GET',f'releases/assets/{a["asset_id"]}');assert asset['digest']=='sha256:'+a['asset_sha256'];data=asset_bytes(T,dict(asset,bytes=a['asset_bytes'],sha256=a['asset_sha256']));proof,body=members(data,record['member']);assert body is not None and len(body)==record['bytes'] and hashlib.sha256(body).hexdigest()==record['sha256']
    if verify_only:return dict(state='RESTORE_BYTES_VERIFIED_IN_RAM',path=requested,bytes=len(body),sha256=record['sha256'])
    assert not Path(requested).exists() and shutil.disk_usage(ROOT).free-len(body)>=39*2**30
    Path(requested).parent.mkdir(parents=True,exist_ok=True);Path(requested).write_bytes(body);assert sha(requested)==record['sha256'];os.chmod(requested,record['mode']);os.utime(requested,ns=(record['mtime_ns'],record['mtime_ns']));return dict(state='RESTORED_REQUESTED_FILE_ONLY',path=requested,bytes=len(body))
if __name__=='__main__':
    try:
        if sys.argv[1]=='restore':print(json.dumps(restore(sys.argv[2],len(sys.argv)>3 and sys.argv[3]=='--verify-only')))
        else:assert sys.argv[1] in ['prepare','run'];globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/'failure.json',dict(error=traceback.format_exc(),native=0,Adam=0,restore_index=str(INDEX)))
        raise
