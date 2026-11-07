"""Fixed ended NPZ lossless reserve, preserving every original archive byte.

Policy snapshots XOR against one retained reference; trajectory arrays XOR
successive rows. No Torch, simulator, optimizer, tensor quantization or sampling.
The original NPZ must regenerate byte-for-byte before it is admitted. All
original NPZ streams are RAM-archived and remotely read back before mutation.
"""
from pathlib import Path
import collections,fcntl,gzip,hashlib,io,json,lzma,os,re,shutil,subprocess,sys,tarfile,time,traceback,urllib.parse
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t409_cold_npz_reserve';DOC=ROOT/'stage/docs/review_20261107'
REF=ROOT/'stage/runs/t196_left_exploration/action_noise/behavior_batch_1.npz'
TARGET=30*2**20;META=1024**2;GLOBAL=2**30
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);tmp=Path(str(p)+'.partial')
    with tmp.open('xb') as f:f.write((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def arrays(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    assert all(v.dtype.kind!='O' for v in a.values())
    check=io.BytesIO();np.savez_compressed(check,**a);assert check.getvalue()==blob,'Original ZIP representation must regenerate exactly'
    return a
def encode(blob,kind,reference=None):
    a=arrays(blob);base=arrays(reference) if kind=='policy_reference_xor' else {};packed={};methods={}
    meta=dict(version=1,kind=kind,original_bytes=len(blob),original_sha256=sha(blob),order=list(a),arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape)) for k,v in a.items()},reference_sha256=None if reference is None else sha(reference))
    for k,v in a.items():
        if kind=='policy_reference_xor' and k in base and base[k].shape==v.shape and base[k].dtype==v.dtype:
            raw=np.bitwise_xor(np.ascontiguousarray(v).view(np.uint8).reshape(-1),np.ascontiguousarray(base[k]).view(np.uint8).reshape(-1));pad=np.pad(raw,(0,(-len(raw))%4));packed[k]=pad.reshape(-1,4).T.copy();methods[k]='reference_byteplane4'
        elif v.dtype.kind=='f' and v.ndim and v.shape[0]>1:
            raw=np.ascontiguousarray(v).view(np.uint8).reshape(v.shape[0],-1).copy();raw[1:]=np.bitwise_xor(raw[1:],raw[:-1]);packed[k]=raw.T.copy();methods[k]='row_xor_byteplane'
        else:packed[k]=v;methods[k]='literal'
    meta['methods']=methods;header=json.dumps(meta,separators=(',',':')).encode();body=io.BytesIO();np.savez(body,**packed)
    return lzma.compress(len(header).to_bytes(8,'little')+header+body.getvalue(),preset=1)
def decode(blob,reference=None):
    raw=lzma.decompress(blob);n=int.from_bytes(raw[:8],'little');m=json.loads(raw[8:8+n]);assert m['version']==1
    base=arrays(reference) if m['kind']=='policy_reference_xor' else {}
    assert m['reference_sha256']==(sha(reference) if reference is not None else None)
    result={}
    with np.load(io.BytesIO(raw[8+n:]),allow_pickle=False) as z:
        for k in m['order']:
            d=m['arrays'][k];v=z[k].copy();method=m['methods'][k];dt=np.dtype(d['dtype']);shape=tuple(d['shape'])
            if method=='reference_byteplane4':
                size=int(np.prod(shape,dtype=np.int64))*dt.itemsize;x=v.T.copy().reshape(-1);assert not x[size:].any();x=np.bitwise_xor(x[:size],np.ascontiguousarray(base[k]).view(np.uint8).reshape(-1));v=x.view(dt).reshape(shape).copy()
            elif method=='row_xor_byteplane':v=np.bitwise_xor.accumulate(v.T.copy(),axis=0).reshape(-1).view(dt).reshape(shape).copy()
            else:assert method=='literal'
            assert v.dtype==dt and v.shape==shape;result[k]=v
    buf=io.BytesIO();np.savez_compressed(buf,**result);out=buf.getvalue();assert len(out)==m['original_bytes'] and sha(out)==m['original_sha256'];return out
def stopped():
    d=ROOT/'stage/runs/t408_joint_wide';s=read(d/'summary.json');assert s['state']=='COMPLETE' and read(d/'audit.json')['state']=='PASS'
    assert read(d/'driver_final_status.json')['state']=='COMPLETE_AUDITED_REMOTE_VERIFIED'
    assert read(d/'source_finalization.json')['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED'
    pids={s['pid'],read(d/'launch.json')['pid'],read(d/'driver_started.json')['pid'],*read(d/'workers.json')['workers']}
    for pid in pids:assert not Path('/proc',str(pid)).exists(),pid
    return sorted(pids)
def protected():
    result={}
    for name in ['t377_full_human_observation_transfer','t404_last_narrow','t406_wide_full_route','t408_joint_wide']:
        p=read(ROOT/'stage/runs'/name/'protocol.json')
        for f,h in p.get('inputs',p.get('input_hashes',{})).items():
            q=Path(f);q=q if q.is_absolute() else ROOT/q;assert sha(q.read_bytes())==h;result[str(q.relative_to(ROOT))]=h
    for name in ['t255_overnight','t404_last_narrow','t406_wide_full_route','t408_joint_wide']:
        for f in (ROOT/'stage/runs'/name).rglob('*'):
            if f.is_file():result[str(f.relative_to(ROOT))]=sha(f.read_bytes())
    for f in [REF,ROOT/'stage/runs/t194_drive_sensitivity/selected_policy.pt',ROOT/'stage/runs/t166_steering_probe/steer_minus0p15.pt',ROOT/'stage/runs/t211_guarded_progress/models/original_round0.pt.gz']:
        result[str(f.relative_to(ROOT))]=sha(f.read_bytes())
    return result
def candidates(keep):
    fs=[]
    for f in (ROOT/'stage/runs').rglob('*.npz'):
        if not f.is_file() or f.is_symlink() or f.stat().st_size<100000 or str(f.relative_to(ROOT)) in keep:continue
        match=re.fullmatch(r't(\d{3})_.+',f.relative_to(ROOT/'stage/runs').parts[0]);number=int(match.group(1)) if match else 0
        if not 177<=number<=219:continue
        if any(t in str(f.relative_to(ROOT)).lower() for t in ['demo','human','teacher','video','retention']):continue
        kind='policy_reference_xor' if number in [196,197,200] and f.name.startswith('behavior_batch') else 'trajectory_row_xor'
        fs.append((f,kind))
    return sorted(fs,key=lambda x:(-x[0].stat().st_size,str(x[0])))
def transport():
    sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as transfer
    transfer.install();return transfer
def backup(manifest):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as archive:
        for row in manifest:
            data=(ROOT/row['path']).read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256'];info=tarfile.TarInfo(row['path']);info.size=len(data);info.mtime=0;archive.addfile(info,io.BytesIO(data))
    data=gzip.compress(buf.getvalue(),compresslevel=3,mtime=0);transfer=transport();transfer.B.verify_tar(data,manifest);release=transfer.B.api('GET','releases/tags/stage-distance-2026-11-01');name='stage-t409-originals-'+sha(data)[:16]+'.tar.gz';found=[]
    for page in range(1,21):
        batch=transfer.B.api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");found.extend(a for a in batch if a['name']==name)
        if len(batch)<100:break
    else:raise AssertionError('Asset inventory bounded')
    assert len(found)<=1
    asset=found[0] if found else json.loads(transfer.req('POST',f"https://uploads.github.com/repos/{transfer.A.SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),data,{'Content-Type':'application/octet-stream'}))
    asset=transfer.B.api('GET',f"releases/assets/{asset['id']}");assert asset['size']==len(data) and asset['digest']=='sha256:'+sha(data);returned=transfer.readback(asset,len(data));assert returned==data;transfer.B.verify_tar(returned,manifest)
    return dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',asset_url=asset['browser_download_url'],asset_id=asset['id'],archive_sha256=sha(data),archive_bytes=len(data),manifest_sha256=sha((json.dumps(dict(files=manifest),indent=2)+'\n').encode()),members=len(manifest),temporary_disk_archive_bytes=0)
def prepare():
    assert not OUT.exists() and not (HERE/'protocol.json').exists();pids=stopped();keep=protected();reference=REF.read_bytes();records=[];skipped=[];start=time.time()
    for f,kind in candidates(keep):
        assert time.time()-start<600;blob=f.read_bytes()
        try:
            a=arrays(blob)
            if kind=='trajectory_row_xor' and 'states' not in a:continue
            z=encode(blob,kind,reference if kind=='policy_reference_xor' else None);assert decode(z,reference if kind=='policy_reference_xor' else None)==blob
        except (AssertionError,ValueError,KeyError) as e:skipped.append(dict(path=str(f.relative_to(ROOT)),reason=repr(e)));continue
        gain=len(blob)-len(z)
        if gain<20000:continue
        records.append(dict(original_path=str(f.relative_to(ROOT)),original_bytes=len(blob),original_sha256=sha(blob),kind=kind,encoded_path=str(f.relative_to(ROOT))+'.t409.nxz',encoded_bytes=len(z),encoded_sha256=sha(z),saving_bytes=gain))
    records.sort(key=lambda r:(-r['saving_bytes'],r['original_path']));chosen=[];gain=0
    for row in records:
        if gain>=TARGET:break
        chosen.append(row);gain+=row['saving_bytes']
    assert gain>=TARGET,('Insufficient fixed reserve: no original mutation',gain)
    assert used()+META<GLOBAL and shutil.disk_usage(ROOT).free>=40*2**30
    manifest=[dict(path=r['original_path'],bytes=r['original_bytes'],sha256=r['original_sha256']) for r in chosen];manifest.append(dict(path=str(REF.relative_to(ROOT)),bytes=len(reference),sha256=sha(reference)));manifest.sort(key=lambda r:r['path'])
    OUT.mkdir();save(DOC/'T409_ORIGINAL_MANIFEST.json',dict(files=manifest));receipt=backup(manifest);save(DOC/'T409_ORIGINAL_RECEIPT.json',receipt)
    p=dict(id='T409_FIXED_ENDED_NPZ_BYTE_EXACT_RESERVE',records=chosen,reference=str(REF.relative_to(ROOT)),reference_sha256=sha(reference),protected_hashes=keep,source_sha256=sha(Path(__file__).read_bytes()),original_receipt_sha256=sha((DOC/'T409_ORIGINAL_RECEIPT.json').read_bytes()),expected_payload_saving=gain,target_payload_saving=TARGET,eligible_candidates=len(records),skipped_noncanonical=skipped,stopped_T408_pids=pids,max_seconds=600,audit_seconds=300,metadata_bytes=META,global_bytes=GLOBAL,minimum_free_bytes=40*2**30,native=0,Adam=0,unique_original_byte_streams_removed=0,scope='Fixed ended Stage T177-T219 real NPZ>=100000B excluding all current protected/source inputs, demos/human/teacher/retention/videos. Eligible trajectory states or T196/197/200 behavior snapshots only. Descending saving minimal files for30MiB; no expansion after registration.',resolver='codec.py load_original(path) or guarded restore; originals remain byte-exact locally encoded plus remote fully verified')
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'preflight.json',dict(state='PASS_READY_FOR_SOURCE_COMMIT_SYNC_ONLY',files=len(chosen),native=0,Adam=0,original_mutations=0,expected_payload_saving=gain,all_original_NPZ_bytes_reconstructed_exact=True,all_original_remote_member_SHA_verified=True,global_before=used(),estimated_after_reserve_plus_41MiB=used()-gain+41*2**20+META,protected_files=len(keep)));save(DOC/'T409_PREFLIGHT.json',read(OUT/'preflight.json'));print('T409_PREFLIGHT_PASS',len(chosen),gain,flush=True)
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes() and sha(Path(__file__).read_bytes())==p['source_sha256']
    for f,h in p['protected_hashes'].items():assert sha((ROOT/f).read_bytes())==h,f
    assert used()<GLOBAL and shutil.disk_usage(ROOT).free>=p['minimum_free_bytes']
def registered(p):
    for f in [Path(__file__),HERE/'protocol.json']:assert subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))],cwd=ROOT)==f.read_bytes()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and r['root_tree']==tree;return dict(root_tree=tree,remote_commit=r['remote_commit'])
def run():
    p=read(OUT/'protocol.json');verify(p);source=registered(p);stopped();assert not (OUT/'status.json').exists() and not (OUT/'summary.json').exists();receipt=read(DOC/'T409_ORIGINAL_RECEIPT.json');assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and sha((DOC/'T409_ORIGINAL_RECEIPT.json').read_bytes())==p['original_receipt_sha256']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();before=used();done=[];reference=REF.read_bytes();save(OUT/'status.json',dict(state='ENCODING',pid=os.getpid(),epoch=start,**source))
    for row in p['records']:
        assert time.time()-start<p['max_seconds'] and used()+META<GLOBAL;f=ROOT/row['original_path'];dest=ROOT/row['encoded_path'];assert not dest.exists();blob=f.read_bytes();assert sha(blob)==row['original_sha256'];z=encode(blob,row['kind'],reference if row['kind']=='policy_reference_xor' else None);assert sha(z)==row['encoded_sha256'] and decode(z,reference if row['kind']=='policy_reference_xor' else None)==blob
        save(OUT/'encoding_ledger.json',dict(state='JOURNALED_BEFORE_RENAME_INPLACE',completed=done,current=row['original_path'],original_sha256=row['original_sha256'],remote_originals=receipt['asset_url'],crash_recovery='Keep partial/ledger; original remote exact archive remains. Do not restart ended training.'))
        f.rename(dest)
        with dest.open('r+b') as stream:stream.write(z);stream.truncate();stream.flush();os.fsync(stream.fileno())
        assert decode(dest.read_bytes(),reference if row['kind']=='policy_reference_xor' else None)==blob;done.append(row['original_path']);save(OUT/'encoding_ledger.json',dict(state='ENCODING',completed=done,current=None))
    verify(p);save(OUT/'encoding_ledger.json',dict(state='ALL_ORIGINAL_BYTE_STREAMS_LOCAL_AND_REMOTE',completed=done,current=None,resolver=p['resolver']));save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,files=len(done),native=0,Adam=0,unique_original_byte_streams_removed=0,released_payload_bytes=p['expected_payload_saving'],global_before=before,global_after=used(),all_workers_stopped=True,**source));save(OUT/'status.json',dict(state='COMPLETE',pid=os.getpid(),files=len(done)));fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print('T409_COMPLETE',p['expected_payload_saving'],flush=True)
def load_original(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f
    if f.exists():return f.read_bytes()
    p=read(OUT/'protocol.json');row=next(r for r in p['records'] if ROOT/r['original_path']==f);z=(ROOT/row['encoded_path']).read_bytes();assert sha(z)==row['encoded_sha256'];blob=decode(z,REF.read_bytes() if row['kind']=='policy_reference_xor' else None);assert sha(blob)==row['original_sha256'];return blob
def audit():
    p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not (OUT/'audit.json').exists()
    start=time.time()
    for row in p['records']:
        assert time.time()-start<p['audit_seconds'];assert not (ROOT/row['original_path']).exists();blob=load_original(ROOT/row['original_path']);assert len(blob)==row['original_bytes'] and sha(blob)==row['original_sha256']
    result=dict(state='PASS',files=len(p['records']),all_original_NPZ_archive_bytes_exact=True,all_current_protected_files_unchanged=True,all_originals_remote_full_member_SHA=True,native=0,Adam=0,unique_original_byte_streams_removed=0,released_payload_bytes=s['released_payload_bytes'],global_bytes=used(),new_encoded_payloads_local_only=True);save(OUT/'audit.json',result);save(DOC/'T409_AUDIT.json',result);print('T409_INDEPENDENT_BYTE_AUDIT_PASS',flush=True)
def restore(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();blob=load_original(f);assert used()+len(blob)+65536<GLOBAL and shutil.disk_usage(ROOT).free-len(blob)>=40*2**30
    with f.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    assert f.read_bytes()==blob;print('T409_ORIGINAL_RESTORED_KEEP_ENCODED',flush=True)
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ['prepare','run','audit','restore']
    try:restore(sys.argv[2]) if mode=='restore' else globals()[mode]()
    except BaseException:
        if OUT.exists():save(DOC/('T409_ERROR_'+mode+'.json'),dict(error=traceback.format_exc(),native=0,Adam=0,original_mutations='zero before run; otherwise durable encoding_ledger',restart=False))
        raise
