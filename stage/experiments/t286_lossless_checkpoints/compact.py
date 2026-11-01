"""Back up fixed ended Stage checkpoints, full readback, then reversible gzip.

Model bytes remain exact locally encoded and remotely original. No navigation/optimization.
"""
from pathlib import Path
import fcntl,gzip,io,json,os,shutil,subprocess,sys,tarfile,time,urllib.parse
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/long_distance_20261101'
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
from backup import readback
from archive import api,req,sha,verify_tar,write,SLUG
LEDGER=DOC/'T286_CHECKPOINT_ENCODING_RECEIPT.json'


def used():return sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())


def no_native_process():
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not args or not Path(os.fsdecode(args[0])).name.startswith('python'):continue
        active=[os.fsdecode(a) for a in args[1:] if str(ROOT/'stage/experiments') in os.fsdecode(a)]
        assert not active,('Another Stage Python process',entry.name,active)


def main():
    assert not LEDGER.exists(),'Inspect existing ledger; do not duplicate compaction'
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    no_native_process();before=used();assert before<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    protected=set()
    for name in ['t285_trajectory_constraints']:
        p=json.loads((ROOT/'stage/experiments'/name/'protocol.json').read_text())
        protected.update(p['input_hashes']);protected.update(p.get('historical_training_logs',[]))
    records=[];compressed={};manifest=[]
    for directory in sorted((ROOT/'stage/runs').iterdir()):
        if not directory.is_dir() or not directory.name[1:4].isdigit() or not 174<=int(directory.name[1:4])<=216 or directory.name.startswith('t194_'):continue
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or f.suffix!='.pt' or f.stat().st_size<2*1024**2 or str(f) in protected:continue
            dest=f.with_suffix(f.suffix+'.gz')
            if dest.exists():continue
            b=f.read_bytes();z=gzip.compress(b,compresslevel=6,mtime=0)
            if len(z)>=len(b):continue
            assert gzip.decompress(z)==b
            relative=str(f.relative_to(ROOT));compressed[relative]=z
            manifest.append(dict(path=relative,bytes=len(b),sha256=sha(b)))
            records.append(dict(original_path=relative,original_bytes=len(b),original_sha256=sha(b),
                encoded_path=str(dest.relative_to(ROOT)),encoded_bytes=len(z),encoded_sha256=sha(z)))
    assert len(records)==43 and sum(r['original_bytes'] for r in records)==224224131
    assert min(before+r['encoded_bytes'] for r in records)<1024**3
    if sys.argv[1:]==['--preflight']:
        write(DOC/'T286_ZERO_STEP_PREFLIGHT.json',dict(state='PASS',members=len(records),original_bytes=sum(r['original_bytes'] for r in records),encoded_bytes=sum(r['encoded_bytes'] for r in records),physics_steps=0,optimizer_calls=0,remote_uploads=0,files_encoded=0,all_T285_inputs_excluded=True,T194_excluded=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print('T286 selection preflight PASS zero mutations',flush=True);return
    assert not sys.argv[1:]
    mp=DOC/'source_manifest_T286.json';write(mp,dict(scope='Ended Stage T174–T216 cold checkpoints excluding T194/current inputs; full originals before reversible gzip, no Phase',files=manifest))
    release=api('GET','releases/tags/stage-distance-2026-11-01')
    assets=[]
    for page in range(1,10):
        batch=api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");assets+=batch
        if len(batch)<100:break
    chunks=[];current=[];size=0
    for r in manifest:
        if current and size+r['bytes']>24*1024**2:chunks.append(current);current=[];size=0
        current.append(r);size+=r['bytes']
    if current:chunks.append(current)
    parts=[]
    for index,entries in enumerate(chunks):
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w') as tar:
            for r in entries:
                b=(ROOT/r['path']).read_bytes();assert sha(b)==r['sha256']
                info=tarfile.TarInfo(r['path']);info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
        data=gzip.compress(buf.getvalue(),compresslevel=6,mtime=0);verify_tar(data,entries)
        name=f'stage-cold-checkpoints-t286-part{index:02d}-'+sha(data)[:12]+'.tar.gz'
        asset=next((a for a in assets if a['name']==name),None)
        if asset is None:
            asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),data,{'Content-Type':'application/octet-stream'}))
        asset=api('GET',f"releases/assets/{asset['id']}")
        assert asset['state']=='uploaded' and asset['size']==len(data) and asset['digest']=='sha256:'+sha(data)
        remote=readback(asset,len(data));assert remote==data;verify_tar(remote,entries)
        parts.append(dict(asset_url=asset['browser_download_url'],asset_id=asset['id'],bytes=len(data),sha256=sha(data),members=len(entries)))
        write(DOC/'T286_REMOTE_PARTS.json',dict(state='PARTIAL_REMOTE_READBACK',parts=parts,local_checkpoints_encoded=0))
        print(json.dumps(dict(state='PART_FULL_READBACK_VERIFIED',part=index,parts=len(chunks),members=len(entries))),flush=True)
    no_native_process()
    receipt=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T286',parts=parts,
        bytes=sum(r['bytes'] for r in parts),members=len(manifest),manifest_sha256=sha(mp.read_bytes()),
        temporary_disk_archive_bytes=0,physics_steps=0,optimizer_updates=0,verified_epoch=time.time())
    write(DOC/'release_receipt_T286.json',receipt)
    ledger=dict(state='REMOTE_VERIFIED_PENDING_LOCAL_ENCODING',records=records,global_bytes_before=before,
        protected_current_inputs=len(protected),source_sha256=sha(Path(__file__).read_bytes()),
        full_remote_original_binary_and_all_members_verified=True,physics_steps=0,optimizer_steps=0,
        restore='gzip.decompress(Path(encoded_path).read_bytes()); verify original_sha256. Reserve original_bytes under global1GiB before writing original_path. Never resume ended experiments.',
        legacy_consumers='Ended checkpoint paths have .pt.gz encoding; decode in RAM or capacity-checked explicit restore. All T285 input hash paths and T194 remain untouched; original protocols unchanged.',
        model_contents_arrays_protocols_and_summaries_changed=0)
    write(LEDGER,ledger)
    for r in sorted(records,key=lambda record:record['encoded_bytes']):
        src=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];assert sha(src.read_bytes())==r['original_sha256']
        assert used()+r['encoded_bytes']<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
        assert not dest.exists();dest.write_bytes(compressed[r['original_path']])
        assert sha(dest.read_bytes())==r['encoded_sha256'] and sha(gzip.decompress(dest.read_bytes()))==r['original_sha256']
        src.unlink()  # Original bytes remain locally encoded AND fully verified remotely.
    for r in records:assert sha(gzip.decompress((ROOT/r['encoded_path']).read_bytes()))==r['original_sha256']
    after=used();assert after<before<1024**3
    ledger.update(state='LOSSLESS_CHECKPOINT_ENCODING_COMPLETE',global_bytes_after=after,logical_bytes_released=before-after,
        all_original_bytes_retained_locally_and_remotely=True,completed_epoch=time.time())
    write(LEDGER,ledger);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps({k:ledger[k] for k in ['state','global_bytes_before','global_bytes_after','logical_bytes_released']}),flush=True)


if __name__=='__main__':main()
