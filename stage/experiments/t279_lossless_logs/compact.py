"""Back up ended Stage text logs, full remote readback, then reversible gzip.

No models/NPZ/protocols/summaries are changed. No navigation or optimization.
"""
from pathlib import Path
import fcntl,gzip,io,json,os,shutil,subprocess,sys,tarfile,time,urllib.parse
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/long_distance_20261101'
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
from backup import readback
from archive import api,req,sha,verify_tar,write,SLUG
LEDGER=DOC/'T279_LOG_ENCODING_RECEIPT.json'


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
    for name in ['t267_causal_credit_course','t278_fixed_goal_starts']:
        p=json.loads((ROOT/'stage/experiments'/name/'protocol.json').read_text())
        protected.update(p['input_hashes']);protected.update(p.get('historical_training_logs',[]))
    records=[];compressed={};manifest=[]
    for directory in sorted((ROOT/'stage/runs').iterdir()):
        if not directory.is_dir() or not directory.name[1:4].isdigit() or not 165<=int(directory.name[1:4])<=267:continue
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or f.suffix not in ['.log','.jsonl'] or f.stat().st_size<32768 or str(f) in protected:continue
            dest=f.with_suffix(f.suffix+'.gz')
            if dest.exists():continue
            b=f.read_bytes();z=gzip.compress(b,compresslevel=6,mtime=0)
            if len(z)>=len(b):continue
            assert gzip.decompress(z)==b
            relative=str(f.relative_to(ROOT));compressed[relative]=z
            manifest.append(dict(path=relative,bytes=len(b),sha256=sha(b)))
            records.append(dict(original_path=relative,original_bytes=len(b),original_sha256=sha(b),
                encoded_path=str(dest.relative_to(ROOT)),encoded_bytes=len(z),encoded_sha256=sha(z)))
    assert records and sum(r['original_bytes'] for r in records)<96*1024**2
    assert min(before+r['encoded_bytes'] for r in records)<1024**3
    mp=DOC/'source_manifest_T279.json';write(mp,dict(scope='Closed Stage T165–T267 text logs only, full originals before reversible encoding; no models or Phase',files=manifest))
    release=api('GET','releases/tags/stage-distance-2026-11-01')
    assets=[]
    for page in range(1,10):
        batch=api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");assets+=batch
        if len(batch)<100:break
    chunks=[];current=[];size=0
    for r in manifest:
        if current and size+r['bytes']>8*1024**2:chunks.append(current);current=[];size=0
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
        name=f'stage-cold-logs-t279-part{index:02d}-'+sha(data)[:12]+'.tar.gz'
        asset=next((a for a in assets if a['name']==name),None)
        if asset is None:
            asset=json.loads(req('POST',f"https://uploads.github.com/repos/{SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),data,{'Content-Type':'application/octet-stream'}))
        asset=api('GET',f"releases/assets/{asset['id']}")
        assert asset['state']=='uploaded' and asset['size']==len(data) and asset['digest']=='sha256:'+sha(data)
        remote=readback(asset,len(data));assert remote==data;verify_tar(remote,entries)
        parts.append(dict(asset_url=asset['browser_download_url'],asset_id=asset['id'],bytes=len(data),sha256=sha(data),members=len(entries)))
        write(DOC/'T279_REMOTE_PARTS.json',dict(state='PARTIAL_REMOTE_READBACK',parts=parts,local_logs_changed=0))
        print(json.dumps(dict(state='PART_FULL_READBACK_VERIFIED',part=index,parts=len(chunks),members=len(entries))),flush=True)
    no_native_process()
    receipt=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T279',parts=parts,
        bytes=sum(r['bytes'] for r in parts),members=len(manifest),manifest_sha256=sha(mp.read_bytes()),
        temporary_disk_archive_bytes=0,physics_steps=0,optimizer_updates=0,verified_epoch=time.time())
    write(DOC/'release_receipt_T279.json',receipt)
    ledger=dict(state='REMOTE_VERIFIED_PENDING_LOCAL_ENCODING',records=records,global_bytes_before=before,
        protected_current_inputs=len(protected),source_sha256=sha(Path(__file__).read_bytes()),
        full_remote_original_binary_and_all_members_verified=True,physics_steps=0,optimizer_steps=0,
        restore='gzip.decompress(Path(encoded_path).read_bytes()); verify original_sha256. Reserve original_bytes under global1GiB before writing original_path. Never resume ended experiments.',
        legacy_consumers='Old text paths have .gz encoding. Consumers must explicitly decode in RAM or restore with capacity check; original protocol hashes/remote members unchanged. Current T267/T278 protected inputs not changed.',
        models_arrays_protocols_and_summaries_changed=0)
    write(LEDGER,ledger)
    for r in records:
        src=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];assert sha(src.read_bytes())==r['original_sha256']
        assert used()+r['encoded_bytes']<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
        assert not dest.exists();dest.write_bytes(compressed[r['original_path']])
        assert sha(dest.read_bytes())==r['encoded_sha256'] and sha(gzip.decompress(dest.read_bytes()))==r['original_sha256']
        src.unlink()  # Original bytes remain locally encoded AND fully verified remotely.
    for r in records:assert sha(gzip.decompress((ROOT/r['encoded_path']).read_bytes()))==r['original_sha256']
    after=used();assert after<before<1024**3
    ledger.update(state='LOSSLESS_LOG_ENCODING_COMPLETE',global_bytes_after=after,logical_bytes_released=before-after,
        all_original_bytes_retained_locally_and_remotely=True,completed_epoch=time.time())
    write(LEDGER,ledger);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps({k:ledger[k] for k in ['state','global_bytes_before','global_bytes_after','logical_bytes_released']}),flush=True)


if __name__=='__main__':main()
