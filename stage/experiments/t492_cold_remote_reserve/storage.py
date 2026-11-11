"""Only ended unpinned Stage cold caches; full remote byte proof before unlink."""
from pathlib import Path
import concurrent.futures,importlib.util,gzip,hashlib,io,json,os,shutil,subprocess,sys,tarfile,time,urllib.error,urllib.parse
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t492_cold_remote_reserve';DOC=ROOT/'stage/docs/overnight_20261111'
sys.path.insert(0,str(ROOT/'stage/experiments/t486_adam_memory_projection'));import t486_cache as K
LIMIT=2**30;TARGET=480*2**20;TAG='stage-cold-reserve-2026-11-11';PART=4*2**20
def sha(b):return hashlib.sha256(b).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def save(f,v):
    f=Path(f);b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();tmp=Path(str(f)+'.partial')
    with tmp.open('xb') as s:s.write(b);s.flush();os.fsync(s.fileno())
    tmp.replace(f)
def used():return K.used()
def idle():K._idle()
def prepare():
    idle();assert not OUT.exists()
    old=read(ROOT/'stage/runs/t491_uncentered_eight/protocol.json')
    keep=K._protected(set(old['inputs'])|set(old['protected_files'])|set(old.get('constructor_reads',[])))
    rows=[];proof={}
    for f in sorted((ROOT/'stage/runs').rglob('*')):
        if not f.is_file() or f.is_symlink() or str(f) in keep or f.suffix not in ['.rxz','.sxz','.dxz','.nxz','.npz']:continue
        root=ROOT/'stage/runs'/f.relative_to(ROOT/'stage/runs').parts[0]
        if root.name.split('_')[0] in ['t145','t146','t159','t194','t255']:continue
        if root not in proof:proof[root]=K._completed(root)
        if proof[root] is None or Path(str(f)+'.stage-ref.json').exists() or Path(str(f)+'.stage-remote.json').exists():continue
        b=f.read_bytes();rows.append(dict(path=str(f.relative_to(ROOT)),bytes=len(b),sha256=sha(b),ended_proof=proof[root]))
    # Prefer large encoded cold payloads; avoid thousands of tiny marker files.
    rows.sort(key=lambda r:(r['path'].endswith('.npz'),-r['bytes'],r['path']))
    chosen=[];release=0
    for r in rows:
        chosen.append(r);release+=r['bytes']-2048
        if release>=TARGET+2**20:break
    assert release>=TARGET+2**20,('Insufficient proof-backed reserve',release,TARGET)
    p=dict(id='T492_ENDED_COLD_CACHE_REMOTE_RESERVE',files=chosen,target_net_release=TARGET,expected_conservative_release=release,
        protected={f:sha(Path(f).read_bytes()) for f in sorted(keep) if Path(f).is_file()},
        source={str(f):sha(f.read_bytes()) for f in HERE.glob('*.py')},max_seconds=1800,part_bytes=PART,max_threads=4,
        native=0,Adam=0,only_encoded_or_NPZ_cache=True,no_retained_current_model_video_or_unique_evidence_deletion=True,
        restore='storage.py restore ENCODED_OR_NPZ_PATH; exact encoded bytes restored before invoking its historical decoder',
        scope='All selected source bytes uploaded and read back in full before ANY unlink. Existing original archives/codec ledgers untouched. Offline availability of selected cache becomes false.')
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(DOC/'T492_PLAN.json',dict(state='REGISTERED_NOT_EXECUTED',files=len(chosen),payload=sum(r['bytes'] for r in chosen),conservative_release=release,max_seconds=1800,native=0,Adam=0))
    print('T492_REGISTERED',len(chosen),release,flush=True)
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in {**p['protected'],**p['source']}.items():assert sha(Path(f).read_bytes())==h,f
    assert used()<=LIMIT and shutil.disk_usage(ROOT).free>=40*2**30
def install():
    spec=importlib.util.spec_from_file_location('_stage492_transport_wrapper',HERE/'transport.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install();return mod.T
def run():
    idle();p=read(OUT/'protocol.json');verify(p);assert not (OUT/'started.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True,cwd=ROOT).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    start=time.time();end=start+p['max_seconds'];save(OUT/'started.json',dict(pid=os.getpid(),epoch=start,deadline=end,native=0,Adam=0));T=install()
    try:release=T.B.api('GET','releases/tags/'+TAG)
    except urllib.error.HTTPError as e:
        if e.code!=404:raise
        release=T.B.api('POST','releases',dict(tag_name=TAG,name='Stage cold caches, exact remote restore',body='Ended unpinned cold byte streams. Manifest/receipt in source repository. No science claim.',draft=False,prerelease=False))
    assets=[]
    for page in range(1,12):
        batch=T.B.api('GET',f"releases/{release['id']}/assets?per_page=100&page={page}");assets+=batch
        if len(batch)<100:break
    bundles=[];current=[];size=0
    for row in p['files']:
        if current and size+row['bytes']>8*2**20:bundles.append(current);current=[];size=0
        current.append(row);size+=row['bytes']
    if current:bundles.append(current)
    def upload(pair):
        index,rows=pair;assert time.time()<end-180;buf=io.BytesIO()
        manifest=[{k:r[k] for k in ['path','bytes','sha256']} for r in rows]
        with tarfile.open(fileobj=buf,mode='w') as archive:
            for r in rows:
                blob=(ROOT/r['path']).read_bytes();assert sha(blob)==r['sha256'];info=tarfile.TarInfo(r['path']);info.size=len(blob);info.mtime=0;archive.addfile(info,io.BytesIO(blob))
        data=gzip.compress(buf.getvalue(),compresslevel=1,mtime=0);T.B.verify_tar(data,manifest);parts=[];returned=[]
        for n,off in enumerate(range(0,len(data),PART)):
            assert time.time()<end-120;part=data[off:off+PART];name=f"stage-t492-{index:04d}-{n:02d}-{sha(part)[:20]}.bin"
            asset=next((a for a in assets if a['name']==name and a.get('digest')=='sha256:'+sha(part)),None)
            if asset is None:
                asset=json.loads(T.req('POST',f"https://uploads.github.com/repos/{T.A.SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(name),part,{'Content-Type':'application/octet-stream'}))
            asset=T.B.api('GET',f"releases/assets/{asset['id']}");assert asset['size']==len(part) and asset['digest']=='sha256:'+sha(part)
            blob=T.readback(asset,len(part));assert blob==part;returned.append(blob)
            parts.append(dict(order=n,asset_id=asset['id'],url=asset['browser_download_url'],bytes=len(blob),sha256=sha(blob)))
        restored=b''.join(returned);assert restored==data;T.B.verify_tar(restored,manifest)
        return dict(index=index,files=rows,archive_sha256=sha(data),archive_bytes=len(data),parts=parts)
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for bundle in pool.map(upload,enumerate(bundles)):
            completed.append(bundle);save(OUT/'upload_progress.json',dict(state='READBACK_IN_PROGRESS_NO_UNLINK',completed_bundles=len(completed),bundles=len(bundles),files_deleted=0));print('T492_REMOTE_BUNDLE',len(completed),len(bundles),flush=True)
    remote_files=[dict(row,bundle_index=bundle['index']) for bundle in completed for row in bundle['files']]
    verify(p);idle();assert time.time()<end-90
    for row in remote_files:assert sha((ROOT/row['path']).read_bytes())==row['sha256']
    receipt=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T492',url=release['html_url'],files=remote_files,bundles=completed,
        encoded_and_NPZ_source_bytes_preserved_exact=True,temporary_disk_archive_bytes=0,files_deleted=0,verified_epoch=time.time())
    save(DOC/'release_receipt_T492.json',receipt)
    save(OUT/'remote_verified_before_unlink.json',dict(state=receipt['state'],receipt_sha256=sha((DOC/'release_receipt_T492.json').read_bytes()),files=len(remote_files),epoch=time.time()))
    before=used();ledger=OUT/'eviction_ledger.jsonl'
    with ledger.open('xb') as log:
        for row in remote_files:
            assert time.time()<end-30;f=ROOT/row['path'];assert sha(f.read_bytes())==row['sha256'];marker=Path(str(f)+'.t492-remote.json')
            entry=dict(original=row['path'],bytes=row['bytes'],sha256=row['sha256'],receipt=str((DOC/'release_receipt_T492.json').relative_to(ROOT)),receipt_sha256=sha((DOC/'release_receipt_T492.json').read_bytes()),offline_available=False,restore=p['restore'])
            save(marker,entry);marker.chmod(0o444);f.unlink();log.write((json.dumps(entry,separators=(',',':'))+'\n').encode());log.flush();os.fsync(log.fileno())
    verify(p);assert before-used()>=TARGET
    save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),files=len(remote_files),released_net_bytes=before-used(),native=0,Adam=0,all_remote_verified_before_any_unlink=True,elapsed_s=time.time()-start))
    print('T492_COMPLETE',before-used(),flush=True)
def audit():
    idle();p=read(OUT/'protocol.json');verify(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    r=read(DOC/'release_receipt_T492.json');assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    by={x['path']:x for x in r['files']}
    for row in p['files']:
        f=ROOT/row['path'];assert not f.exists();m=read(Path(str(f)+'.t492-remote.json'));assert m['sha256']==row['sha256'] and m['bytes']==row['bytes']
        assert by[row['path']]['sha256']==row['sha256'];bundle=next(b for b in r['bundles'] if b['index']==by[row['path']]['bundle_index']);assert any(x['path']==row['path'] and x['sha256']==row['sha256'] for x in bundle['files'])
    save(OUT/'audit.json',dict(state='PASS',files=len(p['files']),native=0,Adam=0,current_inputs_unchanged=True,remote_all_bytes_verified_before_unlink=True,offline_cache_availability=False))
def restore(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f;assert not f.exists();m=read(Path(str(f)+'.t492-remote.json'));r=read(ROOT/m['receipt']);assert sha((ROOT/m['receipt']).read_bytes())==m['receipt_sha256'];row=next(x for x in r['files'] if ROOT/x['path']==f)
    assert used()+row['bytes']+65536<=LIMIT and shutil.disk_usage(ROOT).free-row['bytes']>=40*2**30;T=install();parts=[]
    bundle=next(b for b in r['bundles'] if b['index']==row['bundle_index'])
    for a in bundle['parts']:
        blob=T.readback({'id':a['asset_id']},a['bytes']);assert sha(blob)==a['sha256'];parts.append(blob)
    blob=b''.join(parts);assert sha(blob)==bundle['archive_sha256'];T.B.verify_tar(blob,[{k:x[k] for k in ['path','bytes','sha256']} for x in bundle['files']])
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:data=archive.extractfile(row['path']).read()
    assert sha(data)==row['sha256']
    with f.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
if __name__=='__main__':
    if sys.argv[1]=='restore':restore(sys.argv[2])
    else:globals()[sys.argv[1]]()
