"""Transport-only per-child RAM archives; offline science never depends on this."""
from pathlib import Path
import gzip,importlib.util,io,json,signal,sys,tarfile,time,urllib.parse
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_stage493_old_transport',ROOT/'stage/experiments/t491_uncentered_eight/transport.py');OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
OUT=ROOT/'stage/runs/t493_wide8_night';DOC=ROOT/'stage/docs/long_distance_20261101'
def read(f):return json.loads(Path(f).read_text())
def stopped():
    s=read(OUT/'SEALED.json');assert s['all_scientific_processes_stopped'] and s['finalizer_writes_only_docs_after_this_point']
    assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
def archive(name,files,release,deadline,T):
    receipt=DOC/f'release_receipt_{name}.json';manifest=DOC/f'source_manifest_{name}.json'
    if receipt.exists():
        r=read(receipt);assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';m=read(manifest)['files']
        assert all(T.B.sha((ROOT/x['path']).read_bytes())==x['sha256'] for x in m);return r
    rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=T.B.sha(f.read_bytes())) for f in files]
    T.B.write(manifest,dict(scope='Ended Stage T493 sealed child or root metadata',files=rows,active_runs_included=False))
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for f,row in zip(files,rows):
            b=f.read_bytes();assert T.B.sha(b)==row['sha256'];i=tarfile.TarInfo(row['path']);i.size=len(b);i.mtime=0;tar.addfile(i,io.BytesIO(b))
    data=gzip.compress(buf.getvalue(),compresslevel=1,mtime=0);assert len(data)<96*2**20;T.B.verify_tar(data,rows);parts=[];returned=[]
    for k,off in enumerate(range(0,len(data),4*2**20)):
        assert time.time()<deadline-40;part=data[off:off+4*2**20];label=f'stage-{name.lower()}-{T.B.sha(data)[:12]}-{k:03d}.bin'
        a=json.loads(T.req('POST',f"https://uploads.github.com/repos/{T.A.SLUG}/releases/{release['id']}/assets?name="+urllib.parse.quote(label),part,{'Content-Type':'application/octet-stream'}));a=T.B.api('GET',f"releases/assets/{a['id']}")
        assert a['size']==len(part) and a['digest']=='sha256:'+T.B.sha(part);b=T.readback(a,len(part));assert b==part;returned.append(b)
        parts.append(dict(order=k,asset_id=a['id'],asset_url=a['browser_download_url'],bytes=len(b),sha256=T.B.sha(b)))
    data2=b''.join(returned);assert data2==data;T.B.verify_tar(data2,rows);stopped()
    r=dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group=name,url=release['html_url'],parts=parts,bytes=len(data),sha256=T.B.sha(data),manifest_sha256=T.B.sha(manifest.read_bytes()),members=len(rows),temporary_disk_archive_bytes=0,files_deleted=0,verified_epoch=time.time())
    T.B.write(receipt,r);return r
if __name__=='__main__':
    OLD.install();T=OLD.T
    if sys.argv[1]=='source_sync':
        import sync_source;sync_source.main()
    else:
        assert sys.argv[1]=='T493';stopped();start=time.time();deadline=start+580
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T493 final RAM archive580s')));signal.alarm(580)
        release=T.B.api('GET','releases/tags/stage-eight-only-2026-11-11');done=[]
        for child in sorted(f for f in OUT.iterdir() if f.is_dir()):
            files=sorted(f for f in child.rglob('*') if f.is_file() and f.suffix not in ['.pyc','.lock','.partial'])
            if files:done.append(archive('T493_'+child.name,files,release,deadline,T))
        files=sorted(f for f in OUT.iterdir() if f.is_file() and f.suffix not in ['.pyc','.lock','.partial'])
        done.append(archive('T493_root',files,release,deadline,T));stopped()
        T.B.write(DOC/'release_receipt_T493.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',group='T493',url=release['html_url'],child_receipts=[r['group'] for r in done],members=sum(r['members'] for r in done),bytes=sum(r['bytes'] for r in done),verified_epoch=time.time(),each_child_complete_binary_and_member_SHA=True))
        signal.alarm(0);print('T493_FULL_RAW_BACKUP_PASS',flush=True)
