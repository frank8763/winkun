"""Local reversible encoding of ended, remotely backed Stage JSONL logs.

This module never starts simulators, optimizers or network calls. Plans are
read-only. Encoding runs once before the night's preflight/launch, preserves
original bytes locally and remotely, and never edits a historical ledger.
"""
from pathlib import Path
import gzip,hashlib,json,os,re,shutil,signal,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
RUNS=ROOT/'stage/runs';DOC=ROOT/'stage/docs/long_distance_20261101'
OUT=RUNS/'t468_offline_recovery_night'
LIMIT=2**30;META=512*1024
COMPLETE='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'

def read(f):return json.loads(Path(f).read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()
def used():return sum(f.stat().st_size for f in RUNS.rglob('*') if f.is_file())
def encoded(x):return (json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def path(s):
    f=Path(s);return f if f.is_absolute() else ROOT/f

def exclusive(f,x):
    b=encoded(x);assert len(b)<=META and used()+len(b)<=LIMIT
    with Path(f).open('xb') as out:out.write(b);out.flush();os.fsync(out.fileno())

def protected(items):
    result={str(path(f)) for f in items}
    for f in RUNS.rglob('*.stage-ref.json'):
        ref=read(f);result.add(str(path(ref['canonical'])));result.add(str(f))
    return result

def eligible(f,exclude):
    m=re.match(r'stage/runs/t(\d+)_',str(f.relative_to(ROOT)))
    return bool(m and 299<=int(m[1])<=465 and f.suffix=='.jsonl'
        and not any(s in f.name.lower() for s in ['ledger','fail','error','partial','protocol'])
        and str(f) not in exclude and f.is_file() and not f.is_symlink()
        and f.resolve()==f and not Path(str(f)+'.t468.gz').exists())

def idle():
    for f in Path('/proc').iterdir():
        if not f.name.isdigit() or int(f.name)==os.getpid():continue
        try:args=(f/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        try:env=(f/'environ').read_bytes().split(b'\0')
        except OSError:env=[]
        assert os.fsencode('STAGE_RUN_ROOT='+str(RUNS)) not in env,('OtherStageProcess',f.name)
        assert not any(os.fsdecode(a).startswith((str(ROOT/'stage/experiments')+'/', 'stage/experiments/')) for a in args[1:]),('OtherStageScript',f.name)

def completed(f):
    """Conservative receipt for the closest completed run/subrun, zero mutation."""
    for folder in [f.parent,*f.parent.parents]:
        if folder==RUNS or not folder.is_relative_to(RUNS):break
        s=folder/'summary.json';seal=folder/'SEALED.json'
        if not s.exists():continue
        d=read(s)
        if d.get('state') not in ['COMPLETE','SEALED_FINITE_ROOT','COMPLETE_AUDITED_LOCAL']:return None
        ids=[]
        if d.get('pid'):ids.append(d['pid'])
        if seal.exists():
            z=read(seal)
            if not z.get('all_scientific_processes_stopped'):return None
            ids+=z.get('scientific_pids',[])
        for name in ['driver_started.json','launch.json']:
            q=folder/name
            if q.exists() and read(q).get('pid'):ids.append(read(q)['pid'])
        g=folder/'worker_generations.json'
        if g.exists():
            for z in read(g):ids+=[z['parent']]+z['workers']
        if any(Path('/proc',str(pid)).exists() for pid in ids):return None
        evidence=[s]+([seal] if seal.exists() else [])
        return [dict(path=str(q.relative_to(ROOT)),sha256=sha(q.read_bytes())) for q in evidence]
    return None

def plan(requested,required):
    """Report available release even when requested bytes cannot be satisfied."""
    assert required>=0
    exclude=protected(requested);candidates=[];seen=set()
    for mf in sorted(DOC.glob('source_manifest_T*.json')):
        m=re.search(r'T(\d+)',mf.name)
        if not m or not 299<=int(m[1])<=465:continue
        receipt=DOC/mf.name.replace('source_manifest_','release_receipt_')
        if not receipt.exists():continue
        rc=read(receipt)
        if rc.get('state')!=COMPLETE or rc.get('manifest_sha256')!=sha(mf.read_bytes()):continue
        for row in read(mf).get('files',[]):
            f=ROOT/row['path']
            if str(f) in seen or not eligible(f,exclude):continue
            evidence=completed(f)
            if evidence is None:continue
            b=f.read_bytes()
            if len(b)!=row['bytes'] or sha(b)!=row['sha256']:continue
            z=gzip.compress(b,mtime=0);net=len(b)-len(z)
            if net<=1024:continue
            seen.add(str(f));candidates.append(dict(path=row['path'],bytes=len(b),sha256=sha(b),
                gzip_path=row['path']+'.t468.gz',gzip_bytes=len(z),gzip_sha256=sha(z),net_release=net,
                receipt=str(receipt.relative_to(ROOT)),receipt_sha256=sha(receipt.read_bytes()),
                manifest=str(mf.relative_to(ROOT)),manifest_sha256=sha(mf.read_bytes()),archive_sha256=rc['sha256'],
                completed_evidence=evidence))
    candidates.sort(key=lambda z:(-z['net_release'],z['path']));chosen=[];total=0
    for row in candidates:
        if total>=required+META:break
        chosen.append(row);total+=row['net_release']
    return dict(state='LOCAL_READ_ONLY_ENCODING_PLAN',required_release_bytes=required,
        available_net_release_bytes=sum(z['net_release'] for z in candidates),selected_net_release_bytes=total,
        metadata_reserve_bytes=META,admitted=total>=required+META,rows=chosen,
        original_bytes_stay_local_and_remote=True,models_and_ledger_errors_untouched=True,native=0,optimizer=0)

def validate(p):
    exclude=protected(set(p.get('inputs',{}))|set(p.get('constructor_reads',[]))|set(p.get('protected_files',{})))
    for row in p['storage_rows']:
        f=path(row['path']);assert eligible(f,exclude),(f,'protected/ineligible')
        b=f.read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256']
        z=gzip.compress(b,mtime=0);assert len(z)==row['gzip_bytes'] and sha(z)==row['gzip_sha256']
        assert gzip.decompress(z)==b
        for key in ['receipt','manifest']:
            q=path(row[key]);assert sha(q.read_bytes())==row[key+'_sha256']
        rc=read(path(row['receipt']));assert rc['state']==COMPLETE and rc['sha256']==row['archive_sha256']
        assert any(x['path']==row['path'] and x['bytes']==row['bytes'] and x['sha256']==row['sha256'] for x in read(path(row['manifest']))['files'])
        assert completed(f) is not None
        for evidence in row['completed_evidence']:assert sha(path(evidence['path']).read_bytes())==evidence['sha256']

def execute(p):
    assert OUT.exists() and not any((OUT/f).exists() for f in ['compression_started.json','compression_receipt.json','launch.json','preflight.json'])
    idle();validate(p)
    src=read(RUNS/'t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert shutil.disk_usage(ROOT).free>=40*2**30
    start=time.time();before=used();assert before+META<=LIMIT
    previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T468 reversible encoding300s')));signal.alarm(300)
    try:
        exclusive(OUT/'compression_started.json',dict(pid=os.getpid(),epoch=start,native=0,optimizer=0))
        progress=OUT/'compression_progress.jsonl'
        with progress.open('xb') as log:
            for row in p['storage_rows']:
                assert time.time()-start<290
                f,zpath=path(row['path']),path(row['gzip_path']);b=f.read_bytes();assert sha(b)==row['sha256']
                z=gzip.compress(b,mtime=0);assert len(z)==row['gzip_bytes'] and sha(z)==row['gzip_sha256']
                assert used()+len(z)+4096<=LIMIT and not zpath.exists()
                with zpath.open('xb') as out:out.write(z);out.flush();os.fsync(out.fileno())
                assert gzip.decompress(zpath.read_bytes())==b and sha(f.read_bytes())==row['sha256']
                zpath.chmod(0o444)
                entry=dict(path=row['path'],gzip_path=row['gzip_path'],sha256=row['sha256'],gzip_sha256=row['gzip_sha256'],bytes=row['bytes'],gzip_bytes=row['gzip_bytes'])
                log.write(encoded(entry));log.flush();os.fsync(log.fileno())
                # Exact original bytes now exist in immutable local gzip and
                # the original verified remote archive before unlinking raw.
                f.unlink()
        after=used();own=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
        assert after<before and after-own+p['max_bytes']<=LIMIT
        exclusive(OUT/'compression_receipt.json',dict(state='COMPLETE_LOCAL_LOSSLESS_ENCODING',pid=os.getpid(),files=len(p['storage_rows']),
            before=before,after=after,actual_net_released_bytes=before-after,elapsed_s=time.time()-start,
            progress_sha256=sha(progress.read_bytes()),original_bytes_local_and_remote=True,native=0,optimizer=0))
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)

def audit(p):
    idle();s=read(OUT/'compression_receipt.json');assert s['state']=='COMPLETE_LOCAL_LOSSLESS_ENCODING'
    assert not Path('/proc',str(s['pid'])).exists() and not (OUT/'compression_audit.json').exists()
    records=[json.loads(x) for x in (OUT/'compression_progress.jsonl').read_text().splitlines()]
    assert len(records)==len(p['storage_rows']) and sha((OUT/'compression_progress.jsonl').read_bytes())==s['progress_sha256']
    for row,record in zip(p['storage_rows'],records):
        f,zpath=path(row['path']),path(row['gzip_path']);assert not f.exists() and zpath.is_file() and not zpath.is_symlink()
        z=zpath.read_bytes();b=gzip.decompress(z)
        assert len(z)==row['gzip_bytes'] and sha(z)==row['gzip_sha256'] and len(b)==row['bytes'] and sha(b)==row['sha256']
        assert record['path']==row['path'] and record['sha256']==row['sha256']
    exclusive(OUT/'compression_audit.json',dict(state='PASS_LOCAL_REVERSIBLE_ENCODING',files=len(records),native=0,optimizer=0))

def text(original):
    f=path(original)
    if f.exists():return f.read_text()
    p=read(OUT/'protocol.json');row=next(x for x in p['storage_rows'] if path(x['path'])==f)
    z=path(row['gzip_path']).read_bytes();assert sha(z)==row['gzip_sha256']
    b=gzip.decompress(z);assert len(b)==row['bytes'] and sha(b)==row['sha256'];return b.decode()

def restore(original):
    idle();f=path(original);p=read(OUT/'protocol.json');row=next(x for x in p['storage_rows'] if path(x['path'])==f)
    if f.exists():assert sha(f.read_bytes())==row['sha256'];return
    assert used()+row['bytes']+16384<=LIMIT and shutil.disk_usage(ROOT).free>=40*2**30
    z=path(row['gzip_path']).read_bytes();assert sha(z)==row['gzip_sha256'];b=gzip.decompress(z)
    assert len(b)==row['bytes'] and sha(b)==row['sha256']
    with f.open('xb') as out:out.write(b);out.flush();os.fsync(out.fileno())
    assert sha(f.read_bytes())==row['sha256']
