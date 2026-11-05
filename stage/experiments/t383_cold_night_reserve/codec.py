"""Five fixed ended cold checkpoints, exact raw/gzip/old-XZ streams retained."""
from pathlib import Path
import gzip,hashlib,json,lzma,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t383_cold_night_reserve';DOC=ROOT/'stage/docs/overnight_20261105'
OLD=ROOT/'stage/experiments/t306_reversible_cold_codec/protocol.json'
FIXED=['stage/runs/t165_resume/gamma099/checkpoint_latest.pt.gz.lxz',
       'stage/runs/t165_resume/gamma2s/checkpoint_latest.pt.gz.lxz',
       'stage/runs/t171_reverse_curriculum/checkpoint_latest.pt.gz.lxz',
       'stage/runs/t170_vector_td3/checkpoint_latest.pt.gz.lxz',
       'stage/runs/t167_smdp/learn/checkpoint_latest.pt.gz.lxz']
def read(f):return json.loads(Path(f).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def save(f,x):
    f=Path(f);tmp=f.with_suffix(f.suffix+'.partial')
    with tmp.open('w') as w:json.dump(x,w,indent=2,allow_nan=False);w.write('\n');w.flush();os.fsync(w.fileno())
    tmp.replace(f)
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def prior(r):
    audit=read(ROOT/'stage/docs/overnight_20261102/T306_AUDIT.json');assert audit['state']=='PASS' and audit['all_original_raw_and_gzip_byte_streams_exact']
    receipt=read(ROOT/r['prior_verified_raw_storage_receipt'])
    assert receipt['state']=='LOSSLESS_LOCAL_COMPACTION_COMPLETE'
    assert receipt.get('full_remote_original_archive_verified') or receipt.get('full_remote_original_archive_sha_verified')
    row=next(x for x in receipt['records'] if x['compressed_path']==r['original_gzip_path'])
    assert row['original_sha256']==r['raw_sha256'] and row['compressed_sha256']==r['gzip_sha256']
def check(raw,r):
    assert len(raw)==r['raw_bytes'] and sha(raw)==r['raw_sha256']
    gz=gzip.compress(raw,compresslevel=6,mtime=0);old=lzma.compress(raw,preset=1)
    assert sha(gz)==r['gzip_sha256'] and len(gz)==r['gzip_bytes']
    assert sha(old)==r['encoded_sha256'] and len(old)==r['encoded_bytes']
    return gz,old
def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['protected_hashes'].items():assert sha(Path(f).read_bytes())==h,f
    for f,h in p['source_hashes'].items():assert sha(Path(f).read_bytes())==h,f
    assert used()<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
def prepare():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True);start=time.time();rows=[]
    science=read(ROOT/'stage/runs/t377_full_human_observation_transfer/protocol.json');protected={f:h for f,h in science['input_hashes'].items()}
    for folder in ['t379_human_only_control','t381_native_mppi_teacher','t382_receding_human_teacher']:
        for f in (ROOT/'stage/runs'/folder).rglob('*'):
            if f.is_file():protected[str(f)]=sha(f.read_bytes())
    old=read(OLD)
    for path in FIXED:
        assert time.time()-start<600 and str(ROOT/path) not in protected
        r=next(x for x in old['records'] if x['encoded_path']==path);assert r['transform']=='plain';prior(r)
        data=(ROOT/path).read_bytes();assert sha(data)==r['encoded_sha256'];raw=lzma.decompress(data);check(raw,r)
        z=lzma.compress(raw,preset=9|lzma.PRESET_EXTREME);assert lzma.decompress(z)==raw and len(z)<len(data)
        rows.append(dict(old_record=r,new_path=path+'.t383.rxz',new_bytes=len(z),new_sha256=sha(z)))
    released=sum(r['old_record']['encoded_bytes']-r['new_bytes'] for r in rows)
    assert released>=18*1024**2,('Zero mutation: insufficient fixed reserve',released)
    assert used()+128*1024<1024**3
    p=dict(id='T383',records=rows,protected_hashes=protected,source_hashes={str(f):sha(f.read_bytes()) for f in [Path(__file__),OLD,DOC/'STORAGE_PLAN.md']},
        own_bytes=128*1024,wall_seconds=600,audit_seconds=300,released_payload_bytes=released,native=0,optimizer=0,
        recipe='XZ preset9 EXTREME over original raw; all original raw/gzip6mtime0/XZpreset1 byte streams exact; fixed5 only, no model grid')
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'preflight.json',dict(state='PASS',files=5,modified=0,released_payload_bytes=released,native=0,optimizer=0,all_three_original_streams_exact=True))
    print(json.dumps(dict(state='REGISTERED_PASS',released=released)),flush=True)
def launch():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'main.log').open('xb') as log:proc=subprocess.Popen(['timeout','--kill-after=1s','600s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'codec.py'),'main'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote=receipt['remote_commit']));print(proc.pid)
def main():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'summary.json').exists();start=time.time();done=[];state='FAILED'
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));save(OUT/'encoding_ledger.json',dict(state='PENDING',records=p['records'],completed=[],current=None))
    try:
        for row in p['records']:
            assert time.time()-start<560;prior(row['old_record']);r=row['old_record'];f=ROOT/r['encoded_path'];dest=ROOT/row['new_path'];assert not dest.exists()
            old=f.read_bytes();assert sha(old)==r['encoded_sha256'];raw=lzma.decompress(old);check(raw,r)
            z=lzma.compress(raw,preset=9|lzma.PRESET_EXTREME);assert sha(z)==row['new_sha256'] and lzma.decompress(z)==raw
            save(OUT/'encoding_ledger.json',dict(state='JOURNALED_BEFORE_RENAME_INPLACE_WRITE',records=p['records'],completed=done,current=r['encoded_path'],crash_recovery='Preserve partial file and ledger; original raw in verified remote archive; never restart old experiment'))
            f.rename(dest)
            with dest.open('r+b') as w:w.write(z);w.truncate();w.flush();os.fsync(w.fileno())
            restored=lzma.decompress(dest.read_bytes());assert restored==raw;check(restored,r);done.append(r['encoded_path'])
            save(OUT/'encoding_ledger.json',dict(state='ENCODING',records=p['records'],completed=done,current=None))
            assert used()<1024**3
        state='COMPLETE'
    except BaseException:save(OUT/'failure.json',dict(error=traceback.format_exc(),completed=done))
    save(OUT/'encoding_ledger.json',dict(state=state,records=p['records'],completed=done,current=None,load_original='t383_cold_night_reserve/codec.py load_original(path) returns raw.pt/original.gz/original.lxz exactly'))
    save(OUT/'summary.json',dict(state=state,pid=os.getpid(),elapsed_s=time.time()-start,all_workers_stopped=True,native=0,optimizer=0,files=len(done),released_payload_bytes=sum(x['old_record']['encoded_bytes']-x['new_bytes'] for x in p['records'] if x['old_record']['encoded_path'] in done),global_bytes=used()))
    verify(p);assert state=='COMPLETE';print(json.dumps(read(OUT/'summary.json')))
def audit():
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');verify(p);assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();start=time.time()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path('/proc',str(pid)).exists()
    for row in p['records']:
        assert time.time()-start<300;r=row['old_record'];prior(r);assert not (ROOT/r['encoded_path']).exists();z=(ROOT/row['new_path']).read_bytes();assert sha(z)==row['new_sha256'];check(lzma.decompress(z),r)
    report=dict(state='PASS',files=5,all_original_raw_gzip_prior_lxz_bytes_exact=True,all_current_inputs_unchanged=True,all_remote_original_proofs_verified=True,new_encoded_remote_pending=True,native=0,optimizer=0,released_payload_bytes=s['released_payload_bytes'],global_bytes=used())
    save(OUT/'audit.json',report);save(DOC/'STORAGE_AUDIT.json',report);print(json.dumps(report))
def load_original(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f
    if f.exists():return f.read_bytes()
    relative=str(f.relative_to(ROOT));row=next(x for x in read(OUT/'encoding_ledger.json')['records'] if relative in [x['old_record']['encoded_path'],x['old_record']['original_raw_path'],x['old_record']['original_gzip_path']]);r=row['old_record'];z=(ROOT/row['new_path']).read_bytes();assert sha(z)==row['new_sha256'];raw=lzma.decompress(z);gz,old=check(raw,r)
    return raw if relative==r['original_raw_path'] else gz if relative==r['original_gzip_path'] else old
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T383 bounded storage')));signal.alarm(600)
    assert sys.argv[1] in ['prepare','launch','main','audit'];globals()[sys.argv[1]]()
