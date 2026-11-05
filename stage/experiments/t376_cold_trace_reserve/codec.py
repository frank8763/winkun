"""Two fixed ended JSON traces: byte-exact gzip cold encoding, zero learning."""
from pathlib import Path
import sys,os,json,gzip,hashlib,shutil,time,subprocess,signal
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t376_cold_trace_reserve';DOC=ROOT/'stage/docs/narrow_20261105'
sys.path.insert(0,str(ROOT/'stage/experiments/t260_distance_archive'))
import backup as B
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def used():return B.used()


def prior(r):
    receipt=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T369.json');mf=ROOT/'stage/docs/long_distance_20261101/source_manifest_T369.json'
    assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and sha(mf.read_bytes())==receipt['manifest_sha256']
    row=next(v for v in read(mf)['files'] if v['path']==r['original_path']);assert row['sha256']==r['original_sha256'] and row['bytes']==r['original_bytes']


def prepare():
    assert not OUT.exists() and read(DOC/'T375_CLOSEOUT.json')['state']=='PASS_LOCAL_CLOSURE_AND_FULL_REMOTE_BACKUP'
    old=ROOT/'stage/runs/t369_future_action_supervision';assert read(old/'summary.json')['state']=='COMPLETE' and read(old/'summary.json')['all_workers_stopped']
    for pid in [read(old/'summary.json')['pid'],*read(old/'workers.json')['workers']]:assert not Path('/proc',str(pid)).exists()
    protected={f:sha(Path(f).read_bytes()) for f in read(ROOT/'stage/runs/t375_observation_residual/protocol.json')['input_hashes']}
    for f in (ROOT/'stage/runs/t375_observation_residual').glob('*.pt.gz'):protected[str(f)]=sha(f.read_bytes())
    rows=[]
    for name in ['trace_current.json','trace_future.json']:
        f=old/name;assert str(f) not in protected;b=f.read_bytes();z=gzip.compress(b,compresslevel=9,mtime=0);assert gzip.decompress(z)==b
        r=dict(original_path=str(f.relative_to(ROOT)),original_bytes=len(b),original_sha256=sha(b),encoded_path=str(f.relative_to(ROOT))+'.t376.gz',encoded_bytes=len(z),encoded_sha256=sha(z));prior(r);assert not (ROOT/r['encoded_path']).exists();rows.append(r)
    assert sum(r['original_bytes']-r['encoded_bytes'] for r in rows)>950000
    p=dict(id='T376',records=rows,protected_hashes=protected,source_hashes={str(f):sha(f.read_bytes()) for f in [HERE/'codec.py',DOC/'T376_PLAN.md',Path(B.__file__)]},wall_seconds=180,audit_seconds=120,own_metadata_bytes=128*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,physics_steps=0,optimizer_calls=0,workers_spawned=0,contract=(DOC/'T376_PLAN.md').read_text())
    assert used()+p['own_metadata_bytes']+sum(r['encoded_bytes'] for r in rows)<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'preflight.json',dict(state='PASS',files_modified=0,native=0,Adam=0,original_remote_members_verified=True,decode_reencode_exact=True,expected_release=sum(r['original_bytes']-r['encoded_bytes'] for r in rows)))
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())+16384<p['own_metadata_bytes'];print('T376 PRECHECK_PASS')


def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for name,h in {**p['protected_hashes'],**p['source_hashes']}.items():assert sha(Path(name).read_bytes())==h,name
    assert used()<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']


def launch():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'launch.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as f:proc=subprocess.Popen(['timeout','--kill-after=1s','180s',str(ROOT/'run_stage.sh'),str(HERE/'codec.py'),'run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=proc.pid,epoch=time.time(),source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree']));print(proc.pid)


def run():
    p=read(HERE/'protocol.json');verify(p);assert not (OUT/'summary.json').exists();start=time.time();before=used();done=[]
    save(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    for r in p['records']:
        assert time.time()-start<180;prior(r);f=ROOT/r['original_path'];b=f.read_bytes();assert len(b)==r['original_bytes'] and sha(b)==r['original_sha256']
        z=gzip.compress(b,compresslevel=9,mtime=0);assert len(z)==r['encoded_bytes'] and sha(z)==r['encoded_sha256'] and gzip.decompress(z)==b
        dest=ROOT/r['encoded_path'];assert not dest.exists()
        with dest.open('xb') as w:w.write(z);w.flush();os.fsync(w.fileno())
        assert gzip.decompress(dest.read_bytes())==b and used()<p['global_bytes'];f.unlink();done.append(r);save(OUT/'encoding_ledger.json',dict(state='ENCODING',records=done))
    verify(p);released=sum(r['original_bytes']-r['encoded_bytes'] for r in done)
    save(OUT/'encoding_ledger.json',dict(state='COMPLETE',records=done,load_original='stage/experiments/t376_cold_trace_reserve/codec.py load_original(path)',old_original_remote_archive='release_receipt_T369.json',new_encoded_binaries_not_uploaded=True))
    save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,physics_steps=0,optimizer_calls=0,workers_spawned=0,all_workers_stopped=True,files=2,released_payload_bytes=released,logical_before=before,logical_after=used(),unique_original_bytes_locally_and_remotely_retained=True))
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<p['own_metadata_bytes'];print(json.dumps(read(OUT/'summary.json')))


def audit():
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');verify(p);assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();assert not Path('/proc',str(s['pid'])).exists()
    for r in p['records']:
        prior(r);assert not (ROOT/r['original_path']).exists();z=(ROOT/r['encoded_path']).read_bytes();b=gzip.decompress(z);assert len(z)==r['encoded_bytes'] and sha(z)==r['encoded_sha256'] and len(b)==r['original_bytes'] and sha(b)==r['original_sha256']
    result=dict(state='PASS',new_native=0,new_Adam=0,original_JSON_bytes_exact=True,protected_current_inputs_unchanged=True,prior_full_remote_members_verified=True,new_encoded_binaries_not_uploaded=True,released_payload_bytes=s['released_payload_bytes'],global_bytes=used());save(OUT/'audit.json',result);save(DOC/'T376_AUDIT.json',result);print(json.dumps(result))


def load_original(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f
    if f.exists():return f.read_bytes()
    r=next(v for v in read(OUT/'encoding_ledger.json')['records'] if ROOT/v['original_path']==f);z=(ROOT/r['encoded_path']).read_bytes();assert sha(z)==r['encoded_sha256'];b=gzip.decompress(z);assert sha(b)==r['original_sha256'] and len(b)==r['original_bytes'];return b


def restore(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f;b=load_original(f);assert not f.exists() and used()+len(b)+8192<1024**3
    with f.open('xb') as w:w.write(b);w.flush();os.fsync(w.fileno())
    assert f.read_bytes()==b;print('Original JSON restored; encoded copy and prior remote original retained')


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T376 bounded codec')));signal.alarm(180)
    if sys.argv[1]=='restore':restore(sys.argv[2])
    else:globals()[sys.argv[1]]()
