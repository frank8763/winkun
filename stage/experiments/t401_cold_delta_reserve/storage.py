"""Eight ended checkpoints: remote original readback, then reversible byte delta."""
from pathlib import Path
import sys,io,gzip,lzma,json,hashlib,struct,zipfile,tarfile,time,os,shutil,signal
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t401_cold_delta_reserve';DOC=ROOT/'stage/docs/overnight_20261106'
RAWREF=ROOT/'stage/runs/t166_steering_probe/steer_minus0p15.pt'
GZREF=ROOT/'stage/runs/t211_guarded_progress/models/original_round0.pt.gz'
FIXED=['t166_steering_probe/steer_minus0p3.pt','t166_steering_probe/steer_plus0p15.pt','t166_steering_probe/steer_plus0p3.pt','t168_distance_ablation/scale0p0.pt','t168_distance_ablation/scale0p5.pt','t219_rnd_runtime/control_Actor.pt.gz','t219_rnd_runtime/rnd_Actor.pt.gz','t225_paired_coverage/T221_reconstructed_Actor.pt.gz']
def sha(b):return hashlib.sha256(b).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def save(f,v):
    f=Path(f);tmp=f.with_suffix(f.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def payloads(raw):
    out={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for i in z.infolist():
            assert i.compress_type==0
            pos=i.header_offset+30+sum(struct.unpack_from('<HH',raw,i.header_offset+26));b=z.read(i.filename)
            assert raw[pos:pos+len(b)]==b;out[i.filename.split('/',1)[-1]]=(pos,b)
    return out
def encode(blob,isgzip):
    ref=GZREF if isgzip else RAWREF;rb=ref.read_bytes();raw=gzip.decompress(blob) if isgzip else blob
    base=payloads(gzip.decompress(rb) if isgzip else rb);changed=bytearray(raw);ops=[];level=None
    if isgzip:
        levels=[k for k in [6,9] if gzip.compress(raw,compresslevel=k,mtime=0)==blob];assert levels;level=levels[0]
    for name,(pos,b) in payloads(raw).items():
        cs=[(k,v) for k,(_,v) in base.items() if k.startswith('data/') and len(v)==len(b)]
        if not name.startswith('data/') or not cs:continue
        stride=max(1,len(b)//512);a=np.frombuffer(b,np.uint8)
        k,v=min(cs,key=lambda x:np.count_nonzero(a[::stride]!=np.frombuffer(x[1],np.uint8)[::stride]))
        changed[pos:pos+len(b)]=np.bitwise_xor(a,np.frombuffer(v,np.uint8)).tobytes();ops.append([k,pos,len(b)])
    h=dict(raw_size=len(raw),raw_sha256=sha(raw),original_sha256=sha(blob),gzip=bool(isgzip),level=level,reference=str(ref.relative_to(ROOT)),reference_sha256=sha(rb),ops=ops)
    header=json.dumps(h,separators=(',',':')).encode();pad=bytes(changed)+b'\0'*((-len(changed))%4)
    return lzma.compress(len(header).to_bytes(8,'little')+header+np.frombuffer(pad,np.uint8).reshape(-1,4).T.copy().tobytes(),preset=1)
def decode(z):
    b=lzma.decompress(z);n=int.from_bytes(b[:8],'little');h=json.loads(b[8:8+n]);ref=(ROOT/h['reference']).read_bytes();assert sha(ref)==h['reference_sha256']
    base=payloads(gzip.decompress(ref) if h['gzip'] else ref);raw=np.frombuffer(b[8+n:],np.uint8).reshape(4,-1).T.copy().tobytes();assert raw[h['raw_size']:]==b'\0'*((-h['raw_size'])%4);raw=bytearray(raw[:h['raw_size']])
    for k,pos,size in h['ops']:raw[pos:pos+size]=np.bitwise_xor(np.frombuffer(raw[pos:pos+size],np.uint8),np.frombuffer(base[k][1],np.uint8)).tobytes()
    assert sha(raw)==h['raw_sha256'];out=gzip.compress(bytes(raw),compresslevel=h['level'],mtime=0) if h['gzip'] else bytes(raw);assert sha(out)==h['original_sha256'];return out
def backup():
    assert not OUT.exists();OUT.mkdir();DOC.mkdir(exist_ok=True)
    files=[ROOT/'stage/runs'/p for p in FIXED];manifest=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for f in files]
    save(DOC/'T401_ORIGINAL_MANIFEST.json',dict(files=manifest,scope='Fixed8 cold checkpoint originals, all bytes'))
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as t:
        for f,row in zip(files,manifest):
            info=tarfile.TarInfo(row['path']);info.size=row['bytes'];info.mtime=0;t.addfile(info,io.BytesIO(f.read_bytes()))
    data=gzip.compress(buf.getvalue(),compresslevel=3,mtime=0)
    sys.path.insert(0,str(ROOT/'stage/experiments/t377_full_human_observation_transfer'));import transport as T
    T.install();release=T.B.api('GET','releases/tags/stage-distance-2026-11-01');name='stage-t401-cold-originals-'+sha(data)[:12]+'.tar.gz'
    asset=json.loads(T.req('POST',f"https://uploads.github.com/repos/{T.A.SLUG}/releases/{release['id']}/assets?name="+name,data,{'Content-Type':'application/octet-stream'}))
    asset=T.B.api('GET',f"releases/assets/{asset['id']}");assert asset['size']==len(data) and asset['digest']=='sha256:'+sha(data)
    returned=T.readback(asset,len(data));assert returned==data;T.B.verify_tar(returned,manifest)
    save(DOC/'T401_ORIGINAL_RECEIPT.json',dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',asset_url=asset['browser_download_url'],archive_sha256=sha(data),manifest_sha256=sha((DOC/'T401_ORIGINAL_MANIFEST.json').read_bytes()),members=8,temporary_disk_archive_bytes=0))
    print('T401_ORIGINAL8_FULL_REMOTE_READBACK_PASS',flush=True)
def prepare():
    assert not (OUT/'protocol.json').exists();assert read(DOC/'T401_ORIGINAL_RECEIPT.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest=read(DOC/'T401_ORIGINAL_MANIFEST.json')['files'];records=[]
    for row in manifest:
        f=ROOT/row['path'];b=f.read_bytes();assert sha(b)==row['sha256'];z=encode(b,f.name.endswith('.gz'));assert decode(z)==b
        records.append(dict(row,encoded_path=row['path']+'.t401.dxz',encoded_bytes=len(z),encoded_sha256=sha(z)))
    gain=sum(r['bytes']-r['encoded_bytes'] for r in records);assert gain>13*1024**2
    protected={str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in [RAWREF,GZREF,ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz',ROOT/'stage/runs/t194_drive_sensitivity/selected_policy.pt',ROOT/'stage/runs/t393_safe_demo_pg/candidate.pt.gz',ROOT/'stage/runs/t398_warm_recovery_rl/warm/proposed.pt.gz']}
    p=dict(id='T401_FIXED8_REVERSIBLE_COLD_DELTA',records=records,protected=protected,source_sha256=sha(Path(__file__).read_bytes()),max_seconds=600,own_metadata_bytes=256*1024,native=0,optimizer=0,expected_payload_saving=gain)
    save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);print('T401_PREFLIGHT_PASS',gain,flush=True)
def run():
    p=read(OUT/'protocol.json');assert not (OUT/'summary.json').exists() and not (OUT/'encoding_ledger.json').exists();assert sha(Path(__file__).read_bytes())==p['source_sha256']
    for f,h in p['protected'].items():assert sha((ROOT/f).read_bytes())==h
    start=time.time();done=[];ledger=dict(state='PENDING',completed=[],current=None);save(OUT/'encoding_ledger.json',ledger)
    for row in p['records']:
        assert time.time()-start<550 and used()<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
        f=ROOT/row['path'];dest=ROOT/row['encoded_path'];assert not dest.exists();raw=f.read_bytes();assert sha(raw)==row['sha256'];z=encode(raw,f.name.endswith('.gz'));assert sha(z)==row['encoded_sha256'] and decode(z)==raw
        ledger.update(state='JOURNALED_BEFORE_RENAME',current=row['path']);save(OUT/'encoding_ledger.json',ledger);f.rename(dest)
        with dest.open('r+b') as stream:stream.write(z);stream.truncate();stream.flush();os.fsync(stream.fileno())
        assert decode(dest.read_bytes())==raw;done.append(row['path']);ledger.update(state='ENCODING',completed=done,current=None);save(OUT/'encoding_ledger.json',ledger)
    ledger.update(state='ALL_ORIGINAL_BYTES_PRESERVED');save(OUT/'encoding_ledger.json',ledger)
    save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),native=0,optimizer=0,files=len(done),released_payload_bytes=p['expected_payload_saving'],elapsed_s=time.time()-start,all_workers_stopped=True));print('T401_ENCODE_COMPLETE',p['expected_payload_saving'])
def audit():
    p=read(OUT/'protocol.json');s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for row in p['records']:
        assert not (ROOT/row['path']).exists();z=(ROOT/row['encoded_path']).read_bytes();assert sha(z)==row['encoded_sha256'];b=decode(z);assert len(b)==row['bytes'] and sha(b)==row['sha256']
    for f,h in p['protected'].items():assert sha((ROOT/f).read_bytes())==h
    save(OUT/'audit.json',dict(state='PASS',native=0,optimizer=0,all8_original_byte_streams_exact=True,protected_unchanged=True,remote_originals_full_readback=True,new_encoded_binaries_local_only=True,global_bytes=used()))
    save(DOC/'T401_AUDIT.json',read(OUT/'audit.json'));print('T401_INDEPENDENT_BYTE_AUDIT_PASS')
def load_original(path):
    f=Path(path);f=f if f.is_absolute() else ROOT/f
    if f.exists():return f.read_bytes()
    row=next(r for r in read(OUT/'protocol.json')['records'] if r['path']==str(f.relative_to(ROOT)));b=decode((ROOT/row['encoded_path']).read_bytes());assert sha(b)==row['sha256'];return b
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T401 bounded storage600s')));signal.alarm(600);globals()[sys.argv[1]]()
