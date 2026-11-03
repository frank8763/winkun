"""Fixed reversible coding of ended, fully remotely verified Stage evidence.

No model tensor, optimizer state, or trajectory byte is discarded. NPZ original
bytes must regenerate exactly in this environment before a file is recoded.
"""
from pathlib import Path
import argparse,fcntl,gzip,hashlib,io,json,lzma,os,shutil,subprocess,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t311_reversible_training_reserve';DOC=ROOT/'stage/docs/distance_20261103'
LEDGER=DOC/'T311_ENCODING_LEDGER.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def write(f,x):
    f=Path(f);tmp=f.with_suffix(f.suffix+'.partial');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())

def encode_npz(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    plain=io.BytesIO();np.savez_compressed(plain,**a);assert plain.getvalue()==blob
    assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and 'LAST_NEXT_RESERVED' not in a
    meta=dict(order=list(a),arrays={k:dict(dtype=v.dtype.str,shape=list(v.shape)) for k,v in a.items()})
    a['LAST_NEXT_RESERVED']=a.pop('next_states')[-1:];packed={};transformed=[]
    for k,v in a.items():
        assert v.dtype.kind!='O'
        if v.dtype.kind=='f' and v.ndim>0:
            x=np.ascontiguousarray(v).view(np.uint8).reshape(v.shape[0],-1).copy()
            x[1:]=np.bitwise_xor(x[1:],x[:-1]);packed[k]=x.T.copy();transformed.append(k)
        else:packed[k]=v
    meta['transformed']=transformed;meta['arrays']['LAST_NEXT_RESERVED']=dict(dtype=a['LAST_NEXT_RESERVED'].dtype.str,shape=list(a['LAST_NEXT_RESERVED'].shape))
    header=json.dumps(meta,separators=(',',':')).encode();body=io.BytesIO();np.savez(body,**packed)
    return lzma.compress(len(header).to_bytes(8,'little')+header+body.getvalue(),preset=1)

def decode_npz(blob):
    raw=lzma.decompress(blob);n=int.from_bytes(raw[:8],'little');meta=json.loads(raw[8:8+n]);a={}
    with np.load(io.BytesIO(raw[8+n:]),allow_pickle=False) as z:
        for k in z.files:
            v=z[k].copy()
            if k in meta['transformed']:
                x=v.T.copy();x=np.bitwise_xor.accumulate(x,axis=0);d=meta['arrays'][k]
                v=x.reshape(-1).view(np.dtype(d['dtype'])).reshape(d['shape']).copy()
            a[k]=v
    a['next_states']=np.concatenate([a['states'][1:],a.pop('LAST_NEXT_RESERVED')],axis=0)
    ordered={k:a[k] for k in meta['order']};buf=io.BytesIO();np.savez_compressed(buf,**ordered);return buf.getvalue()

def encode(blob,kind):
    if kind=='npz_chain_xor':return encode_npz(blob)
    assert kind=='gzip_byteplane4';raw=gzip.decompress(blob);assert gzip.compress(raw,compresslevel=6,mtime=0)==blob
    padded=raw+b'\0'*((-len(raw))%4);planes=np.frombuffer(padded,np.uint8).reshape(-1,4).T.copy().tobytes()
    return lzma.compress(planes,preset=1)

def decode(blob,r):
    if r['kind']=='npz_chain_xor':return decode_npz(blob)
    planes=lzma.decompress(blob);raw=np.frombuffer(planes,np.uint8).reshape(4,-1).T.copy().tobytes()
    assert raw[r['raw_bytes']:]==b'\0'*((-r['raw_bytes'])%4);raw=raw[:r['raw_bytes']]
    assert sha(raw)==r['raw_sha256'];return gzip.compress(raw,compresslevel=6,mtime=0)

def prepare():
    assert not OUT.exists() and not LEDGER.exists();records=[];inputs={};protected=set(read(ROOT/'stage/runs/t310_initial_history_recovery/protocol.json')['input_hashes'])
    for group,folder in [('T263','t263_adaptive_goal_course'),('T267','t267_causal_credit_course'),('T280','t280_local_cost')]:
        receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{group}.json';mf=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json'
        rr=read(receipt);assert rr['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and sha(mf.read_bytes())==rr['manifest_sha256']
        refs={v['path']:v for v in read(mf)['files']};inputs[str(receipt)]=sha(receipt.read_bytes());inputs[str(mf)]=sha(mf.read_bytes())
        for f in sorted((ROOT/'stage/runs'/folder/'episodes').glob('*.npz')):
            if f.stat().st_size<60000 or str(f) in protected:continue
            b=f.read_bytes();old=refs[str(f.relative_to(ROOT))];assert sha(b)==old['sha256'] and len(b)==old['bytes']
            try:encoded=encode(b,'npz_chain_xor')
            except (KeyError,AssertionError):continue
            assert decode_npz(encoded)==b
            if len(encoded)>=len(b):continue
            records.append(dict(original_path=str(f.relative_to(ROOT)),kind='npz_chain_xor',original_bytes=len(b),original_sha256=sha(b),encoded_path=str(f.relative_to(ROOT))+'.sxz',encoded_bytes=len(encoded),encoded_sha256=sha(encoded),prior_receipt=str(receipt.relative_to(ROOT)),prior_manifest=str(mf.relative_to(ROOT)),remote_original_path=str(f.relative_to(ROOT))))
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T286.json';mf=ROOT/'stage/docs/long_distance_20261101/source_manifest_T286.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and sha(mf.read_bytes())==read(receipt)['manifest_sha256']
    refs={v['path']:v for v in read(mf)['files']};inputs[str(receipt)]=sha(receipt.read_bytes());inputs[str(mf)]=sha(mf.read_bytes())
    prior=ROOT/'stage/docs/long_distance_20261101/T286_CHECKPOINT_ENCODING_RECEIPT.json';inputs[str(prior)]=sha(prior.read_bytes())
    for old in read(prior)['records']:
        f=ROOT/old['encoded_path']
        if not f.is_file() or f.stat().st_size<3*1024**2 or str(f) in protected:continue
        assert not any(v in str(f) for v in ['t194_','t255_','t299_','t288_'])
        b=f.read_bytes();assert sha(b)==old['encoded_sha256'];raw=gzip.decompress(b);ref=refs[old['original_path']]
        assert sha(raw)==old['original_sha256']==ref['sha256'] and len(raw)==ref['bytes']
        enc=encode(b,'gzip_byteplane4')
        if len(enc)>=len(b):continue
        r=dict(original_path=str(f.relative_to(ROOT)),kind='gzip_byteplane4',original_bytes=len(b),original_sha256=sha(b),raw_bytes=len(raw),raw_sha256=sha(raw),encoded_path=str(f.relative_to(ROOT))+'.sxz',encoded_bytes=len(enc),encoded_sha256=sha(enc),prior_receipt=str(receipt.relative_to(ROOT)),prior_manifest=str(mf.relative_to(ROOT)),remote_original_path=old['original_path'])
        assert decode(enc,r)==b;records.append(r)
    for f in [Path(__file__),DOC/'T311_PLAN.md']:inputs[str(f)]=sha(f.read_bytes())
    for f in protected:
        if Path(f).is_file():inputs[f]=sha(Path(f).read_bytes())
    saving=sum(r['original_bytes']-r['encoded_bytes'] for r in records)
    assert saving>=31*1024**2,('Insufficient reserve; no mutation',saving)
    assert used()+512*1024<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    p=dict(experiment='T311',records=records,input_hashes=inputs,wall_seconds=600,global_bytes=1024**3,min_free_bytes=40*1024**3,own_metadata_bytes=512*1024,expected_released_bytes=saving,native_steps=0,optimizer_calls=0)
    OUT.mkdir();write(HERE/'protocol.json',p);write(OUT/'protocol.json',p);write(OUT/'preflight.json',dict(state='PASS',all_original_byte_streams_exactly_reconstructed=True,remote_prior_member_SHA_verified=True,files_modified=0,native_steps=0,optimizer_calls=0,expected_released_bytes=saving,protocol_sha256=sha((HERE/'protocol.json').read_bytes())))
    print(json.dumps(dict(state='REGISTERED_PREFLIGHT_PASS',files=len(records),expected_saving=saving)),flush=True)

def run():
    p=read(HERE/'protocol.json');assert not (OUT/'status.json').exists() and not LEDGER.exists();start=time.time();before=used();state='FAILED'
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for f,h in p['input_hashes'].items():assert sha(Path(f).read_bytes())==h,f
    ledger=dict(state='PENDING',records=p['records'],completed=[],current=None,all_original_raw_bytes_in_prior_full_readback_archives=True,crash_recovery='Keep any partial file. Recover original member from prior full verified archive using prior_receipt/prior_manifest; for gzip restore raw member then gzip6mtime0. Verify original SHA before any encoding. Never restart old training.')
    write(LEDGER,ledger);write(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        for r in p['records']:
            assert time.time()<start+p['wall_seconds']-10 and used()<p['global_bytes']
            f=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];assert not dest.exists();b=f.read_bytes();assert sha(b)==r['original_sha256']
            z=encode(b,r['kind']);assert sha(z)==r['encoded_sha256'] and decode(z,r)==b
            ledger.update(state='JOURNALED_BEFORE_RENAME',current=r['original_path']);write(LEDGER,ledger)
            f.rename(dest);ledger['state']='RENAMED_ORIGINAL_BEFORE_INPLACE_ENCODING';write(LEDGER,ledger)
            with dest.open('r+b') as stream:stream.write(z);stream.truncate();stream.flush();os.fsync(stream.fileno())
            assert sha(dest.read_bytes())==r['encoded_sha256'] and sha(decode(dest.read_bytes(),r))==r['original_sha256']
            ledger['completed'].append(r['original_path']);ledger.update(state='ENCODING',current=None);write(LEDGER,ledger)
            write(OUT/'status.json',dict(state='ENCODING',pid=os.getpid(),completed=len(ledger['completed']),updated_epoch=time.time()))
        ledger.update(state='ALL_ORIGINAL_BYTES_PRESERVED',released_bytes=before-used());write(LEDGER,ledger);state='COMPLETE'
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),completed=len(ledger['completed']),current=ledger['current']))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    s=dict(state=state,pid=os.getpid(),all_workers_stopped=True,physics_steps=0,native_steps=0,optimizer_calls=0,files_encoded=len(ledger['completed']),released_bytes=before-used(),global_bytes=used(),elapsed_s=time.time()-start,ledger_sha256=sha(LEDGER.read_bytes()))
    write(OUT/'summary.json',s);write(OUT/'status.json',s);print(json.dumps(s),flush=True)

def audit():
    p=read(HERE/'protocol.json');s=read(OUT/'summary.json');dest=DOC/'T311_AUDIT.json';assert s['state']=='COMPLETE' and not dest.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    originals={str(ROOT/r['original_path']) for r in p['records']}
    for r in p['records']:
        assert not (ROOT/r['original_path']).exists();b=(ROOT/r['encoded_path']).read_bytes();assert sha(b)==r['encoded_sha256'] and sha(decode(b,r))==r['original_sha256']
    for f,h in p['input_hashes'].items():
        if f not in originals:assert sha(Path(f).read_bytes())==h,f
    assert used()<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    write(dest,dict(state='PASS',files=len(p['records']),all_original_NPZ_and_raw_model_and_gzip_bytes_exact=True,protected_current_inputs_unchanged=True,prior_full_remote_member_SHA_receipts_verified=True,new_encoded_binaries_not_uploaded=True,all_original_processes_absent=True,native_steps=0,optimizer_calls=0,global_bytes=used(),ledger_sha256=sha(LEDGER.read_bytes())))
    print('T311 independent lossless byte audit PASS')

def launch():
    p=read(HERE/'protocol.json');assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as f:proc=subprocess.Popen(['/usr/bin/timeout','--kill-after=1s','600s','/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=time.time(),remote_commit=receipt['remote_commit'],single_launch=True));print(json.dumps(dict(pid=proc.pid)))

def load_original(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    if path.exists():return path.read_bytes()
    r=next(r for r in read(LEDGER)['records'] if r['original_path']==str(path.relative_to(ROOT)))
    result=decode((ROOT/r['encoded_path']).read_bytes(),r);assert sha(result)==r['original_sha256'];return result

def restore(path):
    path=Path(path);path=path if path.is_absolute() else ROOT/path;assert not path.exists();b=load_original(path)
    assert used()+len(b)+65536<1024**3 and shutil.disk_usage(ROOT).free-len(b)>=40*1024**3
    with path.open('xb') as f:f.write(b)
    assert sha(path.read_bytes())==sha(b);print(json.dumps(dict(restored=str(path),bytes=len(b))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','run','audit','restore']);parser.add_argument('path',nargs='?');args=parser.parse_args()
    if args.mode=='restore':restore(args.path)
    else:globals()[args.mode]()
