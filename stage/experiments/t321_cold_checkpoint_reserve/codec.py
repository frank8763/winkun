"""Lossless storage, never a navigation experiment or model modification.

XOR matching ZIP_STORED payloads against the protected safe checkpoint; preserve
every ZIP header, pickle, optimizer byte and gzip byte exactly. All originals
already have complete remote binary readback receipts. No torch deserialization.
"""
from pathlib import Path
import argparse, fcntl, gzip, hashlib, io, json, lzma, os, shutil, struct
import subprocess, time, traceback, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t321_cold_checkpoint_reserve'; DOC=ROOT/'stage/docs/distance_20261103'
LEDGER=DOC/'T321_ENCODING_LEDGER.json'
REF=ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'
GROUPS={'T256-T258':['t257_distance_policy_gradient'], 'T259':['t259_closed_loop_recovery'],
        'T263':['t263_adaptive_goal_course'], 'T267':['t267_causal_credit_course'],
        'T280':['t280_local_cost'], 'T291':['t291_hard_guard_ablation'],
        'T298':['t298_self_goal_transfer'], 'T302':['t302_balanced_bridge_learning'],
        'T312':['t312_online_goal_ppo'], 'T312R':['t312_remaining_recovery'],
        'T313':['t313_fisher_feasibility'], 'T315':['t315_full_trajectory_gradient']}
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p); t=p.with_suffix(p.suffix+'.partial')
    with t.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def used():return sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
def payloads(raw):
    result={}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for item in z.infolist():
            assert item.compress_type==zipfile.ZIP_STORED
            h=item.header_offset; assert raw[h:h+4]==b'PK\x03\x04'
            n,e=struct.unpack_from('<HH',raw,h+26);start=h+30+n+e
            data=raw[start:start+item.file_size];assert data==z.read(item.filename)
            result[item.filename.split('/',1)[-1]]=(start,data)
    return result
def encode(blob,reference):
    raw=gzip.decompress(blob)
    levels=[level for level in (6,9) if gzip.compress(raw,compresslevel=level,mtime=0)==blob]
    assert levels,'Unknown gzip representation: preserve original, do not encode'
    base=payloads(gzip.decompress(reference));changed=bytearray(raw);ops=[]
    for name,(pos,data) in payloads(raw).items():
        if not name.startswith('data/') or name not in base or len(data)!=len(base[name][1]):continue
        b=base[name][1]
        changed[pos:pos+len(data)]=np.bitwise_xor(np.frombuffer(data,np.uint8),np.frombuffer(b,np.uint8)).tobytes()
        ops.append([name,pos,len(data)])
    padded=bytes(changed)+b'\0'*((-len(changed))%4)
    planes=np.frombuffer(padded,np.uint8).reshape(-1,4).T.copy().tobytes()
    header=dict(version=1,reference_sha256=sha(reference),raw_bytes=len(raw),raw_sha256=sha(raw),
                gzip_level=levels[0],original_sha256=sha(blob),ops=ops)
    h=json.dumps(header,separators=(',',':')).encode()
    return lzma.compress(len(h).to_bytes(8,'little')+h+planes,preset=1)
def decode(blob,reference):
    data=lzma.decompress(blob);n=int.from_bytes(data[:8],'little');h=json.loads(data[8:8+n])
    assert h['version']==1 and sha(reference)==h['reference_sha256']
    raw=np.frombuffer(data[8+n:],np.uint8).reshape(4,-1).T.copy().tobytes()
    assert raw[h['raw_bytes']:]==b'\0'*((-h['raw_bytes'])%4)
    raw=bytearray(raw[:h['raw_bytes']]);base=payloads(gzip.decompress(reference))
    for name,pos,size in h['ops']:
        assert len(base[name][1])==size
        raw[pos:pos+size]=np.bitwise_xor(np.frombuffer(raw[pos:pos+size],np.uint8),np.frombuffer(base[name][1],np.uint8)).tobytes()
    assert sha(raw)==h['raw_sha256'];result=gzip.compress(bytes(raw),compresslevel=h['gzip_level'],mtime=0)
    assert sha(result)==h['original_sha256'];return result
def verify_prior(r):
    receipt=read(ROOT/r['prior_receipt']);manifest=(ROOT/r['prior_manifest']).read_bytes()
    assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert receipt['manifest_sha256']==sha(manifest)
    entry=next(x for x in json.loads(manifest)['files'] if x['path']==r['original_path'])
    assert (entry['sha256'],entry['bytes'])==(r['original_sha256'],r['original_bytes'])
def protected():
    # Keep current scientific inputs in their existing representation. The fixed
    # cold set contains only ended candidate checkpoints, never baseline/data.
    names=[REF,ROOT/'stage/runs/t299_conditional_goal_likelihood/conditional_goal_likelihood_final_checkpoint.pt.gz',
           ROOT/'stage/runs/t250_actuator_policy_gradient/final_Actor.pt.gz',
           ROOT/'stage/runs/t256_distance_frontier/stable_5m_current_states.npz']
    for folder in ['t317_human_demonstration','t318_remaining_human','t319_imperfect_demo_head','t320_human_throttle_augmentation','t288_five_metre_presentation']:
        names.extend(p for p in (ROOT/'stage/runs'/folder).rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in names}
def prepare():
    assert not OUT.exists() and not LEDGER.exists();start=time.time();ref=REF.read_bytes();records=[];skipped=[]
    assert sha(ref)=='63e63e2b444867b26622a2ab0f0c4227bc38d48a7cd2b5a3b378d9b0266006df'
    for group,folders in GROUPS.items():
        for folder in folders:
            directory=ROOT/'stage/runs'/folder;s=read(directory/'summary.json')
            assert s['state'] in ['COMPLETE','FAILED'] and s.get('all_workers_stopped',False)
            if (directory/'workers.json').exists():
                w=read(directory/'workers.json')
                for pid in [w.get('parent'),*w.get('workers',[])]:
                    if pid:assert not Path(f'/proc/{pid}').exists(),('live/reused',pid)
            for f in sorted(directory.glob('*.pt.gz')):
                assert time.time()-start<240 and f.stat().st_size>1000000
                b=f.read_bytes();r=dict(original_path=str(f.relative_to(ROOT)),original_bytes=len(b),original_sha256=sha(b),
                    encoded_path=str(f.relative_to(ROOT))+'.dxz',prior_receipt=f'stage/docs/long_distance_20261101/release_receipt_{group}.json',
                    prior_manifest=f'stage/docs/long_distance_20261101/source_manifest_{group}.json')
                verify_prior(r);enc=encode(b,ref);assert decode(enc,ref)==b
                if len(enc)>=len(b):skipped.append(r['original_path']);continue
                r.update(encoded_bytes=len(enc),encoded_sha256=sha(enc));records.append(r)
    saving=sum(r['original_bytes']-r['encoded_bytes'] for r in records)
    assert saving>=24*1024**2,('Insufficient reserve; zero mutation',saving)
    assert used()+512*1024<=1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
    keep=protected();assert not set(keep)&{r['original_path'] for r in records}
    p=dict(experiment='T321',records=records,protected_hashes=keep,reference=str(REF.relative_to(ROOT)),reference_sha256=sha(ref),
           source_hashes={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__).read_bytes())},
           wall_seconds=600,audit_seconds=300,own_metadata_bytes=512*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,
           expected_released_bytes=saving,physics_steps=0,optimizer_calls=0,Actor_updates=0,skipped_no_gain=skipped)
    OUT.mkdir();write(HERE/'protocol.json',p);write(OUT/'protocol.json',p)
    write(OUT/'preflight.json',dict(state='PASS',files=len(records),all_original_gzip_and_raw_zip_bytes_exact=True,
          prior_remote_every_member_SHA_verified=True,files_modified=0,physics_steps=0,optimizer_calls=0,
          expected_released_bytes=saving,elapsed_s=time.time()-start))
    print(json.dumps(dict(state='PREFLIGHT_PASS',files=len(records),saving=saving)),flush=True)
def run():
    p=read(HERE/'protocol.json');assert not LEDGER.exists() and not (OUT/'status.json').exists()
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for name,h in {**p['source_hashes'],**p['protected_hashes']}.items():assert sha((ROOT/name).read_bytes())==h,name
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();before=used();ref=REF.read_bytes();state='FAILED'
    ledger=dict(state='PENDING',records=p['records'],completed=[],current=None,reference=p['reference'],reference_sha256=p['reference_sha256'],
      recovery='Preserve partial files. For a completed dxz use load_original. For interruption use prior complete remote original member; verify original gzip SHA. Never restart old training; never continue this ended job.')
    write(LEDGER,ledger);write(OUT/'workers.json',dict(parent=os.getpid(),workers=[]))
    try:
        for r in p['records']:
            assert time.time()<start+p['wall_seconds']-20 and used()+65536<=p['global_bytes']
            assert shutil.disk_usage(ROOT).free>=p['min_free_bytes'];verify_prior(r)
            f=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];assert not dest.exists();b=f.read_bytes();assert sha(b)==r['original_sha256']
            z=encode(b,ref);assert sha(z)==r['encoded_sha256'] and decode(z,ref)==b
            ledger.update(state='JOURNALED_BEFORE_RENAME',current=r['original_path']);write(LEDGER,ledger)
            f.rename(dest);fd=os.open(dest.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
            ledger['state']='RENAMED_ORIGINAL_BEFORE_INPLACE_ENCODING';write(LEDGER,ledger)
            with dest.open('r+b') as stream:stream.write(z);stream.truncate();stream.flush();os.fsync(stream.fileno())
            assert decode(dest.read_bytes(),ref)==b
            ledger['completed'].append(r['original_path']);ledger.update(state='ENCODING',current=None);write(LEDGER,ledger)
            write(OUT/'status.json',dict(state='ENCODING',pid=os.getpid(),completed=len(ledger['completed']),updated_epoch=time.time()))
        state='COMPLETE';ledger.update(state='ALL_ORIGINAL_BYTES_PRESERVED',payload_released_bytes=p['expected_released_bytes']);write(LEDGER,ledger)
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),completed=len(ledger['completed']),current=ledger['current']))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    s=dict(state=state,pid=os.getpid(),all_workers_stopped=True,physics_steps=0,native_steps=0,optimizer_calls=0,Actor_updates=0,
           files_encoded=len(ledger['completed']),payload_released_bytes=sum(r['original_bytes']-r['encoded_bytes'] for r in p['records'] if r['original_path'] in ledger['completed']),
           net_runs_released_bytes=before-used(),global_bytes=used(),elapsed_s=time.time()-start,ledger_sha256=sha(LEDGER.read_bytes()))
    write(OUT/'summary.json',s);write(OUT/'status.json',s);print(json.dumps(s),flush=True)
def audit():
    start=time.time();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');f=DOC/'T321_AUDIT.json'
    assert s['state']=='COMPLETE' and not f.exists()
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:assert not Path(f'/proc/{pid}').exists()
    ref=REF.read_bytes()
    for r in p['records']:
        assert time.time()-start<p['audit_seconds'];verify_prior(r);assert not (ROOT/r['original_path']).exists()
        z=(ROOT/r['encoded_path']).read_bytes();assert sha(z)==r['encoded_sha256'];b=decode(z,ref)
        assert sha(b)==r['original_sha256'] and len(b)==r['original_bytes']
    for name,h in {**p['source_hashes'],**p['protected_hashes']}.items():assert sha((ROOT/name).read_bytes())==h,name
    assert used()<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    write(f,dict(state='PASS',files=len(p['records']),original_gzip_and_raw_zip_bytes_exact=True,protected_current_inputs_unchanged=True,
      prior_full_remote_member_SHA_receipts_verified=True,new_encoded_binaries_not_uploaded=True,all_original_processes_absent=True,
      physics_steps=0,optimizer_calls=0,Actor_updates=0,global_bytes=used(),ledger_sha256=sha(LEDGER.read_bytes())))
    print('T321 independent lossless-byte audit PASS',flush=True)
def launch():
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json'])
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as f:
        proc=subprocess.Popen(['/usr/bin/timeout','--kill-after=1s','600s','/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=time.time(),remote_commit=receipt['remote_commit'],single_launch=True));print(proc.pid)
def load_original(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    if p.exists():return p.read_bytes()
    r=next(r for r in read(LEDGER)['records'] if r['original_path']==str(p.relative_to(ROOT)))
    b=decode((ROOT/r['encoded_path']).read_bytes(),REF.read_bytes());assert sha(b)==r['original_sha256'];return b
def restore(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p;assert not p.exists();b=load_original(p)
    assert used()+len(b)+65536<=1024**3 and shutil.disk_usage(ROOT).free-len(b)>=40*1024**3
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert sha(p.read_bytes())==sha(b);print(json.dumps(dict(restored=str(p),bytes=len(b))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','launch','run','audit','restore']);parser.add_argument('path',nargs='?');a=parser.parse_args()
    if a.mode=='restore':restore(a.path)
    else:globals()[a.mode]()
