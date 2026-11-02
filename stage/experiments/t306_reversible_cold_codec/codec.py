"""Cold checkpoint lossless recoding with exact raw AND gzip restoration.

Original raw payloads already have full remotely verified archive receipts.
No models/tensors are removed from the payload. Never touches current inputs.
This module also decodes/restores either original .pt or original .pt.gz bytes.
"""
from pathlib import Path
import argparse,fcntl,gzip,hashlib,io,json,lzma,os,shutil,signal,subprocess,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t306_reversible_cold_codec';DOC=ROOT/'stage/docs/overnight_20261102';LEDGER=DOC/'T306_ENCODING_LEDGER.json'
OLD=ROOT/'stage/docs/results_20261031_archive'
CHOICES=[('overnight_reserve_storage_receipt.json',None,'plain'),
    ('closed_resume_storage_receipt.json',None,'plain'),('reverse_curriculum_storage_receipt.json',None,'plain'),
    ('smdp_storage_receipt.json',None,'plain'),('cold_storage_receipt.json',None,'byteplane4')]


def sha(b):return hashlib.sha256(b).hexdigest()
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())
def write(f,value):
    f=Path(f);tmp=f.with_suffix(f.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def read(f):return json.loads(Path(f).read_text())


def transform(raw,kind):
    if kind=='plain':return raw
    assert kind=='byteplane4'
    padded=raw+b'\0'*((-len(raw))%4)
    return np.frombuffer(padded,np.uint8).reshape(-1,4).T.copy().tobytes()


def untransform(data,kind,size):
    if kind=='plain':assert len(data)==size;return data
    assert kind=='byteplane4' and len(data)%4==0
    raw=np.frombuffer(data,np.uint8).reshape(4,-1).T.copy().tobytes()
    assert raw[size:]==b'\0'*((-size)%4)
    return raw[:size]


def decode_record(record):
    data=(ROOT/record['encoded_path']).read_bytes()
    # A pending ledger may have renamed the original byte-exact gzip before an
    # in-place write starts. That complete original also remains decodable.
    if sha(data)==record['gzip_sha256']:
        raw=gzip.decompress(data)
    else:
        assert sha(data)==record['encoded_sha256']
        raw=untransform(lzma.decompress(data),record['transform'],record['raw_bytes'])
    assert len(raw)==record['raw_bytes'] and sha(raw)==record['raw_sha256']
    compressed=gzip.compress(raw,compresslevel=6,mtime=0)
    assert len(compressed)==record['gzip_bytes'] and sha(compressed)==record['gzip_sha256']
    return raw,compressed


def stopped():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if args and Path(os.fsdecode(args[0])).name.startswith('python'):
            active=[os.fsdecode(a) for a in args[1:] if str(ROOT/'stage/experiments') in os.fsdecode(a)]
            assert not active,('Another Stage process',proc.name,active)


def prepare():
    assert not OUT.exists() and not LEDGER.exists()
    protected=set(read(ROOT/'stage/runs/t299_conditional_goal_likelihood/protocol.json')['input_hashes'])
    for folder in ['t305_goal_conditioned_contract','t305_goal_contract_recovery']:
        f=ROOT/'stage/runs'/folder/'protocol.json'
        if f.exists():protected.update(read(f)['input_hashes'])
    inputs={};records=[]
    for ledgername,_,kind in CHOICES:
        ledgerpath=OLD/ledgername;ledger=read(ledgerpath)
        assert ledger['state']=='LOSSLESS_LOCAL_COMPACTION_COMPLETE'
        assert ledger.get('full_remote_original_archive_verified',False) or ledger.get('full_remote_original_archive_sha_verified',False)
        inputs[str(ledgerpath)]=sha(ledgerpath.read_bytes())
        for old in ledger['records']:
            f=ROOT/old['compressed_path']
            assert str(f) not in protected and f.is_file() and old['remote_original_asset'].startswith('https://github.com/frank8763/winkun/releases/')
            assert not any(name in str(f) for name in ['t194_','t255_','t299_','t288_'])
            b=f.read_bytes();assert sha(b)==old['compressed_sha256'] and len(b)==old['compressed_bytes']
            raw=gzip.decompress(b);assert len(raw)==old['original_bytes'] and sha(raw)==old['original_sha256']
            assert gzip.compress(raw,compresslevel=6,mtime=0)==b,'Byte-exact gzip regeneration required before encoding'
            encoded=lzma.compress(transform(raw,kind),preset=1)
            restored=untransform(lzma.decompress(encoded),kind,len(raw));assert restored==raw
            assert len(encoded)<len(b),'Fixed codec must save space, no codec grid'
            record=dict(original_gzip_path=str(f.relative_to(ROOT)),original_raw_path=old['original_path'],
                gzip_bytes=len(b),gzip_sha256=sha(b),raw_bytes=len(raw),raw_sha256=sha(raw),transform=kind,
                encoded_path=str(f.relative_to(ROOT))+'.lxz',encoded_bytes=len(encoded),encoded_sha256=sha(encoded),
                prior_verified_raw_storage_receipt=str(ledgerpath.relative_to(ROOT)),remote_original_raw_asset=old['remote_original_asset'],
                gzip_regeneration=dict(compresslevel=6,mtime=0),xz_preset=1)
            records.append(record);inputs[str(f)]=sha(b)
            print(json.dumps(dict(candidate=record['original_gzip_path'],saving=len(b)-len(encoded))),flush=True)
    for f in [Path(__file__),DOC/'T306_PLAN.md']:inputs[str(f)]=sha(f.read_bytes())
    for f in sorted(protected):
        if Path(f).is_file():inputs[f]=sha(Path(f).read_bytes())
    expected=sum(r['gzip_bytes']-r['encoded_bytes'] for r in records)
    assert expected>16*1024**2 and used()+512*1024<1024**3
    p=dict(experiment='T306',records=records,input_hashes=inputs,protected_inputs=sorted(protected),wall_seconds=600,
        own_metadata_bytes=512*1024,global_bytes=1024**3,min_free_bytes=40*1024**3,expected_released_bytes=expected,
        native_steps=0,optimizer_calls=0,contract='Fixed coldcheckpoint LOSSLESS recoding only. Raw .pt bytes fully preserved includingActorCriticreplayAdam, existing fullremoteverifiedrawarchives+oldgzipledger prove recoverability. Verify rawSHA and byteexactgzip6mtime0 reconstitution before mutation. Fixed plainXZpreset1 foroldresume/SMDP checkpoints, fixedbyteplane4+XZpreset1 forfourunselectedoldfailedcritics; notmodel/algorithmgrid. CurrentT255/T299/T305inputhashes/protocols/arrays/videos untouched. Every raw.pt and original.gz stream reconstructs exactly fromnewlocalencodedfile andledger; originalrawremotebackup unchanged, no claimnewencodedbinary uploaded. No extra diskcopy exceedingquota: afterRAMencoding/decode+remoteproof, journal pending then renameexistinggziptoencodedpath, write/truncate INPLACE/fsync, verify exactoriginalraw/gzip. A crashduringwrite requires documentedoldverifiedremoterawrecovery; never falselycall partiallocalfileverified. Durableperfileledgerstates andrestorer provided. No activeStagejobs, exclusiveflock, no symlink/relocation bypass, no originalunique-byte deletion. Oldendedconsumers need explicit decode orfullcapacityreservedrestore, neverrestart. Atmost600s, metadata512KiB, global1GiB/free40GiB. Allattempts/errors/perfilespreserved; independentaudit0native/optimizerreconstructsraw+gzip+oldsources/currentinputs. Newmetadataonlybackup mustsay originalmodelraw remainsinpriorverifiedarchives.')
    OUT.mkdir();write(HERE/'protocol.json',p);write(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',files=len(records),expected_released_bytes=expected,native_steps=0)),flush=True)


def preflight():
    p=read(HERE/'protocol.json');assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json','preflight.json'])
    for f,h in p['input_hashes'].items():assert sha(Path(f).read_bytes())==h,f
    assert used()+p['own_metadata_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    write(OUT/'preflight.json',dict(state='PASS',native_steps=0,optimizer_calls=0,files_modified=0,
        all_original_raw_SHA_and_exact_gzip_regeneration_verified=True,expected_released_bytes=p['expected_released_bytes'],
        registered_protocol_sha256=sha((HERE/'protocol.json').read_bytes())))
    print('T306 preflight PASS, no mutation/native/optimizer')


def run():
    p=read(HERE/'protocol.json');assert not (OUT/'status.json').exists() and not LEDGER.exists();assert read(OUT/'preflight.json')['state']=='PASS'
    start=time.time();state='FAILED';lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);stopped()
    for f,h in p['input_hashes'].items():assert sha(Path(f).read_bytes())==h,f
    before=used();ledger=dict(state='ENCODING_PENDING',records=p['records'],completed=[],current=None,
        prior_raw_archives_already_fully_verified=True,original_raw_and_gzip_reconstructable=True,
        restore_in_RAM='codec.py decode supports original.pt andoriginal.pt.gz; restore reserves fulloriginalsize before writing. No endedexperiment restart.',
        crash_recovery='If encodedpath hash is neither gzipSHA nor encodedSHA, keep partial evidence and recover original raw member from prior_verified_raw_storage_receipt/remote_original_raw_asset, verify rawSHA then re-encode. Do not call partial file verified.',global_bytes_before=before)
    write(LEDGER,ledger);write(OUT/'workers.json',dict(parent=os.getpid(),workers=[]));write(OUT/'status.json',dict(state='STARTED',pid=os.getpid(),updated_epoch=time.time(),deadline_epoch=start+p['wall_seconds']))
    try:
        for r in p['records']:
            assert time.time()<start+p['wall_seconds']-10
            src=ROOT/r['original_gzip_path'];dst=ROOT/r['encoded_path'];b=src.read_bytes();assert sha(b)==r['gzip_sha256'] and not dst.exists()
            raw=gzip.decompress(b);assert sha(raw)==r['raw_sha256'];encoded=lzma.compress(transform(raw,r['transform']),preset=1)
            assert sha(encoded)==r['encoded_sha256'] and untransform(lzma.decompress(encoded),r['transform'],r['raw_bytes'])==raw
            assert gzip.compress(raw,compresslevel=6,mtime=0)==b
            ledger.update(current=r['original_gzip_path'],state='PER_FILE_JOURNALED_BEFORE_RENAME');write(LEDGER,ledger)
            src.rename(dst)
            ledger['state']='RENAMED_ORIGINAL_GZIP_BEFORE_REENCODE';write(LEDGER,ledger)
            # Both complete original streams verified in RAM and remote raw
            # archive before this in-place, quota-preserving recoding begins.
            with dst.open('r+b') as f:
                f.seek(0);f.write(encoded);f.truncate(len(encoded));f.flush();os.fsync(f.fileno())
            a,gz=decode_record(r);assert a==raw and gz==b
            ledger['completed'].append(r['original_gzip_path']);ledger.update(current=None,state='PER_FILE_COMPLETE');write(LEDGER,ledger)
            assert used()<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
            write(OUT/'status.json',dict(state='ENCODING',pid=os.getpid(),updated_epoch=time.time(),deadline_epoch=start+p['wall_seconds'],completed=len(ledger['completed']),global_bytes=used()))
            print(json.dumps(dict(encoded=r['original_gzip_path'],saved=r['gzip_bytes']-r['encoded_bytes'])),flush=True)
        original_gz={str(ROOT/r['original_gzip_path']) for r in p['records']}
        for f,h in p['input_hashes'].items():
            if f not in original_gz:assert sha(Path(f).read_bytes())==h,f
        state='COMPLETE';ledger.update(state='ALL_RAW_AND_ORIGINAL_GZIP_BYTES_VERIFIED',global_bytes_after=used(),released_bytes_before_final_metadata=before-used())
        write(LEDGER,ledger)
    except BaseException:write(OUT/'failure.json',dict(traceback=traceback.format_exc(),completed=ledger['completed'],current=ledger['current'],native_steps=0,optimizer_calls=0))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    summary=dict(state=state,pid=os.getpid(),physics_steps=0,native_steps=0,optimizer_calls=0,Actor_updates=0,
        all_workers_stopped=True,files_encoded=len(ledger['completed']),all_models_unique_bytes_preserved=True,
        released_bytes_before_summary=before-used(),global_bytes=used(),elapsed_s=time.time()-start,
        encoding_ledger_sha256=sha(LEDGER.read_bytes()),own_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()))
    write(OUT/'summary.json',summary);write(OUT/'status.json',summary);print(json.dumps(summary),flush=True)


def audit():
    dest=DOC/'T306_AUDIT.json';assert not dest.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json');ledger=read(LEDGER);assert s['state']=='COMPLETE'
    for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live/reused',pid))
    records=[];originals={str(ROOT/r['original_gzip_path']) for r in p['records']}
    for r in p['records']:
        raw,gz=decode_record(r);assert not (ROOT/r['original_gzip_path']).exists()
        old=read(ROOT/r['prior_verified_raw_storage_receipt']);ref=next(x for x in old['records'] if x['compressed_path']==r['original_gzip_path'])
        assert ref['original_sha256']==sha(raw) and ref['compressed_sha256']==sha(gz)
        records.append(dict(path=r['original_gzip_path'],raw_SHA_exact=True,gzip_SHA_exact=True,remote_raw_prior_receipt_matches=True))
    for f,h in p['input_hashes'].items():
        if f not in originals:assert sha(Path(f).read_bytes())==h,f
    assert ledger['state']=='ALL_RAW_AND_ORIGINAL_GZIP_BYTES_VERIFIED' and len(ledger['completed'])==len(records)
    assert used()<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    write(dest,dict(state='PASS',native_steps=0,optimizer_calls=0,all_original_processes_absent=True,
        all_original_raw_and_gzip_byte_streams_exact=True,all_protected_current_inputs_unchanged=True,
        records=records,global_bytes=used(),original_raw_payloads_in_previously_verified_remote_archives=True,
        new_encoded_binary_uploaded=False,ledger_sha256=sha(LEDGER.read_bytes())))
    print('T306 audit PASS: exact raw and gzip recovery, current inputs unchanged')


def launch():
    p=read(HERE/'protocol.json');assert not any((OUT/f).exists() for f in ['launch.json','status.json','summary.json']);assert read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (OUT/'run.log').open('xb') as log:proc=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=1s',str(p['wall_seconds'])+'s',str(ROOT/'run_stage.sh'),str(Path(__file__)),'run'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=proc.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=receipt['remote_commit'],single_launch=True))
    print(json.dumps(dict(pid=proc.pid)),flush=True)


def load_original(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    if path.exists():return path.read_bytes()
    relative=str(path.relative_to(ROOT));ledger=read(LEDGER)
    r=next(x for x in ledger['records'] if relative in [x['original_raw_path'],x['original_gzip_path']])
    raw,gz=decode_record(r);return raw if relative==r['original_raw_path'] else gz


def restore(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    assert not path.exists();data=load_original(path)
    assert used()+len(data)+65536<1024**3 and shutil.disk_usage(ROOT).free-len(data)>=40*1024**3
    with path.open('xb') as f:f.write(data)
    assert sha(path.read_bytes())==sha(data)
    print(json.dumps(dict(restored=str(path),bytes=len(data),sha256=sha(data))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','preflight','launch','run','audit','restore']);parser.add_argument('path',nargs='?');a=parser.parse_args()
    if a.mode=='restore':assert a.path;restore(a.path)
    else:assert not a.path;globals()[a.mode]()
