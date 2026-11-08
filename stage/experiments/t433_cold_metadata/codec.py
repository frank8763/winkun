"""Fixed, remotely verified cold JSON bytes; reversible gzip, zero science.

Old report readers must use load_original() or bounded restore(), because the
old bare JSON path is absent after encoding. No model or trajectory is moved.
"""
from pathlib import Path
import gzip, hashlib, json, os, shutil, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t433_cold_metadata'
DOC=ROOT/'stage/docs/overnight_20261108'
LEDGER=DOC/'T433_ENCODING_LEDGER.json'
FILES=['b/summary.json','b/rounds.json','b/blocks/0/result.json','b/blocks/1/result.json',
       'a/summary.json','a/rounds.json']


def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def used():return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())


def prepare():
    assert not OUT.exists() and not LEDGER.exists()
    receipt_path=ROOT/'stage/docs/long_distance_20261101/release_receipt_T425.json'
    manifest_path=ROOT/'stage/docs/long_distance_20261101/source_manifest_T425.json'
    receipt=read(receipt_path);manifest=read(manifest_path)
    assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert receipt['manifest_sha256']==sha(manifest_path.read_bytes())
    prior=read(ROOT/'stage/runs/t431_actual_recovery/protocol.json')
    rows=[]
    for relative in FILES:
        path=ROOT/'stage/runs/t425_offline_day_chain'/relative
        assert str(path) not in prior['inputs'], 'Do not encode a current scientific input.'
        data=path.read_bytes();json.loads(data);encoded=gzip.compress(data,compresslevel=9,mtime=0)
        assert gzip.decompress(encoded)==data
        original=str(path.relative_to(ROOT))
        entry=next(e for e in manifest['files'] if e['path']==original)
        assert entry['sha256']==sha(data) and entry['bytes']==len(data)
        rows.append(dict(original_path=original,original_sha256=sha(data),original_bytes=len(data),
             encoded_path=original+'.cold.gz',encoded_sha256=sha(encoded),encoded_bytes=len(encoded)))
    released=sum(r['original_bytes']-r['encoded_bytes'] for r in rows)
    assert released>=9*2**20
    assert used()+max(r['encoded_bytes'] for r in rows)+512*1024<2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    OUT.mkdir()
    p=dict(id='T433_FIXED_SIX_COLD_JSON_LOSSLESS_ENCODING',rows=rows,
      source_sha256=sha(Path(__file__).read_bytes()),receipt_path=str(receipt_path),
      receipt_sha256=sha(receipt_path.read_bytes()),manifest_path=str(manifest_path),
      manifest_sha256=sha(manifest_path.read_bytes()),expected_released_bytes=released,
      max_own_bytes=512*1024,max_seconds=120,physics_steps=0,optimizer_calls=0,
      current_protocol_path=str(ROOT/'stage/runs/t431_actual_recovery/protocol.json'),
      current_protocol_sha256=sha((ROOT/'stage/runs/t431_actual_recovery/protocol.json').read_bytes()),
      contract='Six fixed ended archived JSON only; all raw bytes remain locally recoverable and in verified remote originals. No models, trajectories or active scientific inputs changed.')
    write(HERE/'protocol.json',p);write(OUT/'protocol.json',p)
    write(OUT/'preflight.json',dict(state='PASS',files_modified=0,physics_steps=0,
       optimizer_calls=0,expected_released_bytes=released,original_bytes_and_remote_SHA_verified=True))
    print('T433_PREFLIGHT_PASS',released)


def validate(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    assert sha(Path(__file__).read_bytes())==p['source_sha256']
    for prefix in ['receipt','manifest','current_protocol']:
        assert sha(Path(p[prefix+'_path']).read_bytes())==p[prefix+'_sha256']
    assert used()<2**30 and shutil.disk_usage(ROOT).free>=40*2**30


def run():
    p=read(OUT/'protocol.json');validate(p)
    assert not (OUT/'summary.json').exists() and not LEDGER.exists()
    tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    start=time.time();before=used();done=[]
    for row in p['rows']:
        assert time.time()-start<p['max_seconds']
        path=ROOT/row['original_path'];dest=ROOT/row['encoded_path']
        data=path.read_bytes();assert len(data)==row['original_bytes'] and sha(data)==row['original_sha256']
        assert not dest.exists();encoded=gzip.compress(data,compresslevel=9,mtime=0)
        assert sha(encoded)==row['encoded_sha256'] and gzip.decompress(encoded)==data
        assert used()+len(encoded)<2**30
        with dest.open('xb') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        assert gzip.decompress(dest.read_bytes())==data
        path.unlink();done.append(row)
        write(OUT/'progress.json',dict(completed=done,physics_steps=0,optimizer_calls=0))
    write(LEDGER,dict(state='LOCAL_ORIGINAL_BYTES_REVERSIBLY_ENCODED',rows=done,
        released_payload_bytes=p['expected_released_bytes'],raw_models_or_trajectories_changed=0,
        original_remote_readback_receipt=p['receipt_path'],original_remote_complete=True,
        encoded_new_binary_remote_pending=True,
        load='codec.load_original(repo_relative_old_json_path) returns exact original bytes; no old bare path exists.'))
    write(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-start,
        files=len(done),released_payload_bytes=p['expected_released_bytes'],global_before=before,
        global_after=used(),physics_steps=0,optimizer_calls=0,all_workers_stopped=True))
    print('T433_SIX_REPORT_BYTES_ENCODED',p['expected_released_bytes'])


def load_original(relative):
    record=next(r for r in read(LEDGER)['rows'] if r['original_path']==str(relative))
    data=gzip.decompress((ROOT/record['encoded_path']).read_bytes())
    assert len(data)==record['original_bytes'] and sha(data)==record['original_sha256']
    return data


def audit():
    p=read(OUT/'protocol.json');validate(p);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    assert not Path('/proc',str(s['pid'])).exists()
    start=time.time()
    for row in p['rows']:
        assert time.time()-start<120
        assert not (ROOT/row['original_path']).exists()
        assert sha((ROOT/row['encoded_path']).read_bytes())==row['encoded_sha256']
        assert sha(load_original(row['original_path']))==row['original_sha256']
    write(OUT/'audit.json',dict(state='PASS',files=6,original_bytes_SHA_exact=True,
        original_remote_complete_readback_verified=True,physics_steps=0,optimizer_calls=0,
        released_payload_bytes=p['expected_released_bytes'],global_bytes=used()))
    print('T433_BYTE_EXACT_INDEPENDENT_AUDIT_PASS')


def restore(relative):
    data=load_original(relative);path=ROOT/relative
    assert not path.exists() and used()+len(data)<2**30
    path.write_bytes(data);assert path.read_bytes()==data


if __name__=='__main__':
    if sys.argv[1]=='restore':restore(sys.argv[2])
    else:globals()[sys.argv[1]]()
