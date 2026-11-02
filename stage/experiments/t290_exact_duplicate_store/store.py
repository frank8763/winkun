"""Lossless byte-identical Stage NPZ references; never use links for quota."""
from pathlib import Path
import collections, fcntl, hashlib, json, os, shutil, sys, time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t290_exact_duplicate_store'
DOC = ROOT / 'stage/docs/distance_20261102'
GROUPS = {'T280': 't280_local_cost', 'T283': 't283_capped_task_balance',
          'T287': 't287_time_aligned_credit'}


def digest(data): return hashlib.sha256(data).hexdigest()
def used(): return sum(f.stat().st_size for f in (ROOT / 'stage/runs').rglob('*') if f.is_file())
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def resolve(path):
    path = Path(path)
    if not path.is_absolute(): path = ROOT / path
    if path.exists(): return path
    ref = json.loads(Path(str(path) + '.ref.json').read_text())
    target = ROOT / ref['canonical']
    assert target.is_file() and not Path(str(target) + '.ref.json').exists()
    assert target.stat().st_size == ref['bytes'] and digest(target.read_bytes()) == ref['sha256']
    return target


def prepare():
    assert not OUT.exists() and not (HERE / 'protocol.json').exists()
    groups = collections.defaultdict(list); inputs = {}; order = list(GROUPS.values())
    for group, name in GROUPS.items():
        receipt = ROOT / f'stage/docs/long_distance_20261101/release_receipt_{group}.json'
        manifest = ROOT / f'stage/docs/long_distance_20261101/source_manifest_{group}.json'
        r = json.loads(receipt.read_text()); m = json.loads(manifest.read_text())
        assert r['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        assert digest(manifest.read_bytes()) == r['manifest_sha256']
        inputs[str(receipt)] = digest(receipt.read_bytes()); inputs[str(manifest)] = digest(manifest.read_bytes())
        index = {x['path']: x for x in m['files']}
        for f in sorted((ROOT / 'stage/runs' / name / 'episodes').glob('*.npz')):
            relative = str(f.relative_to(ROOT)); b = f.read_bytes(); h = digest(b)
            assert index[relative]['sha256'] == h and index[relative]['bytes'] == len(b)
            groups[(len(b), h)].append(relative)
    records = []
    for (size, h), paths in sorted(groups.items()):
        paths.sort(key=lambda s: (order.index(Path(s).parts[2]), s))
        for duplicate in paths[1:]:
            ref = dict(encoding='exact-byte-alias-v1', canonical=paths[0], bytes=size, sha256=h)
            records.append(dict(original=duplicate, **ref))
    assert len(records) == 523 and sum(r['bytes'] for r in records) == 12042742
    for f in HERE.glob('*.py'): inputs[str(f)] = digest(f.read_bytes())
    p = dict(experiment='T290', scope=GROUPS, records=records, input_hashes=inputs,
             max_seconds=300, metadata_bytes=1024**2, global_bytes=1024**3,
             native_steps=0, optimizer_calls=0,
             contract='Only closed fully remote-readback verified T280/T283/T287 episode NPZ exact byte duplicates. Keep one physical canonical NPZ per SHA, replace additional copies with explicit tiny ref.json. Original bytes remain local and remote, no unique evidence/model/video/snapshot removal. No symlink/hardlink quota bypass. Restore tool checks complete global capacity before exact original path restoration. No native/optimizer. Stage only, Phase isolated.')
    assert used() + p['metadata_bytes'] < p['global_bytes'] and shutil.disk_usage(ROOT).free >= 40*1024**3
    OUT.mkdir(); write(HERE / 'protocol.json', p); write(OUT / 'protocol.json', p)
    print(json.dumps(dict(state='REGISTERED',duplicate_files=len(records),duplicate_bytes=sum(r['bytes'] for r in records),native_steps=0)))


def main():
    assert not (OUT / 'status.json').exists()
    p = json.loads((HERE / 'protocol.json').read_text()); start=time.time()
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items(): assert digest(Path(f).read_bytes()) == h
    lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in GROUPS.values():
        d=ROOT/'stage/runs'/name; assert json.loads((d/'summary.json').read_text())['all_workers_stopped']
        w=json.loads((d/'workers.json').read_text())
        for pid in [w['parent'],*w.get('workers',[])]:
            try: os.kill(pid,0)
            except ProcessLookupError: pass
            else: raise AssertionError(('Process live or reused',pid))
    before=used(); write(OUT/'status.json',dict(state='STARTED',pid=os.getpid(),epoch=start))
    removed={r['original'] for r in p['records']}; encoded=[]
    for r in p['records']:
        assert time.time()-start < p['max_seconds']
        assert r['canonical'] not in removed
        original=ROOT/r['original']; canonical=ROOT/r['canonical']; b=original.read_bytes()
        assert len(b)==r['bytes'] and digest(b)==r['sha256'] and canonical.read_bytes()==b
        ref=Path(str(original)+'.ref.json');assert not ref.exists()
        write(ref,{k:r[k] for k in ['encoding','canonical','bytes','sha256']})
        assert resolve(canonical).read_bytes()==b
        original.unlink(); assert resolve(original).read_bytes()==b
        encoded.append(r['original'])
        assert used() < p['global_bytes'] and shutil.disk_usage(ROOT).free >= 40*1024**3
    result=dict(state='COMPLETE',physics_steps=0,optimizer_calls=0,all_workers_stopped=True,
                files_encoded=len(encoded),unique_bytes_deleted=0,original_bytes_remain_local_and_remote=True,
                logical_bytes_before=before,logical_bytes_after=used(),net_released_before_receipt=before-used(),
                elapsed_s=time.time()-start,protocol_sha256=digest((HERE/'protocol.json').read_bytes()))
    write(OUT/'summary.json',result);write(OUT/'status.json',result)
    write(DOC/'T290_ENCODING_RECEIPT.json',dict(**result,records=p['records']))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print(json.dumps(result))


def audit():
    p=json.loads((OUT/'protocol.json').read_text());s=json.loads((OUT/'summary.json').read_text());assert s['state']=='COMPLETE'
    for r in p['records']:
        f=ROOT/r['original'];assert not f.exists();assert digest(resolve(f).read_bytes())==r['sha256']
    assert used()<p['global_bytes'] and shutil.disk_usage(ROOT).free>=40*1024**3
    write(DOC/'T290_AUDIT.json',dict(state='PASS',all_523_original_byte_streams_locally_recoverable=True,
          all_remote_original_manifests_verified=True,native_steps=0,optimizer_calls=0,
          logical_bytes=used(),free_bytes=shutil.disk_usage(ROOT).free))
    print('T290 audit PASS: all original bytes recoverable, zero native/optimizer')


def restore(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    assert not path.exists();b=resolve(path).read_bytes();assert used()+len(b)+65536<1024**3
    assert shutil.disk_usage(ROOT).free-len(b)>=40*1024**3
    with path.open('xb') as f:f.write(b)
    assert digest(path.read_bytes())==digest(b)
    print(json.dumps(dict(restored=str(path),bytes=len(b),sha256=digest(b))))


if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['run']:main()
    elif sys.argv[1:]==['audit']:audit()
    elif len(sys.argv)==3 and sys.argv[1]=='restore':restore(sys.argv[2])
    else:raise SystemExit('prepare | run | audit | restore ORIGINAL_PATH')
