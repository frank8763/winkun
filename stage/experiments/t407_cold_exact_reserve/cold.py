"""Bounded Stage-only exact NPZ aliases; originals remain local and remote.

prepare freezes the smallest number of largest duplicate files reaching 8MiB,
then backs up every selected canonical and original byte stream in RAM. run is
separate and requires committed, exactly remote-synchronized registered source.
No NumPy, Torch, simulator, optimizer, model transformation or binary disk temp.
"""
from pathlib import Path
import collections
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.parse

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t407_cold_exact_reserve'
DOC = ROOT / 'stage/docs/review_20261107'
T406 = ROOT / 'stage/runs/t406_wide_full_route'
TARGET = 8 * 2**20
META_LIMIT = 256 * 2**10
GLOBAL_LIMIT = 2**30
MAX_SECONDS = 180


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def blob(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def save(path, value):
    path = Path(path)
    temporary = Path(str(path) + '.partial')
    with temporary.open('xb') as stream:
        stream.write(blob(value))
    temporary.replace(path)


def used():
    return sum(f.stat().st_size for f in (ROOT / 'stage/runs').rglob('*') if f.is_file())


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def stopped():
    assert read(T406 / 'status.json')['state'] == 'COMPLETE'
    assert read(T406 / 'summary.json')['state'] == 'COMPLETE'
    assert read(T406 / 'audit.json')['state'] == 'PASS'
    assert read(DOC / 'T406_FINAL_REPORT.json')['state'] == 'COMPLETE_AUDITED_REMOTE_VERIFIED'
    pids = set()
    for name in ['launch.json', 'driver_started.json', 'main_started.json', 'workers.json', 'summary.json']:
        value = read(T406 / name)
        pids.update(value[key] for key in ['pid', 'parent'] if type(value.get(key)) is int)
        pids.update(value.get('workers', []))
    for pid in pids:
        assert not Path('/proc', str(pid)).exists(), ('Live or reused T406 PID', pid)
    return sorted(pids)


def inputs():
    current = read(T406 / 'protocol.json')
    source_path = absolute(current['source_protocol'])
    source = read(source_path)
    maps = [('T406_inputs', current['inputs']), ('source_protocol_input_hashes', source['input_hashes'])]
    excluded = set()
    proof = {}
    for label, mapping in maps:
        for path, digest in mapping.items():
            file = absolute(path)
            assert file.is_file() and sha(file.read_bytes()) == digest, ('Protected input changed', str(file))
            excluded.add(file.resolve())
        proof[label] = dict(files=len(mapping), all_current_bytes_match=True,
                            mapping_sha256=sha(json.dumps(mapping, sort_keys=True).encode()))
    proof.update(excluded_unique_paths=len(excluded),
                 T406_protocol_sha256=sha((T406 / 'protocol.json').read_bytes()),
                 source_protocol=str(source_path.relative_to(ROOT)),
                 source_protocol_sha256=sha(source_path.read_bytes()))
    return excluded, proof


def eligible(path, excluded):
    relative = path.relative_to(ROOT / 'stage/runs')
    match = re.fullmatch(r't(\d{3})_.+', relative.parts[0])
    return bool(match and 166 <= int(match.group(1)) <= 219
                and path.suffix == '.npz' and path.is_file() and not path.is_symlink()
                and not any(parent.is_symlink() for parent in path.parents if parent != ROOT)
                and path.stat().st_size >= 20000 and path.resolve() not in excluded
                and not any(word in str(relative).lower() for word in ['model', 'demo', 'video']))


def discover(excluded):
    groups = collections.defaultdict(list)
    scanned = 0
    for directory in sorted((ROOT / 'stage/runs').iterdir()):
        match = re.fullmatch(r't(\d{3})_.+', directory.name)
        if not match or not 166 <= int(match.group(1)) <= 219 or directory.is_symlink():
            continue
        for file in sorted(directory.rglob('*.npz')):
            if eligible(file, excluded):
                data = file.read_bytes()
                groups[(len(data), sha(data))].append(str(file.relative_to(ROOT)))
                scanned += 1
    pairs = []
    for (size, digest), files in groups.items():
        files.sort()
        for duplicate in files[1:]:
            pairs.append(dict(original=duplicate, canonical=files[0], bytes=size,
                              sha256=digest, encoding='exact-byte-alias-v1'))
    pairs.sort(key=lambda row: (-row['bytes'], row['original']))
    selected, total = [], 0
    for row in pairs:
        if total >= TARGET:
            break
        selected.append(row)
        total += row['bytes']
    assert total >= TARGET, ('Insufficient fixed duplicate bytes; do not expand scope', total)
    assert total - selected[-1]['bytes'] < TARGET
    return selected, dict(scanned_eligible_files=scanned, exact_duplicate_candidates=len(pairs),
                          available_duplicate_bytes=sum(row['bytes'] for row in pairs),
                          selection='descending payload size, then original path; minimum file count reaching 8MiB')


def verify_rows(p, encoded=False):
    excluded, proof = inputs()
    assert proof == p['protected_input_hash_proof']
    originals = {row['original'] for row in p['records']}
    manifest = {}
    for row in p['records']:
        original, canonical = ROOT / row['original'], ROOT / row['canonical']
        assert row['canonical'] < row['original'] and row['canonical'] not in originals
        assert eligible(canonical, excluded)
        data = canonical.read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
        if encoded:
            assert not original.exists()
            assert read(str(original) + '.ref.json') == {key: row[key] for key in ['encoding', 'canonical', 'bytes', 'sha256']}
        else:
            assert eligible(original, excluded) and original.read_bytes() == data
            assert not Path(str(original) + '.ref.json').exists()
        for name in [row['canonical'], row['original']]:
            manifest[name] = dict(path=name, bytes=row['bytes'], sha256=row['sha256'])
    return sorted(manifest.values(), key=lambda row: row['path'])


def transport():
    sys.path.insert(0, str(ROOT / 'stage/experiments/t377_full_human_observation_transfer'))
    import transport as transfer
    transfer.install()
    return transfer


def backup(manifest):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for row in manifest:
            data = (ROOT / row['path']).read_bytes()
            assert len(data) == row['bytes'] and sha(data) == row['sha256']
            info = tarfile.TarInfo(row['path'])
            info.size, info.mtime = len(data), 0
            archive.addfile(info, io.BytesIO(data))
    data = gzip.compress(buffer.getvalue(), compresslevel=3, mtime=0)
    transfer = transport()
    transfer.B.verify_tar(data, manifest)
    release = transfer.B.api('GET', 'releases/tags/stage-distance-2026-11-01')
    name = 'stage-t407-originals-' + sha(data)[:16] + '.tar.gz'
    found = []
    for page in range(1, 21):
        assets = transfer.B.api('GET', f"releases/{release['id']}/assets?per_page=100&page={page}")
        found.extend(asset for asset in assets if asset['name'] == name)
        if len(assets) < 100:
            break
    else:
        raise AssertionError('Bounded asset inventory exceeded; no upload')
    assert len(found) <= 1
    if found:
        asset = found[0]
    else:
        asset = json.loads(transfer.req('POST',
            f"https://uploads.github.com/repos/{transfer.A.SLUG}/releases/{release['id']}/assets?name=" + urllib.parse.quote(name),
            data, {'Content-Type': 'application/octet-stream'}))
    asset = transfer.B.api('GET', f"releases/assets/{asset['id']}")
    assert asset['name'] == name and asset['size'] == len(data) and asset['digest'] == 'sha256:' + sha(data)
    returned = transfer.readback(asset, len(data))
    assert returned == data
    transfer.B.verify_tar(returned, manifest)
    return dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED', asset_id=asset['id'],
                asset_name=name, asset_url=asset['browser_download_url'], bytes=len(data), sha256=sha(data),
                members=len(manifest), original_member_bytes=sum(row['bytes'] for row in manifest),
                manifest_sha256=sha(blob(dict(files=manifest))), temporary_disk_binary_bytes=0,
                exact_asset_name_checked_before_upload=True)


def metadata_bytes(p):
    files = set(HERE.rglob('*')) | set(OUT.rglob('*')) | set(DOC.glob('T407*'))
    files.update(Path(str(ROOT / row['original']) + '.ref.json') for row in p['records'])
    return sum(file.stat().st_size for file in files if file.is_file())


def prepare():
    absent = stopped()  # Verify all prior native/driver processes before any write.
    excluded, proof = inputs()
    if (HERE / 'protocol.json').exists():
        p = read(HERE / 'protocol.json')
        assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
        assert p['source_sha256'] == sha(Path(__file__).read_bytes())
        assert not (OUT / 'status.json').exists()
    else:
        assert not OUT.exists()
        records, discovery = discover(excluded)
        p = dict(id='T407_FIXED_EXACT_NPZ_ALIASES', source_sha256=sha(Path(__file__).read_bytes()),
                 records=records, discovery=discovery, protected_input_hash_proof=proof,
                 stopped_T406_pids=absent, expected_duplicate_bytes=sum(row['bytes'] for row in records),
                 target_duplicate_bytes=TARGET, max_seconds=MAX_SECONDS, metadata_bytes=META_LIMIT,
                 global_bytes=GLOBAL_LIMIT, minimum_free_bytes=40 * 2**30,
                 native=0, simulator_imports=0, Torch=0, Adam=0, unique_bytes_removed=0,
                 scope='Real NPZ files >=20000 bytes in Stage T166–T219 only; current T406 and source-protocol inputs excluded; no models/demos/videos/T255+',
                 resolver='stage/experiments/t290_exact_duplicate_store/store.py resolve or guarded restore',
                 contract='Frozen exact records never expand. Keep lexicographically smallest canonical file per SHA and all unique bytes. Full original canonical+duplicate RAM archive remote readback/member SHA before unlink. No hardlinks/symlinks or binary disk temp. Encode only after source commit and exact remote root-tree synchronization.')
        assert used() + META_LIMIT < GLOBAL_LIMIT
        assert shutil.disk_usage(ROOT).free >= p['minimum_free_bytes']
        OUT.mkdir()
        save(HERE / 'protocol.json', p)
        save(OUT / 'protocol.json', p)
    manifest = verify_rows(p)
    save(DOC / 'T407_ORIGINAL_MANIFEST.json', dict(files=manifest))
    if (DOC / 'T407_ORIGINAL_RECEIPT.json').exists():
        receipt = read(DOC / 'T407_ORIGINAL_RECEIPT.json')
        assert receipt['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        assert receipt['manifest_sha256'] == sha(blob(dict(files=manifest)))
    else:
        save(DOC / 'T407_ORIGINAL_RECEIPT.json', backup(manifest))
    verify_rows(p)
    refs = sum(len(blob({key: row[key] for key in ['encoding', 'canonical', 'bytes', 'sha256']})) for row in p['records'])
    preflight = dict(state='PASS_READY_FOR_SOURCE_COMMIT_SYNC_ONLY', native=0, Adam=0, original_mutations=0,
                     files=len(p['records']), canonical_plus_original_members=len(manifest),
                     duplicate_payload_bytes=p['expected_duplicate_bytes'], reference_bytes=refs,
                     payload_minus_reference_bytes=p['expected_duplicate_bytes'] - refs,
                     global_bytes_now=used(), maximum_global_after_encoding_plus_next15MiB=used() - p['expected_duplicate_bytes'] + META_LIMIT + 15 * 2**20,
                     metadata_bytes_now=metadata_bytes(p), metadata_budget=META_LIMIT,
                     protected_input_hash_proof=proof, T406_pids_absent=absent,
                     original_remote_receipt_sha256=sha((DOC / 'T407_ORIGINAL_RECEIPT.json').read_bytes()),
                     source_sha256=p['source_sha256'], frozen_protocol_sha256=sha((HERE / 'protocol.json').read_bytes()))
    assert preflight['maximum_global_after_encoding_plus_next15MiB'] < GLOBAL_LIMIT
    save(OUT / 'preflight.json', preflight)
    save(DOC / 'T407_PREFLIGHT.json', preflight)
    assert metadata_bytes(p) < META_LIMIT
    print(json.dumps(preflight), flush=True)


def registered_source(p):
    assert sha(Path(__file__).read_bytes()) == p['source_sha256']
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for file in [Path(__file__), HERE / 'protocol.json']:
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(file.relative_to(ROOT))], cwd=ROOT)
        assert committed == file.read_bytes(), ('Registered source not committed', str(file))
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    receipt = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['state'] == 'REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree'] == tree
    return dict(root_tree=tree, remote_commit=receipt['remote_commit'])


def run():
    p = read(OUT / 'protocol.json')
    assert not (OUT / 'status.json').exists() and not (OUT / 'summary.json').exists()
    assert read(OUT / 'preflight.json')['state'] == 'PASS_READY_FOR_SOURCE_COMMIT_SYNC_ONLY'
    receipt = read(DOC / 'T407_ORIGINAL_RECEIPT.json')
    assert receipt['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    source = registered_source(p)
    absent = stopped()
    lock = (ROOT / 'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify_rows(p)
    assert receipt['manifest_sha256'] == sha(blob(dict(files=manifest)))
    assert sha((DOC / 'T407_ORIGINAL_RECEIPT.json').read_bytes()) == read(OUT / 'preflight.json')['original_remote_receipt_sha256']
    assert used() + META_LIMIT < GLOBAL_LIMIT and shutil.disk_usage(ROOT).free >= p['minimum_free_bytes']
    start, before, done = time.time(), used(), []
    save(OUT / 'status.json', dict(state='ENCODING', pid=os.getpid(), epoch=start, **source))
    sys.path.insert(0, str(ROOT / 'stage/experiments/t290_exact_duplicate_store'))
    import store
    for row in p['records']:
        assert time.time() - start < p['max_seconds']
        original, canonical = ROOT / row['original'], ROOT / row['canonical']
        data = original.read_bytes()
        assert data == canonical.read_bytes() and len(data) == row['bytes'] and sha(data) == row['sha256']
        reference = Path(str(original) + '.ref.json')
        assert not reference.exists()
        save(OUT / 'ledger.json', dict(state='ENCODING', completed=done, current=row['original'],
                                      canonical=row['canonical'], original_bytes=row['bytes'], sha256=row['sha256']))
        save(reference, {key: row[key] for key in ['encoding', 'canonical', 'bytes', 'sha256']})
        assert read(reference)['sha256'] == sha(data)
        original.unlink()
        assert store.resolve(original) == canonical and store.resolve(original).read_bytes() == data
        done.append(row['original'])
        assert used() < GLOBAL_LIMIT and metadata_bytes(p) < META_LIMIT
    verify_rows(p, encoded=True)
    save(OUT / 'ledger.json', dict(state='ALL_ORIGINAL_BYTES_REMAIN_LOCAL_AND_REMOTE', completed=done, current=None,
                                  protocol_sha256=sha((OUT / 'protocol.json').read_bytes())))
    save(OUT / 'summary.json', dict(state='COMPLETE', pid=os.getpid(), native=0, Adam=0, unique_bytes_removed=0,
        all_workers_stopped=True, stopped_T406_pids=absent, files=len(done),
        duplicate_payload_released=p['expected_duplicate_bytes'], before=before, after_before_summary=used(),
        elapsed_s=time.time() - start, **source))
    save(OUT / 'status.json', dict(state='COMPLETE', pid=os.getpid(), native=0, Adam=0, files=len(done)))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print('T407_COMPLETE', p['expected_duplicate_bytes'], flush=True)


def audit():
    p, summary = read(OUT / 'protocol.json'), read(OUT / 'summary.json')
    assert summary['state'] == 'COMPLETE' and not Path('/proc', str(summary['pid'])).exists()
    stopped()
    verify_rows(p, encoded=True)
    sys.path.insert(0, str(ROOT / 'stage/experiments/t290_exact_duplicate_store'))
    import store
    for row in p['records']:
        assert store.resolve(ROOT / row['original']) == ROOT / row['canonical']
        assert sha(store.resolve(ROOT / row['original']).read_bytes()) == row['sha256']
    assert used() < GLOBAL_LIMIT and shutil.disk_usage(ROOT).free >= p['minimum_free_bytes']
    result = dict(state='PASS', files=len(p['records']), exact_T290_resolver=True,
                  all_original_bytes_local_canonical_and_remote=True, protected_input_hash_proof=inputs()[1],
                  native=0, Adam=0, unique_bytes_removed=0, global_bytes=used(),
                  metadata_bytes=metadata_bytes(p), metadata_budget=META_LIMIT,
                  restore_guard='T290 restore requires used()+payload+65536<1GiB and free-payload>=40GiB; do not bypass',
                  protocol_sha256=sha((OUT / 'protocol.json').read_bytes()))
    save(OUT / 'audit.json', result)
    save(DOC / 'T407_AUDIT.json', result)
    assert metadata_bytes(p) < META_LIMIT
    print('T407_INDEPENDENT_BYTE_AUDIT_PASS', flush=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) == 2 else ''
    assert command in ['prepare', 'run', 'audit']
    try:
        globals()[command]()
    except BaseException as error:
        if OUT.exists():
            index = len(list(DOC.glob('T407_ERROR_*.json'))) + 1
            save(DOC / f'T407_ERROR_{index:03d}.json', dict(command=command, error=repr(error),
                traceback=traceback.format_exc(), native=0, Adam=0,
                original_mutations=0 if command == 'prepare' else 'see durable ledger',
                protocol_preserved=True, no_automatic_expansion=True))
        raise
