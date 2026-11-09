"""Reversible storage of fixed, archived Stage JSON bytes; never train or simulate.

Old readers must call ``load_original(repo_relative_path)`` or explicitly use
``restore`` under the capacity guard.  This does not change an old scientific
protocol's bytes: those exact bytes remain in a local gzip and verified Release.
Preparation writes only RESERVE_CANDIDATES.json. Encoding is a separate command
which refuses to run before the entire T432 driver and native trees have exited.
"""
from pathlib import Path
import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t435_relay_reserve'
DOC = ROOT / 'stage/docs/overnight_20261108_relay'
CANDIDATES = DOC / 'RESERVE_CANDIDATES.json'
CURRENT = ROOT / 'stage/runs/t432_offline_night/protocol.json'
GLOBAL_BYTES = 2 ** 30
OWN_BYTES = 2 ** 20
FREE_BYTES = 40 * 2 ** 30
TARGET_BYTES = 28 * 2 ** 20
MAX_SECONDS = 180
MAX_FILES = 300
FORBIDDEN = ('case', 'replay', 'optimizer', 'failure', 'error', 'audit',
             'manifest', 'ledger')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def used(folder=ROOT / 'stage/runs'):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


def safe_path(relative):
    path = ROOT / relative
    assert path.is_relative_to(ROOT / 'stage/runs')
    assert path.resolve() == path.absolute(), ('No symlink or traversal', relative)
    return path


def fsync_folder(folder):
    fd = os.open(str(folder), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not exclusive or not path.exists(), ('Refuse overwrite', str(path))
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    assert len(data) < OWN_BYTES
    temporary = path.with_name(path.name + '.tmp')
    assert not temporary.exists(), ('Preserve previous partial metadata', str(temporary))
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_folder(path.parent)


def capacity(extra_bytes=0):
    assert used() + int(extra_bytes) <= GLOBAL_BYTES, 'Global newruns exceeds 1GiB'
    assert not OUT.exists() or used(OUT) <= OWN_BYTES, 'T435 metadata exceeds 1MiB'
    assert shutil.disk_usage(ROOT).free >= FREE_BYTES, 'Free disk is below 40GiB'


def references(protocol_paths):
    values = []
    for path in protocol_paths:
        path = Path(path)
        assert path.is_file(), ('Missing protected protocol', str(path))
        value = read(path)
        # Protect every raw reference, including nested source protocols and the
        # unused-fresh ledger.  Checking only the immediate inputs map is weaker.
        values.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
        for ledger in value.get('fresh_ledger', []):
            values.append(str(ledger))
    return '\n'.join(values)


def remote_members(receipt_path):
    receipt_path = Path(receipt_path)
    receipt = read(receipt_path)
    assert receipt['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest_path = receipt_path.with_name('source_manifest_' + receipt['group'] + '.json')
    raw = manifest_path.read_bytes()
    assert digest(raw) == receipt['manifest_sha256']
    manifest = json.loads(raw)
    assert manifest.get('active_runs_included') is False
    return receipt, manifest_path, {r['path']: r for r in manifest['files']}


def verify_remote(row):
    receipt_path = ROOT / row['receipt_path']
    manifest_path = ROOT / row['manifest_path']
    assert digest(receipt_path.read_bytes()) == row['receipt_sha256']
    assert digest(manifest_path.read_bytes()) == row['manifest_sha256']
    receipt, actual_manifest, members = remote_members(receipt_path)
    assert actual_manifest == manifest_path
    member = members[row['original_path']]
    assert member['bytes'] == row['original_bytes']
    assert member['sha256'] == row['original_sha256']
    return receipt


def prepare(additional_protocols=()):
    previous = None
    if CANDIDATES.exists():
        previous = CANDIDATES.read_bytes()
        old = json.loads(previous)
        assert old['state'] == 'CAPACITY_SHORTFALL_NOT_READY' or (
            old['state'] == 'PREPARED' and old['code_sha256'] != digest(Path(__file__).read_bytes())), (
            'Refuse to replace a prepared unchanged candidate list')
    assert not OUT.exists(), 'Refuse duplicate storage run'
    protected_paths = [CURRENT, *map(Path, additional_protocols)]
    protected = references(protected_paths)
    found = {}
    receipt_dir = ROOT / 'stage/docs/long_distance_20261101'
    for receipt_path in sorted(receipt_dir.glob('release_receipt_*.json')):
        try:
            receipt, manifest_path, members = remote_members(receipt_path)
        except (AssertionError, KeyError, FileNotFoundError, ValueError):
            continue
        for relative, member in members.items():
            match = re.search(r'^stage/runs/t(\d+)_', relative)
            if not match:
                continue
            number = int(match.group(1))
            if not (200 <= number <= 410 or 418 <= number <= 426):
                continue
            path = safe_path(relative)
            if path.suffix != '.json' or not path.is_file() or path.stat().st_size < 20_000:
                continue
            if any(word in path.name.lower() for word in FORBIDDEN):
                continue
            if any(word in relative.lower() for word in ('failed_', 'failure', 'error')):
                continue
            # Protocols are eligible only in the explicitly allowed old set;
            # their scientific raw bytes remain exactly recoverable.
            if relative in protected or str(path) in protected:
                continue
            data = path.read_bytes()
            if len(data) != member['bytes'] or digest(data) != member['sha256']:
                continue
            json.loads(data)
            encoded = gzip.compress(data, compresslevel=9, mtime=0)
            assert gzip.decompress(encoded) == data
            encoded_path = relative + '.cold.gz'
            if safe_path(encoded_path).exists():
                continue
            found[relative] = dict(
                original_path=relative, original_bytes=len(data), original_sha256=digest(data),
                encoded_path=encoded_path, encoded_bytes=len(encoded), encoded_sha256=digest(encoded),
                released_bytes=len(data) - len(encoded), group=receipt['group'],
                receipt_path=str(receipt_path.relative_to(ROOT)),
                receipt_sha256=digest(receipt_path.read_bytes()),
                manifest_path=str(manifest_path.relative_to(ROOT)),
                manifest_sha256=digest(manifest_path.read_bytes()))
    rows = []
    for row in sorted(found.values(), key=lambda r: (-r['released_bytes'], r['original_path'])):
        rows.append(row)
        if sum(r['released_bytes'] for r in rows) >= TARGET_BYTES or len(rows) >= MAX_FILES:
            break
    released = sum(r['released_bytes'] for r in rows)
    value = dict(
        id='T435_FIXED_ARCHIVED_JSON_BYTES_COLD_RESERVE',
        state='PREPARED' if released >= TARGET_BYTES else 'CAPACITY_SHORTFALL_NOT_READY',
        rows=rows, expected_payload_released_bytes=released, target_bytes=TARGET_BYTES,
        max_files=MAX_FILES, max_own_metadata_bytes=OWN_BYTES, max_seconds=MAX_SECONDS,
        code_sha256=digest(Path(__file__).read_bytes()),
        protected_protocols=[dict(path=str(p), sha256=digest(p.read_bytes())) for p in protected_paths],
        prepared_epoch=time.time(), raw_original_files_modified=0,
        physics_steps=0, optimizer_calls=0,
        original_remote_readback_every_member_verified=True,
        old_protocol_storage_only_original_scientific_bytes_unchanged=True,
        old_bare_reader_contract='Use codec.load_original or guarded restore; no symlink indirection.',
        next_protocol_must_be_checked_before_encoding=True,
        run_requires_T432_driver_and_every_native_tree_absent=True)
    if previous is not None:
        value['previous_capacity_shortfall_preparation'] = dict(
            original_sha256=digest(previous), original_bytes=len(previous),
            exact_original_json_gzip_base64=base64.b64encode(
                gzip.compress(previous, compresslevel=9, mtime=0)).decode(),
            original_files_modified=0, physics_steps=0, optimizer_calls=0,
            authorization='Pre-execution eligibility/filter refinement; every prior plan byte retained; no scientific run.')
    write_json(CANDIDATES, value, exclusive=previous is None)
    print(json.dumps(dict(state=value['state'], files=len(rows), released_bytes=released,
                          original_files_modified=0, physics_steps=0, optimizer_calls=0)))


def t432_stopped():
    folder = ROOT / 'stage/runs/t432_offline_night'
    seal = read(folder / 'SEALED.json')
    assert seal['all_scientific_processes_stopped']
    report = read(ROOT / 'stage/docs/overnight_20261108/FINAL_REPORT.json')
    assert report['all_scientific_processes_stopped']
    launch = read(folder / 'launch.json')
    driver = read(folder / 'driver_started.json')
    ids = {launch['pid'], driver['pid'], driver['parent']}
    for row in seal['native_trees']:
        ids.update(row['pids'])
    assert all(not Path('/proc', str(pid)).exists() for pid in ids), 'T432 finalizer or native process remains'
    for stage in ['data', 'b', 'c', 'gate', 'eight']:
        summary = folder / stage / 'summary.json'
        if summary.exists() and read(summary)['state'] == 'COMPLETE':
            assert read(folder / stage / 'audit.json')['state'] == 'PASS'
    return dict(pids=sorted(ids), all_absent=True,
                seal_sha256=digest((folder / 'SEALED.json').read_bytes()))


def verify_candidate(candidate, next_protocols):
    assert candidate['state'] == 'PREPARED'
    assert candidate['code_sha256'] == digest(Path(__file__).read_bytes())
    assert len(candidate['rows']) <= MAX_FILES
    assert candidate['expected_payload_released_bytes'] >= TARGET_BYTES
    protocols = [Path(x['path']) for x in candidate['protected_protocols']]
    for row in candidate['protected_protocols']:
        assert digest(Path(row['path']).read_bytes()) == row['sha256']
    assert next_protocols, 'Supply the actual new relay protocol before encoding'
    protected = references([*protocols, *map(Path, next_protocols)])
    for row in candidate['rows']:
        path = safe_path(row['original_path'])
        safe_path(row['encoded_path'])
        assert row['original_path'] not in protected and str(path) not in protected, (
            'A new scientific protocol consumes this original bare path', row['original_path'])
        verify_remote(row)
    capacity()


def run(next_protocols):
    candidate = read(CANDIDATES)
    verify_candidate(candidate, next_protocols)
    predecessor = t432_stopped()
    source_receipt = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    current_tree = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    assert source_receipt['state'] == 'REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED'
    assert source_receipt['root_tree'] == current_tree, 'Back up exact registered source tree before encoding'
    assert not OUT.exists(), 'Do not restart a completed or partial cold encoding run'
    OUT.mkdir()
    started = time.time()
    before = used()
    done = []
    state = 'FAILED'
    error = None
    write_json(OUT / 'protocol.json', candidate, exclusive=True)
    write_json(OUT / 'main_started.json', dict(pid=os.getpid(), epoch=started,
               predecessor=predecessor, new_protocols=[dict(path=str(Path(p)), sha256=digest(Path(p).read_bytes()))
                   for p in next_protocols]), exclusive=True)
    try:
        for row in candidate['rows']:
            assert time.time() - started < MAX_SECONDS, 'Bounded storage encoding deadline'
            path = safe_path(row['original_path'])
            destination = safe_path(row['encoded_path'])
            verify_remote(row)
            raw = path.read_bytes()
            assert len(raw) == row['original_bytes'] and digest(raw) == row['original_sha256']
            assert not destination.exists(), 'Preserve preexisting encoded bytes for inspection'
            encoded = gzip.compress(raw, compresslevel=9, mtime=0)
            assert len(encoded) == row['encoded_bytes'] and digest(encoded) == row['encoded_sha256']
            assert gzip.decompress(encoded) == raw
            capacity(len(encoded) + 128 * 1024)
            with destination.open('xb') as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            fsync_folder(destination.parent)
            # Independent disk readback is complete before removing any bare
            # original.  Exceptions leave the original and/or verified gzip.
            disk = destination.read_bytes()
            assert digest(disk) == row['encoded_sha256']
            decoded = gzip.decompress(disk)
            assert len(decoded) == row['original_bytes']
            assert digest(decoded) == row['original_sha256'] and decoded == raw
            assert path.read_bytes() == raw
            path.unlink()
            fsync_folder(path.parent)
            done.append(row)
            write_json(OUT / 'encoding_ledger.json', dict(
                state='ENCODING', completed_rows=done, candidate_sha256=digest(CANDIDATES.read_bytes()),
                original_bytes_locally_recoverable=True, original_remote_all_members_SHA_verified=True,
                encoded_new_remote='REMOTE_PENDING', physics_steps=0, optimizer_calls=0))
            capacity()
        state = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        write_json(OUT / 'error.json', dict(epoch=time.time(), error=error,
                   partial_bytes_preserved=True, no_automatic_restart=True), exclusive=True)
    finally:
        released = sum(r['released_bytes'] for r in done)
        write_json(OUT / 'summary.json', dict(
            state=state, pid=os.getpid(), elapsed_s=time.time() - started, completed_files=len(done),
            target_files=len(candidate['rows']), payload_released_bytes=released,
            global_before_bytes=before, global_after_bytes=used(),
            original_local_and_verified_remote_bytes_preserved=True,
            encoded_new_remote='REMOTE_PENDING', physics_steps=0, optimizer_calls=0,
            all_workers_stopped=True, error=error), exclusive=True)
        capacity()
    assert state == 'COMPLETE', 'Encoding failed; inspect preserved original/encoded bytes and error'
    print('T435_LOSSLESS_COLD_JSON_COMPLETE', released)


def load_original(relative):
    relative = str(relative)
    rows = read(CANDIDATES)['rows']
    row = next(r for r in rows if r['original_path'] == relative)
    original = safe_path(relative)
    if original.exists():
        data = original.read_bytes()
    else:
        encoded = safe_path(row['encoded_path']).read_bytes()
        assert digest(encoded) == row['encoded_sha256']
        data = gzip.decompress(encoded)
    assert len(data) == row['original_bytes'] and digest(data) == row['original_sha256']
    return data


def audit():
    candidate = read(OUT / 'protocol.json')
    assert candidate == read(CANDIDATES)
    assert candidate['code_sha256'] == digest(Path(__file__).read_bytes())
    summary = read(OUT / 'summary.json')
    assert summary['state'] == 'COMPLETE'
    assert not Path('/proc', str(summary['pid'])).exists(), 'Run process must exit before independent audit'
    assert not (OUT / 'audit.json').exists(), 'Refuse duplicate audit'
    started = time.time()
    for row in candidate['rows']:
        assert time.time() - started < MAX_SECONDS
        verify_remote(row)
        assert not safe_path(row['original_path']).exists()
        encoded = safe_path(row['encoded_path']).read_bytes()
        assert digest(encoded) == row['encoded_sha256'] and len(encoded) == row['encoded_bytes']
        assert digest(load_original(row['original_path'])) == row['original_sha256']
    capacity()
    write_json(OUT / 'audit.json', dict(
        state='PASS', files=len(candidate['rows']), original_bytes_and_all_SHA_exact=True,
        original_remote_complete_readback_verified=True, local_decompression_independently_verified=True,
        released_payload_bytes=summary['payload_released_bytes'], global_bytes=used(),
        physics_steps=0, optimizer_calls=0, elapsed_s=time.time() - started), exclusive=True)
    print('T435_INDEPENDENT_ORIGINAL_BYTES_SHA_AUDIT_PASS')


def restore(relative):
    data = load_original(relative)
    path = safe_path(relative)
    assert not path.exists(), 'Refuse overwrite of a restored bare original'
    capacity(len(data) + 64 * 1024)
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_folder(path.parent)
    assert digest(path.read_bytes()) == digest(data)
    capacity()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'audit', 'restore'])
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.paths)
    elif args.command == 'run':
        run(args.paths)
    elif args.command == 'restore':
        assert len(args.paths) == 1
        restore(args.paths[0])
    else:
        assert not args.paths
        audit()
