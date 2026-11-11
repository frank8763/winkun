"""Bounded T480 eviction of completed Stage NPZ caches with exact remote restore.

Planning is local and read only. Execution reuses the frozen T460 binary CLI
reader, verifies every member of every chosen archive in RAM before any unlink,
and preserves immutable restore markers and an append-only progress ledger.
Models, current inputs, canonical references and unbacked evidence are excluded.
"""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'stage/runs'
OUT = RUNS / 't480_old_task_rehearsal'
READER = ROOT / 'stage/experiments/t460_closed_loop_coverage/t460_cache.py'
COMPLETE = 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
LIMIT = 2**30
METADATA_RESERVE = 1024 * 1024
PROTECTED_PROTOCOLS = (
    RUNS / 't467_actual_occupancy_audit/protocol.json',
    RUNS / 't465_matched_safe_recovery/protocol.json',
    RUNS / 't464_two_failure_recovery/protocol.json',
)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(f):
    return json.loads(Path(f).read_bytes())


def encoded(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def used():
    return sum(f.stat().st_size for f in RUNS.rglob('*') if f.is_file())


def _exclusive(f, x, check_capacity=True):
    b = encoded(x)
    if check_capacity:
        assert used() + len(b) <= LIMIT, 'Metadata would exceed global logical limit'
    with Path(f).open('xb') as out:
        out.write(b)
        out.flush()
        os.fsync(out.fileno())
    assert Path(f).read_bytes() == b


def _original_json(f):
    if f.exists():
        return read(f)
    cold = Path(str(f) + '.cold.gz')
    return json.loads(gzip.decompress(cold.read_bytes())) if cold.exists() else None


def _path(s):
    f = Path(s)
    return f if f.is_absolute() else ROOT / f


def _protected(requested):
    paths = set(str(_path(f)) for f in requested)
    # Every referenced path in these current protocols is protected, not only
    # the named Actor or teacher. This includes snapshots and historical NPZ.
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str) and (x.startswith(str(ROOT) + '/') or x.startswith('stage/')):
            paths.add(str(_path(x)))
    for f in PROTECTED_PROTOCOLS:
        walk(read(f))
        paths.add(str(f))
    for f in RUNS.rglob('*.stage-ref.json'):
        ref = read(f)
        paths.add(str(_path(ref['canonical'])))
        paths.add(str(f))
    return paths


def _eligible(f, protected):
    rel = f.relative_to(ROOT)
    match = re.match(r'stage/runs/t(\d+)_', rel.as_posix())
    return bool(match and int(match[1]) < 480 and int(match[1]) not in (145, 146, 159, 194, 255)
                and f.suffix == '.npz' and str(f) not in protected
                and f.is_file() and not f.is_symlink() and f.resolve() == f
                and not Path(str(f) + '.stage-ref.json').exists()
                and not Path(str(f) + '.stage-remote.json').exists())


def _marker(group, row):
    return dict(original=row['path'], bytes=row['bytes'], sha256=row['sha256'],
                group=group['group'], receipt=group['receipt'], receipt_sha256=group['receipt_sha256'],
                manifest=group['manifest'], manifest_sha256=group['manifest_sha256'],
                archive_sha256=group['archive_sha256'], local_cache_only=True,
                offline_original_byte_availability=False,
                restore='run_stage.sh stage/experiments/t480_old_task_rehearsal/t480_cache.py restore ORIGINAL_PATH')


def _entry(group, row):
    return dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'],
                marker_sha256=sha(encoded(_marker(group, row))), group=group['group'])


def _completed(root):
    # Only a frozen completed root/substage with every recorded process gone
    # is selectable. A root seal may represent an ended finite relay.
    files = [root / 'SEALED.json', root / 'summary.json', root / 'status.json']
    files += sorted(root.glob('*/summary.json'))
    files += [Path(str(f)[:-8]) for f in sorted(root.glob('*/summary.json.cold.gz'))]
    evidence = None
    for f in files:
        d = _original_json(f)
        if d and (str(d.get('state', '')).startswith('COMPLETE')
                  or d.get('state') == 'SEALED_FINITE_ROOT'
                  or (f.name == 'SEALED.json' and d.get('all_scientific_processes_stopped'))):
            source = f if f.exists() else Path(str(f) + '.cold.gz')
            evidence = dict(path=str(source.relative_to(ROOT)), sha256=sha(source.read_bytes()))
            break
    if evidence is None:
        return None
    ids = []
    # A completed child does not authorize touching any still-active sibling.
    checks = [root / 'summary.json', root / 'SEALED.json', root / 'driver_started.json', root / 'launch.json']
    checks += sorted(root.glob('*/summary.json'))
    checks += sorted(root.glob('*/driver_started.json'))
    checks += sorted(root.glob('*/launch.json'))
    for f in checks:
        d = _original_json(f)
        if not d:
            continue
        if d.get('pid'):
            ids.append(d['pid'])
        ids += d.get('scientific_pids', [])
        ids += d.get('all_scientific_pids', [])
    generations = [root / 'worker_generations.json'] + sorted(root.glob('*/worker_generations.json'))
    for f in generations:
        d = _original_json(f)
        if d:
            for g in d:
                ids += [g['parent']] + g['workers']
    if any(Path('/proc', str(pid)).exists() for pid in ids):
        return None
    return evidence


def _group(receipt):
    d = read(receipt)
    manifest = receipt.with_name(receipt.name.replace('release_receipt', 'source_manifest'))
    if d.get('state') != COMPLETE or not isinstance(d.get('parts'), list) or not manifest.exists():
        return None
    if 'INTERRUPTED' in receipt.name.upper():
        return None
    if not d['parts'] or not all(isinstance(x, dict) and 'order' in x and 'asset_id' in x for x in d['parts']):
        return None  # Older multi-archive receipts are not binary segments.
    assert sha(manifest.read_bytes()) == d['manifest_sha256']
    parts = sorted(d['parts'], key=lambda x: x['order'])
    assert [x['order'] for x in parts] == list(range(len(parts))) and parts
    assert sum(x['bytes'] for x in parts) == d['bytes']
    assert all(x['bytes'] > 0 and isinstance(x['asset_id'], int) and len(x['sha256']) == 64 for x in parts)
    rows = read(manifest)['files']
    assert len({x['path'] for x in rows}) == len(rows)
    roots = {ROOT / '/'.join(x['path'].split('/')[:3]) for x in rows if x['path'].startswith('stage/runs/')}
    if not roots:
        return None
    completed = [_completed(f) for f in sorted(roots)]
    if not all(completed):
        return None
    return dict(group=d['group'], receipt=str(receipt.relative_to(ROOT)), receipt_sha256=sha(receipt.read_bytes()),
                manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest.read_bytes()),
                archive_sha256=d['sha256'], archive_bytes=d['bytes'], segments=len(parts),
                members=len(rows), completed_evidence=completed,
                ended_roots=[str(f.relative_to(ROOT)) for f in sorted(roots)], rows=[])


def plan(protected, required):
    """Choose the fewest completed remote groups, then largest unused members.

    ``required`` is the desired net byte release, after exact compact restore
    markers and a further 1 MiB reservation for this action's audit/ledger.
    No source, protocol, model, run output or cache is written by this function.
    """
    assert isinstance(required, int) and required > 0
    protected = _protected(protected)
    groups = []
    candidate_paths = set()
    for receipt in sorted((ROOT / 'stage/docs').rglob('release_receipt*.json')):
        group = _group(receipt)
        if group is None:
            continue
        for row in read(ROOT / group['manifest'])['files']:
            f = ROOT / row['path']
            if not _eligible(f, protected):
                continue
            row = {k: row[k] for k in ('path', 'bytes', 'sha256')}
            marker_bytes = len(encoded(_marker(group, row)))
            ledger_bytes = len(encoded(_entry(group, row)))
            assert marker_bytes <= 2048
            if row['bytes'] <= marker_bytes + ledger_bytes or f.stat().st_size != row['bytes'] or sha(f.read_bytes()) != row['sha256']:
                continue
            assert row['path'] not in candidate_paths, 'Overlapping archive candidates require explicit review'
            candidate_paths.add(row['path'])
            row['marker_bytes'] = marker_bytes
            row['ledger_bytes'] = ledger_bytes
            group['rows'].append(row)
        group['rows'].sort(key=lambda x: (-(x['bytes'] - x['marker_bytes'] - x['ledger_bytes']), x['path']))
        group['available_net_bytes'] = sum(x['bytes'] - x['marker_bytes'] - x['ledger_bytes'] for x in group['rows'])
        if group['available_net_bytes']:
            groups.append(group)
    groups.sort(key=lambda x: (-x['available_net_bytes'], x['group']))
    available = sum(g['available_net_bytes'] for g in groups) - METADATA_RESERVE
    assert available >= required, ('Insufficient protected remote cache availability', available, required)
    chosen, released = [], 0
    for group in groups:
        rows = []
        for row in group['rows']:
            rows.append(row)
            released += row['bytes'] - row['marker_bytes'] - row['ledger_bytes']
            if released >= required + METADATA_RESERVE:
                break
        chosen.append(dict(group, rows=rows))
        if released >= required + METADATA_RESERVE:
            break
    # Because candidate memberships are disjoint, taking descending capacities
    # reaches the minimum number of remote archive groups required.
    assert sum(g['available_net_bytes'] for g in groups[:len(chosen)-1]) < required + METADATA_RESERVE
    protected_hashes = {f: sha(Path(f).read_bytes()) for f in sorted(protected) if Path(f).is_file()}
    payload = sum(r['bytes'] for g in chosen for r in g['rows'])
    ram_bound = payload + 3 * max(g['archive_bytes'] for g in chosen)
    assert ram_bound <= 250 * 2**20, ('Remote readback RAM reservation exceeds 250MiB', ram_bound)
    return dict(groups=chosen, required_net_bytes=required, conservative_released_bytes=released - METADATA_RESERVE,
                released_payload_bytes=payload, metadata_reserve_bytes=METADATA_RESERVE,
                available_net_bytes=available, available_groups=len(groups), chosen_group_count=len(chosen),
                protected_sha256=protected_hashes, reader=str(READER.relative_to(ROOT)),
                reader_sha256=sha(READER.read_bytes()), maximum_RAM_reservation_bytes=ram_bound,
                maximum_concurrent_segments=4, local_models_deleted=0, native=0, optimizer=0)


candidates = plan


def _idle():
    own = os.getpid()
    for f in Path('/proc').iterdir():
        if not f.name.isdigit() or int(f.name) == own:
            continue
        try:
            args = (f / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        # The independent cache-audit child may have this exact read-only
        # orchestration parent. It waits for the child; it never runs native
        # simulation or mutates evidence concurrently. No other Stage process
        # is exempt, including a driver, worker or another queue instance.
        owner = OUT / 'queue_owner.json'
        queue = ROOT / 'stage/experiments/t480_old_task_rehearsal/t480_queue.py'
        if int(f.name) == os.getppid() and owner.exists():
            r = read(owner)
            if (r['pid'] == int(f.name) and r['script_sha256'] == sha(queue.read_bytes())
                    and os.fsencode(str(queue)) in args[1:]
                    and r['role'] == 'sequential_non_native_orchestrator'):
                continue
        try:
            env = (f / 'environ').read_bytes().split(b'\0')
        except OSError:
            env = []
        assert os.fsencode('STAGE_RUN_ROOT=' + str(RUNS)) not in env, ('Another Stage environment process', f.name)
        for arg in args[1:]:
            a = os.fsdecode(arg)
            # Match actual executable/script arguments, not a shell's quoted
            # command text. Workers invoked by a Stage module remain excluded.
            assert not (a.startswith(str(ROOT / 'stage/experiments') + '/') or a.startswith('stage/experiments/')), ('Another Stage process', f.name)


def _fresh_inputs(p, cache):
    protected = _protected(set(p.get('inputs', {})) | set(p.get('constructor_reads', [])) | set(p.get('protected_files', {})))
    for group in cache['groups']:
        for row in group['rows']:
            assert str(ROOT / row['path']) not in protected
    for name, digest in cache['protected_sha256'].items():
        assert Path(name).is_file() and sha(Path(name).read_bytes()) == digest, ('Protected input changed', name)
    assert sha(READER.read_bytes()) == cache['reader_sha256']
    for group in cache['groups']:
        for root in group['ended_roots']:
            assert _completed(ROOT / root) is not None, ('Recorded run not fully stopped', root)
        for evidence in group['completed_evidence']:
            assert sha((ROOT / evidence['path']).read_bytes()) == evidence['sha256']


def _archive(group, cache, deadline):
    assert sha(READER.read_bytes()) == cache['reader_sha256']
    spec = importlib.util.spec_from_file_location('_t479_frozen_t460_cache', READER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This reader only imports transport when called, and uses at most four
    # existing gh CLI calls. Stderr/tokens are never emitted by this wrapper.
    # One failed CLI request must not discard an otherwise verified archive.
    # Keep the original member verifier unchanged; retry only failed binary
    # transport, at most three calls and within the SAME absolute deadline.
    original_run = subprocess.run
    def bounded_run(args, *a, **kw):
        binary = (isinstance(args, list) and 'Accept: application/octet-stream' in args
                  and any('/releases/assets/' in str(v) for v in args))
        if not binary:
            return original_run(args, *a, **kw)
        for attempt in range(3):
            assert time.time() < deadline - 40, 'Original cache deadline; no further transport attempt'
            result = original_run(args, *a, **kw)
            if result.returncode == 0:
                return result
            with (OUT / 'cache_transport_attempts.jsonl').open('a') as log:
                log.write(json.dumps(dict(attempt=attempt, returncode=result.returncode,
                    epoch=time.time(), stderr_sha256=sha(result.stderr),
                    no_native_no_optimizer=True)) + '\n')
        return result
    subprocess.run = bounded_run
    try:
        return module.archive(group, deadline=deadline, cli=True)
    finally:
        subprocess.run = original_run


def _reserve(p):
    return int(p.get('cache_total_reserved_bytes', p.get('max_bytes', 0)))


def evict(p, recovery=False):
    """Run once before preflight/launch, after a clean exact source sync."""
    assert OUT.exists()
    assert not any((OUT / f).exists() for f in ('cache_summary.json', 'cache_ledger.jsonl', 'launch.json', 'preflight.json'))
    started_name = 'cache_started.json'
    if recovery:
        first = read(OUT / started_name)
        assert not Path('/proc', str(first['pid'])).exists()
        assert not (OUT / 'cache_remote_verified.json').exists(), 'Never resume after cache mutation'
        assert first['hard_stop_epoch'] == p['cache_hard_stop_epoch']
        started_name = 'cache_readback_recovery_started.json'
    assert not (OUT / started_name).exists()
    _idle()
    src = read(RUNS / 't224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    cache = p['remote_cache']
    _fresh_inputs(p, cache)
    start, before = time.time(), used()
    deadline = float(p['cache_hard_stop_epoch'])
    assert start + 50 < deadline <= start + 1200, 'Require fresh original absolute cache deadline within 1200s'
    assert before + METADATA_RESERVE <= LIMIT and shutil.disk_usage(ROOT).free >= 40 * 2**30
    assert _reserve(p) > 0
    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T480 bounded cache deadline')))
    # T460's thread-pool cleanup can wait for an in-flight 35s CLI call.
    # Leave 40s so that even exception cleanup fits the absolute hard stop.
    signal.alarm(max(1, int(deadline - start - 40)))
    try:
        _exclusive(OUT / started_name, dict(pid=os.getpid(), start_epoch=start, hard_stop_epoch=deadline,
                   groups=[g['group'] for g in cache['groups']], before=before, native=0, optimizer=0))
        recovered, verification = {}, []
        for group in cache['groups']:
            assert time.time() < deadline - 45
            b, parts, members = _archive(group, cache, deadline - 45)
            assert not set(recovered).intersection(b)
            recovered.update(b)
            verification.append(dict(group=group['group'], segments=parts, members=members,
                                     archive_sha256=group['archive_sha256']))
        # ALL groups and their ALL members have passed before the first marker
        # or original removal. Recheck all selected local bytes as one batch.
        _idle()
        _fresh_inputs(p, cache)
        selected = [r for g in cache['groups'] for r in g['rows']]
        assert set(recovered) == {r['path'] for r in selected}
        live_protected = _protected(set(p.get('inputs', {})) | set(p.get('constructor_reads', [])) | set(p.get('protected_files', {})))
        for row in selected:
            f = ROOT / row['path']
            assert _eligible(f, live_protected)
            assert f.read_bytes() == recovered[row['path']]
        _exclusive(OUT / 'cache_remote_verified.json', dict(state='ALL_GROUPS_ALL_MEMBERS_VERIFIED_BEFORE_ANY_UNLINK',
                   verified_epoch=time.time(), groups=verification, selected_files=len(selected)))
        ledger = OUT / 'cache_ledger.jsonl'
        # Every selected row releases more than its exact marker+ledger cost.
        # Thus global logical usage declines after every completed row, and the
        # small metadata reservation also covers the moment before unlink.
        assert used() + 2048 <= LIMIT
        with ledger.open('xb') as log:
            for group in cache['groups']:
                for row in group['rows']:
                    assert time.time() < deadline - 40
                    f = ROOT / row['path']
                    assert _eligible(f, set(cache['protected_sha256']))
                    # A fresh second comparison immediately precedes removal.
                    assert f.read_bytes() == recovered[row['path']]
                    marker = Path(str(f) + '.stage-remote.json')
                    ref = _marker(group, row)
                    assert len(encoded(ref)) == row['marker_bytes'] <= 2048
                    _exclusive(marker, ref, check_capacity=False)
                    marker.chmod(0o444)
                    f.unlink()
                    entry = _entry(group, row)
                    assert sha(marker.read_bytes()) == entry['marker_sha256']
                    assert len(encoded(entry)) == row['ledger_bytes']
                    log.write(encoded(entry))
                    log.flush()
                    os.fsync(log.fileno())
        _fresh_inputs(p, cache)
        after = used()
        own = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
        assert after < before and after - own + _reserve(p) <= LIMIT
        _exclusive(OUT / 'cache_summary.json', dict(state='COMPLETE_REMOTE_VERIFIED_CACHE_EVICTION', pid=os.getpid(),
                   files=len(selected), groups=len(cache['groups']), before=before, after=after,
                   actual_net_released_bytes=before-after, released_payload_bytes=sum(r['bytes'] for r in selected),
                   elapsed_s=time.time()-start, all_groups_verified_before_any_unlink=True,
                   ledger_sha256=sha(ledger.read_bytes()), original_bytes_remote_preserved=True,
                   offline_local_bytes_NOT_claimed=True, unbacked_unique_evidence_deleted=0,
                   local_models_deleted=0, native=0, optimizer=0))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def cacheaudit(p):
    """Separate-process local receipt/marker/protected-input audit, zero network."""
    _idle()
    s = read(OUT / 'cache_summary.json')
    assert s['state'] == 'COMPLETE_REMOTE_VERIFIED_CACHE_EVICTION'
    assert not Path('/proc', str(s['pid'])).exists()
    assert not (OUT / 'cache_audit.json').exists()
    cache = p['remote_cache']
    _fresh_inputs(p, cache)
    verified = read(OUT / 'cache_remote_verified.json')
    assert verified['state'] == 'ALL_GROUPS_ALL_MEMBERS_VERIFIED_BEFORE_ANY_UNLINK'
    assert verified['selected_files'] == s['files']
    ledger_file = OUT / 'cache_ledger.jsonl'
    assert sha(ledger_file.read_bytes()) == s['ledger_sha256']
    ledger = [json.loads(x) for x in ledger_file.read_bytes().splitlines()]
    rows = {r['path']: (g, r) for g in cache['groups'] for r in g['rows']}
    assert len(rows) == len(ledger) == s['files'] and {x['path'] for x in ledger} == set(rows)
    for group in cache['groups']:
        for k in ('receipt', 'manifest'):
            assert sha((ROOT / group[k]).read_bytes()) == group[k + '_sha256']
        receipt = read(ROOT / group['receipt'])
        assert receipt['state'] == COMPLETE and receipt['sha256'] == group['archive_sha256']
        assert any(v['group'] == group['group'] and v['segments'] == len(receipt['parts'])
                   and v['members'] == group['members'] for v in verified['groups'])
        for evidence in group['completed_evidence']:
            assert sha((ROOT / evidence['path']).read_bytes()) == evidence['sha256']
    for entry in ledger:
        group, row = rows[entry['path']]
        f = ROOT / row['path']
        marker = Path(str(f) + '.stage-remote.json')
        assert not f.exists() and not f.is_symlink()
        assert marker.is_file() and not marker.is_symlink() and marker.stat().st_size <= 2048
        assert read(marker) == _marker(group, row) and sha(marker.read_bytes()) == entry['marker_sha256']
        assert (entry['bytes'], entry['sha256'], entry['group']) == (row['bytes'], row['sha256'], group['group'])
    own = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
    assert used() - own + _reserve(p) <= LIMIT
    _exclusive(OUT / 'cache_audit.json', dict(state='PASS_REFERENCE_RECEIPT_MARKERS_AND_PROTECTED_INPUTS',
               files=s['files'], groups=s['groups'], all_original_remote_member_SHA_preserved=True,
               all_groups_freshly_read_before_any_unlink=True, current_inputs_canonical_models_untouched=True,
               offline_local_original_bytes_NOT_claimed=True, native=0, optimizer=0))


audit = cacheaudit


def restore(path):
    """Explicit bounded restore of one original, respecting the global limit."""
    _idle()
    f = _path(path)
    assert f.is_relative_to(RUNS) and f.suffix == '.npz' and not f.exists()
    p = read(OUT / 'protocol.json')
    cache_path = Path(p['remote_cache_path'])
    assert sha(cache_path.read_bytes()) == p['remote_cache_sha256']
    cache = read(cache_path)
    matches = [(g, r) for g in cache['groups'] for r in g['rows'] if ROOT / r['path'] == f]
    assert len(matches) == 1
    group, row = matches[0]
    assert read(str(f) + '.stage-remote.json') == _marker(group, row)
    assert used() + row['bytes'] + 131072 <= LIMIT and shutil.disk_usage(ROOT).free >= 40 * 2**30
    recovered, _, _ = _archive(dict(group, rows=[row]), cache, time.time() + 600)
    b = recovered[row['path']]
    assert len(b) == row['bytes'] and sha(b) == row['sha256']
    with f.open('xb') as out:
        out.write(b)
    assert sha(f.read_bytes()) == row['sha256']


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] == 'restore'
    restore(sys.argv[2])
