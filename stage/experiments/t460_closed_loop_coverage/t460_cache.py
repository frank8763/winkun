"""Evict only freshly re-read, archived and unused T427 NPZ cache.

Original remote member bytes and explicit restore addresses remain preserved.
No model, protocol, error, report, current input or unbacked evidence is evicted.
"""
from pathlib import Path
import concurrent.futures, gzip, hashlib, io, json, os, shutil, signal, sys, tarfile, time
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'stage/runs/t460_closed_loop_coverage'
DOC = ROOT/'stage/docs/long_distance_20261101'


def sha(b): return hashlib.sha256(b).hexdigest()
def read(f): return json.loads(Path(f).read_text())
def save(f, x): Path(f).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')
def used(): return sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())


def candidates(protected, required):
    receipt = DOC/'release_receipt_T427.json'
    manifest = DOC/'source_manifest_T427.json'
    d = read(receipt)
    assert d['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert sha(manifest.read_bytes()) == d['manifest_sha256']
    protected = set(protected)
    for f in (ROOT/'stage/runs').rglob('*.stage-ref.json'):
        protected.add(str(ROOT/read(f)['canonical']))
    rows = []
    for r in read(manifest)['files']:
        f = ROOT/r['path']
        if f.suffix != '.npz' or not f.exists() or str(f) in protected:
            continue
        assert f.is_relative_to(ROOT/'stage/runs/t427_recurrent_plain') or '/t427_' in str(f)
        if f.stat().st_size == r['bytes'] and sha(f.read_bytes()) == r['sha256']:
            rows.append(r)
    rows.sort(key=lambda r: (-r['bytes'], r['path']))
    chosen, total = [], 0
    for r in rows:
        chosen.append(r)
        total += r['bytes']-2048  # reserve explicit per-file restore marker overhead
        if total >= required:
            break
    assert total >= required and len(chosen) <= 500
    return dict(group='T427', receipt=str(receipt.relative_to(ROOT)), receipt_sha256=sha(receipt.read_bytes()),
        manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest.read_bytes()),
        rows=chosen, conservative_released_bytes=total, local_models_deleted=0,
        archive_sha256=d['sha256'], archive_bytes=d['bytes'])


def archive(plan):
    receipt = ROOT/plan['receipt']
    mf = ROOT/plan['manifest']
    assert sha(receipt.read_bytes()) == plan['receipt_sha256']
    assert sha(mf.read_bytes()) == plan['manifest_sha256']
    d = read(receipt)
    assert d['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    sys.path.insert(0, str(ROOT/'stage/experiments/t377_full_human_observation_transfer'))
    import transport as T
    T.install()
    deadline = time.time()+540
    def fetch(part):
        assert time.time() < deadline-125, 'Insufficient bounded readback time; no eviction'
        b = T.readback({'id': part['asset_id']}, part['bytes'])
        assert sha(b) == part['sha256']
        return b
    parts = sorted(d['parts'], key=lambda x: x['order'])
    assert [p['order'] for p in parts] == list(range(len(parts)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(fetch, parts))
    blob = b''.join(chunks)
    assert len(blob) == d['bytes'] and sha(blob) == d['sha256'] == plan['archive_sha256']
    wanted = {r['path']: r for r in read(mf)['files']}
    selected = {r['path'] for r in plan['rows']}
    restored, seen = {}, set()
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as t:
        for m in t:
            assert m.isfile() and m.name in wanted and m.name not in seen
            b = t.extractfile(m).read()
            r = wanted[m.name]
            assert len(b) == r['bytes'] and sha(b) == r['sha256']
            if m.name in selected:
                restored[m.name] = b
            seen.add(m.name)
    assert seen == set(wanted) and set(restored) == selected
    return restored, len(parts), len(seen)


def evict(p):
    assert not (OUT/'cache_summary.json').exists()
    assert not any((OUT/name).exists() for name in ['launch.json', 'preflight.json'])
    for f in Path('/proc').iterdir():
        if not f.name.isdigit() or int(f.name) == os.getpid():
            continue
        try: args = (f/'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        assert not any(str(ROOT/'stage/experiments') in os.fsdecode(a) for a in args[1:]), ('Another Stage process', f.name)
    import subprocess
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Fresh remote cache verification600s')))
    signal.alarm(600)
    plan = p['remote_cache']
    before, start = used(), time.time()
    recovered, parts, members = archive(plan)  # zero file removal until full remote readback
    completed = []
    for r in plan['rows']:
        f = ROOT/r['path']
        assert str(f) not in p['inputs'] and str(f) not in p['constructor_reads']
        assert f.read_bytes() == recovered[r['path']]
        marker = Path(str(f)+'.stage-remote.json')
        assert not marker.exists()
        ref = dict(original=r['path'], bytes=r['bytes'], sha256=r['sha256'],
            receipt=plan['receipt'], receipt_sha256=plan['receipt_sha256'],
            manifest=plan['manifest'], manifest_sha256=plan['manifest_sha256'],
            archive_sha256=plan['archive_sha256'], group='T427',
            restore='run_stage.sh stage/experiments/t460_closed_loop_coverage/run.py restore ORIGINAL_PATH',
            local_cache_only=True, offline_original_byte_availability=False)
        save(marker, ref)
        assert marker.stat().st_size <= 2048 and read(marker) == ref
        # The signed group receipt, this freshly verified archive and marker
        # identify the exact original bytes; this only removes a local cache.
        f.unlink()
        completed.append(r)
        save(OUT/'cache_ledger.json', dict(completed=completed, remote_originals_preserved=True,
            current_inputs_and_models_untouched=True, archive_sha256=plan['archive_sha256']))
    after = used()
    assert after < before and after-(sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()))+p['max_bytes'] <= 2**30
    save(OUT/'cache_summary.json', dict(state='COMPLETE_REMOTE_VERIFIED_CACHE_EVICTION', pid=os.getpid(),
        files=len(completed), released_payload_bytes=sum(r['bytes'] for r in completed),
        before=before, after=after, elapsed_s=time.time()-start, freshly_verified_segments=parts,
        freshly_verified_archive_members=members, original_bytes_remote_preserved=True,
        unbacked_unique_evidence_deleted=0, local_models_deleted=0, native=0, optimizer=0))
    signal.alarm(0)


def audit(p):
    s = read(OUT/'cache_summary.json')
    assert s['state'] == 'COMPLETE_REMOTE_VERIFIED_CACHE_EVICTION'
    assert not Path('/proc', str(s['pid'])).exists() and not (OUT/'cache_audit.json').exists()
    for r in p['remote_cache']['rows']:
        f = ROOT/r['path']
        ref = read(str(f)+'.stage-remote.json')
        assert not f.exists() and (ref['original'], ref['bytes'], ref['sha256']) == (r['path'], r['bytes'], r['sha256'])
    for k in ['receipt', 'manifest']:
        assert sha((ROOT/p['remote_cache'][k]).read_bytes()) == p['remote_cache'][k+'_sha256']
    save(OUT/'cache_audit.json', dict(state='PASS_REFERENCE_AND_FRESH_REMOTE_ARCHIVE_RECEIPT',
        files=s['files'], fresh_archive_verified_before_eviction=True,
        original_remote_member_SHA_preserved=True, offline_local_bytes_NOT_claimed=True,
        current_inputs_models_untouched=True, native=0, optimizer=0))


def restore(path):
    f = Path(path)
    if not f.is_absolute(): f = ROOT/f
    assert f.is_relative_to(ROOT/'stage/runs') and not f.exists()
    ref = read(str(f)+'.stage-remote.json')
    p = read(OUT/'protocol.json')
    assert ref['original'] in {r['path'] for r in p['remote_cache']['rows']}
    assert used()+ref['bytes']+131072 <= 2**30 and shutil.disk_usage(ROOT).free >= 40*2**30
    recovered, _, _ = archive(p['remote_cache'])
    b = recovered[ref['original']]
    assert sha(b) == ref['sha256']
    with f.open('xb') as out:
        out.write(b)
    assert sha(f.read_bytes()) == ref['sha256']
