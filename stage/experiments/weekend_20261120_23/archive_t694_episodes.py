"""One deterministic T694 full-history evidence batch; no science and no local deletion."""
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import signal
import time

import archive_after_t695 as Guard

ROOT, DOC = Guard.ROOT, Guard.DOC
PREFIX = 'WEEKEND_T694_FULL_EPISODES_'


def main():
    started = time.time()
    auth = Guard.read(DOC / 'AUTHORIZATION.json')
    ledger = Guard.read(DOC / 'LEDGER.json')
    assert auth['approved']
    assert ledger['transport_seconds'] + 180 + 300 < auth['max_transport_seconds']
    assert started + 480 < auth['absolute_all_stop']
    assert not any(j['state'] == 'RESERVED' for j in ledger['jobs'])
    assert shutil.disk_usage(ROOT).free > auth['minimum_free_bytes']
    assert sum(f.stat().st_size for f in (ROOT / 'stage/runs').rglob('*')
               if f.is_file()) < auth['archive_trigger_bytes']
    Guard.stopped()
    with (ROOT / 'stage/runs/.stage_writer.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        restore = Guard.read(DOC / 'RESTORE_INDEX.json')
        prior = [g['group'] for g in restore['groups'] if g['group'].startswith(PREFIX)]
        index = max((int(name.rsplit('_', 1)[1]) for name in prior), default=-1) + 1
        GROUP = PREFIX + str(index)
        evidence = ROOT / 'stage/docs/long_distance_20261101'
        receipt = evidence / f'release_receipt_{GROUP}.json'
        manifest = evidence / f'source_manifest_{GROUP}.json'
        assert not receipt.exists() and not manifest.exists()
        verified = set()
        for group in restore['groups']:
            assert group['remote_receipt']['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            verified.update(x['path'] for x in group['members'])
        files, size = [], 0
        for f in sorted((ROOT / 'stage/runs/t694_full_history_compatibility/evaluation/episodes').glob('*.npz')):
            if str(f.relative_to(ROOT)) in verified:
                continue
            if size + f.stat().st_size > 8 * 2**20:
                break
            files.append(f)
            size += f.stat().st_size
        assert files and size <= 8 * 2**20
        hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T694 evidence archive180s')))
        signal.alarm(180)
        Guard.Client.A.stopped = Guard.stopped
        client = Guard.Client.client()
        release = json.loads(client.req('GET', 'https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
        result = Guard.Client.A.archive(GROUP, files, release, min(started + 175, auth['absolute_all_stop'] - 30), client)
        Guard.stopped()
        assert result['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        assert result['bytes'] <= 8 * 2**20
        members = Guard.read(manifest)['files']
        assert {x['path']: x['sha256'] for x in members} == hashes
        for item in members:
            f = ROOT / item['path']
            assert f.stat().st_size == item['bytes']
            assert hashlib.sha256(f.read_bytes()).hexdigest() == item['sha256']
        restore['groups'].append(dict(group=GROUP, receipt=str(receipt), manifest=str(manifest),
                                      remote_receipt=result, members=members))
        restore.update(epoch=time.time(), files_deleted=0, other_raw_pending=True)
        (DOC / 'RESTORE_INDEX.json').write_text(json.dumps(restore, indent=2) + '\n')
        report = dict(state=result['state'], group=GROUP, members=len(members), source_bytes=size,
                      compressed_bytes=result['bytes'], elapsed_seconds=time.time() - started,
                      deleted=0, all_weekend_science_driver_helper_units_checked=True,
                      no_new_optimizer_or_native=True)
        (DOC / f'T694_FULL_EPISODE_BACKUP_{index}.json').write_text(json.dumps(report, indent=2) + '\n')
        signal.alarm(0)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    started = time.time()
    try:
        main()
    finally:
        path = DOC / 'LEDGER.json'
        value = Guard.read(path)
        value['transport_seconds'] += time.time() - started
        path.write_text(json.dumps(value, indent=2) + '\n')
