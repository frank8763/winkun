"""Stage-only, RAM-only completed-run backup with full byte readback.

No navigation, learning, deletion, compaction, or active-run snapshots.
"""
from pathlib import Path
import argparse
import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / 'stage/docs/long_distance_20261101'
sys.path.insert(0, str(ROOT / 'stage/experiments/t224_archive'))
from archive import OPENER, SLUG, api, req, sha, token, verify_tar, write

GROUPS = {
    'T256-T258': ['t256_distance_frontier', 't257_distance_policy_gradient',
                  't258_closed_loop_retention'],
    'T259': ['t259_closed_loop_recovery'],
    'T261': ['t261_distance_credit_audit'],
    'T262': ['t262_event_contract'],
}


def used():
    return sum(p.stat().st_size for p in (ROOT / 'stage/runs').rglob('*')
               if p.is_file())


def require_stopped(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['state'] in ('COMPLETE', 'FAILED', 'FAILED_ZERO_STEP_ENTRYPOINT')
    if summary['state'] == 'FAILED_ZERO_STEP_ENTRYPOINT':
        assert summary['physics_steps'] == summary['optimizer_calls'] == summary['workers_spawned'] == 0
    else:
        assert summary['all_workers_stopped'] is True
    if (directory / 'workers.json').exists():
        w = json.loads((directory / 'workers.json').read_text())
        for pid in [w.get('parent'), *w.get('workers', [])]:
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                raise AssertionError(f'Refuse live or reused process id {pid}')


def readback(asset, size):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        f'https://api.github.com/repos/{SLUG}/releases/assets/{asset["id"]}',
        headers={'Authorization': 'Bearer ' + token(),
                 'Accept': 'application/octet-stream', 'User-Agent': 'Stage-evidence-backup'})
    try:
        with opener.open(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        assert error.code == 302
        url = error.headers['Location']
    assert urllib.parse.urlparse(url).hostname in (
        'release-assets.githubusercontent.com', 'objects.githubusercontent.com')
    parts = []
    for begin in range(0, size, 1024**2):
        end = min(size, begin + 1024**2) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={
                    'Range': f'bytes={begin}-{end}', 'User-Agent': 'Stage-evidence-backup'})
                with OPENER.open(request, timeout=30) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'].startswith(f'bytes {begin}-{end}/')
                    data = response.read()
                    assert len(data) == end - begin + 1
                break
            except (OSError, http.client.HTTPException):
                if attempt == 4:
                    raise
        parts.append(data)
    return b''.join(parts)


def main(group):
    receipt = DOC / f'release_receipt_{group}.json'
    assert not receipt.exists(), 'Already verified; inspect receipt rather than duplicating'
    directories = [ROOT / 'stage/runs' / name for name in GROUPS[group]]
    for directory in directories:
        require_stopped(directory)
    before = used()
    assert before < 1024**3 and shutil.disk_usage(ROOT).free >= 40 * 1024**3
    files = sorted(p for directory in directories for p in directory.rglob('*')
                   if p.is_file() and p.suffix not in ('.lock', '.partial', '.pyc'))
    manifest = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                     sha256=sha(p.read_bytes())) for p in files]
    assert sum(r['bytes'] for r in manifest) < 96 * 1024**2
    manifest_path = DOC / f'source_manifest_{group}.json'
    write(manifest_path, dict(scope=f'Stage {group} completed runs, all failures and models',
                             files=manifest, active_runs_included=False))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', compresslevel=3) as archive:
        for p, record in zip(files, manifest):
            assert sha(p.read_bytes()) == record['sha256']
            archive.add(p, arcname=record['path'], recursive=False)
    data = buffer.getvalue()
    verify_tar(data, manifest)
    tag = 'stage-distance-2026-11-01'
    try:
        release = api('GET', 'releases/tags/' + tag)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        source = api('GET', 'git/ref/heads/main')['object']['sha']
        release = api('POST', 'releases', dict(
            tag_name=tag, target_commitish=source, draft=False,
            name='Stage bounded distance experiments (2026-11-01)',
            body='Stage-only original evidence, including failures and final Actor/Adam. '
                 'Read per-run summaries and distance reports for actual results. '
                 'No generalized-stability or algorithm-novelty claim. '
                 'Completed runs only; all local originals retained.'))
    assets = api('GET', f'releases/{release["id"]}/assets?per_page=100')
    name = f'stage-distance-{group.lower()}-{sha(data)[:12]}.tar.gz'
    asset = next((a for a in assets if a['name'] == name), None)
    if asset is None:
        asset = json.loads(req('POST',
            f'https://uploads.github.com/repos/{SLUG}/releases/{release["id"]}/assets?name='
            + urllib.parse.quote(name), data, {'Content-Type': 'application/octet-stream'}))
    asset = api('GET', f'releases/assets/{asset["id"]}')
    assert asset['size'] == len(data) and asset['digest'] == 'sha256:' + sha(data)
    print(json.dumps(dict(state='UPLOADED_READBACK_PENDING', group=group,
                          members=len(manifest), bytes=len(data))), flush=True)
    remote = readback(asset, len(data))
    assert remote == data
    verify_tar(remote, manifest)
    for p, record in zip(files, manifest):
        assert p.stat().st_size == record['bytes'] and sha(p.read_bytes()) == record['sha256']
    for directory in directories:
        require_stopped(directory)
    after = used()
    assert after < 1024**3 and shutil.disk_usage(ROOT).free >= 40 * 1024**3
    write(receipt, dict(state='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED',
        group=group, url=release['html_url'], asset_url=asset['browser_download_url'],
        asset_id=asset['id'], bytes=len(data), sha256=sha(data), members=len(manifest),
        manifest_sha256=sha(manifest_path.read_bytes()), temporary_disk_archive_bytes=0,
        files_deleted=0, physics_steps=0, optimizer_updates=0,
        newruns_logical_bytes_before=before, newruns_logical_bytes_after=after,
        local_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                   cwd=ROOT, text=True).strip(),
        verified_epoch=time.time()))
    print(json.dumps(dict(state='BACKUP_VERIFIED', group=group,
                          url=release['html_url'], members=len(manifest))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=GROUPS)
    main(parser.parse_args().group)
