"""Bounded Stage-only archive with local member verification and GitHub SHA receipt.

Usage: python stage/tools/archive_mechanism_audit.py package|publish
Authentication is read in memory from the already authorized GitHub CLI.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'stage/runs'
TRANSFER = RUNS / '_transfer_mechanism_audit_20261030'
DOCS = REPO / 'stage/docs/results_20261030_mechanism_audit'
MANIFEST = REPO / 'stage/manifests/mechanism_audit_20261030_files.json'
TAG = 'stage-mechanism-audit-2026-10-30'
SLUG = 'frank8763/winkun'
ROOTS = ['t179_mechanism_audit']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def used():
    return sum(p.stat().st_size for p in RUNS.rglob('*') if p.is_file())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def package():
    assert not (TRANSFER / 'package_receipt.json').exists(), 'Existing package must be inspected'
    for name, suffix in [(name, 'status.json') for name in ROOTS]:
        assert json.loads((RUNS / name / suffix).read_text())['state'] == 'COMPLETE', name
    assert json.loads((RUNS/'t179_mechanism_audit/gradient_audit.json').read_text())['state']=='COMPLETE'
    assert json.loads((RUNS/'t179_mechanism_audit/contact_recheck/status.json').read_text())['state']=='COMPLETE'
    TRANSFER.mkdir(exist_ok=True)
    files = []
    for name in ROOTS:
        for p in sorted((RUNS / name).rglob('*')):
            if not p.is_file() or 'cache' in p.parts or p.suffix in ('.lock', '.partial', '.pyc'):
                continue
            assert not p.is_symlink(), p
            assert p.name != 'checkpoint_previous.pt', 'Only evaluated final checkpoint retained'
            files.append({'path': str(p.relative_to(REPO)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    total = sum(r['bytes'] for r in files)
    assert used() + total + 32 * 1024**2 < 1024**3, 'Worst-case package must fit new-run1GiB cap'
    assert shutil.disk_usage(REPO).free > 40 * 1024**3
    write(MANIFEST, {'schema': 1, 'scope': 'Completed Stage T179 frozen radius, contact provenance and offline gradient diagnostics; no Phase, caches, credentials, or original chats',
                     'source_bytes': total, 'source_files': len(files), 'files': files})
    partial = TRANSFER / 'stage-mechanism-audit-20261030.tar.zst.partial'
    with partial.open('wb') as output:
        compressor = subprocess.Popen(['zstd', '-T2', '-3', '-q', '-c'], stdin=subprocess.PIPE, stdout=output)
        try:
            with tarfile.open(fileobj=compressor.stdin, mode='w|') as archive:
                archive.add(MANIFEST, arcname=str(MANIFEST.relative_to(REPO)), recursive=False)
                for row in files:
                    p = REPO / row['path']
                    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
                    archive.add(p, arcname=row['path'], recursive=False)
            compressor.stdin.close()
            assert compressor.wait() == 0
        finally:
            if compressor.poll() is None:
                compressor.terminate()
                compressor.wait()
    expected = {r['path']: r for r in files}
    expected[str(MANIFEST.relative_to(REPO))] = {'bytes': MANIFEST.stat().st_size, 'sha256': sha(MANIFEST)}
    seen = set()
    decompressor = subprocess.Popen(['zstd', '-d', '-q', '-c', str(partial)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decompressor.stdout, mode='r|') as archive:
            for member in archive:
                assert member.isfile() and member.name in expected and member.name not in seen
                stream = archive.extractfile(member)
                h, count = hashlib.sha256(), 0
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                    count += len(block)
                row = expected[member.name]
                assert count == row['bytes'] and h.hexdigest() == row['sha256'], member.name
                seen.add(member.name)
        # Consume tar end padding so zstd does not receive SIGPIPE.
        while decompressor.stdout.read(1024 * 1024):
            pass
        assert decompressor.wait() == 0 and seen == set(expected)
    finally:
        if decompressor.poll() is None:
            decompressor.terminate()
            decompressor.wait()
    digest = sha(partial)
    archive_path = TRANSFER / f'stage-mechanism-audit-20261030-{digest[:12]}.tar.zst'
    os.replace(partial, archive_path)
    assert used() < 1024**3
    receipt = {'archive': str(archive_path.relative_to(REPO)), 'bytes': archive_path.stat().st_size,
               'sha256': digest, 'source_files': len(files), 'source_bytes': total,
               'locally_verified_members': len(seen), 'manifest_sha256': sha(MANIFEST),
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
               'state': 'LOCAL_FULL_MEMBER_HASH_VERIFIED'}
    write(TRANSFER / 'package_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def publish():
    local = json.loads((TRANSFER / 'package_receipt.json').read_text())
    receipt_path = DOCS / 'release_receipt.json'
    assert not receipt_path.exists(), 'Inspect existing successful release receipt'
    archive_path = REPO / local['archive']
    assert sha(archive_path) == local['sha256']
    cli = REPO.parent / 'outputs/stage_migration/tools/gh'
    token = subprocess.check_output([str(cli), 'auth', 'token', '--hostname', 'github.com'],
                                    text=True, stderr=subprocess.DEVNULL).strip()
    headers = {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'Stage-verified-archive'}

    def request(method, url, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        extra = {'Content-Type': 'application/json'} if data is not None else {}
        for attempt in range(3):
            req = urllib.request.Request(url, data=data, method=method, headers={**headers, **extra})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f'GitHub API HTTP{error.code} {method}') from None
                time.sleep(2 + attempt)
            except (OSError, ValueError):
                if attempt == 2:
                    raise RuntimeError(f'GitHub network/JSON failure {method}') from None
                time.sleep(2 + attempt)

    api = f'https://api.github.com/repos/{SLUG}'
    releases = request('GET', api + '/releases?per_page=100')
    release = next((r for r in releases if r['tag_name'] == TAG), None)
    if release is None:
        release = request('POST', api + '/releases', {
            'tag_name': TAG, 'target_commitish': local['source_commit'], 'draft': True,
            'name': 'Stage T179 failure mechanism audit — 2026-10-30',
            'body': 'Stage only. Source/protocols/summaries are in Git. Preserves 12 frozen radius cases, 3 exact contact-provenance rechecks, offline gradient diagnostics, failed outcomes and input hashes. Zero optimizer updates, no new trained model or automatic promotion; original5m remains0/2,T159 incumbent. Per-file SHA256 manifest included. No Phase research, account credentials or chat originals.'})
    assets = request('GET', api + f'/releases/{release["id"]}/assets?per_page=100')
    upload_files = [archive_path, MANIFEST, TRANSFER / 'package_receipt.json']
    verified = []
    for p in upload_files:
        name = p.name
        asset = next((a for a in assets if a['name'] == name), None)
        if asset is None:
            url = f'https://uploads.github.com/repos/{SLUG}/releases/{release["id"]}/assets?name=' + urllib.parse.quote(name)
            with p.open('rb') as stream:
                req = urllib.request.Request(url, data=stream, method='POST', headers={
                    **headers, 'Content-Type': 'application/octet-stream', 'Content-Length': str(p.stat().st_size)})
                try:
                    with urllib.request.urlopen(req, timeout=300) as response:
                        asset = json.load(response)
                except (OSError, ValueError):
                    raise RuntimeError('Asset upload interrupted; inspect server assets before retry') from None
        remote = request('GET', api + f'/releases/assets/{asset["id"]}?verify=' + str(time.time_ns()))
        assert remote['state'] == 'uploaded' and remote['size'] == p.stat().st_size
        digest = sha(p)
        assert remote['digest'] == 'sha256:' + digest, 'Remote content digest mismatch'
        verified.append({'name': name, 'bytes': remote['size'], 'sha256': digest,
                         'asset_id': remote['id'], 'url': remote['browser_download_url'],
                         'verification': 'GitHub server SHA256 and size match local file'})
        print(json.dumps({'verified_asset': name, 'bytes': remote['size']}), flush=True)
    release = request('PATCH', api + f'/releases/{release["id"]}', {'draft': False})
    assert not release['draft']
    receipt = {'state': 'PUBLISHED_VERIFIED', 'url': release['html_url'], 'release_id': release['id'],
               'tag': TAG, 'source_commit': local['source_commit'], 'assets': verified,
               'local_archive_members_verified': local['locally_verified_members'],
               'full_remote_download_verification': False,
               'verification_scope': 'All local decompressed members hashed; server asset SHA256 and size verified. Full remote binary readback not performed.',
               'temporary_archive_deleted_after_verification': False}
    write(receipt_path, receipt)
    assert archive_path.parent == TRANSFER and sha(archive_path) == local['sha256']
    archive_path.unlink()
    receipt['temporary_archive_deleted_after_verification'] = True
    receipt['remaining_new_runs_bytes'] = used()
    write(receipt_path, receipt)
    print(json.dumps({'state': receipt['state'], 'url': receipt['url'],
                      'remaining_new_runs_bytes': receipt['remaining_new_runs_bytes']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['package', 'publish'])
    action = parser.parse_args().action
    package() if action == 'package' else publish()
