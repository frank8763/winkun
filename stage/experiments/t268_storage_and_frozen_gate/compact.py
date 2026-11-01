"""Lossless encoding of backed-up, stopped Stage text logs; no navigation."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import time

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/long_distance_20261101'
GROUPS={'T259':'t259_closed_loop_recovery','T263':'t263_adaptive_goal_course','T267':'t267_causal_credit_course'}
RECEIPT=DOC/'T268_LOG_ENCODING_RECEIPT.json'


def sha(data):return hashlib.sha256(data).hexdigest()


def used():return sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())


def main():
    assert not RECEIPT.exists(),'No duplicate encoding; use restoration ledger'
    records=[];packed={};before=used()
    for group,name in GROUPS.items():
        directory=ROOT/'stage/runs'/name
        receipt=json.loads((DOC/f'release_receipt_{group}.json').read_text())
        manifestpath=DOC/f'source_manifest_{group}.json'
        assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        assert sha(manifestpath.read_bytes())==receipt['manifest_sha256']
        manifest={r['path']:r for r in json.loads(manifestpath.read_text())['files']}
        assert json.loads((directory/'summary.json').read_text())['all_workers_stopped']
        workers=json.loads((directory/'workers.json').read_text())
        for pid in [workers['parent'],*workers['workers']]:
            try:os.kill(pid,0)
            except ProcessLookupError:continue
            raise AssertionError('Process exists or reused: '+str(pid))
        for p in sorted(directory.glob('*.jsonl')):
            # Later curricula read this log directly as an immutable input.
            if p.name=='training.jsonl':continue
            b=p.read_bytes();relative=str(p.relative_to(ROOT));m=manifest[relative]
            assert len(b)==m['bytes'] and sha(b)==m['sha256']
            z=gzip.compress(b,compresslevel=6,mtime=0);assert gzip.decompress(z)==b
            dest=p.with_suffix('.jsonl.gz');assert not dest.exists()
            assert len(z)<len(b)
            packed[relative]=z
            records.append(dict(original_path=relative,original_bytes=len(b),original_sha256=sha(b),
                encoded_path=str(dest.relative_to(ROOT)),encoded_bytes=len(z),encoded_sha256=sha(z),
                original_remote_asset=receipt['asset_url']))
    ledger=dict(state='PREVERIFIED_PENDING_ENCODING',physics_steps=0,optimizer_calls=0,
        all_original_payloads_retained_losslessly=True,records=records,global_bytes_before=before,
        restore='gzip.decompress(Path(encoded_path).read_bytes()); verify original_sha256; reserve original bytes within1GiB before writing original_path. Never rerun old experiments.',
        warning='Ended historical jsonl files now have .gz suffix. Consumers must explicitly read gzip or restore with capacity check; original hashes and remote tar members unchanged.')
    RECEIPT.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    for r in records:
        p=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];b=p.read_bytes()
        assert sha(b)==r['original_sha256']
        assert used()+r['encoded_bytes']<1024**3 and shutil.disk_usage(ROOT).free>=40*1024**3
        dest.write_bytes(packed[r['original_path']])
        assert sha(gzip.decompress(dest.read_bytes()))==r['original_sha256']
        p.unlink()  # Every byte remains locally in verified gzip and in verified remote tar.
    ledger.update(state='LOSSLESS_LOG_ENCODING_COMPLETE',global_bytes_after=used(),
        logical_bytes_released=before-used(),completed_epoch=time.time())
    assert ledger['global_bytes_after']<1024**3
    RECEIPT.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:ledger[k] for k in ['state','logical_bytes_released','global_bytes_before','global_bytes_after']}))


if __name__=='__main__':main()
