"""Encode only fixed-manifest files already in fully read-back old archives.
Original fresh229-file upload failed; do not claim it completed. No extra upload,
no new selection outside manifest, preserve all original bytes locally in gzip.
"""
from compact import *

def main():
    deadline=(DOC/'source_manifest_T300.json').stat().st_mtime+600;assert time.time()<deadline and not LEDGER.exists();lock=(ROOT/'stage/runs/.stage_native_exclusive.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);no_native_process()
    coverage=ROOT/'stage/docs/distance_20261102/T300_EXISTING_BACKUP_COVERAGE.json';data=json.loads(coverage.read_text());assert data['eligible_count']==44;manifest=json.loads((DOC/'source_manifest_T300.json').read_text())['files'];keys={(r['path'],r['bytes'],r['sha256']) for r in manifest};records=[];compressed={};before=used()
    for r in data['eligible']:
        assert (r['path'],r['bytes'],r['sha256']) in keys;receipt=json.loads((ROOT/r['receipt']).read_text());mf=ROOT/r['manifest'];assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and sha(mf.read_bytes())==receipt['manifest_sha256'];assert any((v['path'],v['bytes'],v['sha256'])==(r['path'],r['bytes'],r['sha256']) for v in json.loads(mf.read_text())['files'])
        f=ROOT/r['path'];b=f.read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'];z=gzip.compress(b,compresslevel=6,mtime=0);dest=Path(str(f)+'.gz');assert not dest.exists();compressed[r['path']]=z;records.append(dict(original_path=r['path'],original_bytes=len(b),original_sha256=sha(b),encoded_path=str(dest.relative_to(ROOT)),encoded_bytes=len(z),encoded_sha256=sha(z),prior_complete_backup_receipt=r['receipt'],prior_complete_manifest=r['manifest']))
    ledger=dict(state='VERIFIED_SUBSET_ENCODING_PENDING',requested_files229_fresh_upload_complete=False,requested_files229=229,encoded_subset_files=44,original_upload_error='write timeout, then bounded45s retry408/write timeout; only2freshpackages complete. No original local change until this separately documented existing-backup fallback.',records=records,global_bytes_before=before,physics_steps=0,optimizer_steps=0,unique_bytes_deleted=0,restore='gzip decode inRAM,verify originalSHA,capacityreserve original_bytes before exactpath restore; no endedtraining restart',original_deadline_epoch=deadline,all_original_bytes_retained_locally_and_in_prior_full_readback_verified_archives=True)
    write(LEDGER,ledger)
    for r in records:
        assert time.time()<deadline and used()+r['encoded_bytes']<1024**3;src=ROOT/r['original_path'];dest=ROOT/r['encoded_path'];assert sha(src.read_bytes())==r['original_sha256'];dest.write_bytes(compressed[r['original_path']]);assert sha(dest.read_bytes())==r['encoded_sha256'] and sha(gzip.decompress(dest.read_bytes()))==r['original_sha256'];src.unlink()
    ledger.update(state='VERIFIED_SUBSET_ENCODING_COMPLETE',global_bytes_after=used(),logical_bytes_released=before-used(),completed_epoch=time.time());write(LEDGER,ledger)
    write(DOC/'release_receipt_T300.json',dict(state='SUBSET_PREVIOUS_FULL_READBACK_RECEIPTS_RECONCILED_NOT_NEW229_BACKUP',members=44,requested_fresh229_upload_failed=True,original_byte_sources=sorted({r['prior_complete_backup_receipt'] for r in records}),encoding_ledger_sha256=sha(LEDGER.read_bytes()),new_uploads_by_this_adapter=0,original_manifest_deadline_epoch=deadline,physics_steps=0,optimizer_calls=0));print(json.dumps(dict(state=ledger['state'],files=44,released=ledger['logical_bytes_released'])),flush=True)

if __name__=='__main__':main()
