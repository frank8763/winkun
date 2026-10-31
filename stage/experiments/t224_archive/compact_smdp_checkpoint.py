"""Byte-exact lossless local storage for one ended historical T167 checkpoint."""
from archive import *
def main():
    receipt=DOC/'smdp_storage_receipt.json';assert not receipt.exists(),'Inspect completed compaction'
    name='stage/runs/t167_smdp/learn/checkpoint_latest.pt';p=ROOT/name
    # T167 completed its locked256000-frame run and frozen acceptance; never resumed. This is only
    # storage encoding, preserving Actor/Critic/replay/optimizer bytes unchanged.
    manifest=json.loads((ROOT/'stage/manifests/progress_20261030_files.json').read_text());row=next(r for r in manifest['files'] if r['path']==name)
    old=json.loads((ROOT/'stage/docs/results_20261030/release_receipt.json').read_text());a=old['assets'][0]
    status=json.loads((p.parent/'status.json').read_text());pipeline=json.loads((p.parent.parent/'pipeline_status.json').read_text());assert status['state']=='COMPLETED' and pipeline['state']=='COMPLETE' and not pipeline['automatic_promotion']
    raw=p.read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
    packed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(packed)==raw and len(packed)<len(raw)
    print(json.dumps(dict(state='LOCAL_LOSSLESS_ENCODING_VERIFIED',saved_MiB=(len(raw)-len(packed))/1024**2)),flush=True)
    remote=req('GET',f"https://api.github.com/repos/{SLUG}/releases/assets/{a['asset_id']}",headers={'Accept':'application/octet-stream'});assert sha(remote)==a['sha256'] and len(remote)==a['bytes']
    process=subprocess.run(['zstd','-d','-q','-c'],input=remote,capture_output=True,check=True);found=False
    with tarfile.open(fileobj=io.BytesIO(process.stdout),mode='r:') as archive:
        for member in archive:
            if member.name==name:assert archive.extractfile(member).read()==raw;found=True
    assert found
    dest=p.with_suffix('.pt.gz');assert not dest.exists()
    record=dict(original_path=name,original_bytes=len(raw),original_sha256=sha(raw),compressed_path=str(dest.relative_to(ROOT)),compressed_bytes=len(packed),compressed_sha256=sha(packed),remote_original_asset=a['url'])
    write(receipt,dict(state='VERIFIED_BEFORE_ENCODING_RELOCATION',records=[record],full_remote_original_archive_sha_verified=True,original_model_member_byte_exact_verified=True,reason='EndedT167checkpointneverresumed; all original Actor/Critic/replay/optimizerpayload preserved locally gzip and verified remote raw archive. CurrentretainedT159/T194/T221/T228/T231/T239untouched.',restore='archive.py restore accepts all storage ledgers after this source update; budget checked'))
    assert sha(p.read_bytes())==record['original_sha256'];p.unlink();dest.write_bytes(packed);assert sha(gzip.decompress(dest.read_bytes()))==record['original_sha256']
    data=json.loads(receipt.read_text());data.update(state='LOSSLESS_LOCAL_COMPACTION_COMPLETE',logical_bytes_freed=len(raw)-len(packed),original_payload_preserved_locally_and_remotely=True);write(receipt,data)
    print(json.dumps(dict(state=data['state'],bytes_freed=data['logical_bytes_freed'])),flush=True)
if __name__=='__main__':main()
