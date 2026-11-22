"""Read back exactly two fixed historical successful NPZ members, no science."""
from pathlib import Path
import importlib.util, json, os, signal, sys, time, hashlib
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/weekend_20261120_23/t697'
sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'))
import archive_after_t695 as Guard
spec=importlib.util.spec_from_file_location('t697_oldstream',ROOT/'stage/experiments/t626_multitask_closed_loop/t626_stream.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

def main():
    assert not (DOC/'RESTORE_REPORT.json').exists()
    Guard.stopped()
    ledger=Guard.read(DOC.parent/'LEDGER.json');auth=Guard.read(DOC.parent/'AUTHORIZATION.json')
    assert ledger['transport_seconds']+180+300<auth['max_transport_seconds']
    assert time.time()+180<auth['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed historical restore180s')))
    signal.alarm(180)
    selection=Guard.read(DOC/'SOURCE_INVENTORY.json')['selected'];client=Guard.Client.client();results=[]
    folder=ROOT/'stage/runs/t697_verified_inputs';folder.mkdir(exist_ok=False)
    for row in selection:
        receipt=Guard.read(row['archive_receipt']);assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        pieces=[]
        for part in receipt['parts']:
            asset=client.B.api('GET',f"releases/assets/{part['asset_id']}")
            blob=client.readback(asset,part['bytes']);assert S.digest(blob)==part['sha256'];pieces.append(blob)
        blob=b''.join(pieces);assert S.digest(blob)==receipt['sha256']
        files,manifest=S.unpack(blob)
        rel=str(Path(row['original_path']).relative_to(ROOT));data=files[rel]
        assert S.digest(data)==row['original_sha256']
        path=Path(row['restored_path']);assert path.parent==folder
        with path.open('xb') as f:f.write(data)
        results.append(dict(id=row['id'],path=str(path),original_path=row['original_path'],sha256=S.digest(data),bytes=len(data),receipt=row['archive_receipt'],archive_bytes=len(blob),entire_archive_every_member_verified=True))
    Guard.stopped();signal.alarm(0)
    (DOC/'RESTORE_REPORT.json').write_text(json.dumps(dict(state='EXACT_ORIGINAL_TWO_SOURCE_BYTES_REMOTE_READBACK_VERIFIED',rows=results,new_native=0,new_optimizer=0,new_Actor=0),indent=2)+'\n')
    print(json.dumps(dict(state='PASS',members=len(results),bytes=sum(x['bytes'] for x in results))),flush=True)

if __name__=='__main__':
    started=time.time()
    try:main()
    finally:
        f=DOC.parent/'LEDGER.json';v=Guard.read(f);v['transport_seconds']+=time.time()-started;f.write_text(json.dumps(v,indent=2)+'\n')
