"""One <=8MiB backup of the13 audited same-policy supplementary successes."""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,signal,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/weekend_20261120_23'
spec=importlib.util.spec_from_file_location('all_stage_stopped',Path(__file__).with_name('archive_after_t695.py'))
Guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(Guard)
read=Guard.read


def main():
    start=time.time();auth=read(DOC/'AUTHORIZATION.json');ledger=read(DOC/'LEDGER.json')
    assert auth['approved'] and ledger['transport_seconds']+180+300<auth['max_transport_seconds']
    Guard.stopped();lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=ROOT/'stage/runs/t695_frozen_neighborhood';assert read(folder/'audit.json')['state']=='PASS'
    source=read(DOC/'t695/SUPPLEMENTARY_SUCCESS_SET.json')
    assert source['safe_count']==13 and len(source['sources'])==13
    retained=read(DOC/'RETAINED_MODEL.json');assert source['policy_hash']==retained['model']['policy_hash']
    files=[Path(row['path']) for row in source['sources']]
    assert len(files)==len(set(files))==13
    for row in source['sources']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    size=sum(f.stat().st_size for f in files);assert size==2791591 and size<=8*2**20
    # Do not retransmit any member already covered by a verified receipt.
    verified=set()
    for item in read(DOC/'RESTORE_INDEX.json')['groups']:
        if read(item['receipt'])['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED':
            verified.update(row['path'] for row in read(item['manifest'])['files'])
    assert all(str(f.relative_to(ROOT)) not in verified and str(f) not in verified for f in files)
    group='WEEKEND_T695_SUPPLEMENTARY_SUCCESS_0';assert not (DOC/'T695_SUCCESS_BACKUP.json').exists()
    client=Guard.Client.client();release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
    Guard.Client.A.stopped=Guard.stopped
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite T695 supplementary backup180s')));signal.alarm(180)
    result=Guard.Client.A.archive(group,files,release,min(start+175,auth['absolute_all_stop']-30),client)
    Guard.stopped();assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json';receipt=manifest.with_name(manifest.name.replace('source_manifest_','release_receipt_'))
    restore=read(DOC/'RESTORE_INDEX.json');assert group not in [g['group'] for g in restore['groups']]
    restore['groups'].append(dict(group=group,manifest=str(manifest),receipt=str(receipt),remote_receipt=result,members=read(manifest)['files']))
    restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True)
    (DOC/'RESTORE_INDEX.json').write_text(json.dumps(restore,indent=2)+'\n')
    value=dict(state=result['state'],group=group,members=result['members'],compressed_bytes=result['bytes'],source_bytes=size,elapsed_seconds=time.time()-start,deleted=0,all13_supplementary_success_full_histories=True,other_raw_pending=True,all_science_driver_helper_units_source_stopped=True)
    (DOC/'T695_SUCCESS_BACKUP.json').write_text(json.dumps(value,indent=2)+'\n')
    signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print(json.dumps(value),flush=True)


if __name__=='__main__':
    started=time.time()
    try:main()
    finally:
        f=DOC/'LEDGER.json';value=read(f);value['transport_seconds']+=time.time()-started;f.write_text(json.dumps(value,indent=2)+'\n')
