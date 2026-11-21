"""One finite RAM-only retained-model primary-evidence backup, no deletion."""
from pathlib import Path
import fcntl,importlib.util,json,signal,sys,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/weekend_20261120_23'
spec=importlib.util.spec_from_file_location('all_stage_stopped',Path(__file__).with_name('archive_after_t694r.py'))
Guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(Guard)
read=Guard.read


def main():
    start=time.time();auth=read(DOC/'AUTHORIZATION.json');ledger=read(DOC/'LEDGER.json')
    assert auth['approved'] and ledger['transport_seconds']+180<auth['max_transport_seconds']
    Guard.stopped();lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=ROOT/'stage/runs/t694_full_history_compatibility';recovered=ROOT/'stage/runs/t694r_audit_recovery'
    assert read(recovered/'audit.json')['state']=='PASS'
    files=[parent/name for name in ['fit/proposal.pt.gz','protocol.json','fit/LEARNING.json','fit/audit.json','fit/DATA_MANIFEST.json','fit/fit_calls.jsonl',
        'evaluation/JOINT407.json.gz','evaluation/summary.json','evaluation/SCREEN.json','evaluation/cases.jsonl.gz','SEALED.json','FINAL_REPORT.json']]
    files+=[recovered/name for name in ['audit.json','FINAL_REPORT.json','SEALED.json']]
    size=sum(f.stat().st_size for f in files);assert size<=8*2**20
    group='WEEKEND_T694_RETAINED_PRIMARY_0';assert not (DOC/'T694_RETAINED_BACKUP.json').exists()
    client=Guard.Client.client();release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
    Guard.Client.A.stopped=Guard.stopped
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite retainedmodel backup180s')));signal.alarm(180)
    result=Guard.Client.A.archive(group,files,release,min(start+175,auth['absolute_all_stop']-30),client)
    Guard.stopped();assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json';receipt=manifest.with_name(manifest.name.replace('source_manifest_','release_receipt_'))
    restore=read(DOC/'RESTORE_INDEX.json');assert group not in [g['group'] for g in restore['groups']]
    restore['groups'].append(dict(group=group,manifest=str(manifest),receipt=str(receipt),remote_receipt=result,members=read(manifest)['files']))
    restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True)
    (DOC/'RESTORE_INDEX.json').write_text(json.dumps(restore,indent=2)+'\n')
    report=dict(state=result['state'],group=group,members=result['members'],compressed_bytes=result['bytes'],source_bytes=size,
                elapsed_seconds=time.time()-start,deleted=0,named_retained_model_included=True,all_scientific_PIDs_units_and_source_stopped=True)
    (DOC/'T694_RETAINED_BACKUP.json').write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print(json.dumps(report),flush=True)


if __name__=='__main__':
    started=time.time()
    try:main()
    finally:
        f=DOC/'LEDGER.json';value=read(f);value['transport_seconds']+=time.time()-started;f.write_text(json.dumps(value,indent=2)+'\n')
