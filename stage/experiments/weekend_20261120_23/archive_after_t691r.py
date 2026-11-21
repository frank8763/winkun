"""One <=8MiB RAM-only archive after T691R; includes all actual root/helper units."""
from pathlib import Path
import fcntl,importlib.util,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/weekend_20261120_23'
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
spec=importlib.util.spec_from_file_location('_weekend_t690_transport',ROOT/'stage/experiments/t654_verified_deviation_recovery/transport.py')
Client=importlib.util.module_from_spec(spec);spec.loader.exec_module(Client)
read=lambda f:json.loads(Path(f).read_text())

def stopped():
    ledger=read(DOC/'LEDGER.json');folders={ROOT/'stage/runs/t665_remaining_verification'}
    for item in ledger['jobs']:
        if item.get('out'):folders.add(Path(item['out']))
    for folder in folders:
        assert folder.is_relative_to(ROOT/'stage/runs') and folder.exists(),folder
        seal=read(folder/'SEALED.json');assert seal['all_scientific_processes_stopped'],folder
        pids=set(seal['scientific_pids']);driver=read(folder/'DRIVER_STARTED.json');pids.add(driver['pid'])
        report=read(folder/'FINAL_REPORT.json')
        if report.get('helper_pid'):pids.add(report['helper_pid'])
        assert all(not Path('/proc',str(pid)).exists() for pid in pids),(folder,pids)
    # Includes numerical jobs not launched: their unit must also be inactive.
    for index in [str(i) for i in range(665,693)]+['691r']:
        props=subprocess.check_output(['systemctl','--user','show',f'stage-t{index}.service','-p','MainPID','-p','ActiveState'],text=True)
        fields=dict(line.split('=',1) for line in props.splitlines())
        assert fields['MainPID']=='0' and fields['ActiveState'] in ['inactive','failed'],(index,fields)
    for doc in DOC.glob('t*/AFFINITY_GUARD.json'):
        if doc.parent.name in ['t691r'] or (doc.parent.name[1:].isdigit() and int(doc.parent.name[1:])>=665):
            value=read(doc);assert not Path('/proc',str(value['pid'])).exists(),doc
    # Read only command metadata for own Stage transport/source processes.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(ROOT) in command and ('sync_source.py' in command or '/transport.py' in command):
            raise AssertionError(('Another Stage source/archive process active',proc.name))

def main():
    started=time.time();auth=read(DOC/'AUTHORIZATION.json');ledger=read(DOC/'LEDGER.json')
    assert auth['approved'] and ledger['transport_seconds']+180<auth['max_transport_seconds']
    deadline=min(started+175,auth['absolute_all_stop']-30);stopped()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    Client.A.stopped=stopped
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite raw archive180s')));signal.alarm(180)
    client=Client.client();release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
    verified=set();names=[]
    for manifest in (ROOT/'stage/docs/long_distance_20261101').glob('source_manifest_WEEKEND_T665_PRIMARY_EVIDENCE_*.json'):
        names.append(int(manifest.stem.rsplit('_',1)[1]));receipt=manifest.with_name(manifest.name.replace('source_manifest_','release_receipt_'))
        if receipt.exists() and read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED':verified.update(x['path'] for x in read(manifest)['files'])
    files=[];size=0
    for f in sorted((ROOT/'stage/runs/t665_remaining_verification').glob('*/episodes/*.npz')):
        if str(f.relative_to(ROOT)) in verified:continue
        if size+f.stat().st_size>8*2**20:break
        files.append(f);size+=f.stat().st_size
    assert len(files)>0 and size<=8*2**20
    group=f'WEEKEND_T665_PRIMARY_EVIDENCE_{max(names,default=-1)+1}'
    result=Client.A.archive(group,files,release,deadline,client);stopped()
    assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{group}.json'
    manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{group}.json'
    restore=read(DOC/'RESTORE_INDEX.json');assert group not in [g['group'] for g in restore['groups']]
    restore['groups'].append(dict(group=group,receipt=str(receipt),manifest=str(manifest),remote_receipt=result,members=read(manifest)['files']))
    restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True)
    (DOC/'RESTORE_INDEX.json').write_text(json.dumps(restore,indent=2)+'\n')
    report=dict(state=result['state'],group=group,members=result['members'],compressed_bytes=result['bytes'],source_bytes=size,
        elapsed_seconds=time.time()-started,deleted=0,all_weekend_science_driver_helper_units_checked=True,no_new_optimizer_or_native=True)
    (DOC/f'T665_RAW_BACKUP_{group.rsplit("_",1)[1]}.json').write_text(json.dumps(report,indent=2)+'\n')
    signal.alarm(0);fcntl.flock(lock,fcntl.LOCK_UN);lock.close();print(json.dumps(report),flush=True)

if __name__=='__main__':
    started=time.time()
    try:main()
    finally:
        f=DOC/'LEDGER.json';value=read(f);value['transport_seconds']+=time.time()-started;f.write_text(json.dumps(value,indent=2)+'\n')
