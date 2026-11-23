"""One-shot resource watch and post-exit evidence save; never science/retries."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t701_postexit_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
DOC,OUT,read,save,sha=C.DOC,C.OUT,C.read,C.save,C.sha
GROUP='DAY_T701_PRIMARY_0'

def charge(commit,amount=300):
    path=C.C.WEEKEND/'LEDGER.json';l=read(path)
    if commit not in l['charged_source_sync_commits']:
        assert l['transport_seconds']+amount<=18000
        l['transport_seconds']+=amount;l['charged_source_sync_commits'].append(commit);l['transport_remote_commits_accounted'].append(commit)
        save(path,l)

def source():
    C.stopped();p=read(OUT/'protocol.json');l=read(C.C.WEEKEND/'LEDGER.json')
    assert l['transport_seconds']+300<=18000 and time.time()+300<p['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Final single source300')));signal.alarm(295)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Archive
    Archive.stopped=C.stopped;Archive.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();C.stopped();signal.alarm(0)

def primary_backup():
    start=time.time();p=read(OUT/'protocol.json');l=read(C.C.WEEKEND/'LEDGER.json')
    assert l['transport_seconds']+120+300<=18000 and time.time()+450<p['absolute_all_stop']
    C.stopped();sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Guard
    restore=read(Guard.DOC/'RESTORE_INDEX.json')
    if any(g['group']==GROUP for g in restore['groups']):return
    choices=[OUT/f for f in ['protocol.json','fit/quant32.pt.gz','fit/replay_refined108.pt.gz','fit/summary.json','fit/audit.json',
         'fit/fit_calls.jsonl','FIT_LEDGER.json','LEDGER.json','FINAL_REPORT.json','SEALED.json','status.json']]
    for name in ['known_quant32','new_quant32','known_replay_refined108','new_replay_refined108']:
        choices += [OUT/name/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    choices += [ROOT/'stage/runs/t700_plain_external_review/audit.json',ROOT/'stage/runs/t700r_import_isolation/audit.json']
    files=[f for f in choices if f.exists()];missing=[str(f.relative_to(ROOT)) for f in choices if not f.exists()]
    size=sum(f.stat().st_size for f in files)
    assert size<=8*2**20,'Primary fixed set exceeds one8MiB batch; retain local, no splitting/rescue'
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),source_bytes=size,missing=missing,selection_fixed_before_completion=True))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('One primary archive120s')));signal.alarm(120)
    try:
        with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);Guard.Client.A.stopped=C.stopped
            client=Guard.Client.client();release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
            result=Guard.Client.A.archive(GROUP,files,release,min(start+115,p['absolute_all_stop']-330),client)
            C.stopped();assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json'
            receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json'
            members=read(manifest)['files'];assert {r['path']:r['sha256'] for r in members}==hashes
            assert all(sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==r['bytes'] for r in members)
            restore['groups'].append(dict(group=GROUP,receipt=str(receipt),manifest=str(manifest),remote_receipt=result,members=members))
            restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True);save(Guard.DOC/'RESTORE_INDEX.json',restore)
            save(DOC/'PRIMARY_BACKUP.json',dict(state=result['state'],members=len(members),source_bytes=size,compressed_bytes=result['bytes'],missing=missing,deleted=0))
    finally:
        signal.alarm(0);l=read(C.C.WEEKEND/'LEDGER.json');l['transport_seconds']+=time.time()-start
        assert l['transport_seconds']+300<=18000;save(C.C.WEEKEND/'LEDGER.json',l)

def analysis():
    rows=[]
    for folder in ['known_quant32','new_quant32','known_replay_refined108','new_replay_refined108']:
        s=OUT/folder/'summary.json';a=OUT/folder/'audit.json'
        if s.exists():
            x=read(s);rows.append(dict(block=folder,state=x['state'],verified=a.exists() and read(a)['state']=='PASS',
                success=x.get('success'),total=x.get('total'),core=x.get('core_success'),
                lost403=len(x['lost403']) if x.get('lost403') is not None else None,source_success=len(x.get('source_success_ids',[])),
                summary_sha256=sha(s),audit_sha256=sha(a) if a.exists() else None))
        else:rows.append(dict(block=folder,state='MISSING_NOT_ZERO_FAILURE',verified=False))
    fit=read(OUT/'fit/summary.json') if (OUT/'fit/summary.json').exists() else None
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],rows=rows,fit=fit,retained='T698',
                no_novelty_or_unknown_map_claim=True,new78_is_exposed_fixed_regression_set=True,
                all_units_and_processes_stopped_guard=True,epoch=time.time())
    save(DOC/'FINDINGS.json',report)
    lines=['# T701 bounded compact comparison','',f"Scientific state: {report['state']}. T698 remains retained. All scientific processes/helper/driver/unit stopped, guarded independently.",'',
           '| Block | Verified | Success | Core | Lost old403 |','|---|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['block']} | {row['verified']} | {row.get('success')} / {row.get('total')} | {row.get('core')} | {row.get('lost403')} |")
    lines += ['','Original T699 and T700 failures stay unchanged; T700R independently reviewed plain391/407 with7old losses under new Monday authorization.',
              'Compare with T696396core/0old loss/47new and historical small replay382core/16old loss/52new. Those are historical references, not new independent runs.',
              'Compressed center support/source-command gate is finite memory compatibility, not generalization. Refined replay is one capped optimization pilot; budget exhaustion is not a BC-capacity impossibility proof.',
              'Full loss/gradient curve, exhaustion flag, tensor bytes, decoded workspace, batch1 latency and peak process RSS are in FINDINGS.json. Peak RSS includes training cache and Python. New78 is already exposed and is a fixed same-map regression/transfer set, not an untouched holdout.',
              'A positive result does not prove minimum capacity, algorithm novelty or top-journal readiness. Missing/failed checks are never successes or zero-percent navigation rates. All local raw/models preserved; only PRIMARY_BACKUP receipt can establish named remote coverage.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')

def watch():
    p=read(OUT/'protocol.json');assert time.time()<p['absolute_all_stop']
    save(DOC/'FINALIZER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),one_shot=True,absolute_stop=p['absolute_all_stop']))
    fixed=set();real_affinity=os.sched_setaffinity
    # Historical imports wrap Python affinity; the documented syscall remains
    # available as the native guard's original function for metadata-only repair.
    real_affinity=C.N.N.G._real_affinity
    while time.time()<p['absolute_all_stop']-10:
        props=subprocess.check_output(['systemctl','--user','show','stage-t701.service','-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        status=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        if status.get('child')=='fit' and status.get('child_pid'):
            pid=status['child_pid']
            try:
                C.C.resources();assert '/stage-t701.service' in Path('/proc',str(pid),'cgroup').read_text()
                for item in Path('/proc',str(pid),'task').iterdir():
                    tid=int(item.name);mask=os.sched_getaffinity(tid)
                    if mask=={14,15}:
                        real_affinity(tid,set(range(8,16)));assert os.sched_getaffinity(tid)==set(range(8,16));fixed.add(tid)
            except (FileNotFoundError,ProcessLookupError):pass
            except AssertionError as error:
                save(DOC/'RESOURCE_WATCH.json',dict(state='RESOURCE_CONFLICT_OR_REPAIR_REJECTED',error=str(error),epoch=time.time()))
        time.sleep(5)
    C.stopped();analysis()
    pending=read(DOC/'TRANSPORT_PENDING.json');charge(pending['remote_commit']);pending.update(accounted=True,state='ACCOUNTED_AFTER_ALL_SCIENCE_EXIT');save(DOC/'TRANSPORT_PENDING.json',pending)
    try:primary_backup()
    except BaseException:save(DOC/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_raw_retained=True,no_retry=True))
    C.stopped();save(DOC/'FINALIZER_PROGRESS.json',dict(state='SCIENTIFICALLY_STOPPED_FINAL_SOURCE_PENDING',repaired_thread_ids=sorted(fixed),epoch=time.time()))
    # Only task-owned metadata/source are staged. Never include raw binaries or
    # unrelated existing edits; exact source sync rejects other tracked dirt.
    paths=['stage/docs/day_20261123','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json',
           'stage/docs/RESEARCH_STATUS.md','stage/experiments/t701_compact_retention/finalize.py']
    for name in [f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json',f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json']:
        if (ROOT/name).exists():paths.append(name)
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Record guarded T701 compact comparison and primary evidence'],cwd=ROOT,check=True)
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    start=time.time()
    with (DOC/'final_source.log').open('xb') as log:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=log,stderr=log)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==target
        charge(r['remote_commit']);save(DOC/'FINAL_SOURCE.json',dict(state='ACTUAL_SHELL_EXIT0_TREE_EXACT_SOURCE300_UNIQUE',remote_commit=r['remote_commit'],root_tree=target,accounted=True))
    else:
        l=read(C.C.WEEKEND/'LEDGER.json');l['transport_seconds']+=min(300,time.time()-start);save(C.C.WEEKEND/'LEDGER.json',l)
        save(DOC/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_METADATA_RETAINED',returncode=code,no_retry=True))
    save(DOC/'FINALIZER_DONE.json',dict(state='STOPPED',science=read(OUT/'FINAL_REPORT.json')['state'],epoch=time.time(),no_more_work=True,no_recurring_automation=True))

if __name__=='__main__':
    if sys.argv[1]=='source':source()
    else:
        try:watch()
        except BaseException:
            save(DOC/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),all_local_raw_retained=True,epoch=time.time(),no_retry=True));raise
