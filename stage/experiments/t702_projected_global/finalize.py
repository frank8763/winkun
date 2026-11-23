"""One-shot post-exit analysis, named evidence readback and final source."""
from pathlib import Path
import fcntl,importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t702_final_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
OUT,DOC,read,save,sha=C.OUT,C.DOC,C.read,C.save,C.sha
GROUP='DAY_T702_PRIMARY_0'

def charge(commit):
    p=DOC/'TRANSPORT_LEDGER.json';l=read(p)
    if commit not in l['source_commits']:
        assert l['charged_seconds']+300<=l['cap'];l['charged_seconds']+=300;l['source_commits'].append(commit);save(p,l)

def source():C.sync(final=True)

def backup():
    C.stopped();start=time.time();l=read(DOC/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+180+300<=l['cap']
    choices=[OUT/f for f in ['fit/proj8.pt.gz','fit/summary.json','fit/audit.json','fit/fit_calls.jsonl','FIT_LEDGER.json','LEDGER.json','SEALED.json','FINAL_REPORT.json','status.json']]
    for name in ['known_proj8','new_proj8']:choices += [OUT/name/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    files=[p for p in choices if p.exists()];missing=[str(p.relative_to(ROOT)) for p in choices if not p.exists()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files};size=sum(p.stat().st_size for p in files)
    assert size<=8*2**20,'One fixed primary set exceeds8MiB; retain local, no split or rescue'
    save(DOC/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),source_bytes=size,missing=missing))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T702 primary180')));signal.alarm(175)
    try:
        sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
        G.stopped=G.Client.A.stopped=C.stopped;client=G.Client.client();restore=read(G.DOC/'RESTORE_INDEX.json')
        assert not any(g['group']==GROUP for g in restore['groups'])
        with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
            result=G.Client.A.archive(GROUP,files,release,min(start+170,read(OUT/'protocol.json')['absolute_all_stop']-330),client)
            C.stopped();assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json';receipt=ROOT/f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json'
            members=read(manifest)['files'];assert {r['path']:r['sha256'] for r in members}==hashes
            assert all(sha(ROOT/r['path'])==r['sha256'] and (ROOT/r['path']).stat().st_size==r['bytes'] for r in members)
            restore['groups'].append(dict(group=GROUP,receipt=str(receipt),manifest=str(manifest),remote_receipt=result,members=members))
            restore.update(epoch=time.time(),files_deleted=0,other_raw_pending=True);save(G.DOC/'RESTORE_INDEX.json',restore)
            save(DOC/'PRIMARY_BACKUP.json',dict(state=result['state'],members=len(members),source_bytes=size,compressed_bytes=result['bytes'],missing=missing,deleted=0))
    finally:
        signal.alarm(0);l=read(DOC/'TRANSPORT_LEDGER.json');l['charged_seconds']+=time.time()-start
        assert l['charged_seconds']+300<=l['cap'];save(DOC/'TRANSPORT_LEDGER.json',l)

def analysis():
    rows=[]
    for name in ['known_proj8','new_proj8']:
        sf,af=OUT/name/'summary.json',OUT/name/'audit.json';s=read(sf) if sf.exists() else {};a=read(af) if af.exists() else {}
        rows.append(dict(block=name,state=s.get('state','MISSING'),audit=a.get('state','MISSING'),success=s.get('success'),total=s.get('total'),
            core=s.get('core_success'),lost403=len(s['lost403']) if s.get('lost403') is not None else None,source_success=len(s.get('source_success_ids',[]))))
    fit=read(OUT/'fit/summary.json') if (OUT/'fit/summary.json').exists() else None
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],rows=rows,fit=fit,retained='T698',new78_exposed_regression=True,
        raw='ALL_LOCAL_RETAINED_PRIMARY_ONLY_REMOTE_IF_RECEIPT_PASS',no_novelty_or_unknown_map_claim=True,epoch=time.time())
    save(DOC/'FINDINGS.json',report)
    lines=['# T702 fixed projected/global-radius comparison','',f"State: {report['state']}. T698 remains retained. Original tasks not restarted.",'',
        '| Block | Audit | Success | Core407 | Lost403 | Sources |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['block']} | {r['audit']} | {r['success']}/{r['total']} | {r['core']} | {r['lost403']} | {r['source_success']} |")
    lines += ['','Historical T696:396core/0oldloss/47local; T701quant32:396/0/45 at1320120B added; refined108:387/11/53 at161584B added.',
        'This candidate has a pre-fixed8D projection and one global radius, nominal150744B added, unchanged2621976B common parent. Actual tensor/latency/RSS and numerical construction gates are in fit metadata.',
        'Projection plus shared radius is a combined simpler counterexample, not an attribution to either factor. Projected supports extend along the nullspace; finite-history command preservation gives no neighborhood or unknown-map guarantee.',
        'All missing/failed audits remain explicit. Known500 saves full69+compact431; local85 saves full78+compact7. Independent audit checks actual saved payloads; it does not reintegrate every trajectory. No blind-test, capacity lower-bound, BC impossibility, RL or novelty claim.',
        'All original models, failures and local raw retained; remote coverage is exactly the verified PRIMARY_BACKUP receipt.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-23 T702单一8维投影/共享半径对照结束：{report['state']}，原始结果与独立审计见day_20261123/t702/FINDINGS。T698保持；同T694父链控制研究、固定78已曝光，不自动下一任务。\n\n"+f.read_text())

def watch():
    p=read(OUT/'protocol.json');save(DOC/'FINALIZER_STARTED.json',dict(pid=os.getpid(),one_shot=True,epoch=time.time(),absolute_stop=p['absolute_all_stop']))
    while time.time()<p['absolute_all_stop']-10:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();analysis();pending=read(DOC/'TRANSPORT_PENDING.json');charge(pending['remote_commit']);pending.update(accounted=True);save(DOC/'TRANSPORT_PENDING.json',pending)
    try:backup()
    except BaseException:save(DOC/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,no_retry=True))
    C.stopped()
    paths=['stage/docs/day_20261123/t702','stage/experiments/t702_projected_global','stage/docs/RESEARCH_STATUS.md',
        'stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']
    for name in [f'stage/docs/long_distance_20261101/release_receipt_{GROUP}.json',f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json']:
        if (ROOT/name).exists():paths.append(name)
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Record T702 finite capacity falsifier and primary evidence'],cwd=ROOT,check=True)
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (DOC/'final_source.log').open('xb') as log:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=log,stderr=log)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==target;charge(r['remote_commit'])
        save(DOC/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_SOURCE300_ACCOUNTED',remote_commit=r['remote_commit'],root_tree=target))
    else:
        l=read(DOC/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;save(DOC/'TRANSPORT_LEDGER.json',l)
        save(DOC/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_PRESERVED',returncode=code,no_retry=True))
    save(DOC/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_next_science_or_recurring_automation=True))

if __name__=='__main__':
    try:source() if sys.argv[1]=='source' else watch()
    except BaseException:save(DOC/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),epoch=time.time(),all_local_preserved=True));raise
