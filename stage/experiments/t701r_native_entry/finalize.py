"""One post-exit60s source save; models already backed up, new raw local."""
from pathlib import Path
import importlib.util,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
# Historical imports may own generic run: load the controller by explicit path.
spec=importlib.util.spec_from_file_location('t701r_finish_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
DOC,OUT,read,save=C.DOC,C.OUT,C.read,C.save

def charge(commit):
    f=C.C.WEEKEND/'LEDGER.json';l=read(f)
    if commit not in l['charged_source_sync_commits']:
        assert l['transport_seconds']+60<=18000;l['transport_seconds']+=60
        l['charged_source_sync_commits'].append(commit);l['transport_remote_commits_accounted'].append(commit);save(f,l)

def source():
    C.stopped();p=read(OUT/'protocol.json');l=read(C.C.WEEKEND/'LEDGER.json')
    assert l['transport_seconds']+60<=18000 and time.time()+60<p['absolute_all_stop']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Final source60')));signal.alarm(55)
    sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as Archive
    Archive.stopped=C.stopped;Archive.Client.client()
    sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'));import t626_sync_source
    t626_sync_source.main();C.stopped();signal.alarm(0)

def watch():
    p=read(OUT/'protocol.json');save(DOC/'FINALIZER_STARTED.json',dict(pid=os.getpid(),one_shot=True,epoch=time.time()))
    while time.time()<p['absolute_all_stop']-10:
        props=subprocess.check_output(['systemctl','--user','show','stage-t701r.service','-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();rows=[]
    for name in ['known_quant32','new_quant32','known_replay_refined108','new_replay_refined108']:
        sf=OUT/name/'summary.json';af=OUT/name/'audit.json'
        s=read(sf) if sf.exists() else {};a=read(af) if af.exists() else {}
        rows.append(dict(block=name,state=s.get('state','MISSING_NOT_ZERO_FAILURE'),audit=a.get('state','MISSING'),
            success=s.get('success'),total=s.get('total'),core=s.get('core_success'),
            lost403=len(s['lost403']) if s.get('lost403') is not None else None,source_success=len(s.get('source_success_ids',[]))))
    report=dict(state=read(OUT/'FINAL_REPORT.json')['state'],rows=rows,new_Actor=0,new_optimizer=0,
        original_T701_stays_FAILED=True,retained='T698',models_primary_remote_verified=True,new_native_raw='LOCAL_RETAINED_NOT_REMOTE_BACKED_UP',epoch=time.time())
    save(DOC/'FINDINGS.json',report)
    lines=['# T701R original fixed comparison completed or explicitly missing','',f"State: {report['state']}. Original T701 helper failure unchanged; fitted models and independent audit reused by SHA, zero new training.",'',
        '| Block | Audit | Success | Core | Lost403 |','|---|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['block']} | {r['audit']} | {r['success']}/{r['total']} | {r['core']} | {r['lost403']} |")
    lines += ['','Compare T696 historical396core/0old loss/47new, small replay historical382core/16old loss/52new. New78 is now an exposed same-map regression set, not blind or new-map generalization.',
        'Original T701 fit metrics: quant32 added1320120B, offline403+7 owncarry commands exact; refined108 added161584B,80 full-batch gradient evaluations, budget reached rather than proven converged. See original fit/audit and DAY_T701_PRIMARY_0 for verified model evidence.',
        'Same finite retention requirements and all failures reported; no capacity lower bound, universal BC failure, method novelty or top-journal claim. T698 still retained. New native raw stays local because transport is limited; no original model/raw deletion.']
    (DOC/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-23 T701R固定四块原生验收结束：{report['state']}，0新拟合，逐块独立审计与核心/旧403保持/固定78见day_20261123/t701r/FINDINGS。原T701助手解释器FAILED保留；模型/fit已DAY_T701_PRIMARY_0远端核验，新native raw仅本地未称归档。T698保持，不自动下一实验。\n\n"+f.read_text())
    pending=read(DOC/'TRANSPORT_PENDING.json');charge(pending['remote_commit']);pending.update(accounted=True);save(DOC/'TRANSPORT_PENDING.json',pending)
    paths=['stage/docs/day_20261123','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/experiments/t701r_native_entry']
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Record T701R guarded native results and original failure'],cwd=ROOT,check=True)
    target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (DOC/'final_source.log').open('xb') as log:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=log,stderr=log)
        try:code=child.wait(timeout=60)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        r=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert r['root_tree']==target;charge(r['remote_commit'])
        save(DOC/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_SOURCE60_ACCOUNTED',remote_commit=r['remote_commit'],root_tree=target))
    else:
        l=read(C.C.WEEKEND/'LEDGER.json');l['transport_seconds']+=60;save(C.C.WEEKEND/'LEDGER.json',l)
        save(DOC/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_METADATA_RETAINED',returncode=code,no_retry=True))
    save(DOC/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_new_science_or_recurring_automation=True))

if __name__=='__main__':
    try:source() if sys.argv[1]=='source' else watch()
    except BaseException:
        save(DOC/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),epoch=time.time(),no_retry=True));raise
