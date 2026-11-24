"""Bounded post-exit recovery qualification, receipts and no further learning."""
from pathlib import Path
import importlib.util,json,gzip,time,subprocess,sys,signal,traceback,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t720_final_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
D,O=C.DOC,C.OUT
GROUP='DAY_T720_COMPLETE_EXPOSED_PANELS_0'
def charge(commit):
    x=C.read(D/'TRANSPORT_LEDGER.json')
    if commit not in x['source_commits']:x['source_commits'].append(commit);x['charged_seconds']+=300;assert x['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',x)
def analysis():
    p=C.root_plan();selection=C.read(D/'FIXED_SELECTION.json');final=C.read(O/'FINAL_REPORT.json');complete=final['state']=='COMPLETE';panels={};union={};baseline={};sourceids={r['job_id'] for v in selection['panels'].values() for r in v['reused_rows']}
    for name,panel in selection['panels'].items():
        folder=O/name;sm=C.read(folder/'summary.json') if (folder/'summary.json').exists() else {};au=C.read(folder/'audit.json') if (folder/'audit.json').exists() else {};valid=sm.get('state')=='COMPLETE' and au.get('state')=='PASS';complete &=valid
        rows=[json.loads(v) for v in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (folder/'cases.jsonl.gz').exists() else [];by={r['job_id']:r for r in panel['reused_rows']+rows if not r.get('exact')};base={r['job_id']:r for r in panel['baseline_rows']};union.update(by);baseline.update(base)
        safe={k for k,r in by.items() if C.M.N.T.Prior.safe(r)};old={k for k,r in base.items() if C.M.N.T.Prior.safe(r)}
        panels[name]=dict(state=sm.get('state','MISSING'),audit=au.get('state','MISSING'),total=len(panel['all_jobs']),complete=len(by)==len(panel['all_jobs']) and valid,safe=len(safe) if valid else None,baseline_safe=len(old),gained=sorted(safe-old),lost=sorted(old-safe),collision_ids=sorted(k for k,r in by.items() if r['collision']),baseline_collision_ids=sorted(k for k,r in base.items() if r['collision']),timeout_ids=sorted(k for k,r in by.items() if not r['success'] and not r['collision']),boundary_ids=sorted(k for k,r in by.items() if 'fullfield_start_pose' in r['request']['job'] and r.get('field_boundary')),reused_original=len(panel['reused_rows']),reused_exact=len(panel['reused_exact_rows']),success_ids=sorted(safe))
    safe={k for k,r in union.items() if C.M.N.T.Prior.safe(r)};old={k for k,r in baseline.items() if C.M.N.T.Prior.safe(r)}
    report=dict(state=final['state'],complete_panel=complete,panels=panels,total=174,safe=len(safe) if complete else None,baseline_safe=len(old),gained=sorted(safe-old),lost=sorted(old-safe),source18_safe=len(safe&sourceids) if complete else None,remaining156_safe=len(safe-sourceids) if complete else None,baseline_remaining156_safe=len(old-sourceids),collision=sum(r['collision'] for r in union.values()),baseline_collision=sum(r['collision'] for r in baseline.values()),old403='NOT_MEASURED',core407='NOT_MEASURED',T719_acquisition_gate_still_failed=True,retained='T698',integration='NOT_RUN',new_learning=0,all174_exposed_conditions_not_independent_worlds=True,model=selection['model'],cost=C.read(O/'LEDGER.json'),epoch=time.time());C.save(D/'PAIRED_RESULTS.json',report)
    text=['# T720 complete exposed panels for frozen T719','',f"State {final['state']}; complete panel {complete};174ID safe {report['safe']} versus B0 {report['baseline_safe']}.",'','| Fixed panel | T719 safe | B0 safe | Gains | Losses | Collision | B0 collision |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in panels.items():text.append(f"| {name} | {v['safe']} | {v['baseline_safe']} | {len(v['gained'])} | {len(v['lost'])} | {len(v['collision_ids'])} | {len(v['baseline_collision_ids'])} |")
    text+=['','All18samecandidate audited original-startT719rows reused, includingfailures; exact overlaps reused aswell, no repeatedconsumednative. Everyremainingfresh/exactcasefullowncarry/initialpairing/physical/density/sourceSHA audited. These174starts are alreadyexposed same-scene conditions; no independentworld/sampleinflation ornewtrainingseed/statisticalsuperiorityclaim. PerIDsafe gained/lost andcollisions expose regressioncost; old403/core407/fullT698412retention unmeasured, T719acquisitiongate remainsfailed, nointegration/promotion. Alloriginalraw/model/failures preserved. Noautomaticnextscientificjob.'];(D/'FINDINGS.md').write_text('\n'.join(text)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-24 T720冻结T719完整曝光面板验收结束：{final['state']}，完整={complete}，174起点安全{report['safe']}vsB0{len(old)}；完整逐ID gains/losses/collisions见day_20261124/t720/PAIRED_RESULTS。18已审原起点与重叠exact复用，无新学习，旧403/core407仍未测，不撤销T719能力门失败，T698保持。\n\n"+f.read_text())

def backup():
    C.stopped();start=time.time();p=C.root_plan();l=C.read(D/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=900
    files=[O/f for f in ['protocol.json','LEDGER.json','FINAL_REPORT.json','SEALED.json']]+[D/f for f in ['PAIRED_RESULTS.json','FINDINGS.md','FIXED_SELECTION.json','PREFLIGHT.json']]
    for name in ['panel78','panel96']:files +=[O/name/f for f in ['summary.json','audit.json','cases.jsonl.gz','REUSED.json']]
    files=[f for f in files if f.exists()];hashes={str(f.relative_to(ROOT)):C.sha(f) for f in files};size=sum(f.stat().st_size for f in files);assert size<=8*2**20
    C.save(D/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),bytes=size,full_histories_local_pending=True,frozen_model_already_verified_T719=True))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T720 primary180')));signal.alarm(175)
    try:
        sys.path.insert(0,str(ROOT/'stage/experiments/weekend_20261120_23'));import archive_after_t695 as G
        G.stopped=G.Client.A.stopped=C.stopped;client=G.Client.client();index=C.read(G.DOC/'RESTORE_INDEX.json');assert not any(x['group']==GROUP for x in index['groups'])
        with (ROOT/'stage/runs/.stage_writer.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'));result=G.Client.A.archive(GROUP,files,release,min(start+170,p['absolute_all_stop']-330),client)
            assert result['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';C.stopped();manifest=ROOT/f'stage/docs/long_distance_20261101/source_manifest_{GROUP}.json';receipt=manifest.with_name('release_receipt_'+GROUP+'.json');members=C.read(manifest)['files'];assert {r['path']:r['sha256'] for r in members}==hashes and all(C.sha(ROOT/r['path'])==r['sha256'] for r in members)
            index['groups'].append(dict(group=GROUP,manifest=str(manifest),receipt=str(receipt),members=members,remote_receipt=result));index.update(epoch=time.time(),files_deleted=0,other_raw_pending=True);C.save(G.DOC/'RESTORE_INDEX.json',index);C.save(D/'PRIMARY_BACKUP.json',dict(state=result['state'],members=len(members),source_bytes=size,deleted=0,full_histories_local_pending=True))
    finally:
        signal.alarm(0);l=C.read(D/'TRANSPORT_LEDGER.json');l['charged_seconds']+=time.time()-start;assert l['charged_seconds']+300<=900;C.save(D/'TRANSPORT_LEDGER.json',l)
def watch():
    p=C.root_plan();C.save(D/'FINALIZER_STARTED.json',dict(pid=__import__('os').getpid(),epoch=time.time(),one_shot=True))
    while time.time()<p['absolute_all_stop']-600:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        time.sleep(5)
    C.stopped();charge(C.read(D/'TRANSPORT_PENDING.json')['remote_commit']);analysis()
    try:backup()
    except BaseException:C.save(D/'BACKUP_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,no_retry=True))
    C.stopped();assert time.time()+300<p['absolute_all_stop']
    paths=['stage/docs/day_20261124/t720','stage/experiments/t720_complete_exposed_panels','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']+[str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],check=True,cwd=ROOT);dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,dirty
    subprocess.run(['git','commit','-m','Record T720 complete exposed-start regression of frozen T719'],check=True,cwd=ROOT);target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    with (D/'final_source.log').open('xb') as f:
        child=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'finalize.py'),'source'],cwd=ROOT,stdout=f,stderr=f)
        try:code=child.wait(timeout=300)
        except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=3);code=-15
    if code==0:
        receipt=C.read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==target;charge(receipt['remote_commit']);C.save(D/'FINAL_SOURCE.json',dict(state='ACTUAL_EXIT0_EXACT_TREE_VERIFIED',remote_commit=receipt['remote_commit'],root_tree=target))
    else:
        l=C.read(D/'TRANSPORT_LEDGER.json');l['charged_seconds']+=300;assert l['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',l);C.save(D/'FINAL_SOURCE.json',dict(state='FAILED_LOCAL_PRESERVED',returncode=code))
    C.save(D/'FINALIZER_DONE.json',dict(state='STOPPED',epoch=time.time(),no_next_science=True))
if __name__=='__main__':
    try:C.sync(final=True) if sys.argv[1]=='source' else watch()
    except BaseException:C.save(D/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,epoch=time.time()));raise
