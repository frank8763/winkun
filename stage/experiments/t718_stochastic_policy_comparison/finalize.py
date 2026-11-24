"""Bounded post-exit recovery qualification, receipts and no further learning."""
from pathlib import Path
import importlib.util,json,gzip,time,subprocess,sys,signal,traceback,fcntl
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t718_final_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
D,O=C.DOC,C.OUT
GROUP='DAY_T718_STOCHASTIC_CONTROL_0'
def charge(commit):
    x=C.read(D/'TRANSPORT_LEDGER.json')
    if commit not in x['source_commits']:x['source_commits'].append(commit);x['charged_seconds']+=300;assert x['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',x)
def analysis():
    p=C.root_plan();report=C.read(O/'FINAL_REPORT.json');arms={};complete=report['state']=='COMPLETE';ids=[j['id'] for j in p['recovery_jobs']];per={key:{} for key in ids};np=C.A.np
    for arm in ['baseline','trained']:
        folder=O/('recovery_'+arm);sm=C.read(folder/'summary.json') if (folder/'summary.json').exists() else {};audit=C.read(folder/'audit.json') if (folder/'audit.json').exists() else {};valid=sm.get('state')=='COMPLETE' and audit.get('state')=='PASS';complete &=valid
        rows=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (folder/'cases.jsonl.gz').exists() else [];fresh=[x for x in rows if not x.get('exact')]
        values=[]
        for row in fresh:
            arr=C.M.N.T.R.E.N.S.arrays(row['path']);k=int(arr['recovery_first']);discount=1.;value=0.
            for reward,count in zip(arr['rewards'][k:],arr['native_counts'][k:]):value+=discount*float(reward);discount*=float(p['gamma_native']**count)
            v=dict(id=row['job_id'],seed=row['seed'],safe=bool(C.M.N.T.Prior.safe(row)),collision=bool(row['collision']),timeout=not row['success'] and not row['collision'],boundary=bool('fullfield_start_pose' in row['request']['job'] and row.get('field_boundary')),discounted_suffix_native_return=value,seconds=row['physics']*.002)
            values.append(v);per[v['id']].setdefault(arm,[]).append(v)
        arms[arm]=dict(state=sm.get('state','MISSING'),audit=audit.get('state','MISSING'),model=sm.get('model'),fresh=len(fresh),safe=sum(v['safe'] for v in values) if valid else None,collision=sum(v['collision'] for v in values),timeout=sum(v['timeout'] for v in values),boundary=sum(v['boundary'] for v in values),validation_safe=sum(v['safe'] for v in values if v['id'] in p['validation_ids']) if valid else None,validation_collision=sum(v['collision'] for v in values if v['id'] in p['validation_ids']),mean_suffix_return=sum(v['discounted_suffix_native_return'] for v in values)/len(values) if values else None,rows=values)
    comparisons=[]
    if complete:
        for key,v in per.items():
            aa,bb=v['baseline'],v['trained'];assert len(aa)==len(bb)==3 and [x['seed'] for x in aa]==[x['seed'] for x in bb];comparisons.append(dict(id=key,split='validation' if key in p['validation_ids'] else 'training',baseline_safe=sum(x['safe'] for x in aa),trained_safe=sum(x['safe'] for x in bb),paired_safe_gain=sum(int(y['safe'])-int(x['safe']) for x,y in zip(aa,bb)),paired_suffix_return_gain=sum(y['discounted_suffix_native_return']-x['discounted_suffix_native_return'] for x,y in zip(aa,bb))/3))
    base,trained=arms['baseline'],arms['trained'];flag=bool(complete and trained['safe']-base['safe']>=6 and trained['validation_safe']-base['validation_safe']>=2 and trained['collision']<=base['collision'] and trained['validation_collision']<=base['validation_collision'])
    result=dict(state=report['state'],complete_panel=complete,arms=arms,per_ID_paired=sorted(comparisons,key=lambda x:x['id']),descriptive_added_stochastic_benefit_flag=flag,T717_deterministic_acquisition_gate_still_failed=True,independent_worlds=1,independent_training_seeds=0,exact_replays_are_audit_only=True,no_formal_generalization_or_significance_claim=True,integration='NOT_AUTHORIZED_BY_THIS_DIAGNOSTIC',retained='T698',new_learning=0,cost=C.read(O/'LEDGER.json'),epoch=time.time());C.save(D/'PAIRED_RESULTS.json',result)
    lines=['# T718 frozen stochastic policy versus exploration-only control','',f"State {report['state']}; complete panel {complete}; descriptive added-benefit flag {flag}.",'','| Same sigma .2 | Safe/54 | Validation safe/18 | Collision | Timeout | Boundary overlap |','|---|---:|---:|---:|---:|---:|']
    for name,v in arms.items():lines.append(f"| {name} | {v['safe']} | {v['validation_safe']} | {v['collision']} | {v['timeout']} | {v['boundary']} |")
    lines+=['','Same18conditions, three shared new rollout seeds each, same frozen B0 prefixes and original horizon. All216episodes include108exact repeats solely for audit. PerID paired counts and returns are descriptive;54draws are not54independenttasks,6validationIDs were excluded from T717training but already exposed inventory, no unseenmap or algorithm superiority claim. This study does not undo T717 deterministic acquisition failure, learn new weights, or authorize integration. T698 retained; alloriginalraw/errors/models preserved. No automatic next study.'];(D/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-24 T718冻结随机策略vs同噪声B0对照结束：{report['state']}，完整面板={complete}，描述性附加收益标记={flag}；各54fresh与54exact、18条件内配对结果见day_20261124/t718/PAIRED_RESULTS。0学习，不撤销T717确定性能力门失败或启动整合，T698保持。\n\n"+f.read_text())

def backup():
    C.stopped();start=time.time();p=C.root_plan();l=C.read(D/'TRANSPORT_LEDGER.json');assert l['charged_seconds']+480<=900
    files=[O/f for f in ['protocol.json','LEDGER.json','FINAL_REPORT.json','SEALED.json','fit/summary.json','fit/audit.json']]+[D/f for f in ['PAIRED_RESULTS.json','FINDINGS.md','FIXED_SELECTION.json','PREFLIGHT.json','T717_RATIONALE.json']]
    for arm in ['baseline','trained']:files +=[O/('recovery_'+arm)/f for f in ['summary.json','audit.json','cases.jsonl.gz']]
    files +=[Path(model['path']) for model in p['models'].values()]
    files=[f for f in files if f.exists()];hashes={str(f.relative_to(ROOT)):C.sha(f) for f in files};size=sum(f.stat().st_size for f in files);assert size<=8*2**20
    C.save(D/'PRIMARY_BACKUP_SET.json',dict(group=GROUP,files=list(hashes),bytes=size,full_histories_local_pending=True,frozen_models_included=True))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T718 primary180')));signal.alarm(175)
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
    paths=['stage/docs/day_20261124/t718','stage/experiments/t718_stochastic_policy_comparison','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json','stage/docs/weekend_20261120_23/RESTORE_INDEX.json']+[str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*'+GROUP+'.json')]
    subprocess.run(['git','add','--',*paths],check=True,cwd=ROOT);dirty=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();assert not dirty,dirty
    subprocess.run(['git','commit','-m','Record frozen T718 stochastic versus exploration control'],check=True,cwd=ROOT);target=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
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
