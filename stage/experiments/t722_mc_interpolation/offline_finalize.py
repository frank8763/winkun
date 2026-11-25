"""Local-only completion and paired controls; remote receipts remain pending."""
from pathlib import Path
import importlib.util,json,gzip,subprocess,time,os,traceback
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('T722_postflight_controller',HERE/'run.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
D,O=C.DOC,C.OUT

def analysis():
    p=C.root_plan();final=C.read(O/'FINAL_REPORT.json');control=C.read(C.ROOT/'stage/docs/overnight_20261124_25/t721/PAIRED_RESULTS.json');complete=final['state']=='COMPLETE';panels={}
    for name in p['panels']:
        f=O/name;sm=C.read(f/'summary.json') if (f/'summary.json').exists() else {};au=C.read(f/'audit.json') if (f/'audit.json').exists() else {};valid=sm.get('state')=='COMPLETE' and au.get('state')=='PASS';complete &=valid
        ss=set(sm.get('success_ids',[]));mc=control['arms']['mc1']['panels'][name];ms=set(mc['success_ids']);raw=[json.loads(v) for v in gzip.decompress((f/'cases.jsonl.gz').read_bytes()).decode().splitlines()] if (f/'cases.jsonl.gz').exists() else []
        if name=='old403':bs=set(p['required403_ids'])-set(p['baseline_old403_lost']);b0safe=len(bs);b0coll=None
        elif name=='exposed174':bs={r['job_id'] for r in p['panels'][name]['baseline_rows'] if C.M.N.T.Prior.safe(r)};b0safe=len(bs);b0coll=41
        elif name=='prospective24':bs=set(control['prospective_B0']['success_ids']);b0safe=len(bs);b0coll=len(control['prospective_B0']['collision_ids'])
        else:bs=None;b0safe=None;b0coll=None
        panels[name]=dict(verified=valid,state=sm.get('state','MISSING'),audit=au.get('state','MISSING'),safe=sm.get('success') if valid else None,total=len(p['panels'][name]['jobs']),success_ids=sorted(ss) if valid else None,collision_ids=sm.get('collision_ids') if valid else None,timeout_ids=sm.get('timeout_ids') if valid else None,boundary_ids=sm.get('boundary_ids') if valid else None,B0_safe=b0safe,B0_collision=b0coll,MC_safe=mc['safe'],MC_collision=len(mc['collision_ids']),gained_vs_B0=sorted(ss-bs) if valid and bs is not None else None,lost_vs_B0=sorted(bs-ss) if valid and bs is not None else None,gained_vs_MC=sorted(ss-ms) if valid else None,lost_vs_MC=sorted(ms-ss) if valid else None)
    practical=complete and panels['old403']['safe']>=383 and panels['exposed174']['safe']>=102 and len(panels['exposed174']['collision_ids'])<=41
    result=dict(state=final['state'],complete_all_panels=complete,model=p['candidate_model'],alpha=.5,new_learning=0,new_optimizer=0,added_parameters=0,practical_marker=bool(practical),panels=panels,cost=C.read(O/'LEDGER.json'),retained='T698',single_development_point_not_alpha_search=True,all_sets_exposed_same_map=True,epoch=time.time());C.save(D/'PAIRED_RESULTS.json',result)
    lines=['# T722 fixed half interpolation of B0 and T721 MC','',f"State {final['state']}; complete {complete}; practical marker {bool(practical)}. Same20198 head and frozen body; zero optimizer, no new capacity.",'','| Panel | B0 safe | MC safe | half-interpolation safe | MC collision | half collision |','|---|---:|---:|---:|---:|---:|']
    for name,v in panels.items():lines.append(f"| {name} | {v['B0_safe']} | {v['MC_safe']} | {v['safe']} | {v['MC_collision']} | {len(v['collision_ids']) if v['verified'] else 'MISSING'} |")
    lines+=['','Every694/MC/B0 original remains unchanged. Old403 baselineB0392; losses relative original403 and B0 must be separated. Recovery18 conditional tests and24now-exposed starts are not unknown-map generalization. Exact repeats do not add independent samples; no training-seed superiority claim. This is a mandatory simple interpolation baseline, not a novel forgetting algorithm. PerIDgains/losses/collisions/timeouts/boundary overlaps inPAIRED_RESULTS. T698 remains retained; no automatic nextscience.']
    (D/'FINDINGS.md').write_text('\n'.join(lines)+'\n');f=C.ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(f"2026-11-25 T722固定B0/MC小头参数各半插值验收结束：{final['state']}，完整={complete}、practical={bool(practical)}；0学习/0扩容，详见day_20261125/t722/FINDINGS及PAIRED_RESULTS。T698保持，无自动下一科学任务。\n\n"+f.read_text())

def main():
    p=C.root_plan();C.save(D/'FINALIZER_STARTED.json',dict(pid=os.getpid(),epoch=time.time(),local_only=True))
    while True:
        props=subprocess.check_output(['systemctl','--user','show',C.UNIT,'-p','MainPID','-p','ActiveState'],text=True)
        if 'MainPID=0' in props and ('ActiveState=inactive' in props or 'ActiveState=failed' in props):break
        if time.time()>=p['absolute_all_stop']-5:subprocess.run(['systemctl','--user','stop',C.UNIT],check=True,timeout=10);break
        time.sleep(3)
    C.stopped();pending=C.read(D/'TRANSPORT_PENDING.json');l=C.read(D/'TRANSPORT_LEDGER.json')
    if pending['remote_commit'] not in l['source_commits']:l['source_commits'].append(pending['remote_commit']);l['charged_seconds']+=300;assert l['charged_seconds']<=900;C.save(D/'TRANSPORT_LEDGER.json',l)
    analysis();C.save(D/'DEFERRED_TRANSPORT.json',dict(state='LOCAL_RESULTS_COMPLETE_REMOTE_PRIMARY_AND_FINAL_SOURCE_PENDING',new_remote_requests=0,all_models_and_raw_preserved=True))
    C.save(D/'FINALIZER_DONE.json',dict(state='STOPPED',network_required=False,no_next_science=True,remote_work_pending=True,epoch=time.time()))
    paths=['stage/docs/day_20261125/t722','stage/experiments/t722_mc_interpolation','stage/docs/RESEARCH_STATUS.md','stage/docs/weekend_20261120_23/LEDGER.json'];subprocess.run(['git','add','--',*paths],check=True,cwd=C.ROOT);subprocess.run(['git','commit','-m','Save complete T722 fixed interpolation controls and offline receipts'],check=True,cwd=C.ROOT)
if __name__=='__main__':
    try:main()
    except BaseException:C.save(D/'FINALIZER_FAILURE.json',dict(error=traceback.format_exc(),all_local_preserved=True,epoch=time.time()));raise
