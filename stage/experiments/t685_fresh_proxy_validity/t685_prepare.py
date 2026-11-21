"""Fresh/fit source disjointness, original surrogate identity and real loaders."""
from pathlib import Path
import ast,json,sys,time
import t685_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    last=ROOT/'stage/runs/t684_temporal_noise_collection';rep=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert rep['state']=='COMPLETE' and rep['audit']['state']=='PASS' and rep['gate']=='SCIENTIFIC_GATE_FAILED'
    ids=seal['scientific_pids']+[rep['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    source=ROOT/'stage/runs/t678_reference_target_pilot';science=read(source/'protocol.json');fresh=read(last/'protocol.json')
    auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time();candidates=[]
    for arm in ['executed','self']:
        folder=source/arm;s=read(folder/'summary.json');assert s['state']=='COMPLETE' and read(folder/'SCREEN.json')['lost_ids']
        assert s['retained']['policy_hash']==science['initial']['policy_hash'] and s['full407'] is None
        candidates.append(dict(name=arm,model=s['proposal'],summary=str(folder/'summary.json'),screen=str(folder/'SCREEN.json'),learning=str(folder/'LEARNING.json')))
    p=dict(id='T685_FROZEN_FRESH_PROXY_VALIDITY',authorization_approved=False,out=str(OUT),weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'),
        science_protocol=str(source/'protocol.json'),initial=science['initial'],raw_sigma=[.05,.05],score_scale=512,ratio_clip=.2,
        candidates=candidates,panel_ids=fresh['collection_ids'],old_ids=fresh['old_collection_ids'],train_summaries=science['summaries'],
        fresh_summary=str(last/'summary.json'),new_Actor=0,new_Value=0,new_optimizer=0,max_physics=0,max_commands=0,
        main_seconds=600,audit_seconds=600,max_bytes=32*2**20,child_bytes=32*2**20,global_bytes=10*2**30,
        absolute_learning_stop=min(now+600,auth['absolute_learning_stop']),absolute_native_stop=min(now+600,auth['absolute_native_stop']),
        absolute_all_stop=min(now+1800,auth['absolute_all_stop']),finalization_seconds=60)
    datasets=S.data(p);assert len(datasets['fresh'])==8
    # Fresh data were generated after both immutable candidates were evaluated;
    # they did not participate in either one-step fit.
    assert all(read(source/'FINAL_REPORT.json')['epoch']<read(last/'DRIVER_STARTED.json')['epoch'] for unused in [0])
    science0=S.read(p['science_protocol']);base=S.M.L.frozen(science0,p['initial']['path'])
    result,arrays=S.compute(p,True)
    assert result['identity_control'] and result['scientific_learning_admission'] is False
    fixture=S.is_metrics([0.,0.,0.,0.],[-1.,0.,1.,1.]);assert fixture['ESS']==4 and fixture['ordinary_IS_change']==0. and fixture['SNIS_change']==0.
    concentrated=S.is_metrics([20.,-20.,-20.,-20.],[1.,-1.,0.,1.]);assert not concentrated['concentration_gate'] and concentrated['ESS']<1.01
    delta=S.np.array([.01,-.02,.03]);adv=2.;ratio=S.np.exp(delta)
    direct=float((ratio-1).sum()*adv/512);assert abs(direct-float(S.np.expm1(delta).sum()*adv/512))<1e-17
    resources=S.Wait.Raw.resources();assert S.T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',fresh32_distinct_seed_hash_and_later_than_candidates=True,
        original48_plus_fresh32_complete_history_scope=True,both_real_fixed_candidate_loaders_and_frozen_backbone=True,
        identity_zero_proxy_ratio_KL_and_ESS_fixture=True,raw_vs_clip_logp_schema_explicit=True,concentration_and_peraction_definitions_fixture=True,
        full_train_proxy_numeric_identity_pending_main=True,new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in science['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T685_PRIMARY_REVIEW.md'}
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json','summary.json','TEMPORAL_RESULTS.json']}
    deps|={source/'protocol.json',source/'FINAL_REPORT.json'}|set(map(Path,p['train_summaries']))
    for item in candidates:deps|={Path(item[k]) for k in ['summary','screen','learning']}|{Path(item['model']['path'])}
    deps|={Path(row['path']) for by in datasets.values() for rows in by.values() for row in rows}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+12
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
