"""Read-only result, first-batch credit and update-rejection description."""
from t280_core import *
import collections,gzip,json,time
OUT=ROOT/'stage/runs/t280_local_cost';DEST=ROOT/'stage/docs/long_distance_20261101/T280_MECHANISM_DESCRIPTION.json'


def readrows(path):
    path=Path(path)
    if not path.exists():path=Path(str(path)+'.gz')
    text=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
    return list(map(json.loads,text.splitlines()))


def main():
    assert not DEST.exists();s=json.loads((OUT/'summary.json').read_text());assert s['all_workers_stopped'] and not s['inflight_jobs']
    p=json.loads((OUT/'protocol.json').read_text());rr=readrows(OUT/'cases.jsonl');events=readrows(OUT/'proposal_events.jsonl')
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    outcome=lambda r:'GOAL' if safe(r) else 'CONTACT' if r['collision'] else 'TIMEOUT' if r['timeout'] else 'CENSORED'
    descriptions={};credit={}
    for arm in p['arms']:
        groups={};trialrows=[a for e in events if e['arm']==arm and e['phase']=='live' for a in e.get('attempts',[])]
        for role in ['training','old_regression','development']:
            rows=[r for r in rr if r['arm']==arm and r['role']==role]
            groups[role]=dict(n=len(rows),outcomes=dict(collections.Counter(outcome(r) for r in rows)))
        dev=[r for r in rr if r['arm']==arm and r['role']=='development']
        for side in ['LEFT','RIGHT']:
            rows=[r for r in dev if side in r['source_reset_id']]
            groups['six_m_'+side]=dict(n=len(rows),outcomes=dict(collections.Counter(outcome(r) for r in rows)),
                remaining_min_m=min(r['remaining_m'] for r in rows),remaining_max_m=max(r['remaining_m'] for r in rows))
        groups['success_ids']=[r['source_reset_id'] for r in dev if safe(r)]
        descriptions[arm]=dict(groups=groups,live_trials=len(trialrows),numerical_or_surrogate_rejected_before_native=sum(a.get('native_old_task_guard') is None for a in trialrows),
            native_accepted=sum(a['native_old_task_guard']['all_safe'] for a in trialrows if a.get('native_old_task_guard')),
            native_rejected=sum(not a['native_old_task_guard']['all_safe'] for a in trialrows if a.get('native_old_task_guard')),
            old_failure_ids=dict(collections.Counter(name for a in trialrows if a.get('native_old_task_guard') for name in a['native_old_task_guard']['failed_ids'])))
        batch=[r for r in rr if r['arm']==arm and r['role']=='training' and r['id'].startswith(arm+'_r0_')]
        assert len(batch)==16;batch_arrays=[];returns=[];old_raw=[];by_reset={}
        for row in batch:
            assert C.sha(row['path'])==row['sha256']
            with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
            batch_arrays.append(a);returns.append(S.monte_carlo(a['rewards'],a['native_counts']))
        for index,(row,a) in enumerate(zip(batch,batch_arrays)):
            others=[returns[k][0] for k,r in enumerate(batch) if k!=index and r['source_reset_id']==row['source_reset_id']]
            assert len(others)==3
            baseline=float(np.mean(others));old_raw.extend((returns[index]-baseline).tolist())
            by_reset.setdefault(row['source_reset_id'],[]).append(float(returns[index][0]-baseline))
        scale=max(float(np.sqrt(np.mean(np.square(old_raw)))),.05)
        q=policy(p).eval().requires_grad_(False);details=[]
        for reset,coefficients in sorted(by_reset.items()):
            ids=[i for i,r in enumerate(batch) if r['source_reset_id']==reset]
            xs=np.concatenate([batch_arrays[i]['states'] for i in ids]);actual=np.concatenate([batch_arrays[i]['actions'] for i in ids]);modes=[];raws=[]
            with torch.no_grad():
                for x in xs:
                    mode,_,_,raw=q.step(torch.tensor(x[None]));modes.append(mode[0].numpy());raws.append(raw[0].numpy())
            modes=np.asarray(modes);raws=np.asarray(raws)
            details.append(dict(reset=reset,trajectory_G0=[float(returns[i][0]) for i in ids],
                weighted_LOO_coefficient=list(map(float,coefficients)),common_old_estimator_RMS_scale=scale,
                normalized_weighted_coefficient_RMS=float(np.sqrt(np.mean(np.square(coefficients)))/scale),
                sampled_command_std=actual.std(axis=0).tolist(),deterministic_mode_saturated_fraction=(np.abs(modes)>=.99999).mean(axis=0).tolist(),
                raw_output_abs_gt1_fraction=(np.abs(raws)>1).mean(axis=0).tolist(),
                limitation='Saturation/credit scale describe frozen first-batch states only, not proof of zero likelihood gradient or unique causal root.'))
        credit[arm]=details
    historical={}
    for group,name in [('T257','t257_distance_policy_gradient'),('T259','t259_closed_loop_recovery')]:
        historical[group]=[r.get('source_reset_id',r['id']) for r in readrows(ROOT/'stage/runs'/name/'development.jsonl') if safe(r)]
    C.save(DEST,dict(state='READ_ONLY_COMPLETED',physics_added=0,optimizer_calls_added=0,navigation_Actor_updates_added=0,
        arms=descriptions,first_batch_credit_and_deployment_description=credit,historical_six_m_success_ids=historical,
        current_success_is_historically_known_RIGHT18=True,all_raw_complete=sum(not r['complete'] for r in rr)==0,
        missing_fresh16_and_reload78='Not executed by gate, not failures',summary_sha256=C.sha(OUT/'summary.json'),created_epoch=time.time(),
        conclusion='New objective creates distinguishable failure returns but did not expand final success set in this pilot. Retention rejections and absent training successes coexist; no proof that either alone is sufficient.'))
    print(json.dumps(dict(state='PASS',arms=descriptions,credit=credit),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
