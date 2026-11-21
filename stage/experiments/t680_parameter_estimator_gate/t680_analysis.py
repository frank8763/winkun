"""Fixed four-pair parameter estimator diagnostics; no Actor update or inference claim."""
from pathlib import Path
import fcntl,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t680_parameter_estimator_gate';DOC=ROOT/'stage/docs/weekend_20261120_23/t680'
sys.path.insert(0,str(ROOT/'stage/experiments/t679_episode_parameter_exploration'))
import t679_calibration as C
from status_io import save
np,read,sha,digest=C.np,C.read,C.sha,C.digest

def cosine(a,b):
    den=float(np.linalg.norm(a)*np.linalg.norm(b));return float(a@b/den) if den>1e-24 else None

def estimate(differences,epsilon,scale):return np.mean(differences[:,None]*epsilon,axis=0)/(2*scale)

def compute(p,limit):
    start=time.time();C.B.A.guard(p,start,limit);s=read(p['summary']);source=read(p['source_protocol']);cal=read(p['calibration'])
    assert s['state']=='COMPLETE' and s['gate']=='PARAMETER_EXPLORATION_SUPPORT_PASS' and cal['state']=='PASS'
    epsilon=np.stack([C.direction(seed) for seed in source['evaluation_direction_seeds']]);assert epsilon.shape==(4,11906)
    by={(r['role'],r['job_id']):r for r in s['rows']};assert len(by)==64
    models={item['name']:item['model'] for item in s['models']};assert len(models)==8
    for model in models.values():assert sha(model['path'])==model['sha256']
    deltas=[];details={}
    for key in source['collection_ids']:
        y=[]
        for index in range(4):
            pair=[]
            for sign in ['minus','plus']:
                r=by[(f'direction{index:02d}_{sign}_fresh',key)];assert sha(r['path'])==r['sha256'] and r['seed'] is None and not r.get('teacher')
                model=models[f'direction{index:02d}_{sign}'];assert r['loaded_model_sha256']==model['sha256'] and r['policy_hash']==model['policy_hash']
                pair.append(C.M.outcome(r,'collision_cost'))
            y.append(pair[1]-pair[0])
        assert y==s['support'][key]['antithetic_return_differences'];diff=np.array(y);g=estimate(diff,epsilon,cal['scale'])
        details[key]=dict(pair_differences=y,nonzero_pairs=int((diff!=0).sum()),norm=float(np.linalg.norm(g)),
                          zero_signal_not_retention=True,at_least2_nonzero_pairs=bool((diff!=0).sum()>=2));deltas.append(diff)
    deltas=np.stack(deltas);names=source['collection_ids'];field=np.array([k in source['field_collection_ids'] for k in names]);old=~field
    means={'field4':deltas[field].mean(0),'old4':deltas[old].mean(0),'joint8':deltas.mean(0)};reports={};arrays={'epsilon':epsilon,'pair_differences':deltas}
    for name,diff in means.items():
        g=estimate(diff,epsilon,cal['scale']);paircontrib=diff[:,None]*epsilon/(8*cal['scale']);norms=np.linalg.norm(paircontrib,axis=1)
        deletions=[]
        for i in range(4):
            mask=np.arange(4)!=i;v=estimate(diff[mask],epsilon[mask],cal['scale']);deletions.append(dict(removed=i,norm=float(np.linalg.norm(v)),cosine=cosine(v,g)))
        first=estimate(diff[:2],epsilon[:2],cal['scale']);last=estimate(diff[2:],epsilon[2:],cal['scale'])
        reports[name]=dict(differences=diff.tolist(),norm=float(np.linalg.norm(g)),pair_contribution_norms=norms.tolist(),
            largest_fraction=float(norms.max()/max(norms.sum(),1e-30)),first_half_norm=float(np.linalg.norm(first)),last_half_norm=float(np.linalg.norm(last)),
            halves_cosine=cosine(first,last),leave_one_pair_out=deletions,not_confidence_interval=True)
        arrays[name+'_gradient']=g
    r=reports['field4'];checks=dict(all_source_models_and_pairs_verified=True,
        at_least2_field_tasks_with2_nonzero_pairs=sum(details[k]['at_least2_nonzero_pairs'] for k in source['field_collection_ids'])>=2,
        field_mean_nonzero=r['norm']>1e-12,
        field_delete_any_pair_cosine_at_least_half=all(v['cosine'] is not None and v['cosine']>=.5 and v['norm']>1e-12 for v in r['leave_one_pair_out']),
        field_fixed_half_cosine_at_least_half=r['halves_cosine'] is not None and r['halves_cosine']>=.5,
        finite=all(np.isfinite(a).all() for a in arrays.values()))
    result=dict(state='PARAMETER_ESTIMATOR_STABILITY_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,tasks=details,means=reports,
        parameter_dimension=11906,independent_Gaussian_direction_count=4,linear_model_relative_MSE_reference=(11906+1)/4,
        MSE_reference_not_empirical_variance_estimate=True,rank_of_direction_span=int(np.linalg.matrix_rank(epsilon)),
        Gaussian_smoothed_objective_not_center_return=True,no_optimizer_or_candidate=True,no_new_physical_or_independent_samples=True,
        center_model_unchanged=True,no_automatic_learning=True,new_native=0,new_optimizer=0,array_hashes={k:digest(v) for k,v in arrays.items()})
    return result,arrays

def execute(action):
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    folder=Path(p['source_run']);seal=read(folder/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
    assert not Path('/proc',str(read(folder/'DRIVER_STARTED.json')['pid'])).exists()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();result,arrays=compute(p,p[action+'_seconds'])
    if action=='main':
        with (OUT/'estimates.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result)
    else:
        assert result==read(OUT/'RESULTS.json')
        with np.load(OUT/'estimates.npz',allow_pickle=False) as f:assert set(f.files)==set(arrays) and all(np.array_equal(f[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all_antithetic_estimators_and_delete_pair_checks_exact=True,new_native=0,new_optimizer=0))
    assert C.T.R.used(OUT)<p['max_bytes'] and C.T.R.used(ROOT/'stage/runs')<p['global_bytes']
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',elapsed_s=time.time()-start,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':execute(sys.argv[1])
