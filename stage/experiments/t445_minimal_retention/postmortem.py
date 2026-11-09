"""One zero-native scoped reconstruction; retain original failed geometry audit."""
from pathlib import Path
import importlib.util,json,time,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('stage445_completed_raw',HERE/'run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
K,E,np,torch=R.K,R.E,R.np,R.torch
OUT=R.OUT;DOC=R.DOC
def main():
    import mujoco as mj
    start=time.time();p=R.read(OUT/'protocol.json');s=R.read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not (OUT/'postmortem.json').exists()
    ids=[s['pid']]+R.read(OUT/'workers.json')['workers']+[R.read(OUT/'driver_started.json')['pid'],R.read(OUT/'launch.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    original_error=DOC/'T445_DRIVER_ERROR.json';assert original_error.exists() and not (OUT/'audit.json').exists()
    E.N.offline();R.C.ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    step_attempts=[]
    def forbidden(*args,**kwargs):step_attempts.append(1);raise AssertionError('Postmortem forbids physical integration')
    for name in ['mj_step','mj_step1','mj_step2']:setattr(mj,name,forbidden)
    rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()];replays=[json.loads(x) for x in (OUT/'replays.jsonl').read_text().splitlines()]
    geom=R.read(OUT/'geometry.json');model=R.ram_model(p);data=mj.MjData(model);geometry=[]
    for row in rows:
        assert time.time()-start<180
        assert R.sha(row['path'])==row['sha256'];a=E.N.S.arrays(row['path']);assert R.digest(a)==row['array_digest']
        q=K.policy(p,row['kind']);E.NATIVE_AUDIT.audit_array(q,a,p,seed=None)
        assert int(a['native_counts'].sum())==row['physics']
        differences=[];pair_mismatches=0
        for i,qp in enumerate(a['qpos_before']):
            data.qpos[:]=qp;mj.mj_forward(model,data);distance,pair=K.clearance(model,data,geom['pairs'])
            differences.append(distance-float(a['body_clearance_capped_m'][i]));pair_mismatches+=not np.array_equal(pair,a['body_geom_pair'][i])
        bad=np.flatnonzero(np.abs(differences)>=1e-10)
        geometry.append(dict(job_id=row['job_id'],kind=row['kind'],first_mismatch_decision=None if not len(bad) else int(bad[0]),
            maximum_abs_reconstructed_distance_difference_m=float(np.max(np.abs(differences))),pair_mismatches=int(pair_mismatches),
            original_1e_10_threshold_not_relaxed=True,not_an_audit_PASS=bool(len(bad) or pair_mismatches)))
    assert len(rows)==len(replays)==4 and s['physics']==sum(x['physics'] for x in rows+replays) and s['commands']==18
    for row in replays:
        ref=next(x for x in rows if x['path']==row['reference_path']);assert row['array_digest']==ref['array_digest'] and row['integration_sha256']==ref['integration_sha256']
    for row in R.read(OUT/'comparisons.json'):
        assert R.sha(row['comparison_path'])==row['comparison_sha256'];values=E.N.S.arrays(row['comparison_path'])
        for i,kind in enumerate(['safe_reset','student']):
            for j,actualkind in enumerate(['safe_reset','student']):
                a=E.N.S.arrays(next(r for r in rows if r['job_id']==row['job_id'] and r['kind']==actualkind)['path'])
                command,loc=K.responses(p,a,kind);assert np.array_equal(values[f'A{i}{j}'],command[:len(values['time_s'])].astype(np.float64))
        assert np.array_equal(values['total'],values['A11']-values['A00'])
        assert np.array_equal(values['symmetric_policy'],((values['A10']-values['A00'])+(values['A11']-values['A01']))/2)
        assert np.array_equal(values['symmetric_history'],((values['A01']-values['A00'])+(values['A11']-values['A10']))/2)
        assert np.max(np.abs(values['total']-values['symmetric_policy']-values['symmetric_history']))<1e-12
    assert not step_attempts
    for f,h in p['protected_files'].items():assert R.sha(f)==h
    result=dict(state='PASS_SCOPED_ACTION_HISTORY_REWARD_AND_EXACT_REPLAY_ONLY',overall_registered_audit='FAIL_GEOMETRY_RECONSTRUCTION',
        original_audit_failure_sha256=R.sha(original_error),geometry=geometry,
        geometry_distance_values_EXCLUDED_from_scientific_inference=True,
        physical_primary=4,physical_exact_replays=4,reported_physics=s['physics'],reported_all_commands=18,
        historical_same_policy_trajectories_repeated_NOT_new_independent_samples=True,
        new_native=0,native_step_function_attempts=0,new_Actor_Adam=0,new_V_Adam=0,
        all_full120_history_hidden_tape_actions_rewards_and_comparisons_exact=True,
        missing_physical_geometry_cache_information_is_only_a_hypothesis=True,
        original_models_unchanged=True,pid=__import__('os').getpid(),elapsed_s=time.time()-start)
    R.save(OUT/'postmortem.json',result);R.save(DOC/'T445_SCOPED_RECONSTRUCTION.json',result)
    print(json.dumps(dict(state=result['state'],overall_audit=result['overall_registered_audit'],geometry=geometry)),flush=True)
if __name__=='__main__':main()
