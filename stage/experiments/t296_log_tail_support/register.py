from pathlib import Path
import json,sys
from run import ROOT,HERE,OUT,C,S

def main():
    assert not OUT.exists();OUT.mkdir();prior=ROOT/'stage/runs/t295_hindsight_support';old=json.loads((prior/'protocol.json').read_text());rows=json.loads((prior/'cases.json').read_text());assert json.loads((ROOT/'stage/docs/distance_20261102/T295_AUDIT.json').read_text())['state']=='PASS';assert (ROOT/'stage/docs/long_distance_20261101/release_receipt_T295.json').exists()
    sources=[];inputs=dict(old['input_hashes'])
    for f in [prior/'protocol.json',prior/'cases.json',prior/'summary.json',ROOT/'stage/docs/distance_20261102/T295_AUDIT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T295.json']+list(HERE.glob('*.py')):inputs[str(f)]=C.sha(f)
    for s,r in zip(old['sources'],rows):
        s=dict(s,witness_payload=r['path'],alternate_outcome=r['alternate']);sources.append(s);inputs[r['path']]=C.sha(r['path'])
    p=dict(experiment='T296_SAME_AR_DENSITY_LOG_TAIL_NUMERICAL_SUPPORT_AUDIT',sources=sources,source_protocol=old['source_protocol'],policy_hash=old['policy_hash'],input_hashes=inputs,reset_ids=old['reset_ids'],ESS_min=old['ESS_min'],max_normalized_weight=old['max_normalized_weight'],max_abs_terminal_log_weight=old['max_abs_terminal_log_weight'],old_logp_abs_tolerance=1e-8,
        wall_seconds=300,audit_wall_seconds=300,max_commands=0,max_physics=0,episode_bytes=48*1024,final_bytes=1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        scientific_scope='Numerical audit only,0native0optimizer0Actor. Same8 endedT295 historical data/goals/sigma(.035,.03)/rho.98/Actor/AR conditional density/firsthit/objective/support thresholds. Distinguish finite-precision CDF rejection from actual importance-weight overlap. Loginterval by reflected Gaussian tails, inverse logCDF solved by12 checked Newton iterations, no CDF/probability/density/weight clipping or changes in policy.14 extreme scalar cases against90digit mpmath inverse/marginal and current/previousmean gradient finite differences. Stable original means exact and logp within1e-8 historical; report all prefix/terminal weights, logstable ESS/maxdominance same screen. Not a new sampler/Actor-learning method or evidence of original6m success, not an unbiased estimator for data-adaptive continuous auxiliary goals or original6m. No new seed/goal/noise/grid or physical repeat. Finite four trajectories yield only exploratory support evidence; cannot infer universal impossibility of HPG. Independently re-evaluate all data/inputs/policy/marginals after producer stop, preserve all failures/T295 originals; complete remote RAM memberSHA readback then source sync. No Phase/planner/waypoint/runtime teacher/router/promotion/7/8.')
    assert S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes']<p['global_bytes']
    for f in [HERE/'protocol.json',OUT/'protocol.json']:f.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(registered=True,native_steps=0,optimizer_calls=0,worst_global_bytes=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'])))

if __name__=='__main__':main()
