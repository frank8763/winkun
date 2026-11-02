"""Posthoc read-only comparison and independent fixed-scale recomputation."""
from pathlib import Path
import sys,json,gzip,os,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t299_conditional_goal_likelihood'))
from t299_learning import *

def main():
    out=ROOT/'stage/runs/t299_conditional_goal_likelihood';doc=ROOT/'stage/docs/distance_20261102';dest=doc/'T299_POSTHOC_COMPARISON.json';assert not dest.exists()
    p=json.loads((out/'protocol.json').read_text());s=json.loads((out/'summary.json').read_text());w=json.loads((out/'workers.json').read_text());assert s['all_workers_stopped'] and not s['inflight_jobs']
    for pid in [w['parent'],*w['workers'],json.loads((out/'launch.json').read_text())['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('Live/reused',pid))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);safe=policy(p);items=auxiliary(p)
    scale=gradient_scale(safe,items);saved=json.loads((out/'gradient_scale.json').read_text());assert scale==saved and scale['factor']==p['auxiliary_NLL_scale']
    safe.eval().requires_grad_(False);models=[('safeT255',safe)];source_models={}
    for name,folder,arm in [('T298','t298_self_goal_transfer','self_goal_transfer'),('T299','t299_conditional_goal_likelihood','conditional_goal_likelihood')]:
        summary=json.loads((ROOT/'stage/runs'/folder/'summary.json').read_text());exp=summary['arms'][arm]['export'];assert C.sha(exp['path'])==exp['sha256'];q=policy(p).eval().requires_grad_(False);q.load_state_dict(L.read(exp['path'])['policy']);assert S.digest(q.state_dict())==exp['policy_hash'];models.append((name,q));source_models[name]=exp['sha256']
    results=[]
    for name,q in models:
        rows=[]
        for source,item in zip(p['auxiliary_sources'],items):
            x,y=item
            with torch.no_grad():
                _,_,_,raw=q.step(x);mode=q.mode(raw);error=(mode-y).square();nll=torch.cat([conditional_nll(q,item,np.arange(k,min(k+32,len(x)))) for k in range(0,len(x),32)])
                mean=exact_mean(q,x);_,marginal=D.terms(mean,y);marginal_nll=-marginal.sum(-1)
            rows.append(dict(id=source['id'],MSE=float(error.mean()),MSE_drive_steer=error.mean(0).tolist(),conditional_NLL=float(nll.mean()),marginal_NLL=float(marginal_nll.mean()),clipped_drive_steer_fraction=(raw.abs()>.999999).float().mean(0).tolist(),mean_absolute_mode_action_difference=(mode-y).abs().mean(0).tolist()))
        results.append(dict(policy=name,policy_hash=S.digest(q.state_dict()),trajectory_equal_MSE=float(np.mean([r['MSE'] for r in rows])),trajectory_equal_conditional_NLL=float(np.mean([r['conditional_NLL'] for r in rows])),trajectory_equal_marginal_NLL=float(np.mean([r['marginal_NLL'] for r in rows])),rows=rows))
    rows=list(map(json.loads,gzip.decompress((out/'cases.jsonl.gz').read_bytes()).decode().splitlines()));counts=[]
    for role,expected in [('parent_auxiliary_centers',2),('auxiliary_centers',2),('old20_guard',20),('old_regression',275),('development',54),('heldout',16),('independent_reload',78)]:
        rr=[r for r in rows if r['role']==role];counts.append(dict(role=role,executed=len(rr),expected=expected,safe=sum(r['complete'] and r['success'] and not r['collision'] for r in rr),contact=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),censored=sum(not r['complete'] for r in rr),error=sum(bool(r['error']) for r in rr),not_executed=expected-len(rr)))
    old_failed=[dict(id=r['source_reset_id'],contact=r['collision'],timeout=r['timeout']) for r in rows if r['role']=='old_regression' and not(r['complete'] and r['success'] and not r['collision'])]
    six=[dict(id=r['source_reset_id'],success=r['success'] and not r['collision'],contact=r['collision'],timeout=r['timeout'],physics_steps=r['physics_steps'],remaining_m=r['remaining_m']) for r in rows if r['role']=='development'];parent=[r for r in rows if r['role']=='parent_auxiliary_centers'];aux=[r for r in rows if r['role']=='auxiliary_centers']
    assert len(parent)==len(aux)==2 and [r['initial120_sha256'] for r in parent]==[r['initial120_sha256'] for r in aux]
    # Exact AR latent-space illustration; no fitted parameter or new sample.
    # Correct joint likelihood can improve without centering every action at
    # the deterministic deployment readout. This is not a density bug/theorem
    # about the actual nonlinear trained policy.
    rho=.98;length=128;shift=2.;zero=np.zeros(length);constant=np.full(length,shift);decaying=shift*rho**np.arange(length)
    def latent_energy(z):return float(.5*z[0]**2+.5*np.sum((z[1:]-rho*z[:-1])**2)/(1-rho**2))
    exact_constant=.5*shift**2*(1+(length-1)*(1-rho)/(1+rho));exact_decay=.5*shift**2
    assert abs(latent_energy(constant)-exact_constant)<1e-12 and abs(latent_energy(decaying)-exact_decay)<1e-12
    ar_illustration=dict(rho=rho,decisions=length,latent_shift=shift,constant_shift_joint_NLL_increase=latent_energy(constant),constant_shift_independent_marginal_NLL_increase=float(.5*np.sum(constant**2)),decaying_shift_joint_NLL_increase=latent_energy(decaying),scope='Gaussian latent-space identity with unchanged Jacobians. It illustrates weighting of shared versus innovation errors under correct correlated density; not a replacement for native causal evidence or a proposed change of sampler/rho.')
    used=S.used(ROOT/'stage/runs');assert used<=p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    C.save(dest,dict(scope='Posthoc fixed8 consumed auxiliary-data comparison, no new optimizer/native, independent exact gradient-scale recomputation. Conditional/marginal/MSE changes descriptive, not pre-registered new selection criteria or navigation causality.',native_steps=0,optimizer_calls=0,navigation_updates=0,gradient_scale_recomputed_exact=scale,source_model_SHA=source_models,metrics=results,counts=counts,old_failure_ids=old_failed,six_rows=six,parent_and_candidate_auxiliary_initial120_exact=True,AR_latent_likelihood_illustration=ar_illustration,source_sha256=C.sha(Path(__file__)),input_protocol_sha256=C.sha(out/'protocol.json'),global_bytes=used,free_GiB=shutil.disk_usage(ROOT).free/1024**3));print('T299 posthoc compare and frozen gradient-scale recomputation exact,0native0optimizer')

if __name__=='__main__':main()
