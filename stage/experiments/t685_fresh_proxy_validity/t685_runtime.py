"""Posthoc fixed-candidate proxy validation; no simulator, fitting or selection."""
from pathlib import Path
import fcntl,json,math,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t685_fresh_proxy_validity';DOC=ROOT/'stage/docs/weekend_20261120_23/t685'
import t685_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
T,M,torch,np=B.T,B.M,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

def data(p):
    train={k:[] for k in p['panel_ids']}
    for f in p['train_summaries']:
        for r in read(f)['rows']:
            if r['job_id'] in train:train[r['job_id']].append(r)
    new={k:[] for k in p['panel_ids']}
    for r in read(p['fresh_summary'])['rows']:
        if r['role']=='iid_fresh':new[r['job_id']].append(r)
    assert sum(map(len,train.values()))==48 and sum(map(len,new.values()))==32
    assert all(len(train[k])==(8 if k.startswith('t502_field') else 4) and len(new[k])==4 for k in train)
    for k in train:
        assert not {tuple(r['seed']) for r in train[k]}&{tuple(r['seed']) for r in new[k]}
        assert not {r['sha256'] for r in train[k]}&{r['sha256'] for r in new[k]}
    return dict(train=train,fresh=new)

def is_metrics(ell,y):
    ell=np.asarray(ell,np.float64);y=np.asarray(y,np.float64);assert np.isfinite(ell).all()
    maximum=float(ell.max());z=np.exp(ell-maximum);weights=z/z.sum();ess=float(1/np.square(weights).sum())
    logmean=maximum+math.log(float(z.mean()));snis=float(weights@y);base=float(y.mean())
    # Never silently clip or renormalize ordinary weights. Retain log values
    # and explicit unsupported overflow instead of producing JSON infinities.
    raw_weight=np.exp(ell) if maximum<700 else None
    delta=float(np.mean(np.expm1(ell)*y)) if maximum<700 else None
    return dict(logmean_weight=logmean,normalized_weights=weights.tolist(),ESS=ess,max_weight=float(weights.max()),
        ordinary_weights=raw_weight.tolist() if raw_weight is not None else None,ordinary_IS_change=delta,
        ordinary_weight_underflow_count=int((ell<math.log(np.nextafter(0.,1.))).sum()),
        ordinary_IS_finite=delta is not None and math.isfinite(delta),SNIS_return=snis,SNIS_change=snis-base,
        behavior_sample_return=base,SNIS_biased=True,n=len(ell),concentration_gate=ess>=3 and float(weights.max())<=.5)

def group(p,baseline,candidate,rows,fresh,short=False):
    ys=[M.outcome(r,'collision_cost') for r in rows];n=len(rows);adv=np.asarray(ys)-(sum(ys)-np.asarray(ys))/(n-1)
    proxy=clipped=0.;ell=[];meta=[];all_delta=[]
    for i,r in enumerate(rows):
        assert r['full'] and r['seed'] is not None and sha(r['path'])==r['sha256'] and not r.get('teacher')
        a=T.R.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        assert not a['ownership'].any() and a['stochastic'].all()
        x=torch.tensor(a['states'][:32] if short else a['states']);raw=torch.tensor(a['raw_actions'][:len(x)]);atom=torch.tensor(a['atom'][:len(x)])
        with torch.no_grad():
            oldmu=baseline.sequence(x);mu=candidate.sequence(x)
            assert float((oldmu-torch.tensor(a['locations'][:len(x)])).abs().max())<1e-10
            old=M.P.L.density(oldmu,raw,atom,p['raw_sigma']);new=M.P.L.density(mu,raw,atom,p['raw_sigma'])
        if fresh:
            assert r['request']['job']['temporal_tau']==0. and a['logp_is_raw_latent'].all()
            saved=a['conditional_clip_logp'][:len(x)]
        else:
            assert r['policy_hash']==p['initial']['policy_hash'];saved=a['logp'][:len(x)]
        assert np.max(np.abs(old.numpy()-saved))<1e-9
        delta=(new-old).numpy();ratio=np.exp(delta)
        proxy+=float(np.sum(np.expm1(delta))*adv[i])/(n*p['score_scale'])
        clipped+=float((np.minimum(ratio*adv[i],np.clip(ratio,.8,1.2)*adv[i])-adv[i]).sum())/(n*p['score_scale'])
        ell.append(float(delta.sum()));all_delta.append(delta)
        kl=.5*((mu-oldmu)/mu.new_tensor(p['raw_sigma'])).square().sum(-1)
        meta.append(dict(source_sha256=r['sha256'],path=r['path'],seed=r['seed'],outcome=ys[i],advantage=float(adv[i]),decisions=len(x),
            atoms=int(np.count_nonzero(a['atom'][:len(x)])),conditional_KL_mean=float(kl.mean()),conditional_KL_sum=float(kl.sum())))
    weights=is_metrics(ell,ys)
    result=dict(outcomes=ys,LOO_advantages=adv.tolist(),all_same_return_no_signal=len(set(ys))==1,
        per_action_unclipped_proxy=proxy,per_action_clipped_proxy=clipped,trajectory_logratios=ell,trajectory_weights=weights,
        task_episode_mean_conditional_KL=sum(m['conditional_KL_mean'] for m in meta)/n,
        task_episode_mean_sum_KL=sum(m['conditional_KL_sum'] for m in meta)/n,episodes=meta,
        KL_is_latent_Gaussian_conditional_not_clipped_command_KL=True)
    return result,all_delta

def compute(p,short=False,guard=lambda:None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    science=read(p['science_protocol']);base=M.L.frozen(science,p['initial']['path']);initial=digest(base.state_dict())
    datasets=data(p);arrays={};reports={};cross={};support=True;identity=True
    models=[dict(name='identity',model=p['initial'])]+p['candidates']
    for item in models:
        guard();q=M.L.frozen(science,item['model']['path']);assert sha(item['model']['path'])==item['model']['sha256']
        assert M.L.fixed_hash(q)==M.L.fixed_hash(base);before=digest(q.state_dict());name=item['name'];reports[name]={}
        for split,by in datasets.items():
            reports[name][split]={}
            for k,rr in by.items():
                guard();v,deltas=group(p,base,q,rr,split=='fresh',short)
                reports[name][split][k]=v
                for i,delta in enumerate(deltas):arrays[f'{name}_{split}_{k}_{i}']=delta
                if name=='identity':
                    check=all(np.array_equal(delta,np.zeros_like(delta)) for delta in deltas) and v['per_action_unclipped_proxy']==v['per_action_clipped_proxy']==0.
                    check=check and abs(v['trajectory_weights']['ESS']-len(rr))<1e-12 and abs(v['trajectory_weights']['SNIS_change'])<1e-12
                    assert check;identity=identity and check
                elif split=='fresh':support=support and v['trajectory_weights']['concentration_gate']
        assert digest(q.state_dict())==before
        if name!='identity':
            stored=read(item['summary'])['fit']['nonlinear_panel']
            cross[name]={}
            for k in p['old_ids']:
                train=reports[name]['train'][k];fresh=reports[name]['fresh'][k]
                recorded=stored[k]['sample_surrogate'] if k in stored else None
                error=abs(recorded-train['per_action_unclipped_proxy']) if recorded is not None else None
                if not short and recorded is not None:assert error<1e-10,(name,k,error)
                if recorded is None:assert train['all_same_return_no_signal'] and train['per_action_unclipped_proxy']==0.
                lost=k in read(item['screen'])['lost_ids'];positive=train['per_action_unclipped_proxy']>1e-12
                label='NOT_TRAIN_POSITIVE_DETERMINISTIC_FAILURE'
                if lost and positive:
                    if fresh['all_same_return_no_signal']:label='FRESH_NO_OUTCOME_SIGNAL'
                    elif fresh['per_action_unclipped_proxy']>1e-12:label='POSITIVE_PROXY_DESPITE_DETERMINISTIC_FAILURE_PERSISTS'
                    else:label='TRAIN_POSITIVE_NOT_CONFIRMED_ON_FRESH'
                cross[name][k]=dict(deterministic_screen_lost=lost,recorded_train_proxy=recorded,train_reconstruction_error=error,
                    train_proxy=train['per_action_unclipped_proxy'],fresh_proxy=fresh['per_action_unclipped_proxy'],interpretation=label)
    assert digest(base.state_dict())==initial and all(v.grad is None for v in base.parameters())
    finite_metrics=all(math.isfinite(v['per_action_unclipped_proxy']) and math.isfinite(v['per_action_clipped_proxy'])
        and v['trajectory_weights']['ordinary_IS_finite'] and math.isfinite(v['trajectory_weights']['SNIS_change'])
        for bysplit in reports.values() for bytask in bysplit.values() for v in bytask.values())
    result=dict(state='PROXY_DIAGNOSTIC_COMPUTATION_PASS' if finite_metrics else 'NONFINITE_METRIC_SUPPORT_FAILED',identity_control=identity,
        all_train_and_fresh_metrics_finite=finite_metrics,fresh_all_candidate_task_concentration_gate=support,
        trajectory_IS_support='CONCENTRATION_GATE_PASS_NOT_CONFIDENCE' if support and finite_metrics else 'INSUFFICIENT_CONCENTRATION_SUPPORT',
        original_vs_fresh_deterministic_counterexamples=cross,models=reports,raw_sigma=p['raw_sigma'],
        clipped_probability_rebuilt_not_T684_latent_logp=True,AR_tau2_excluded=True,no_population_or_retention_claim=True,
        train_adaptive_in_sample_IS_not_unbiased_estimate=True,fresh_frozen_candidate_fixed_prior_to_draws=True,
        scientific_learning_admission=False,new_native=0,new_optimizer=0,array_hashes={k:digest(v) for k,v in arrays.items()})
    return result,arrays

def verify(p):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');Wait.resources()
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    for f,h in p['inputs'].items():assert sha(f)==h,f

def execute(action):
    p=read(OUT/'protocol.json');verify(p);start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['main_seconds' if action=='collect' else 'audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>3:verify(p);last[0]=time.time()
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result,arrays=compute(p,False,guard)
    if action=='collect':
        with (OUT/'logratios.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result);save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),physics=0,commands=0,gate=result['state'],new_optimizer=0))
    else:
        assert not Path('/proc',str(read(OUT/'summary.json')['pid'])).exists() and result==read(OUT/'RESULTS.json')
        with np.load(OUT/'logratios.npz',allow_pickle=False) as saved:assert set(saved.files)==set(arrays) and all(np.array_equal(saved[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all_train_fresh_ratios_proxy_and_identity_exact=True,raw_vs_clip_logp_schema_verified=True,new_native=0,new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':execute(sys.argv[1])
