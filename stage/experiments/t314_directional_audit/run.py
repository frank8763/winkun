"""Read-only full/minibatch gradient audit. No optimizer or proposals."""
from pathlib import Path
import sys,json,os,time,math,copy,io,gzip,signal,traceback,argparse,shutil
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t313_fisher_feasibility'))
import t313_core as U
np,torch=U.np,U.torch
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t314_directional_audit';DOC=ROOT/'stage/docs/distance_20261103'
read=U.read

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert U.K.C.sha(f)==h,f
    assert U.K.S.used(ROOT/'stage/runs')-U.K.S.used(OUT)+p['final_bytes']<=p['global_bytes']
    assert U.K.S.used(OUT)<=p['final_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def compute(p,guard,progress):
    items,fitted=U.baseline_batch(p);q=U.K.policy(p);initial=U.K.S.digest(q.state_dict());x0=U.flat(q);params=[v for v in q.parameters() if v.requires_grad];calls=0
    parent=copy.deepcopy(q).eval().requires_grad_(False);ax,ay,offset=U.K.T.anchors(p,parent)
    def gradient(value):
        nonlocal calls
        guard();gg=torch.autograd.grad(value,params,allow_unused=True);calls+=1
        v=torch.cat([torch.zeros_like(z).reshape(-1) if g is None else g.detach().reshape(-1) for z,g in zip(params,gg)]).double()
        assert torch.isfinite(v).all();return v
    rng=np.random.default_rng(p['seed']);indices=[rng.integers(0,len(i['returns']),16) for i in items['mc']]
    oi=np.concatenate([rng.integers(offset[k],offset[k+1],16) for k in range(8)])
    keep_full=torch.zeros_like(x0)
    for k in range(8):keep_full+=gradient((q.step(ax[offset[k]:offset[k+1]])[0]-ay[offset[k]:offset[k+1]]).square().mean()*p['retention_weight']/8)
    keep_mini=gradient((q.step(ax[oi])[0]-ay[oi]).square().mean()*p['retention_weight'])
    measurements=read(ROOT/'stage/runs/t313_fisher_feasibility/metrics.json');results=[]
    for arm in ['mc','gae']:
        batch,target=fitted[arm];full=torch.zeros_like(x0);mini=torch.zeros_like(x0);six={};scale=max(.05,float(np.sqrt(np.mean([np.mean(i['returns']**2) for i in batch]))))
        for i,t,ix in zip(batch,target,indices):
            a=i['arrays'];lp=U.K.lp(q,a);assert np.array_equal(lp.detach().numpy(),a['logp'])
            c=torch.tensor(t[arm]*t['prefix']/scale);gi=gradient(torch.dot(lp,c)/len(c));full+=gi/16
            if '_d6_' in i['row']['goal_family']:six[i['row']['goal_family']]=six.get(i['row']['goal_family'],torch.zeros_like(x0))+gi/2
            lm=U.K.lp(q,a,ix);mini+=gradient(torch.dot(lm,c[ix])/len(ix))/16
            progress(arm,calls)
        gf=full-keep_full;gm=mini-keep_mini;reference=next(u for u in measurements['historical_Adam_reconstructions'] if u['arm']==arm)['steps'][0]['gradient_norm']
        relative=abs(float(gm.norm())-reference)/max(1.,reference);assert relative<=p['streamed_gradient_norm_relative_tolerance']
        for method in ['adam','empirical_fisher']:
            ref=next(r for r in measurements['results'] if r['arm']==arm and r['method']==method);model=ref['model'];assert U.K.C.sha(model['path'])==model['sha256']
            payload=torch.load(io.BytesIO(gzip.decompress(Path(model['path']).read_bytes())),map_location='cpu',weights_only=False)
            xc=torch.cat([payload['policy'][n].reshape(-1) for n,v in q.named_parameters() if v.requires_grad]).double();d=xc-x0;assert float(d.norm())>0
            cosine=lambda a,b:float(torch.dot(a,b)/(a.norm()*b.norm()))
            pred=float(torch.dot(gf,d));last=ref['trials'][-1]
            results.append(dict(arm=arm,method=method,candidate_policy_hash=model['policy_hash'],candidate_delta_sha256=U.K.S.digest(d),candidate_delta_norm=float(d.norm()),
                full_gradient_sha256=U.K.S.digest(gf),minibatch_gradient_sha256=U.K.S.digest(gm),full_gradient_norm=float(gf.norm()),minibatch_gradient_norm=float(gm.norm()),
                original_minibatch_gradient_norm=reference,streamed_norm_relative_error=relative,full_minibatch_gradient_cosine=cosine(gf,gm),
                full_objective_directional_derivative=pred,minibatch_objective_directional_derivative=float(torch.dot(gm,d)),
                full_surrogate_directional_derivative=float(torch.dot(full,d)),full_retention_directional_derivative=float(torch.dot(keep_full,d)),
                full_direction_cosine=cosine(gf,d),minibatch_direction_cosine=cosine(gm,d),six_goal_surrogate_directional_derivatives={g:float(torch.dot(v,d)) for g,v in six.items()},
                actual_finite_objective_change=last['improvement'],finite_change_minus_local_linear_prediction=last['improvement']-pred,
                finite_retention_cost=last['value']['weighted_retention']-ref['base']['weighted_retention'],proxy_accepted=ref['accepted']))
        assert U.K.S.digest(q.state_dict())==initial and all(v.grad is None for v in params)
    assert calls==p['max_autograd_calls']==73
    return dict(results=results,autograd_calls=calls,source_model_unchanged=True,Adam_calls=0,parameter_proposals=0,native_steps=0,
        gradient_evaluation_semantics='At original Actor: clip ratio1, exact old density; streaming algebra with double accumulation, bounded numerical norm check vs original batched gradient, not original bitwise gradient identity.',
        finite_remainder_semantics='Includes all nonlinear policy/clip/retention/float effects, not a Hessian or causal safety estimate.',finite_fixed_data_not_independent_seeds=True)

def forbid_adam(*args,**kwargs):raise AssertionError('T314 forbids every Adam.step')

def preflight():
    assert not (OUT/'preflight.json').exists() and not (OUT/'launch.json').exists();p=read(HERE/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=torch.optim.Adam.step;torch.optim.Adam.step=forbid_adam
    try:
        items,_=U.baseline_batch(p);q=U.K.policy(p);h=U.K.S.digest(q.state_dict());a=items['mc'][0]['arrays'];value=U.K.lp(q,a,np.asarray([0]));assert float(value.detach())==float(a['logp'][0]);torch.autograd.grad(value.sum(),[v for v in q.parameters() if v.requires_grad],allow_unused=True)
        assert h==U.K.S.digest(q.state_dict())
    finally:torch.optim.Adam.step=old
    U.K.C.save(OUT/'preflight.json',dict(state='PASS',native_steps=0,Adam_calls=0,parameter_proposals=0,autograd_calls=1,real_density_exact=True,Actor_hash_unchanged=True,Adam_step_explicitly_forbidden=True,protocol_sha256=U.K.C.sha(HERE/'protocol.json')))
    print('T314 zero-step gradient preflight PASS')

def execute(audit=False):
    p=read(HERE/'protocol.json');start=time.time();dest=DOC/'T314_AUDIT.json';meter=None;state='FAILED';result=None;old=torch.optim.Adam.step;torch.optim.Adam.step=forbid_adam
    if audit:
        assert not dest.exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
        for pid in [s['pid'],read(OUT/'launch.json')['pid']]:
            try:os.kill(pid,0)
            except ProcessLookupError:continue
            raise AssertionError(('Process still exists',pid))
    else:assert not (OUT/'status.json').exists()
    def guard():assert time.time()<start+(p['audit_seconds'] if audit else p['wall_seconds'])-10
    def progress(arm,calls):
        if not audit:U.K.C.save(OUT/'status.json',dict(state=arm,pid=os.getpid(),main_epoch=start,updated_epoch=time.time(),autograd_calls=calls,Adam_calls=0,native_steps=0))
    try:
        verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        if not audit:meter=U.K.Budget(ROOT,OUT,p,start_epoch=start)
        result=compute(p,guard,progress)
        if audit:
            assert result==read(OUT/'metrics.json');U.K.C.save(dest,dict(state='PASS',native_added=0,Adam_calls=0,parameter_proposals=0,autograd_calls=73,all_vectors_derivatives_model_inputs_exact=True,all_source_SHA_exact=True,elapsed_s=time.time()-start,summary_sha256=U.K.C.sha(OUT/'summary.json')))
        else:U.K.C.save(OUT/'metrics.json',result);meter.checkpoint()
        state='COMPLETE'
    except BaseException:
        if audit:raise
        U.K.C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),native_steps=0,Adam_calls=0,parameter_proposals=0))
    finally:
        torch.optim.Adam.step=old
        if meter:meter.close()
    if not audit:
        s=dict(state=state,pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,all_workers_stopped=True,workers_spawned=0,commands=0,physics_steps=0,Adam_calls=0,parameter_proposals=0,
            autograd_calls=None if result is None else result['autograd_calls'],own_bytes=U.K.S.used(OUT),global_bytes=U.K.S.used(ROOT/'stage/runs'),six_metre_navigation_evaluated=False,deployed_Actor_changed=False,automatic_training=False)
        U.K.C.save(OUT/'summary.json',s);U.K.C.save(OUT/'status.json',s)
    print('T314 independent audit PASS' if audit else json.dumps(s))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main','audit']);a=parser.parse_args();preflight() if a.mode=='preflight' else execute(a.mode=='audit')
