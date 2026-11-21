"""Full original Actor BPTT contract and first-Adam formula, no optimizer."""
from pathlib import Path
import fcntl,json,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t687_full_actor_gradient_contract';DOC=ROOT/'stage/docs/weekend_20261120_23/t687'
import t687_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
T,M,torch,np=B.T,B.M,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

def layout(q):
    parts=[];offset=0
    for name,v in q.named_parameters():parts.append(dict(name=name,first=offset,n=v.numel(),shape=list(v.shape)));offset+=v.numel()
    assert offset==64772
    adapter=np.concatenate([np.arange(x['first'],x['first']+x['n']) for x in parts if x['name'].startswith(('drive_adapter.','steer_adapter.'))])
    assert len(adapter)==11906
    return parts,adapter

def flatgrad(q):return torch.cat([(v.grad if v.grad is not None else torch.zeros_like(v)).detach().flatten() for v in q.parameters()]).numpy().copy()
def vector(q):return torch.cat([v.detach().flatten() for v in q.parameters()]).numpy().copy()
def move(q,theta):
    k=0
    with torch.no_grad():
        for v in q.parameters():v.copy_(torch.tensor(theta[k:k+v.numel()]).reshape_as(v));k+=v.numel()
    assert k==64772

def arrays(row):
    assert sha(row['path'])==row['sha256'];a=T.R.E.N.S.arrays(row['path'])
    assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
    assert not a['ownership'].any() and a['stochastic'].all()
    return a

def pg(q,rows,p,verify_base=False):
    y=np.array([M.outcome(r,'collision_cost') for r in rows]);adv=y-(y.sum()-y)/(len(y)-1)
    total=next(q.parameters()).new_zeros(())
    for row,weight in zip(rows,adv):
        a=arrays(row);x=torch.tensor(a['states']);mu=q.sequence(x)
        density=M.P.L.density(mu,torch.tensor(a['raw_actions']),torch.tensor(a['atom']),p['raw_sigma'])
        if verify_base:
            assert np.max(np.abs(mu.detach().numpy()-a['locations']))<1e-10
            assert np.max(np.abs(density.detach().numpy()-a['logp']))<1e-9
        ratio=(density-torch.tensor(a['logp'])).exp();w=float(weight)
        total-=(torch.minimum(ratio*w,ratio.clamp(.8,1.2)*w)-w).sum()/(len(y)*p['score_scale'])
    return total

def reference_loss(q,row):
    assert sha(row['source_path'])==row['sha256'];a=T.R.E.N.S.arrays(row['source_path'])
    assert not a['ownership'].any() and not a['stochastic'].any() and a['end_flags'][-1,1] and not a['end_flags'][:-1].any()
    x=torch.tensor(a['states']);target=torch.tensor(a['actions'],dtype=torch.float64)
    return M.P.action_loss(q.sequence(x),target).mean()/.2**2

def numeric(q,p,science,rows,refs):
    before=vector(q);parts,adapter=layout(q)
    def probe():return pg(q,rows,science)+science['replay_weight']*sum(reference_loss(q,r) for r in refs)/len(refs)
    q.zero_grad(set_to_none=True);loss=probe();loss.backward();gradient=flatgrad(q);out=[]
    for index,scope in enumerate(['all','core.','encoder.']):
        direction=np.random.default_rng([68720261121,index]).standard_normal(64772)
        if scope!='all':
            mask=np.zeros(64772,dtype=bool)
            for part in parts:
                if part['name'].startswith(scope):mask[part['first']:part['first']+part['n']]=True
            direction[~mask]=0
        direction/=np.linalg.norm(direction);eps=p['numeric_epsilon']
        try:
            move(q,before+eps*direction);plus=float(probe().detach())
            move(q,before-eps*direction);minus=float(probe().detach())
        finally:move(q,before)
        actual=(plus-minus)/(2*eps);expected=float(gradient@direction);error=abs(actual-expected)
        out.append(dict(scope=scope,analytic=expected,numeric=actual,error=error,passed=error<=p['numeric_atol']+p['numeric_rtol']*abs(expected)))
    # Reset h0 at each complete history; sequence output and step output must
    # agree, including the untrained-but-differentiated recurrent backbone.
    a=arrays(rows[0]);q.sample=False;q.reset_noise();hidden=None;outputs=[]
    with torch.no_grad():
        for x in a['states']:
            unused,mu,hidden,extra=q.step(torch.tensor(x[None]),hidden);outputs.append(mu[0].numpy().copy())
        full=q.sequence(torch.tensor(a['states'])).numpy()
    assert np.max(np.abs(np.stack(outputs)-full))<1e-10 and np.array_equal(vector(q),before)
    q.zero_grad(set_to_none=True)
    return out

def evaluate(q,groups,science,guard):
    report={}
    with torch.no_grad():
        for key,rows in groups.items():
            guard('formula_validation',key);episodes=[];surrogate=0.;y=np.array([M.outcome(r,'collision_cost') for r in rows]);adv=y-(y.sum()-y)/(len(y)-1)
            for row,w in zip(rows,adv):
                a=arrays(row);mu=q.sequence(torch.tensor(a['states']));initial=torch.tensor(a['locations'])
                kl=.5*((mu-initial)/mu.new_tensor(science['raw_sigma'])).square().sum(-1)
                density=M.P.L.density(mu,torch.tensor(a['raw_actions']),torch.tensor(a['atom']),science['raw_sigma'])
                surrogate+=float(((density-torch.tensor(a['logp'])).exp()-1).sum())*float(w)/(len(y)*science['score_scale'])
                episodes.append(dict(mean_KL=float(kl.mean()),sum_KL=float(kl.sum()),max_KL=float(kl.max()),source_sha256=row['sha256']))
            report[key]=dict(task_episode_mean_KL=sum(x['mean_KL'] for x in episodes)/len(episodes),mean_episode_sum_KL=sum(x['sum_KL'] for x in episodes)/len(episodes),
                sample_surrogate=surrogate,episodes=episodes,latent_Gaussian_not_clipped_command_KL=True)
    return report

def compute(p,guard,short=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    science=read(p['science_protocol']);q=M.L.frozen(science,p['initial']['path']);q.requires_grad_(True)
    before=vector(q);basehash=digest(q.state_dict());parts,adapter=layout(q);groups=B.dataset(science)
    refs=read(science['reference_contract'])['reference_rows'];assert len(refs)==387
    probe_refs=[next(r for r in refs if r['origin']==name) for name in ['T501_original_success','T653_mean_success']]
    guard('numeric','');numbers=numeric(q,p,science,groups['dev_task6_00'],probe_refs)
    if short:
        assert all(x['passed'] for x in numbers)
        return dict(numeric=numbers,parameter_dimension=64772,adapter_dimension=11906),{}
    q.zero_grad(set_to_none=True);pg_value=0.;verified=0
    for key,rows in groups.items():
        guard('full_BPTT',key);y=[M.outcome(r,'collision_cost') for r in rows]
        if len(set(y))>1:
            term=pg(q,rows,science,True)/407;term.backward();pg_value+=float(term.detach());del term
        else:
            with torch.no_grad():pg(q,rows,science,True)
        verified+=len(rows)
    gradient_pg=flatgrad(q);retention=0.
    for index,row in enumerate(refs):
        guard('reference_BPTT',row['task_id']);term=science['replay_weight']*reference_loss(q,row)/387
        term.backward();retention+=float(term.detach());del term
    total=flatgrad(q)
    with np.load(science['outcome_gradients'],allow_pickle=False) as saved:expected_pg=-saved['task_gradients'].mean(0)
    with np.load(science['prior_target_gradients'],allow_pickle=False) as saved:expected_keep=saved['executed_retention_gradient']
    pg_error=float(np.max(np.abs(gradient_pg[adapter]-expected_pg)))
    total_error=float(np.max(np.abs(total[adapter]-(expected_pg+science['replay_weight']*expected_keep))))
    norms={part['name']:float(np.linalg.norm(total[part['first']:part['first']+part['n']])) for part in parts}
    outside=np.ones(64772,dtype=bool);outside[adapter]=False
    semantic=dict(all1724_full_causal_histories=verified==1724,full64772_finite=bool(np.isfinite(total).all()),
        adapter_PG_matches_original=pg_error<1e-10,adapter_joint_matches_original=total_error<1e-10,
        numerical_full_GRU_encoder_all=all(x['passed'] for x in numbers),new_scope_gradient_nonzero=float(np.linalg.norm(total[outside]))>1e-12)
    norm=float(np.linalg.norm(total));clipped=total*min(1.,1./max(norm,1e-30))
    formula=-science['fit_lr']*clipped/(np.abs(clipped)+science['Adam_eps']);validation=None;displacement=None
    try:
        move(q,before+formula);displacement=vector(q)-before
        representation_error=float(np.max(np.abs(displacement-formula)));assert representation_error<1e-12
        validation=evaluate(q,groups,science,guard)
    finally:move(q,before);q.zero_grad(set_to_none=True)
    assert np.array_equal(vector(q),before) and digest(q.state_dict())==basehash
    panel=p['KL_panel_ids'];assert set(panel)<=set(validation)
    maxkl=max(validation[k]['task_episode_mean_KL'] for k in panel)
    admitted=all(semantic.values()) and np.isfinite(displacement).all() and maxkl<=science['max_task_latent_KL']
    result=dict(state='FULL_ACTOR_GRADIENT_CONTRACT_PASS' if all(semantic.values()) else 'GRADIENT_CONTRACT_FAILED',checks=semantic,
        full_scope_pilot_numerically_admitted=bool(admitted),same_T678_fixed_lr=science['fit_lr'],norm_before_clip=norm,
        adapter_pg_error=pg_error,adapter_joint_error=total_error,new_scope_gradient_norm=float(np.linalg.norm(total[outside])),
        per_tensor_gradient_norms=norms,parameter_layout=parts,numeric=numbers,PG_loss_initial=pg_value,weighted_retention_initial=retention,
        represented_first_Adam_formula_error=representation_error,max_original_panel_task_mean_latent_KL=maxkl,
        max_all407_task_mean_latent_KL=max(v['task_episode_mean_KL'] for v in validation.values()),KL_limit=science['max_task_latent_KL'],
        full407_history_formula_validation=validation,no_optimizer_or_native=True,no_checkpoint_exported=True,
        first_Adam_formula_NOT_executed_Adam=True,scope_geometry_not_capacity_or_safety_proof=True,all_parameter_state_restored=True)
    arrays_out=dict(full_pg_loss_gradient=gradient_pg,full_joint_loss_gradient=total,full_formula_displacement=displacement,adapter_indices=adapter)
    result['array_hashes']={k:digest(v) for k,v in arrays_out.items()}
    return result,arrays_out

def verify(p):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json');Wait.resources()
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    for f,h in p['inputs'].items():assert sha(f)==h,f

def execute(action):
    p=read(OUT/'protocol.json');verify(p);start=time.time();last=[0.]
    def guard(phase,key):
        assert time.time()<min(start+p['main_seconds' if action=='collect' else 'audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:
            verify(p);last[0]=time.time();save(OUT/'status.json',dict(state='ACTIVE',action=action,phase=phase,task=key,child_pid=os.getpid(),epoch=time.time()))
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result,values=compute(p,guard)
    if action=='collect':
        with (OUT/'gradients.npz').open('xb') as f:np.savez_compressed(f,**values)
        save(OUT/'RESULTS.json',result);save(OUT/'summary.json',dict(state='COMPLETE',pid=os.getpid(),physics=0,commands=0,gate=result['state'],new_optimizer=0))
    else:
        assert not Path('/proc',str(read(OUT/'summary.json')['pid'])).exists() and result==read(OUT/'RESULTS.json')
        with np.load(OUT/'gradients.npz',allow_pickle=False) as saved:assert set(saved.files)==set(values) and all(np.array_equal(saved[k],v) for k,v in values.items())
        save(OUT/'audit.json',dict(state='PASS',full_BPTT_and_adapter_gradient_identity_and_formula_exact=True,all_arrays_numeric_and_full407_history_metrics_reconstructed=True,
            new_native=0,new_optimizer=0,full_scope_pilot_numerically_admitted=result['full_scope_pilot_numerically_admitted']))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':execute(sys.argv[1])
