"""Same histories/loss, two fixed reference targets; no optimizer or native calls."""
from pathlib import Path
import fcntl,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t677_reference_target_contract';DOC=ROOT/'stage/docs/weekend_20261120_23/t677'
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
from status_io import save
T,M,torch,np=B.T,B.M,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

def own_targets(q,refs):
    with torch.no_grad():return [dict(g,target=M.location(q,g).detach().clone().clamp(-1,1)) for g in refs]

def loss_gradient(q,variables,refs):
    value,_=M.retention(q,M.pack(refs),False)
    return float(value.detach()),B.A.flatten_gradient(value,variables)

def compute(p,limit):
    start=time.time();last=[0.]
    def guard(phase,key=''):
        B.A.guard(p,start,limit)
        if time.time()-last[0]>3:save(OUT/'status.json',dict(state='ACTIVE',action=sys.argv[1],phase=phase,task=key,child_pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0));last[0]=time.time()
    guard('setup');science=read(p['science_protocol']);science.update(out=str(OUT),arm='t677');T.setup(science)
    q=M.L.frozen(science,p['initial']['path']);q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];assert sum(v.numel() for v in variables)==11906
    base=q.phash();before=[v.detach().clone() for v in variables]
    contract=read(science['reference_contract']);refs=M.references(science,q,lambda:guard('references'))
    provenance={r['task_id']:r['origin'] for r in contract['reference_rows']};assert len(provenance)==387
    assert sum(v=='T501_original_success' for v in provenance.values())==384 and sum(v=='T653_mean_success' for v in provenance.values())==3
    selfrefs=own_targets(q,refs);full={};gradients={};byorigin={};numeric=[]
    for name,rr in [('executed',refs),('self',selfrefs)]:
        guard('initial',name);value,grad=loss_gradient(q,variables,rr);gradients[name]=grad
        full[name]=dict(loss=value,gradient_norm=float(np.linalg.norm(grad)))
        reconstructed=np.zeros(11906);byorigin[name]={}
        for origin in ['T501_original_success','T653_mean_success']:
            group=[g for g in rr if provenance[g['id']]==origin];v,h=loss_gradient(q,variables,group)
            reconstructed+=len(group)/387*h;byorigin[name][origin]=dict(n=len(group),loss=v,gradient_norm=float(np.linalg.norm(h)),weighted_gradient_norm=float(np.linalg.norm(h)*len(group)/387))
        assert np.max(np.abs(grad-reconstructed))<1e-12
        direction=grad/max(np.linalg.norm(grad),1e-30);eps=p['finite_parameter_epsilon']
        # The self-reference target stays frozen while q is perturbed.
        if name=='self':direction=gradients['executed']/max(np.linalg.norm(gradients['executed']),1e-30)
        try:
            B.A.move(q,variables,before,direction,eps);plus=float(M.retention(q,M.pack(rr),False)[0].detach())
            B.A.move(q,variables,before,direction,-eps);minus=float(M.retention(q,M.pack(rr),False)[0].detach())
        finally:B.A.move(q,variables,before,np.zeros(11906))
        actual=(plus-minus)/(2*eps);expected=float(grad@direction)
        numeric.append(dict(target=name,analytic=expected,numeric=actual,error=abs(actual-expected),passed=abs(actual-expected)<=p['numeric_atol']+p['numeric_rtol']*abs(expected)))
    oldresult=read(p['outcome_result']);ids=oldresult['ids']
    with np.load(p['outcome_gradients'],allow_pickle=False) as f:
        outcome=f['task_gradients'].copy();prior_keep=f['nominal387_gradient'].copy();prior_displacement=f['all407_displacement'].copy()
    keep_error=float(np.max(np.abs(gradients['executed']-prior_keep)));assert keep_error<1e-12
    goal=outcome.mean(0);directions={};represented_error={};reference_after={}
    for name in ['executed','self']:
        gl=-goal+science['replay_weight']*gradients[name];gc=gl*min(1.,1./max(np.linalg.norm(gl),1e-30))
        formula=-science['fit_lr']*gc/(np.abs(gc)+science['Adam_eps'])
        try:
            B.A.move(q,variables,before,formula);d=np.concatenate([(v.detach()-b).flatten().numpy().copy() for v,b in zip(variables,before)])
            reference_after[name]={target:float(M.retention(q,M.pack(rr),False)[0].detach()) for target,rr in [('executed',refs),('self',selfrefs)]}
        finally:B.A.move(q,variables,before,np.zeros(11906))
        directions[name]=d;represented_error[name]=float(np.max(np.abs(d-formula)));assert represented_error[name]<1e-12
    assert np.max(np.abs(directions['executed']-prior_displacement))<1e-12
    del refs,selfrefs
    groups=B.dataset(p);nonlinear={name:{} for name in directions}
    panel=[k for k in ids if k.startswith('t502_field_') or (k in p['protected384'] and oldresult['support'][k]['gradient_norm']>1e-12)]
    for key in panel:
        guard('nonlinear',key);g,y=B.A.group(q,groups[key],science)
        with torch.no_grad():
            for item in g:item['initial']=M.location(q,item).detach().clone()
        for name,d in directions.items():
            try:B.A.move(q,variables,before,d);nonlinear[name][key]=B.metrics(q,g,science)
            finally:B.A.move(q,variables,before,np.zeros(11906))
    field=np.array([k.startswith('t502_field_') for k in ids]);old=np.array([k in p['protected384'] for k in ids]);summaries={}
    for name,d in directions.items():
        margins=outcome@d
        summaries[name]=dict(displacement_norm=float(np.linalg.norm(d)),old_negative_ids=[k for k,m,o in zip(ids,margins,old) if o and m<-1e-12],
            field_linear_ascent=float(outcome[field].mean(0)@d),field_nonlinear_mean=sum(nonlinear[name][k]['sample_surrogate'] for k in panel if k.startswith('t502_field_'))/24,
            old_nonlinear_negative=[k for k in panel if k in p['protected384'] and nonlinear[name][k]['sample_surrogate']<-1e-12])
    checks=dict(same387_and_provenance=True,executed_matches_prior_gradient=keep_error<1e-12,self_initial_loss_zero=full['self']['loss']<1e-20,
                self_initial_gradient_zero=full['self']['gradient_norm']<1e-12,source_group_recomposition=True,numeric_all=all(x['passed'] for x in numeric),
                finite=all(np.isfinite(x).all() for x in gradients.values()))
    arrays=dict(executed_retention_gradient=gradients['executed'],self_retention_gradient=gradients['self'],executed_formula_displacement=directions['executed'],self_formula_displacement=directions['self'])
    result=dict(state='REFERENCE_CONTRACT_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,initial=full,origin_groups=byorigin,numeric=numeric,
                prior_gradient_max_error=keep_error,representation_errors=represented_error,reference_losses_after=reference_after,
                relative_direction_change=float(np.linalg.norm(directions['self']-directions['executed'])/np.linalg.norm(directions['executed'])),
                direction_summaries=summaries,nonlinear=nonlinear,only_target_changes_not_loss_metric=True,self_reference_not_extra3_acquisition=True,
                zero_initial_gradient_not_retention_guarantee=True,no_failed_boundary_cost_added=True,no_automatic_learning_admission=True,
                new_native=0,new_optimizer=0,array_hashes={k:digest(v) for k,v in arrays.items()})
    assert q.phash()==base and all(v.grad is None for v in q.parameters());return result,arrays

def execute(action):
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    for folder in p['sealed_sources']:
        seal=read(Path(folder)/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
        assert not Path('/proc',str(read(Path(folder)/'DRIVER_STARTED.json')['pid'])).exists()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();result,arrays=compute(p,p[action+'_seconds'])
    if action=='main':
        with (OUT/'gradients.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        save(OUT/'RESULTS.json',result)
    else:
        assert result==read(OUT/'RESULTS.json')
        with np.load(OUT/'gradients.npz',allow_pickle=False) as f:assert set(f.files)==set(arrays) and all(np.array_equal(f[k],v) for k,v in arrays.items())
        save(OUT/'audit.json',dict(state='PASS',all_reference_targets_gradients_directions_nonlinear_exact=True,new_native=0,new_optimizer=0))
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',elapsed_s=time.time()-start,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':execute(sys.argv[1])
