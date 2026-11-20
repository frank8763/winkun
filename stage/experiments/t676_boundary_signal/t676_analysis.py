"""Recorded boundary severity and finite outcome-score support, zero policy steps."""
from pathlib import Path
import fcntl,gzip,os,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t676_boundary_signal';DOC=ROOT/'stage/docs/weekend_20261120_23/t676'
sys.path.insert(0,str(ROOT/'stage/experiments/t673_all_task_signal_gate'))
import t673_analysis as B
from status_io import save
T,M,torch,np=B.T,B.M,B.torch,B.np
read,sha,digest=B.read,B.sha,B.digest

def boundary(row,a):
    poses=a['poses'];nextposes=a['next_poses'];assert poses.shape==nextposes.shape and poses.shape[1]==3
    assert np.isfinite(poses).all() and np.isfinite(nextposes).all()
    excess=np.maximum(np.abs(poses[:,:2])-7.,0.).max(axis=1)
    peak=float(excess.max());cross=bool(peak>0);assert cross==row['field_boundary']
    applies='fullfield_start_pose' in row['request']['job']
    indexes=np.flatnonzero(excess>0)
    final_excess=float(np.maximum(np.abs(nextposes[-1,:2])-7.,0.).max())
    return dict(applicable=applies,raw_boundary=cross,cost=peak/7. if applies else 0.,recorded_peak_excess_m=peak,
                first_recorded_crossing=int(indexes[0]) if len(indexes) else None,recorded_peak_index=int(excess.argmax()),
                final_next_pose_excess_m=final_excess,terminal_extra_crossing=bool(final_excess>0 and not cross),
                source_sha256=row['sha256'])

def cost_group(q,rows,science):
    group,ys=B.A.group(q,rows,science);values=[]
    for row,g in zip(rows,group):
        a=T.R.E.N.S.arrays(row['path']);values.append(boundary(row,a))
    costs=np.array([v['cost'] for v in values]);n=len(costs)
    for i,g in enumerate(group):g['adv']=float(-costs[i]+(costs.sum()-costs[i])/(n-1))
    return group,costs,values,ys

def compute(p,limit):
    start=time.time();last=[0.]
    def guard(phase,key):
        B.A.guard(p,start,limit)
        if time.time()-last[0]>3:save(OUT/'status.json',dict(state='ACTIVE',action=sys.argv[1],phase=phase,task=key,child_pid=os.getpid(),epoch=time.time(),new_native=0,new_optimizer=0));last[0]=time.time()
    guard('setup','');science=read(p['science_protocol']);science.update(out=str(OUT),arm='t676');T.setup(science)
    q=M.L.frozen(science,p['initial']['path']);q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];before=[v.detach().clone() for v in variables];base=q.phash()
    allgroups=B.dataset(p);field={k:v for k,v in allgroups.items() if 'fullfield_start_pose' in v[0]['request']['job']};assert len(field)==24
    oldsource=read(p['outcome_result']);oldids=oldsource['ids']
    with np.load(p['outcome_gradients'],allow_pickle=False) as f:oldgrad=f['task_gradients'].copy();nominal=f['nominal387_gradient'].copy()
    exact={}
    import json
    for filename in p['exact_cases']:
        for r in [json.loads(line) for line in gzip.decompress(Path(filename).read_bytes()).decode().splitlines()]:
            if r.get('exact'):assert r['reference_path'] not in exact;exact[r['reference_path']]=r
    # All nonfield tasks must remain outside the field-only cost scope,
    # even when a valid original 8m trajectory leaves the 7m square.
    nonfield_outside=0
    for key,rows in allgroups.items():
        if key in field:continue
        for row in rows:
            if row['field_boundary']:nonfield_outside+=1
            assert 'fullfield_start_pose' not in row['request']['job']
    results={};grads=[];halves=[];numeric=[];omitted={}
    for key,rows in field.items():
        guard('score',key);group,costs,values,ys=cost_group(q,rows,science);assert len(group)==8
        for row in rows:
            e=exact[row['path']];assert sha(e['path'])==e['sha256']
            assert e['array_hashes']==row['array_hashes'] and e['integration_sha256']==row['integration_sha256']
            assert e['policy_hash']==base and e['seed']==row['seed']
        scores=[]
        if float(np.ptp(costs))>p['variation_floor']:
            for g in group:scores.append(B.A.flatten_gradient(M.P.L.density(M.location(q,g),g['raw'],g['atom'],science['raw_sigma']).sum()/science['score_scale'],variables))
            scores=np.stack(scores);gradient=B.combination(scores,-costs)
            direct=B.A.flatten_gradient(B.A.outcome_surrogate(q,group,science),variables);assert np.max(np.abs(direct-gradient))<1e-10
            first=B.combination(scores[:4],-costs[:4]);lastg=B.combination(scores[4:],-costs[4:])
            contribution=np.linalg.norm(scores*((-costs+(costs.sum()-costs)/7)[:,None])/8,axis=1)
            omitted[key]=[];omission_rows=[]
            for i in range(8):
                selected=np.arange(8)!=i;v=B.combination(scores[selected],-costs[selected]);vn=float(np.linalg.norm(v));den=float(vn*np.linalg.norm(gradient))
                omitted[key].append(v);omission_rows.append(dict(removed=i,norm=vn,cosine=float(v@gradient/den) if den>1e-24 else None))
        else:gradient=np.zeros(11906);first=gradient.copy();lastg=gradient.copy();contribution=np.zeros(8);omission_rows=[]
        norm=float(np.linalg.norm(gradient));dot=float(first@lastg);den=float(np.linalg.norm(first)*np.linalg.norm(lastg))
        results[key]=dict(measurements=values,costs=costs.tolist(),outcomes=ys,cost_range=float(np.ptp(costs)),gradient_norm=norm,
            first_half_norm=float(np.linalg.norm(first)),last_half_norm=float(np.linalg.norm(lastg)),halves_inner_product=dot,
            halves_cosine=dot/den if den>1e-24 else None,split_supported=dot>0 and norm>1e-12,
            split_is_not_independent_new_data=True,contribution_norms=contribution.tolist(),
            maximum_contribution_fraction=float(contribution.max()/max(contribution.sum(),1e-30)),leave_one_trajectory_out=omission_rows)
        grads.append(gradient);halves.extend([first,lastg])
        if key in p['numeric_ids']:
            direction=gradient/max(norm,1e-30);eps=p['finite_parameter_epsilon']
            try:
                B.A.move(q,variables,before,direction,eps);plus=float(B.A.outcome_surrogate(q,group,science).detach())
                B.A.move(q,variables,before,direction,-eps);minus=float(B.A.outcome_surrogate(q,group,science).detach())
            finally:B.A.move(q,variables,before,np.zeros(11906))
            expected=float(gradient@direction);actual=(plus-minus)/(2*eps)
            numeric.append(dict(id=key,analytic=expected,numeric=actual,passed=norm>1e-12 and abs(actual-expected)<=p['numeric_atol']+p['numeric_rtol']*abs(expected),error=abs(actual-expected)))
    grads=np.stack(grads);rewardgrad=grads.mean(0);gl=-rewardgrad+science['replay_weight']*nominal
    gc=gl*min(1.,1./max(np.linalg.norm(gl),1e-30));formula=-science['fit_lr']*gc/(np.abs(gc)+science['Adam_eps'])
    try:
        B.A.move(q,variables,before,formula);d=np.concatenate([(v.detach()-b).flatten().numpy().copy() for v,b in zip(variables,before)])
    finally:B.A.move(q,variables,before,np.zeros(11906))
    assert np.max(np.abs(d-formula))<1e-12
    for key,values in omitted.items():
        for item,v in zip(results[key]['leave_one_trajectory_out'],values):item['proposal_directional_margin']=float(v@d)
    nonlinear_cost={};nonlinear_old={};oldnegative=[]
    oldmixed=[k for k in oldids if k in p['protected384'] and oldsource['support'][k]['gradient_norm']>1e-12];assert len(oldmixed)==44
    for label,key in [('cost',k) for k in field]+[('old',k) for k in oldmixed]:
        guard('nonlinear',key)
        if label=='cost':g,_,_,_=cost_group(q,field[key],science)
        else:g,_=B.A.group(q,allgroups[key],science)
        with torch.no_grad():
            for item in g:item['initial']=M.location(q,item).detach().clone()
        try:B.A.move(q,variables,before,d);value=B.metrics(q,g,science)
        finally:B.A.move(q,variables,before,np.zeros(11906))
        if label=='cost':nonlinear_cost[key]=value
        else:
            nonlinear_old[key]=value
            if value['sample_surrogate']<-1e-12:oldnegative.append(key)
    mean_sur=sum(nonlinear_cost[k]['sample_surrogate'] for k in field)/24
    varying=[k for k,v in results.items() if v['cost_range']>p['variation_floor']]
    stable=[k for k,v in results.items() if v['cost_range']>p['variation_floor'] and v['split_supported']]
    checks=dict(all24_measurement_and_exact=True,nonfield_scope_exempt=nonfield_outside>0,at_least2_varying=len(varying)>=2,
                at_least2_split_supported=len(stable)>=2,numeric_all=len(numeric)==len(p['numeric_ids']) and all(x['passed'] for x in numeric),
                cost_direction_ascent=bool(rewardgrad@d>1e-12),cost_nonlinear_ascent=mean_sur>1e-12,finite=bool(np.isfinite(grads).all()))
    goalgrad=oldgrad[[i for i,k in enumerate(oldids) if k.startswith('t502_field_')]].mean(0)
    result=dict(state='BOUNDARY_SIGNAL_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,field_ids=list(field),tasks=results,
                varying_ids=varying,split_supported_ids=stable,numeric=numeric,nonfield_outside_unpenalized=nonfield_outside,
                negative_cost_linear=float(rewardgrad@d),negative_cost_nonlinear=mean_sur,field_goal_linear=float(goalgrad@d),
                old44_linear_margins={k:float(oldgrad[oldids.index(k)]@d) for k in oldmixed},old44_nonlinear_negative=oldnegative,nonlinear_cost=nonlinear_cost,nonlinear_old=nonlinear_old,
                recorded_cost_not_continuous_native_safety=True,first_Adam_formula_not_Actor_step=True,does_not_authorize_learning=True,new_native=0,new_optimizer=0)
    arrays=dict(field_cost_gradients=grads,split_gradients=np.stack(halves),formula_displacement=d)
    result['array_hashes']={k:digest(v) for k,v in arrays.items()};assert q.phash()==base and all(v.grad is None for v in q.parameters())
    return result,arrays

def execute(action):
    p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    for folder in p['sealed_sources']:
        s=read(Path(folder)/'SEALED.json');assert s['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
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
        save(OUT/'audit.json',dict(state='PASS',all_boundary_cost_score_split_numeric_nonlinear_exact=True,new_native=0,new_optimizer=0))
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')<p['global_bytes']
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',elapsed_s=time.time()-start,new_native=0,new_optimizer=0));fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':execute(sys.argv[1])
