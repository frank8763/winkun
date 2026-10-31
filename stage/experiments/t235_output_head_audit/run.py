"""One bounded linear solve; only a validated offline fit may be exported."""
from pathlib import Path
import sys,os,io,gzip,json,time,copy,fcntl,shutil,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t235_output_head_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t219_rnd_runtime'))
import support as S
C,R,np,torch=S.C,S.R,S.np,S.torch

def load(p):return torch.load(io.BytesIO(gzip.decompress(Path(p).read_bytes())),map_location='cpu',weights_only=False)['policy']
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'summary.json').exists(),'No duplicateT235';plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();solves=0;export=None
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);state=load(plan['checkpoint']);p=S.policy(state).eval().requires_grad_(False);teacher=S.policy(load(plan['teacher'])).eval().requires_grad_(False)
    feature=lambda x:p.body.net[:4](x).double();native=lambda q,x:C.ParentPolicy.forward(q,x)
    groups=[]
    with np.load(plan['retention_targets']) as z:x=torch.tensor(z['states']);offset=z['offsets']
    for i in range(7):groups.append([x[offset[i]:offset[i+1]][::plan['retain_stride']]])
    with np.load(plan['original_left_states']) as z:groups.append([torch.tensor(z['states'])[::plan['retain_stride']]])
    for r in plan['retained_sources']:
        with np.load(r['path']) as z:groups[r['task']].append(torch.tensor(z['states'])[::plan['retain_stride']])
    groups=[torch.cat(g) for g in groups];rx=torch.cat(groups);rw=torch.cat([torch.full((len(g),),plan['retention_weight']/8/len(g),dtype=torch.float64) for g in groups])
    trainx=[];holdx=[];tw=[];hw=[]
    for source in plan['hard_sources']:
        with np.load(source['path']) as z:x=torch.tensor(z['states'])
        ind=np.arange(len(x));mask=(ind%plan['hard_holdout_every'])==0;trainx.append(x[~mask]);holdx.append(x[mask]);tw.append(torch.full((int((~mask).sum()),),.5/int((~mask).sum()),dtype=torch.float64));hw.append(torch.full((int(mask.sum()),),.5/int(mask.sum()),dtype=torch.float64))
    hx=torch.cat(trainx);tx=torch.cat(holdx);w=torch.cat(tw)
    with torch.no_grad():
        fr=feature(rx);fh=feature(hx);ft=feature(tx);mean=torch.cat([fr,fh]).mean(0);std=torch.cat([fr,fh]).std(0,unbiased=False).clamp_min(plan['feature_std_floor'])
        design=lambda f:torch.cat([(f-mean)/std,torch.ones((len(f),1),dtype=torch.float64)],dim=1)
        zr,zh,zt=map(design,[fr,fh,ft]);delta=(native(teacher,hx)-native(p,hx)).double();baseline=(teacher(tx)-p(tx)).double()
        A=zh.T@(w[:,None]*zh)+zr.T@(rw[:,None]*zr)+plan['ridge_lambda']*torch.eye(zh.shape[1],dtype=torch.float64);rhs=zh.T@(w[:,None]*delta)
        coef=torch.linalg.solve(A,rhs);solves=1;assert bool(torch.isfinite(coef).all());residual=float((A@coef-rhs).norm()/rhs.norm().clamp_min(1e-30))
        q=copy.deepcopy(p);raw=coef[:-1]/std[:,None];bias=coef[-1]-mean@raw;q.direct_head.weight.add_(raw.T.float());q.direct_head.bias.add_(bias.float())
        assert all(torch.equal(v,q.state_dict()[k]) for k,v in p.state_dict().items() if not k.startswith('direct_head.'))
        for xx in [hx,tx,rx]:assert torch.isfinite(q(xx)).all()
        d=(q(tx)-teacher(tx)).double();retain=(q(rx)-p(rx)).abs().double();ratio=float(d.square().mean().sqrt()/baseline.square().mean().sqrt().clamp_min(1e-30));hardmax=float(d.abs().max());r99=float(torch.quantile(retain.flatten(),.99));rmax=float(retain.max());fold_error=float((native(q,tx).double()-native(p,tx).double()-zt@coef).abs().max())
    gates=plan['gates'];passed=ratio<=gates['hard_heldout_RMSE_fraction_max'] and hardmax<=gates['hard_heldout_native_max'] and r99<=gates['retention_native_q99_max'] and rmax<=gates['retention_native_max'] and residual<1e-8 and time.monotonic()-start<plan['wall_seconds']
    if passed:
        def guard(n):assert S.used(OUT)+n<plan['final_bytes'] and S.used(ROOT/'stage/runs')+n<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
        export=S.save_gzip(OUT/'offline_candidate_Actor.pt.gz',dict(policy=q.state_dict(),source='T235onefrozenfeatureridgesolve',protocol_sha256=C.sha(HERE/'protocol.json'),RL_updates=0),guard)
    summary=dict(state='OFFLINE_GATE_PASS' if passed else 'OFFLINE_GATE_FAILED',physics_steps=0,optimizer_calls=0,supervised_linear_solves=solves,train_states=len(hx),correlated_heldout_states=len(tx),retention_states=len(rx),linear_system_dimension=len(coef),normal_equation_residual=residual,baseline_hard_heldout_RMSE=float(baseline.square().mean().sqrt()),new_hard_heldout_RMSE=float(d.square().mean().sqrt()),hard_heldout_RMSE_fraction=ratio,hard_heldout_native_max=hardmax,retention_native_q99=r99,retention_native_max=rmax,fold_to_float32_native_max_error=fold_error,all_non_head_parameters_unchanged=True,changed_parameter_count=q.direct_head.weight.numel()+q.direct_head.bias.numel(),export=export,automatic_promotion=False,navigation_test_started=False,stable_5m_claim=False,elapsed_s=time.monotonic()-start,own_bytes=S.used(OUT),runs_MiB=S.used(ROOT/'stage/runs')/1024**2)
    for f,h in plan['input_hashes'].items():assert C.sha(f)==h,f
    C.save(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
    assert solves<=plan['max_solves'] and time.monotonic()-start<plan['wall_seconds'] and S.used(OUT)<plan['final_bytes'] and S.used(ROOT/'stage/runs')<plan['global_bytes']
if __name__=='__main__':main()
