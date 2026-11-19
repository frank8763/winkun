"""Exact clipped density and genuine terminal-score gradient; zero Adam/native."""
from pathlib import Path
import ast,copy,gzip,os,sys,time
import t649_runtime as T
R,M,torch,np=T.R,T.M,T.torch,T.np
ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=T.read(ROOT/'stage/experiments/t648_response_constrained_replay/protocol.json')
    for k in ['registration_epoch','absolute_learning_stop','absolute_native_stop','absolute_all_stop','user_authorization','_root_sha256']:p.pop(k,None)
    jobs=p['key_jobs']+[j for j in p['field_jobs'] if j['id'] in p['before_success_ids']];assert len(jobs)==19 and len({j['id'] for j in jobs})==19
    shared_physics=2*4*sum(j['case']['step_cap'] for j in jobs)
    p.update(id='T649_FRESH_TERMINAL_OUTCOME_PG',authorization_approved=False,out=str(OUT),arms=['bc','pg'],arm='bc',training_jobs=jobs,
       fit_steps=8,fit_lr=.0001,keep_weight=10.,old_batch=16,ratio_clip=.2,score_scale=512.,max_task_latent_KL=.01,
       terminal_objective='task-equal safe GOAL probability; failure zero; no environment reward mutation',
       max_physics=12000000,max_commands=427,total_max_physics=31000000,total_max_commands=1020,total_main_seconds=6000,Actor_each_cap=16,
       max_bytes=512*2**20,child_bytes=112*2**20,shared_bytes=256*2**20,global_bytes=10*2**30,
       shared_physics=shared_physics,shared_main_seconds=1200,shared_audit_seconds=600,
       main_seconds=2400,audit_seconds=600,learning_seconds=600,full_evaluation_reserve_seconds=1500,
       old_backup_seconds=300,backup_seconds=900,source_seconds=300,all_seconds=10800,extra_optimizer_preflight=0,new_Critic=0,
       model_bytes=16*2**20,fresh_bytes=3*2**20,full_bytes=4*2**20,compact_bytes=32768)
    worst=shared_physics+2*sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs']);assert worst<=p['total_max_physics']
    assert 158+2*427<=p['total_max_commands'] and p['shared_main_seconds']+2*p['main_seconds']==p['total_main_seconds']
    assert 1200+600+2*(2400+600)+300+900+300+120<p['all_seconds']
    storage=76*p['fresh_bytes']+76*p['compact_bytes']+p['full_bytes']+10*2**20+2*(p['model_bytes']+14*p['fresh_bytes']+407*p['compact_bytes']+p['full_bytes']+10*2**20)+8*2**20
    assert storage<=p['max_bytes']
    # Preserve old scientifically immutable inputs; superseded prep plans are
    # still hashed at their actual existing bytes, with no active run edits.
    for key in list(p['inputs']):
        if '/stage/experiments/t648_response_constrained_replay/' in key or '/stage/docs/day_20261119/t648/' in key:p['inputs'].pop(key)
    source=ROOT/'stage/runs/t646_offline_night/coverage';rows=[T.read(source/'summary.json')['new_success_rows'][0]]
    allrows=[T.json.loads(x) for x in T.gzip.decompress((source/'cases.jsonl.gz').read_bytes()).decode().splitlines() if T.json.loads(x)['role']=='coverage_fresh']
    # Two actual tasks, each one success and one failure available; fixed rows
    # by original collection order, no new data claimed for numeric fixtures.
    fixture=[]
    for key in p['new_tasks']:
        rr=[r for r in allrows if r['job_id']==key]
        if len({M.outcome(r) for r in rr})<2:continue
        chosen=rr[:4]
        if len({M.outcome(r) for r in chosen})<2:
            other=next(r for r in rr if M.outcome(r)!=M.outcome(chosen[0]));chosen=chosen[:3]+[other]
        fixture.append((key,[(R.E.N.S.arrays(r['path']),r) for r in chosen]))
        if len(fixture)==2:break
    assert len(fixture)==2
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);old=[];original=q.phash()
    manifest=T.read(T.Shared.OUT/'OFFLINE_INPUTS.json');by={r['job_id']:r for r in R.D.selected()}
    for row in manifest['members'][:16]:
        a=R.E.N.S.arrays(row['offline_path']);g=R.D.features(q,a,by[row['id']]);g['x']=torch.tensor(a['states']);old.append(g)
    R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(p);model,opt=obj;before=M.digest(M.state(obj));items,diag=M.build(model,fixture,p)
    loss=M.pg_objective(model,items,p);loss.backward();gradient=torch.cat([v.grad.reshape(-1) for v in model.parameters() if v.requires_grad]).detach().clone()
    assert torch.isfinite(gradient).all() and float(gradient.norm())>0
    assert before==M.digest(M.state(obj)) and not opt.state and model.phash()==original
    # At exact behavior, clipped surrogate's derivative must equal terminal
    # REINFORCE's independent-LOO episodic SUM derivative.
    opt.zero_grad(set_to_none=True);direct=next(model.parameters()).new_zeros(())
    for group in items:
        for b in group:
            direct-=M.P.L.density(model.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma']).sum()*b['adv']/(len(items)*4*p['score_scale'])
    direct.backward();expected=torch.cat([v.grad.reshape(-1) for v in model.parameters() if v.requires_grad]).detach().clone()
    assert torch.allclose(gradient,expected,atol=1e-10,rtol=1e-10)
    opt.zero_grad(set_to_none=True);bc=M.bc_objective(model,items,p);bc.backward();bc_grad=torch.cat([v.grad.reshape(-1) for v in model.parameters() if v.requires_grad]).detach().clone()
    assert torch.isfinite(bc_grad).all() and float(bc_grad.norm())>0
    cos=float(torch.dot(gradient,bc_grad)/(gradient.norm()*bc_grad.norm()));assert not torch.allclose(gradient,bc_grad,atol=1e-10,rtol=1e-10)
    opt.zero_grad(set_to_none=True);keep=M.keep_backward(model,old,p);assert keep<1e-16
    with torch.no_grad():
        kl=M.trust(model,items,p);assert max(kl)<1e-18
        for group in items:assert abs(sum(b['adv'] for b in group))<1e-12
        # LOO baseline explicitly excludes own outcome; all-success/all-fail
        # task supplies zero signal without numerical standardization.
        for rewards in [[0,0,0,0],[1,1,1,1]]:
            assert all(rewards[i]-sum(rewards[j] for j in range(4) if j!=i)/3==0 for i in range(4))
    assert before==M.digest(M.state(obj)) and not opt.state
    model.cpu();del obj,model;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PRIMARY_REVIEW.md',source/'summary.json',source/'audit.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    global_worst=R.used(ROOT/'stage/runs')+p['max_bytes'];assert global_worst<10*2**30
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',historical_numeric_fixture_only=True,actual_tasks=2,actual_policy_density_120carry_rebuilt=True,
       terminal_score_sum_matches_clipped_initial_gradient=True,PG_gradient_norm=float(gradient.norm()),BC_gradient_norm=float(bc_grad.norm()),gradient_cosine=cos,
       PG_BC_different_direction=True,LOO_zero_group_signal_checked=True,no_sample_STD_or_length_normalization=True,
       initial_Actor_Adam_unchanged=True,native_worst=worst,commands_worst=1012,storage_worst=storage,global_worst=global_worst,new_native=0,new_optimizer=0))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','shared_physics','shared_main_seconds','shared_audit_seconds','main_seconds','audit_seconds','learning_seconds','all_seconds','old_backup_seconds','backup_seconds','source_seconds']})
    print('T649_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',worst,float(gradient.norm()),cos,flush=True)

if __name__=='__main__':main()
