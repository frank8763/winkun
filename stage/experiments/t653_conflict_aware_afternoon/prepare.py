"""Preflight active-set projection and actual canonical score gradients, no steps."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast,copy,hashlib,json,os,sys,time
import t653_runtime as T
ROOT,HERE,OUT,DOC=T.ROOT,T.HERE,T.OUT,T.DOC
M,R,torch,np=T.M,T.R,T.torch,T.np

def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=T.read(ROOT/'stage/runs/t652_adapters_pg/protocol.json');s=T.read(ROOT/'stage/runs/t652_adapters_pg/pg/summary.json')
    assert T.read(ROOT/'stage/runs/t652_adapters_pg/pg/audit.json')['state']=='PASS'
    old,new=T.DataSource.data(p,lambda:None);assert len(old)==384 and len(new)==19
    target=['dev_task6_00','dev_task6_03'];extra=[copy.deepcopy(next(j for j in p['evaluation_jobs'] if j['id']==key)) for key in target]
    # Add original per-ID scene/120/integration/goal constraints, which the
    # existing canonical worker enforces BEFORE the first physical step.
    source=T.read(ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json');by={r['id']:r for r in source['members']}
    for j in extra:
        a=R.E.N.S.arrays(by[j['id']]['offline_path'])
        j['initial_contract']=dict(scene_sha256=bytes(a['scene_sha256']).hex(),initial120_sha256=hashlib.sha256(a['states'][0].tobytes()).hexdigest(),integration_sha256=hashlib.sha256(a['initial_native_integration'].tobytes()).hexdigest(),goal_xy=a['task_goal_xy'].tolist())
    profile=[];deltas=[(.01,0,0),(-.01,0,0),(0,.01,0),(0,-.01,0),(0,0,.5),(0,0,-.5),(.01,.01,.5),(-.01,-.01,-.5)]
    field=next(j for j in p['field_jobs'] if j['id']=='t502_field_sector5_offset1')
    for base in [*extra,field]:
        for i,(dx,dy,da) in enumerate(deltas):
            j=copy.deepcopy(base);j.pop('initial_contract',None);j['id']=f"pair_{base['id']}_{i}";j['expected_initial120_sha256']=None
            for k,d in [('delta_x_m',dx),('delta_y_m',dy),('delta_yaw_deg',da)]:j[k]=j.get(k,0)+d
            profile.append(j)
    assert len(profile)==24
    def epoch(s):return datetime.fromisoformat(s).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    p.update(id='T653_AFTERNOON_SAMPLED_OLD_OUTCOME_CONSTRAINT',out=str(OUT),authorization_approved=False,arms=['mean','protected'],arm='mean',
        protected_tasks=target,extra_jobs=extra,paired_jobs=profile,
        fit_steps=8,fit_lr=.00003,keep_weight=10.,old_batch=16,max_task_latent_KL=.01,
        constraint_gradient_floor=1e-12,projection_tolerance=1e-11,old_surrogate_tolerance=1e-10,minimum_displacement=1e-12,constraint_scales=[1.,.5,.25,.125],
        total_max_physics=30000000,total_max_commands=1100,total_main_seconds=7500,Actor_each_cap=16,
        max_bytes=640*2**20,shared_bytes=112*2**20,child_bytes=128*2**20,paired_bytes=240*2**20,global_bytes=10*2**30,
        shared_physics=32*sum(j['case']['step_cap'] for j in extra),paired_physics=6*sum(j['case']['step_cap'] for j in profile),
        shared_main_seconds=1200,shared_audit_seconds=300,main_seconds=2700,audit_seconds=600,paired_main_seconds=900,paired_audit_seconds=600,
        max_physics=12100000,max_commands=433,learning_seconds=600,full_evaluation_reserve_seconds=1500,
        backup_seconds=900,source_seconds=240,
        absolute_learning_stop=epoch('2026-11-19T15:30:00'),absolute_native_stop=epoch('2026-11-19T16:10:00'),absolute_all_stop=epoch('2026-11-19T17:00:00'),
        new_Critic=0,preflight_optimizer=0,new_teacher=0)
    p.pop('registration_epoch',None)
    eval_worst=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+p['key_jobs'])
    worst=p['shared_physics']+2*eval_worst+p['paired_physics'];assert worst<=p['total_max_physics']
    assert 70+2*433+150<=1100
    storage=(32*p['fresh_bytes']+32*p['compact_bytes']+p['full_bytes']+8*2**20)+2*(20*p['fresh_bytes']+p['model_bytes']+407*p['compact_bytes']+p['full_bytes']+20*2**20)+(72*p['fresh_bytes']+72*p['compact_bytes']+p['full_bytes']+8*2**20)+8*2**20
    assert storage<=p['max_bytes']
    R.B.setup(p);R.gpu_guard();os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    obj=M.make(p);q,opt=obj;initial=M.digest(M.state(obj));body=M.L.fixed_hash(q)
    # Use existing four-outcome actual groups only as numeric fixtures for the
    # variable-n builder, including a16-row repeated fixture NOT new data.
    fixture=list(new)+[(target[0],new[2][1]*4),(target[1],new[3][1]*4)]
    items,diag=M.build(q,fixture,p);variables=M.variables(q)
    losses=[];grads=[]
    for group in items:
        loss=M.task_loss(q,group,p);gg=torch.autograd.grad(loss,variables);losses.append(float(loss.detach()));grads.append(M.flat([g.detach() for g in gg]))
    gradients=torch.stack(grads);mean=gradients.mean(0)
    assert torch.isfinite(mean).all() and float(mean.norm())>0
    # Actual average of21 per-task gradients equals original scalar objective
    # with21groups and each group's own sample count, not group-length weighting.
    opt.zero_grad(set_to_none=True)
    for group in items:(M.task_loss(q,group,p)/len(items)).backward()
    expected=M.flat([v.grad for v in variables]);assert torch.allclose(mean,expected,atol=1e-10,rtol=1e-9)
    all_adv=[x['advantages'] for x in diag];assert all(abs(sum(v))<1e-12 for v in all_adv)
    # Closed-form QP fixtures force two active constraints, one active, and
    # degenerate collinear constraints; zero matrices correctly rejected.
    fixtures=[]
    for d,g in [(torch.tensor([1.,1.,2.]),torch.tensor([[1.,0.,0.],[0.,1.,0.]])),(torch.tensor([1.,-1.,2.]),torch.tensor([[1.,0.,0.],[0.,1.,0.]])),(torch.tensor([1.,2.,3.]),torch.tensor([[1.,0.,0.],[2.,0.,0.]]))]:
        d=d.to(device='cuda',dtype=torch.float64);g=g.to(device='cuda',dtype=torch.float64);v,detail=M.project(d,g,p);assert bool((g@v<=1e-10).all())
        fixtures.append(detail)
    try:M.project(torch.ones(3,dtype=torch.float64,device='cuda'),torch.zeros((2,3),dtype=torch.float64,device='cuda'),p);raise RuntimeError('zero gradient accepted')
    except AssertionError:pass
    # The live fixture old2 both have nonzero task gradients; inspect actual
    # projected negative mean direction without taking any Adam step.
    pg=gradients[-2:];candidate=-mean*.00003;projected,actual=M.project(candidate,pg,p)
    assert bool((pg@projected<=1e-10).all())
    # Mature synthetic Adam serialization + projected parameter application,
    # no optimizer.step or simulation. Rejection restores all moments exactly.
    before=M.state(obj)
    with torch.no_grad():
        for v in variables:opt.state[v]=dict(step=torch.tensor(5.),exp_avg=torch.full_like(v,.03125),exp_avg_sq=torch.full_like(v,.0625))
        mature=M.state(obj);mh=M.digest(mature)
        q.steer_adapter[-1].bias.add_(.01)
        proposal=M.state(obj)
        displacement=M.flat([(v-mature['actor'][name]).detach() for name,v in q.named_parameters() if v.requires_grad])
        M.set_displacement(obj,mature,proposal,displacement,.5)
        assert M.digest(opt.state_dict())==M.digest(proposal['actor_opt'])
        M.set_displacement(obj,mature,proposal,displacement,0.);assert M.digest(M.state(obj))==mh
        M.load(obj,before);assert M.digest(M.state(obj))==initial and not opt.state and M.L.fixed_hash(q)==body
    q.cpu();del obj,q;torch.cuda.empty_cache()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PRIMARY_REVIEW.md',DOC/'T652_TWO_FAILURES.json',ROOT/'stage/runs/t652_adapters_pg/pg/summary.json',ROOT/'stage/runs/t652_adapters_pg/pg/audit.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):T.sha(f) for f in files})
    for f,h in p['inputs'].items():assert T.sha(f)==h,f
    global_worst=R.used(ROOT/'stage/runs')+p['max_bytes'];assert global_worst<10*2**30
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',existing76_full_history_and_density_rebuilt=True,
        numeric_repeated16_fixtures_not_new_samples=True,n4_n16_task_balancing_and_LOO_checked=True,scalar_and_mean_gradients_match=True,
        projection_active_set_fixtures=fixtures,actual_projection=actual,zero_constraint_gradient_rejected=True,
        projected_parameter_and_Adam_full_rollback_fixture=True,initial11906andfrozenbodyunchanged=True,
        native_worst=worst,commands_worst=1086,storage_worst=storage,global_worst=global_worst,new_native=0,new_optimizer=0))
    T.save(DOC/'BUDGET.json',{k:p[k] for k in ['total_max_physics','total_max_commands','total_main_seconds','Actor_each_cap','max_bytes','global_bytes','shared_physics','paired_physics','shared_main_seconds','shared_audit_seconds','main_seconds','audit_seconds','paired_main_seconds','paired_audit_seconds','learning_seconds','backup_seconds','source_seconds','absolute_learning_stop','absolute_native_stop','absolute_all_stop']})
    print('T653_ZERO_NATIVE_ZERO_OPTIMIZER_PASS',worst,storage,flush=True)

if __name__=='__main__':main()
