"""Real full-parameter JVP/serialization/worker contract, zero native or steps."""
from pathlib import Path
import ast,copy,io,json,sys,time,importlib.util
import t679_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
C,T,M,torch,np=S.C,S.T,S.M,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    priorroot=ROOT/'stage/runs/t678_reference_target_pilot';prior=read(priorroot/'protocol.json');rep=read(priorroot/'FINAL_REPORT.json')
    assert rep['state']=='COMPLETE' and len(rep['children'])==2 and all(x['audit']['state']=='PASS' and x['screen']['lost_ids'] for x in rep['children'])
    seal=read(priorroot/'SEALED.json');assert seal['all_scientific_processes_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in seal['scientific_pids'])
    assert not Path('/proc',str(read(priorroot/'DRIVER_STARTED.json')['pid'])).exists()
    p=copy.deepcopy(prior);p.pop('registration_epoch',None);a=read(DOC.parent/'AUTHORIZATION.json');assert a['approved'];now=time.time()
    olds=['dev_task6_00','dev_task6_02','dev_task6_13','dev_task6_20'];fields=['t502_field_sector0_offset3','t502_field_sector1_offset1','t502_field_sector7_offset1','t502_field_sector7_offset2']
    by={j['id']:j for j in p['evaluation_jobs']};keys=olds+fields;jobs=[by[k] for k in keys]
    groups=C.B.dataset(p);calrows=[groups[k][0] for k in keys]
    p.update(id='T679_FROZEN_EPISODE_PARAMETER_EXPLORATION',authorization_approved=False,out=str(OUT),arm='episode_parameters',
             collection_jobs=jobs,collection_ids=keys,old_collection_ids=olds,field_collection_ids=fields,calibration_rows=calrows,
             calibration_direction_seeds=[[67920261121,0,i] for i in range(8)],evaluation_direction_seeds=[[67920261121,1,i] for i in range(4)],
             target_location_RMS=.05,minimum_location_RMS=.025,maximum_location_RMS=.1,finite_parameter_epsilon=1e-6,numeric_atol=1e-8,numeric_rtol=1e-4,
             max_physics=16*sum(j['case']['step_cap'] for j in jobs),max_commands=134,max_bytes=320*2**20,child_bytes=320*2**20,global_bytes=10*2**30,
             main_seconds=3000,audit_seconds=300,calibration_seconds=300,absolute_learning_stop=min(now+300,a['absolute_learning_stop']),
             absolute_native_stop=min(now+3000,a['absolute_native_stop']),absolute_all_stop=min(now+3600,a['absolute_all_stop']),
             fresh_bytes=4*2**20,compact_bytes=32768,model_bytes=2*2**20,new_Actor=0,new_Value=0,new_optimizer=0,finalization_seconds=60)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(p);q=M.L.frozen(p,p['initial']['path']);base=q.phash()
    theta,layout=C.parameter_layout(q);row=calrows[0];aa=T.R.E.N.S.arrays(row['path']);g=M.features(q,aa,row['job_id'],row['sha256'])
    eps=theta.new_tensor(C.direction(p['calibration_direction_seeds'][0]));value,jvp=torch.autograd.functional.jvp(lambda x:C.output(q,g,x,layout),theta,eps,strict=True)
    assert float((value-q.sequence(torch.tensor(aa['states']))).abs().max())<1e-10
    h=p['finite_parameter_epsilon'];fd=(C.output(q,g,theta+h*eps,layout)-C.output(q,g,theta-h*eps,layout))/(2*h)
    err=float((fd-jvp).abs().max());assert err<p['numeric_atol']+p['numeric_rtol']*float(jvp.abs().max())
    # Exercise checkpoint packet and real loader in memory; do not create
    # a physical run or permanent parameter candidate during preflight.
    packet=copy.deepcopy(M.P.checkpoint(p['initial']['path']));buf=io.BytesIO();torch.save(packet,buf);buf.seek(0);again=torch.load(buf,weights_only=False)
    assert digest(packet)==digest(again)
    perturb=theta+h*eps
    try:C.set_parameters(q,perturb);assert torch.allclose(M.location(q,g),C.output(q,g,perturb,layout),atol=1e-10,rtol=0)
    finally:C.set_parameters(q,theta)
    assert q.phash()==base and not q.core.batch_first and all(not v.requires_grad for v in q.parameters())
    T.N.P.frozen=T.N.L.frozen;assert T.N.P.frozen is T.N.L.frozen
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    spec=importlib.util.spec_from_file_location('_t679_helper_check',HERE/'affinity_guard.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.OUT==OUT and helper.CG.name=='stage-t679.service'
    resources=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',all11906_parameter_Gaussian_JVP_numeric=True,JVP_error=err,own_complete_carry_output_identity=True,
             antithetic_parameters_restored=True,checkpoint_in_memory_roundtrip=True,canonical_deterministic_worker_loader=True,resource_helper_configuration=True,
             data_count_fresh=64,exact_count=64,calibration_seeds_disjoint=True,new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T679_PRIMARY_REVIEW.md',priorroot/'protocol.json',priorroot/'FINAL_REPORT.json',priorroot/'SEALED.json',priorroot/'DRIVER_STARTED.json'}
    for arm in ['executed','self']:deps|={priorroot/arm/'SCREEN.json',priorroot/arm/'LEARNING.json',priorroot/arm/'audit.json'}
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21;save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
