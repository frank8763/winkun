"""Actual first-divergence selection and frozen teacher full-carry precheck."""
from pathlib import Path
import ast,copy,gzip,json,time
import t643_runtime as T
R=T.R;ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    base=ROOT/'stage/runs/t642_tail_retention';s=R.read(base/'summary.json');assert s['state']=='COMPLETE' and not s['result']['accepted']
    assert R.read(base/'audit.json')['state']=='PASS'
    rows=[json.loads(z) for z in gzip.decompress((base/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    source=next(r for r in rows if r['job_id']=='wide117_80_dev6_LEFT_18')
    assert source['full'] and source['collision'] and R.sha(source['path'])==source['sha256']
    old=[json.loads(z) for z in (ROOT/'stage/runs/t636_joint_best_reproduction/cases.jsonl').read_text().splitlines() if json.loads(z)['role']=='fresh']
    ref=next(r for r in old if r['job_id']==source['job_id']);assert ref['success'] and not ref['collision']
    p=R.read(ROOT/'stage/experiments/t642_tail_retention/protocol.json')
    for k in ['registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop','user_authorization']:p.pop(k,None)
    a=R.E.N.S.arrays(source['path']);b=R.E.N.S.arrays(ref['path']);np=R.np
    time_a=np.r_[0,np.cumsum(a['native_counts'])[:-1]];time_b=np.r_[0,np.cumsum(b['native_counts'])[:-1]];n=min(len(time_a),len(time_b));assert np.array_equal(time_a[:n],time_b[:n])
    xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1);dy=a['poses'][:n,2]-b['poses'][:n,2];yaw=np.abs(np.arctan2(np.sin(dy),np.cos(dy)))*180/np.pi
    k=int(np.flatnonzero((xy>=.02)|(yaw>=1.))[0]);assert k==7
    selection=dict(rule='first physical-time aligned XY>=.02m OR yaw>=1degree',decision=k,seconds=float(time_a[k]*.002),xy_m=float(xy[k]),yaw_degrees=float(yaw[k]))
    job=dict(source['request']['job'],source_path=source['path'],student_takeover_first=k)
    p.update(id='T643_OLD_POLICY_ON_ACTUAL_CANDIDATE_DRIFT',out=str(OUT),source_row=source,reference_row=ref,handoff_job=job,
       first_divergence=selection,max_physics=80000,max_commands=8,max_bytes=12*2**20,
       main_seconds=360,audit_seconds=240,transport_seconds=90,source_seconds=90,all_seconds=1200,
       new_optimizer=0,new_Actor_Adam=0,shadow_Actor_Adam=0,independent_historical_Actor_Adam=0,new_Critic=0,new_RL=0,authorization_approved=False)
    R.B.setup(p);q=R.L.frozen(p,p['initial']['path']);q.sample=False;q.reset_noise();before=R.digest(q.state_dict());h=None;outputs=[]
    with R.torch.no_grad():
        x=R.torch.tensor(a['states'][:k+1]);expected=q.sequence(x)
        for z in x:
            u,_,h,_=q.step(z[None],h);outputs.append(u[0])
        err=float((R.torch.stack(outputs)-expected.clamp(-1,1).float()).abs().max())
        assert err<=2e-6 and h is not None and bool(h.abs().max()>0)
    assert R.digest(q.state_dict())==before and not R.torch.cuda.is_initialized()
    assert 2*job['case']['step_cap']==p['max_physics']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    files=set(HERE.glob('*.py'))|{DOC/'PLAN.md',Path(source['path']),Path(ref['path']),base/'audit.json',base/'summary.json',base/'SEALED.json',base/'protocol.json'}
    for module in list(T.sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    p['inputs'].update({str(f):R.sha(f) for f in files})
    parent=R.read(base/'protocol.json');assert R.used(base)+p['max_bytes']<=parent['max_bytes']
    assert s['physics']+p['max_physics']<=parent['max_physics'] and s['commands']+p['max_commands']<=parent['max_commands']
    glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
    assert time.time()+900<parent['absolute_all_stop']
    R.save(HERE/'protocol.json',p)
    R.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',first_divergence=selection,actual_T642_source_SHA=source['sha256'],
       reference_T501_success_SHA=ref['sha256'],teacher_own_carry_step_sequence_error=err,teacher_weights_unchanged=True,
       original_no_runtime_teacher_model_changed=True,combined_physics=s['physics']+p['max_physics'],combined_commands=s['commands']+p['max_commands'],
       own_full_and_partial_worst_bytes=12*2**20,global_worst=glob,new_optimizer=0))
    R.save(DOC/'BUDGET.json',{v:p[v] for v in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','transport_seconds','source_seconds','new_optimizer']})
    print('T643_PREFLIGHT_PASS_ZERO_NATIVE_ZERO_OPTIMIZER',selection,glob,flush=True)


if __name__=='__main__':main()
