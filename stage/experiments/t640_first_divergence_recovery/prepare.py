"""Fix the first observed trajectory divergence, before any new outcome."""
from pathlib import Path
import ast,gzip,json,sys,time
import t640_runtime as T
B=T.B;ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    p=B.read(ROOT/'stage/experiments/t639_neural_recovery/protocol.json')
    for key in ['authorization_approved','user_authorization','registration_epoch','absolute_native_stop','absolute_learning_stop','absolute_all_stop']:p.pop(key,None)
    old=ROOT/'stage/runs/t639_neural_recovery'
    rows=[json.loads(z) for z in gzip.decompress((old/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    nominal=next(r for r in rows if r['role']=='nominal')
    assert B.read(old/'audit.json')['state']=='PASS' and B.read(old/'summary.json')['result']['qualified'] is False
    a=B.R.B.R.E.N.S.arrays(p['source_row']['path']);b=B.R.B.R.E.N.S.arrays(nominal['path']);np=B.np
    ta=np.r_[0,np.cumsum(a['native_counts'])[:-1]];tb=np.r_[0,np.cumsum(b['native_counts'])[:-1]]
    n=min(len(ta),len(tb));assert np.array_equal(ta[:n],tb[:n])
    xy=np.linalg.norm(a['poses'][:n,:2]-b['poses'][:n,:2],axis=1)
    yaw=np.abs(np.arctan2(np.sin(a['poses'][:n,2]-b['poses'][:n,2]),np.cos(a['poses'][:n,2]-b['poses'][:n,2])))*180/np.pi
    first=int(np.flatnonzero((xy>=.02)|(yaw>=1.))[0]);assert first==5
    selection=dict(rule='first actual physical-time-aligned XY>=.02m OR yaw>=1degree',decision=first,
       seconds=float(ta[first]*.002),xy_m=float(xy[first]),yaw_degrees=float(yaw[first]),
       T501_source_SHA=p['source_row']['sha256'],teacher_nominal_source_SHA=nominal['sha256'])
    handoff=dict(p['handoff_job'],student_takeover_first=first)
    p.update(id='T640_FIRST_DIVERGENCE_NEURAL_RECOVERY',out=str(OUT),handoff_job=handoff,first_divergence=selection,
         nominal_control_reused=nominal,max_physics=69394,max_commands=8,max_bytes=12*2**20,
         main_seconds=360,audit_seconds=240,transport_seconds=90,source_seconds=90,all_seconds=1200,authorization_approved=False)
    pc=B.read(p['actor_configuration']);pc.update(p);q=B.M.frozen(pc,p['reference_model']['path'])
    assert q.phash()==p['reference_model']['policy_hash'];before=B.digest(q.state_dict())
    q.sample=False;q.reset_noise();h=None;outputs=[]
    with B.torch.no_grad():
        x=B.torch.tensor(a['states'][:first+1]);loc=q.sequence(x)
        for z in x:
            u,_,h,_=q.step(z[None],h);outputs.append(u[0])
    error=float((B.torch.stack(outputs)-loc.clamp(-1,1).float()).abs().max())
    assert error<=2e-6 and B.digest(q.state_dict())==before and h is not None and bool(h.abs().max()>0)
    assert not B.torch.cuda.is_initialized()
    for f in HERE.glob('*.py'):ast.parse(f.read_text());p['inputs'][str(f)]=B.sha(f)
    for f in [old/'audit.json',old/'summary.json',old/'SEALED.json',Path(nominal['path']),DOC/'PLAN.md']:
        p['inputs'][str(f)]=B.sha(f)
    parent=B.read(ROOT/'stage/runs/t638_memory_ready_recovery/protocol.json')
    actuals=[B.read(ROOT/'stage/runs'/name/'summary.json') for name in ['t638_memory_ready_recovery','t639_neural_recovery']]
    combined_physics=sum(s['physics'] for s in actuals)+p['max_physics'];combined_commands=sum(s['commands'] for s in actuals)+p['max_commands']
    assert combined_physics<=parent['max_physics'] and combined_commands<=parent['max_commands']
    own=sum(B.R.used(ROOT/'stage/runs'/name) for name in ['t638_memory_ready_recovery','t639_neural_recovery'])+p['max_bytes']
    assert own<=parent['max_bytes']
    glob=B.R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
    assert time.time()+900<parent['absolute_all_stop']
    B.save(HERE/'protocol.json',p)
    B.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',first_divergence=selection,
        actual_teacher_own_history_step_sequence_max_error=error,nonzero_carry=True,
        teacher_unchanged=True,prior_nominal_and_recovery_audit_PASS=True,
        combined_physics=combined_physics,combined_commands=combined_commands,parent_limits=[parent['max_physics'],parent['max_commands']],
        combined_reserved_bytes=own,parent_bytes=parent['max_bytes'],global_worst=glob,new_optimizer=0))
    B.save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','transport_seconds','source_seconds','new_optimizer']})
    print('T640_ZERO_NATIVE_PREFLIGHT_PASS',first,error,combined_physics,combined_commands,flush=True)


if __name__=='__main__':main()
