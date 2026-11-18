"""Actual teacher checkpoint/history/IO checks, no session or optimizer."""
from pathlib import Path
import ast,copy,sys,time,json
import t639_runtime as T
ROOT,HERE,DOC,OUT=T.ROOT,T.HERE,T.DOC,T.OUT


def main():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    basefile=ROOT/'stage/runs/t604_reverse_initialization/protocol.json'
    nightfile=ROOT/'stage/experiments/t629_offline_night_relay/protocol.json'
    config=T.read(basefile);night=T.read(nightfile);config.update(night)
    # Keep the full original effective teacher constructor configuration,
    # separate from this small operational protocol.
    T.save(HERE/'teacher_configuration.json',config)
    source_root=ROOT/'stage/runs/t638_memory_ready_recovery'
    import gzip
    rr=[json.loads(z) for z in gzip.decompress((source_root/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    source=next(r for r in rr if r['role']=='baseline')
    reference=next(r for r in T.read(ROOT/'stage/runs/t629_offline_night_relay/profile_0/summary.json')['result']['rows'] if r['job_id']==source['job_id'])
    assert reference['success'] and not reference['collision'] and source['timeout']
    handoff=dict(source['request']['job'],source_path=source['path'],student_takeover_first=208,explore_after_prefix=False)
    p=dict(id='T639_SINGLE_NEURAL_TEACHER_RECOVERY',out=str(OUT),actor_configuration=str(HERE/'teacher_configuration.json'),
        source_row=source,nominal_reference=reference,reference_model=night['reference_model'],
        nominal_job=source['request']['job'],handoff_job=handoff,
        max_physics=138788,max_commands=10,max_bytes=16*2**20,full_bytes=2**21,fresh_bytes=2**21,compact_bytes=12288,
        main_seconds=600,audit_seconds=300,transport_seconds=120,source_seconds=90,all_seconds=1800,
        stage_cpus=[8,10,12,14],field_limit_m=7.,new_Actor=0,new_Critic=0,new_RL=0,new_optimizer=0,
        authorization_approved=False,inputs={})
    pc=dict(config,**p)
    visited=[];original=T.M.L.P.checkpoint
    def traced(path):visited.append(Path(path));return original(path)
    T.M.L.P.checkpoint=traced
    try:q=T.M.frozen(pc,p['reference_model']['path'])
    finally:T.M.L.P.checkpoint=original
    assert q.phash()==p['reference_model']['policy_hash']
    assert T.sha(p['reference_model']['path'])==p['reference_model']['sha256']
    a=T.R.B.R.E.N.S.arrays(source['path']);k=208
    assert all(T.digest(v)==source['array_hashes'][key] for key,v in a.items())
    q.sample=False;q.reset_noise();h=None;outputs=[];before=T.digest(q.state_dict())
    with T.torch.no_grad():
        x=T.torch.tensor(a['states'][:k+1]);seq=q.sequence(x)
        for z in x:
            command,location,h,_=q.step(z[None],h);outputs.append(command[0])
        actual=T.torch.stack(outputs);error=float((actual-seq.clamp(-1,1).float()).abs().max())
        assert error<=2e-6 and T.digest(q.state_dict())==before
        assert h is not None and bool(h.abs().max()>0)
        assert hasattr(q,'extra_hidden') and hasattr(q,'anchor_hidden')
        memories=dict(base64=T.digest(h),extra32=T.digest(q.extra_hidden),anchor32=T.digest(q.anchor_hidden))
    assert not T.torch.cuda.is_initialized()
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    parent=T.read(source_root/'protocol.json');actual_parent=T.read(source_root/'summary.json')
    assert actual_parent['physics']+p['max_physics']<=parent['max_physics']
    assert actual_parent['commands']+p['max_commands']<=parent['max_commands']
    assert T.R.used(source_root)+p['max_bytes']<=parent['max_bytes']
    glob=T.R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
    assert time.time()+1230<parent['absolute_all_stop']
    files=set(visited)|{Path(source['path']),basefile,nightfile,HERE/'teacher_configuration.json',
        source_root/'protocol.json',source_root/'summary.json',source_root/'SEALED.json',
        ROOT/'stage/runs/t629_offline_night_relay/profile_0/summary.json',DOC/'PLAN.md',
        Path(source['request']['job']['exact_snapshot'])}
    science=T.R.B.R.E.N.science(pc);files|={Path(f) for f in science['input_hashes']};files.add(Path(pc['source_protocol']))
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):files.add(Path(f).resolve())
    files|=set(HERE.glob('*.py'));p['inputs']={str(f):T.sha(f) for f in sorted(files)}
    T.save(HERE/'protocol.json',p)
    T.save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',actual_T501_prefix_decisions=k,
        teacher_policy_hash=q.phash(),actual_teacher_own_full_history_step_sequence_max_error=error,
        carry64plus32plus32=memories,empty_hidden_not_used=True,teacher_weights_unchanged=True,
        checkpoint_dependencies=[str(f) for f in sorted(set(visited))],global_worst=glob,
        combined_physics=actual_parent['physics']+p['max_physics'],combined_commands=actual_parent['commands']+p['max_commands'],
        remaining_authorized_window=True,new_optimizer=0))
    T.save(DOC/'BUDGET.json',{key:p[key] for key in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','transport_seconds','source_seconds','new_optimizer']})
    print('T639_READY_ZERO_NATIVE_ZERO_OPTIMIZER',error,glob,flush=True)


if __name__=='__main__':main()
