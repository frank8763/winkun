"""Actual zero-update entry contracts for one frozen-parent extension."""
from pathlib import Path
import ast, copy, inspect, json, os, resource, sys, time, unittest.mock as mock
import t696_runtime as S

A, N, T, torch, np = S.A, S.N, S.T, S.torch, S.np
ROOT, HERE, DOC = S.ROOT, S.HERE, S.ROOTDOC
read, sha, digest = S.read, S.sha, S.digest


def main():
    started = time.time()
    os.sched_setaffinity(0, {14})
    assert not S.ROOTOUT.exists() and not (DOC / 'PREFLIGHT.json').exists()
    pre = read(DOC / 'source_config_recovery/PREPARATION.json')
    assert pre['state'] == 'GEOMETRY_SOURCE_AND_403_OWNCARRY_PASS_NO_COEFFICIENT_COMMIT'
    assert (DOC.parent / 'EXISTING_SUCCESS_SOURCE_QUALIFICATION.md').exists()
    p = copy.deepcopy(read(ROOT / 'stage/runs/t694_full_history_compatibility/protocol.json'))
    p.pop('inputs'); p.pop('registration_epoch', None)
    retained = read(DOC.parent / 'RETAINED_MODEL.json')
    p['original_T501'] = p['initial']
    p['initial'] = retained['model']
    p['protected403_rows'] = [{k: r[k] for k in ['id', 'path', 'sha256', 'original384']} for r in pre['protected']]
    inventory = read(DOC / 'SOURCE_INVENTORY.json')
    sources = []
    for selected in inventory['selected']:
        row = selected['row']
        parent = Path(selected['source_audit']).parent.parent
        if not (parent / 'protocol.json').exists(): parent = Path(selected['source_audit']).parent
        source_protocol = read(parent / 'protocol.json')
        sources.append(dict(id=selected['id'], path=selected['path'], sha256=selected['sha256'],
                            source_model=row['request']['model'], source_raw_sigma=source_protocol['raw_sigma']))
    p['novel_rows'] = sources
    core_jobs = p['evaluation_jobs']
    neighbors = read(ROOT / 'stage/runs/t695_frozen_neighborhood/protocol.json')['neighborhood_jobs']
    jobs = {j['id']: j for j in core_jobs + neighbors}
    required403 = {r['id'] for r in p['protected403_rows']}
    source_ids = {r['id'] for r in sources}
    screen_ids = {j['id'] for j in p['screen_jobs']} | source_ids | {j['id'] for j in neighbors if j['id'] in required403}
    screen = [jobs[k] for k in sorted(screen_ids)]
    evaluation = core_jobs + neighbors
    assert len(evaluation) == len(jobs) == 431 and len(required403) == 403 and len(source_ids) == 7
    maximum_physics = sum(j['case']['step_cap'] for j in evaluation + screen)
    maximum_commands = len(evaluation) + len(screen) + 6
    auth = read(DOC.parent / 'AUTHORIZATION.json')
    now = time.time()
    p.update(id='T696_FIXED_EXISTING_AUTONOMOUS_SUCCESS_FROZEN_T694_EXTENSION', authorization_approved=False,
        out=str(S.ROOTOUT), arm='root', protected403_rows=p['protected403_rows'], novel_rows=sources,
        before_success_ids=read(DOC.parent / 'CAMPAIGN_STATE.json')['future_core407_protection_success_ids'],
        required403_ids=sorted(required403), source_ids=sorted(source_ids), core407_ids=[j['id'] for j in core_jobs],
        screen_jobs=screen, required_screen_ids=sorted((required403 & screen_ids) | source_ids), evaluation_jobs=evaluation,
        support_fraction=.49, minimum_feature_separation=1e-10, maximum_centers=4096, model_bytes=8*2**20,
        new_Actor=1, new_Value=0, new_optimizer=0, preflight_optimizer=0,
        fit_main_seconds=3300, fit_audit_seconds=1800, learning_seconds=3300,
        child_main_seconds=6500, child_audit_seconds=600, child_max_physics=maximum_physics,
        child_max_commands=maximum_commands, night_max_physics=maximum_physics, night_max_commands=maximum_commands,
        night_main_seconds=9800, audit_total_seconds=2400, fit_bytes=96*2**20,
        child_bytes=512*2**20, max_bytes=768*2**20, global_bytes=10*2**30,
        final_storage_reserve=32*2**20, full_evaluation_reserve_seconds=4200, root_seconds=13200,
        absolute_all_stop=min(now+13200,auth['absolute_all_stop']),
        absolute_learning_stop=auth['absolute_learning_stop'], absolute_native_stop=auth['absolute_native_stop'],
        campaign_learning_stop=auth['absolute_learning_stop'], campaign_native_stop=auth['absolute_native_stop'],
        finalization_seconds=60, weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'), learner_device='cpu')
    assert maximum_physics / p['admission_physics_per_second'] + 120 < p['child_main_seconds']
    assert T.R.used(ROOT/'stage/runs') + p['max_bytes'] < auth['archive_trigger_bytes']
    S.Wait.Raw.resources(); S.setup(p)
    def forbidden(*args, **kwargs): raise AssertionError('No fit/native/optimizer in entry preflight')
    torch.optim.Adam.step = forbidden; A.fit = forbidden
    import mujoco
    mujoco.mj_step = forbidden
    parent = A.Parent.frozen(p, p['initial']['path'])
    q = A.Actor(p, len(parent.compat_radii), pre['geometry']['centers'])
    missing, extra = q.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {'new_centers', 'new_radii', 'new_weights'} and not extra
    q.new_centers.fill_(1e6)
    assert not q.new_weights.any()
    blob, joint = A.serialized(q, p)
    fd = os.memfd_create('T696_zero_new_readout_loader_only', flags=os.MFD_CLOEXEC)
    try:
        os.write(fd, blob); restored = A.frozen(p, f'/proc/self/fd/{fd}')
        assert restored.phash() == q.phash() and digest(restored.state_dict()) == digest(q.state_dict())
        row = p['protected403_rows'][0]; a = T.R.E.N.S.arrays(row['path'])
        _, original = A.trajectory(parent, a, row['id'], a['actions'])
        _, zero = A.trajectory(restored, a, row['id'], a['actions'])
        assert torch.equal(original, zero)
    finally: os.close(fd)
    assert all(torch.equal(q.state_dict()[k], v) for k, v in parent.state_dict().items())
    N.P.frozen = A.frozen
    rr = dict(job_id=row['id'], request=dict(model=p['initial']['path']), seed=None, teacher=False,
              physics=int(a['native_counts'].sum()), array_hashes={k:digest(v) for k,v in a.items()})
    N.inspect(p, a, rr); N.audit_source_hashes(p, rr)
    broken=copy.deepcopy(rr); broken['array_hashes']['actions']='0'*64
    try: N.audit_source_hashes(p, broken)
    except AssertionError: pass
    else: raise AssertionError('Corrupted source hash accepted')
    seen=[]; previous=os.sched_setaffinity
    def witness(pipe,index,pc,counts):
        assert N.G.T.Session is N.C.Session and N.P.frozen is A.frozen; seen.append(index)
    try:
        with mock.patch.object(N.G.T,'worker',witness): N.worker(None,0,p,None)
    finally: os.sched_setaffinity=previous
    assert seen == [0] and T.R.N.Pool.__init__.__globals__['worker'] is N.worker
    constructor=inspect.getsource(N.C.Session.__init__)
    assert 'self.steps==0' in constructor and 'self.hidden.any()' in constructor and 'self.scene==expected' in constructor
    assert 2*len(screen)+(431-len(screen))+6 == maximum_commands
    upper = len(screen)*p['full_bytes']+431*p['compact_bytes']+3*p['full_bytes']+32*2**20
    assert upper < p['child_bytes']
    for f in HERE.glob('*.py'): ast.parse(f.read_text())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024; assert peak<4*2**30
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_NO_COEFFICIENT_COMMIT',
        frozen_parent_all_state_exact=True, zero_new_readout_actual_RAM_export_loader_batch1=True,
        actual_source_physical_auditor_and_corruption_rejected=True, worker_canonical_factory_loader=True,
        same_original431_jobs_no_new_geometry=True, new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,
        screen_count=len(screen),maximum_physics=maximum_physics,maximum_commands=maximum_commands,
        storage_upper=upper,peak_rss_bytes=peak,elapsed_seconds=time.time()-started)
    (DOC/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    prior=read(ROOT/'stage/runs/t694_full_history_compatibility/protocol.json')
    deps={Path(f) for f in prior['inputs']} | set(HERE.glob('*.py'))
    deps |= {Path(r['path']) for r in p['protected403_rows']+sources}
    deps |= {Path(r['source_model']) for r in sources}
    deps |= {DOC/'PLAN.md',DOC/'SOURCE_INVENTORY.json',DOC/'source_config_recovery/PREPARATION.json',DOC/'PREFLIGHT.json',
             DOC.parent/'EXISTING_SUCCESS_SOURCE_QUALIFICATION.md',Path(p['initial']['path']),
             ROOT/'stage/runs/t695_frozen_neighborhood/audit.json'}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'): deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)}
    p['preflight_seconds']=time.time()-started+10
    (HERE/'protocol.json').write_text(json.dumps(p,separators=(',',':'))+'\n')
    assert (HERE/'protocol.json').stat().st_size<2**21
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
