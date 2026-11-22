"""Actual zero-update entry contracts for one frozen-parent extension."""
from pathlib import Path
import ast, copy, inspect, json, os, resource, sys, time, unittest.mock as mock
import t698_runtime as S

A, N, T, torch, np = S.A, S.N, S.T, S.torch, S.np
ROOT, HERE, DOC = S.ROOT, S.HERE, S.ROOTDOC
read, sha, digest = S.read, S.sha, S.digest


def main():
    started = time.time()
    os.sched_setaffinity(0, {14})
    assert not S.ROOTOUT.exists() and not (DOC / 'PREFLIGHT.json').exists()
    pre = read(DOC / 'cpu_allocation_recovery/PREPARATION.json')
    assert pre['state'] == 'GPU_GEOMETRY_AND_LEGACY_SOURCE410_PROTECTION_PASS_NO_COEFFICIENT_COMMIT'
    assert (DOC.parent / 'T698_CROSS_REPRESENTATION_SOURCE_REVIEW.md').exists()
    p = copy.deepcopy(read(ROOT / 'stage/runs/t696_existing_success_compatibility/protocol.json'))
    p.pop('inputs'); p.pop('registration_epoch', None)
    retained = read(DOC.parent / 'RETAINED_MODEL.json')
    p['original_T501'] = p['original_T501']
    p['initial'] = retained['model']
    p.pop('protected403_rows',None)
    p['protected410_rows'] = pre['protected410_rows']
    sources = pre['novel_rows']
    core_jobs = p['evaluation_jobs'][:407]
    neighbors = read(ROOT / 'stage/runs/t695_frozen_neighborhood/protocol.json')['neighborhood_jobs']
    jobs = {j['id']: j for j in core_jobs + neighbors}
    required410 = {r['id'] for r in p['protected410_rows']}
    source_ids = {r['id'] for r in sources}
    screen_ids = {j['id'] for j in p['screen_jobs']} | source_ids | {j['id'] for j in neighbors if j['id'] in required410}
    screen = [jobs[k] for k in sorted(screen_ids)]
    evaluation = core_jobs + neighbors
    assert len(evaluation) == len(jobs) == 431 and len(required410) == 410 and len(source_ids) == 2
    maximum_physics = sum(j['case']['step_cap'] for j in evaluation + screen)
    maximum_commands = len(evaluation) + len(screen) + 14
    auth = read(DOC.parent / 'AUTHORIZATION.json')
    now = time.time()
    p.update(id='T698_CROSS_REPRESENTATION_EXISTING_SUCCESS_FROZEN_T696', authorization_approved=False,
        out=str(S.ROOTOUT), arm='root', protected410_rows=p['protected410_rows'], novel_rows=sources,
        before_success_ids=read(DOC.parent / 'CAMPAIGN_STATE.json')['future_core407_protection_success_ids'],
        required410_ids=sorted(required410), source_ids=sorted(source_ids), core407_ids=[j['id'] for j in core_jobs],
        screen_jobs=screen, required_screen_ids=sorted((required410 & screen_ids) | source_ids), evaluation_jobs=evaluation,
        support_fraction=.49, minimum_feature_separation=1e-10, maximum_centers=4096, model_bytes=8*2**20,
        new_Actor=1, new_Value=0, new_optimizer=0, preflight_optimizer=0,
        fit_main_seconds=2400, fit_audit_seconds=1500, learning_seconds=2400,
        child_main_seconds=5300, child_audit_seconds=1800, child_max_physics=maximum_physics,
        child_max_commands=maximum_commands, night_max_physics=maximum_physics, night_max_commands=maximum_commands,
        night_main_seconds=7700, audit_total_seconds=3300, fit_bytes=96*2**20,
        child_bytes=512*2**20, max_bytes=768*2**20, global_bytes=10*2**30,
        final_storage_reserve=32*2**20, full_evaluation_reserve_seconds=2500, root_seconds=12000,
        absolute_all_stop=min(now+12000,auth['absolute_all_stop']),
        absolute_learning_stop=auth['absolute_learning_stop'], absolute_native_stop=auth['absolute_native_stop'],
        campaign_learning_stop=auth['absolute_learning_stop'], campaign_native_stop=auth['absolute_native_stop'],
        finalization_seconds=60, weekend_authorization=str(DOC.parent/'AUTHORIZATION.json'), learner_device='cpu', geometry_device='cuda:0', worker_count=7, admission_physics_per_second=3000, stage_cpus=[8,9,10,11,12,13,14,15])
    assert maximum_physics / 3000 + 120 < p['child_main_seconds']
    assert 2400+1500+60<7200 and 5300+1800+60<7200
    assert T.R.used(ROOT/'stage/runs') + p['max_bytes'] < auth['archive_trigger_bytes']
    S.Wait.Raw.resources(); S.setup(p)
    def forbidden(*args, **kwargs): raise AssertionError('No fit/native/optimizer in entry preflight')
    torch.optim.Adam.step = forbidden; A.fit = forbidden
    import mujoco
    mujoco.mj_step = forbidden
    parent = A.Parent.frozen(p, p['initial']['path'])
    q = A.Actor(p, len(parent.compat_radii), len(parent.new_radii), pre['geometry']['centers'])
    missing, extra = q.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing) == {'last_centers', 'last_radii', 'last_weights'} and not extra
    q.last_centers.fill_(1e6)
    assert not q.last_weights.any()
    blob, joint = A.serialized(q, p)
    fd = os.memfd_create('T698_zero_new_readout_loader_only', flags=os.MFD_CLOEXEC)
    try:
        os.write(fd, blob); restored = A.frozen(p, f'/proc/self/fd/{fd}')
        assert restored.phash() == q.phash() and digest(restored.state_dict()) == digest(q.state_dict())
        row = p['protected410_rows'][0]; a = T.R.E.N.S.arrays(row['path'])
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
    assert seen == [0] and T.R.N.Pool is N.Pool
    constructor=inspect.getsource(N.C.Session.__init__)
    assert 'self.steps==0' in constructor and 'self.hidden.any()' in constructor and 'self.scene==expected' in constructor
    assert 2*len(screen)+(431-len(screen))+14 == maximum_commands
    upper = len(screen)*p['full_bytes']+431*p['compact_bytes']+7*p['full_bytes']+32*2**20
    assert upper < p['child_bytes']
    for f in HERE.glob('*.py'): ast.parse(f.read_text())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024; assert peak<8*2**30
    result=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_NO_COEFFICIENT_COMMIT',
        frozen_parent_all_state_exact=True, zero_new_readout_actual_RAM_export_loader_batch1=True,
        actual_source_physical_auditor_and_corruption_rejected=True, worker_canonical_factory_loader=True,
        same_original431_jobs_no_new_geometry=True, new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,
        screen_count=len(screen),maximum_physics=maximum_physics,maximum_commands=maximum_commands,
        storage_upper=upper,peak_rss_bytes=peak,elapsed_seconds=time.time()-started)
    result['new_resources']=__import__('t698_resources').resources()
    result['seven_workers_separate_counter_slots']=len(p['stage_cpus'])==8 and p['worker_count']==7
    result['GPU_geometry_with_CPU_exact_deployed_arithmetic']=pre['geometry']['actual_cpu_basis_all_old_zero_and_K_identity']
    (DOC/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    prior=read(ROOT/'stage/runs/t696_existing_success_compatibility/protocol.json')
    deps={Path(f) for f in prior['inputs']} | set(HERE.glob('*.py'))
    deps |= {Path(r['path']) for r in p['protected410_rows']+sources}
    deps |= {Path(r['source_model']) for r in sources}|{Path(r['source_audit']) for r in sources}
    deps |= {DOC/'PLAN.md',DOC.parent/'t697/SOURCE_INVENTORY.json',DOC/'cpu_allocation_recovery/PREPARATION.json',DOC/'PREFLIGHT.json',
             DOC.parent/'T698_CROSS_REPRESENTATION_SOURCE_REVIEW.md',Path(p['initial']['path']),
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
