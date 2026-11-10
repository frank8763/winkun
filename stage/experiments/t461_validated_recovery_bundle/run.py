"""One fixed validated off-policy imitation bundle, then full development.

T460 gate failure remains unchanged. No queries, failed labels or new PPO are
used here; all six source recovery trajectories already reached safe GOAL.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t461_validated_recovery_bundle'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage461_pinned460', ROOT/'stage/experiments/t460_closed_loop_coverage/run.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
H, M, P, R, E, torch, np = G.H, G.M, G.P, G.R, G.E, G.torch, G.np
read, save, sha, digest = G.read, G.save, G.sha, G.digest
make, state, load, export, frozen, step = G.make, G.state, G.load, G.export, G.frozen, G.step


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def worker(pipe, index, p, counts):
    P.frozen = frozen
    M.worker(pipe, index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe(); c = ctx.Process(target=worker, args=(b, i, p, counts))
                c.start(); b.close(); self.children.append(c); self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(r['state'] == 'READY' for r in self.ready)
            f = Path(p['out'])/'worker_generations.json'
            generations = read(f) if f.exists() else []
            generations.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, generations)
        except BaseException:
            self.close(); raise


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def data(p):
    return G.data(p, p['new_original_own5'], p['validated_recovery6'])


def validate_sources(p):
    assert len(p['new_original_own5']) == 5 and len(p['validated_recovery6']) == 6
    assert sorted(r['job_id'] for r in p['validated_recovery6']) == p['recovery_ids']
    assert p['validated_recovery6'][0] == p['initial_correction']
    assert not p['validated_recovery6'][0]['policy_hash'] == p['query_model']['policy_hash']
    assert len({r['path'] for r in p['validated_recovery6']}) == 6
    for r in p['new_original_own5']+p['validated_recovery6']:
        assert r['success'] and not r['collision'] and sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert digest(a) == r['array_digest']
        assert {k:digest(v) for k,v in a.items()} == r['array_hashes']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].tolist() == [0, 1, 0]
        if r['teacher']:
            assert r['teacher_first'] == 20 and not a['ownership'][:20].any() and a['ownership'][20:].all()
        else:
            assert r['policy_hash'] == p['research_initial']['policy_hash'] and not a['ownership'].any()


def prepare():
    assert not OUT.exists(), 'Never restart'
    previous = ROOT/'stage/runs/t460_closed_loop_coverage'
    assert read(previous/'audit.json')['state'] == 'PASS'
    assert read(DOC/'T460_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc', str(pid)).exists() for pid in read(previous/'SEALED.json')['scientific_pids'])
    assert not Path('/proc', str(read(previous/'driver_started.json')['pid'])).exists()
    base = read(previous/'protocol.json'); s = read(previous/'summary.json')
    assert s['state'] == 'COMPLETE' and s['Actor_BC_live'] == 0 and not s['gate']['all6_safe']
    valid = [r for r in s['query_rows'] if r['success'] and not r['collision']]
    assert len(valid) == 5 and all(r['job_id'] != 'dev_task6_00' for r in valid)
    p = copy.deepcopy(base)
    for k in ['remote_cache', 'cache_hard_stop_epoch', 'cache_cli_transport', 'cache_transport_recovery']:
        p.pop(k, None)
    p.update(id='T461_FIXED_VALIDATED_MULTI_BEHAVIOR_RECOVERY_BUNDLE', out=str(OUT),
        validated_recovery6=[base['initial_correction']]+sorted(valid, key=lambda r:r['job_id']),
        new_original_own5=s['own_rows'], recovery_ids=sorted(r['job_id'] for r in s['query_rows']),
        mechanism='Original328 zero-residual model; same1074network/seed/freshAdam128BC asT459, but old00 uses original328 safe4sec recovery and5others use T460 safe current-student recovery; original5own histories added. Final329.',
        new_mechanism='Fixed multi-behavior success-verified bundle explicitly excludes failed T46000; no new queries or repeat of stopped T460 gate.',
        training='128BC+128shadow, sixvalidatedexistingrecovery histories, old218 and15own function anchors;0newPPO/Value/teacher/native training.',
        comparison='Coverage and source-behavior mixture jointly change versusT459. No causal one-factor/independent training replicate/originality/retention guarantee.',
        user_authorization='Nov10 user inspect and demand real progress; keep original T4600BC gate rejection and assemble all-safe existing bundle under a new finite actual learning protocol, no old-directory restart.',
        max_bytes=10*2**20, max_physics=7000000, max_commands=400,
        maximum_teacher_trajectories=0, maximum_teacher_exact_replays=0,
        maximum_parent_preservation_trajectories=0, maximum_parent_exact_replays=0,
        maximum_new_training_episodes=0, maximum_exact_training_replays=0,
        preflight_discarded_Actor_Adam=2, all_six_teacher_recoveries_must_be_safe_before_BC=True,
        main_seconds=2700, learning_seconds=1200, audit_seconds=600, backup_seconds=600, all_seconds=4500,
        final_selection='Fixed128th proposal full329, original328 successIDs plus00 required. No teacher/epoch/seed/weight grid. Failure preserves original328.',
        original_T460_complete_gate_failure_0BC_and_raw_unchanged=True,
        six_gate=False, fresh16=None, reload78=None, eight_started=False)
    p['worst_physics'] = sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
    assert p['worst_physics'] < p['max_physics']
    for k in ['research_initial', 'residual_seed', 'warmup_steps', 'repair_lr', 'raw_sigma', 'base_actor_digest']:
        assert p[k] == base[k]
    validate_sources(p)
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {r['path'] for r in p['new_original_own5']+p['validated_recovery6']}
    files |= {str(DOC/'T460_FINDINGS_AND_NEXT.md'), str(DOC/'T461_PLAN.md'), str(DOC/'T461_PRIMARY_SOURCES.md'),
        str(previous/'protocol.json'), str(previous/'summary.json'), str(previous/'audit.json'),
        str(previous/'cases.jsonl'), str(previous/'replays.jsonl'), str(previous/'SEALED.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T460.json')}
    p['inputs'] = {f:sha(f) for f in sorted(files)}; p['constructor_reads'] = sorted(files)
    storage = 329*p['compact_bytes']+8*p['full_bytes']+p['model_bytes']+4*2**20
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] < 2**30
    OUT.mkdir(); save(HERE/'protocol.json', p); save(OUT/'protocol.json', p); verify(p, True)
    save(DOC/'T461_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'], storage_worst=storage,
        global_full=M.used()-M.used(OUT)+p['max_bytes'], max_physics=p['max_physics'], worst_physics=p['worst_physics'],
        max_commands=400, maximum_all_commands=369, estimated_main_minutes=[10,25], hard_cap_NOT_queued_duration=True,
        main_seconds=2700, learning_seconds=1200, audit_seconds=600, all_seconds=4500,
        Actor_BC_live=128, Actor_BC_shadow=128, PPO=0, new_teacher=0, new_native_training=0,
        independent_discarded_history_Adam=128, fresh16_reload78_NOT_reserved=True, unique_bytes_deleted=0))
    print('T461_REGISTERED', flush=True)


def preflight():
    p = read(OUT/'protocol.json'); verify(p, True)
    assert not (OUT/'preflight.json').exists(); setup_runtime(); validate_sources(p)
    a, b = make(p), make(p); initial = state(*a)
    zero = read(ROOT/'stage/runs/t459_nonlinear_residual_repair/preflight.json')['zero_model']
    assert a[0].phash() == zero['policy_hash'] and digest(initial) == zero['joint_hash']
    train = data(p)
    start = time.monotonic()
    va, vb = step(a, train), step(b, train)
    torch.cuda.synchronize(); estimate = (time.monotonic()-start)*128
    assert estimate < 1000 and va == vb and digest(state(*a)) == digest(state(*b))
    assert va['reference218'] < 1e-24 and va['own10'] < 1e-24
    load(a, initial); load(b, initial)
    assert digest(state(*a)) == digest(initial) == digest(state(*b))
    counts = R.M.mp.get_context('spawn').RawArray('q',9); folder = OUT/'preflight_native'; folder.mkdir()
    pool = Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
    try:
        replies = pool.request([dict(model=zero['path'],job=j,seed=None,cancel_probe=True) for j in p['old8_jobs']])
        assert all(r['state']=='CANCEL_PASS' for r in replies) and sum(counts)==0
    finally: pool.close()
    save(OUT/'preflight.json',dict(state='PASS',native=0,commands=24,discarded_supervised_clone_Adam=2,
        verified_six_safe_offpolicy_recovery_and_fifteen_own_histories=True,
        old_T460_failed00_excluded_original_successful00_included=True, initial_retention=va,
        live_shadow_and_rollback_exact=True, estimated128_live_shadow_seconds=estimate,
        same_initial_zero_model=zero, eight_workers_cancelled_and_stopped=True))
    print('T461_PREFLIGHT_PASS',flush=True)


def main():
    p=read(OUT/'protocol.json');verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'summary.json').exists()
    setup_runtime();E.N.offline();E.L.install_stop_handlers();rt=M.Runtime(p)
    status,error,candidate,proposal='FAILED',None,None,None
    obj=shadow=initial=None;completed=0
    try:
        validate_sources(p);obj,shadow=make(p),make(p);initial=state(*obj);train=data(p);trace=[]
        for k in range(128):
            rt.role='VALIDATED_BUNDLE_BC';rt.guard(True);values=[]
            for which,objects in [('live',obj),('shadow',shadow)]:
                M.events('bundle_BC',which)('BEFORE',k);values.append(step(objects,train));M.events('bundle_BC',which)('AFTER',k)
            assert values[0]==values[1] and digest(state(*obj))==digest(state(*shadow))
            completed+=1;trace.append(values[0])
            if k%16==0:save(OUT/'fit_progress.json',dict(steps=completed,latest=values[0],epoch=time.time()))
        save(OUT/'repair_trace.json',trace);proposal=export(obj,OUT/'proposal.pt.gz',p['model_bytes'])
        rt.pool=Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        candidate,_=M.evaluate(rt,p,proposal,'VALIDATED_BUNDLE_FINAL329')
        candidate['preserves_initial328']=set(p['initial_success_ids'])<=set(candidate['success_ids'])
        candidate['repairs00']=p['failed_job']['id'] in candidate['success_ids']
        candidate['development_admitted']=len(candidate['success_ids'])==329
        status='COMPLETE'
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and initial is not None:
            load(obj,initial);load(shadow,initial);assert digest(state(*obj))==digest(initial)
        save(OUT/'failure.json',dict(error=error,original_model_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=8
        ev=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines())) if (OUT/'optimizer_calls.jsonl').exists() else []
        counts={w:sum(r[1:3]==[w,'AFTER'] for r in ev) for w in ['live','shadow']}
        save(OUT/'summary.json',dict(state=status,pid=os.getpid(),elapsed_s=time.time()-rt.start,physics=sum(rt.counts),commands=rt.commands+p['preflight_commands'],error=error,
            candidate=candidate,proposal=proposal,Actor_BC_live=counts['live'],Actor_BC_shadow=counts['shadow'],completed_paired_BC=completed,
            PPO_Actor_live=0,PPO_Value_live=0,new_native_RL_training=0,new_teacher=0,
            retained_model=p['research_initial'],research_proposal_not_promoted=True,
            fresh16=None,reload78=None,six_gate=False,eight_started=False,all_native_workers_stopped=True))
        save(OUT/'status.json',dict(state=status,epoch=time.time()))
        R.M.fcntl.flock(rt.lock,R.M.fcntl.LOCK_UN);rt.lock.close()
    assert status=='COMPLETE',error
    print('T461_MAIN_COMPLETE',flush=True)


def audit():
    p=read(OUT/'protocol.json');verify(p,True);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists()
    ids=[s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    validate_sources(p);obj=make(p);train=data(p)
    trace=[step(obj,train) for _ in range(128)]
    assert trace==read(OUT/'repair_trace.json') and digest(state(*obj))==s['proposal']['joint_hash']
    assert H.base_digest(obj[0])==p['base_actor_digest'] and obj[0].phash()==s['proposal']['policy_hash']
    assert sha(s['proposal']['path'])==s['proposal']['sha256'] and frozen(p,s['proposal']['path']).phash()==obj[0].phash()
    ev=list(map(json.loads,(OUT/'optimizer_calls.jsonl').read_text().splitlines()));assert len(ev)==512
    for w in ['live','shadow']:
        assert [r[2:] for r in ev if r[:2]==['bundle_BC',w]]==[[where,k] for k in range(128) for where in ['BEFORE','AFTER']]
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()))
    assert len(rows)==len({r['job_id'] for r in rows})==329
    assert s['physics']==sum(r['physics'] for r in rows) and s['commands']==24+16+329
    for r in rows:
        assert sha(r['path'])==r['sha256'] and r['policy_hash']==obj[0].phash() and r['loaded_model_sha256']==s['proposal']['sha256']
        assert not r['teacher'] and r['seed'] is None and not r['administrative_censor']
        a=E.N.S.arrays(r['path']);assert {k:digest(v) for k,v in a.items()}=={k:r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum())==r['physics'] and np.abs(a['actions']).max()<=1
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert [bool(v) for v in a['end_flags'][-1]]==[r['collision'],r['success'],r['timeout']]
    assert R.M.tally(rows[:54])==s['candidate']['wide54'] and R.M.tally(rows[54:])==s['candidate']['old275']
    original=set(p['initial_success_ids']);success={r['job_id'] for r in rows if r['success']}
    comparison=dict(initial328_preserved=sorted(original&success),lost_original_success=sorted(original-success),repaired_original00=p['failed_job']['id'] in success)
    save(OUT/'success_ID_comparison.json',comparison)
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,discarded_historical_residual_Adam=128,
        all_live_shadow_historical_Actor_residualAdam_reconstruction_exact=True,full_source_arrays_SHA_and_all_safe_labels=True,
        frozen_original52866_and_parent_ActorValueAdam_file_unchanged=True,
        final329_compact_actions_counts_termination_SHA_exact=True,full120_hidden329_NOT_reconstructed=True,
        physics=s['physics'],commands=s['commands'],fresh16=None,reload78=None,six_gate=False,eight_started=False))
    print('T461_AUDIT_PASS',flush=True)


def launch():
    p=read(OUT/'protocol.json');verify(p,True)
    assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists()
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T461','--why=Validated recovery bundle finite Stage learning',
            'timeout','--signal=TERM','--kill-after=2',str(p['all_seconds']),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,learning_stop=now+p['learning_seconds'],native_stop=now+p['main_seconds'],all_stop=now+p['all_seconds'],protocol_sha256=sha(OUT/'protocol.json'),source_remote=src['remote_commit']))
    print('T461_ACTUALLY_LAUNCHED',c.pid,flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),epoch=time.time()))
        raise
