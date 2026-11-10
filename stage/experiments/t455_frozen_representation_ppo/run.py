"""Matched early-correction/PPO pilot with frozen encoder and GRU.

Only the existing 130 Actor head parameters learn. The unchanged mature Adam
groups remain serializable; frozen parameters and their moments must be exact.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t455_frozen_representation_ppo'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage455_pinned453', ROOT/'stage/experiments/t453_early_correction_ppo/run.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
P, F, M, R, E, torch = D.P, D.F, D.M, D.R, D.E, D.torch
read, save, sha, digest = P.read, P.save, P.sha, P.digest
ORIGINAL_MAKE = P.make
ORIGINAL_WARM_STEP = D.warm_step
ORIGINAL_PPO = D.ppo


def frozen_digest(obj):
    names = [name for name, v in obj[0].named_parameters()]
    state = P.state(*obj)
    keys = [k for k, name in enumerate(names) if not name.startswith('head.')]
    return digest(dict(parameters={n: v for n, v in state['actor'].items() if not n.startswith('head.')},
                       optimizer={k: state['actor_opt']['state'].get(k) for k in keys}))


def make(p, device='cuda'):
    obj = ORIGINAL_MAKE(p, device)
    for name, v in obj[0].named_parameters():
        v.requires_grad_(name.startswith('head.'))
    assert sum(v.numel() for v in obj[0].parameters() if v.requires_grad) == 130
    assert sum(v.numel() for v in obj[0].parameters() if not v.requires_grad) == 52736
    return obj


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def worker(pipe, index, p, counts):
    M.worker(pipe, index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe()
                c = ctx.Process(target=worker, args=(b, i, p, counts))
                c.start()
                b.close()
                self.children.append(c)
                self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(r['state'] == 'READY' for r in self.ready)
            f = Path(p['out'])/'worker_generations.json'
            rows = read(f) if f.exists() else []
            rows.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, rows)
        except BaseException:
            self.close()
            raise


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def warm_step(obj, *args):
    before = frozen_digest(obj)
    result = ORIGINAL_WARM_STEP(obj, *args)
    assert before == frozen_digest(obj)
    return result


def ppo(obj, *args):
    before = frozen_digest(obj)
    result = ORIGINAL_PPO(obj, *args)
    assert before == frozen_digest(obj)
    return result


def bind():
    P.make = make
    D.OUT, D.HERE, D.DOC = OUT, HERE, DOC
    D.verify, D.Pool, D.setup_runtime = verify, Pool, setup_runtime
    D.warm_step, D.ppo = warm_step, ppo


def prepare():
    assert not OUT.exists(), 'Never restart'
    prior = ROOT/'stage/runs/t453_early_correction_ppo'
    diagnosis = ROOT/'stage/runs/t454_frozen_function_audit'
    assert read(prior/'audit.json')['state'] == 'PASS'
    assert read(diagnosis/'audit.json')['state'] == 'PASS_FROZEN_DESCRIPTOR_ONLY'
    assert read(DOC/'T454_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for directory in [prior, diagnosis]:
        assert all(not Path('/proc', str(pid)).exists() for pid in read(directory/'SEALED.json')['scientific_pids'])
        assert not Path('/proc', str(read(directory/'driver_started.json')['pid'])).exists()
    base = read(prior/'protocol.json')
    p = copy.deepcopy(base)
    p.pop('registration_metadata_correction', None)
    p.update(id='T455_MATCHED_FROZEN_REPRESENTATION_EARLY_CORRECTION_PPO', out=str(OUT),
        architecture='plain input120 encoder128Tanh GRU64 head2 1.25tanh; encoder/core52736 frozen, head130 trainable; no runtime teacher/router',
        actor_trainable_parameters=130, actor_frozen_parameters=52736,
        trainable_prefixes=['head.'], frozen_prefixes=['encoder.', 'core.'],
        trainable_parameter_mask_is_the_only_scientific_change_to_T453=True,
        mechanism='Same T453 parent/matureAdam128maskedBC+2round4freshPPO/weights/seeds/11panels/final329. Only freeze52736encoder/core parameters and their Adam moments; update existing130head parameters.',
        new_mechanism='Restrict Actor learning to existing action readout while preserving complete causal recurrent representation and its mature Adam state exactly.',
        training='128BC+128shadow, up to8PPOActor8Value and matching shadow, existing130head-only Actor, same independentValue and complete-history computation.',
        comparison='Matched T453 algorithm/input/sampling-seed budget except Actor trainable mask; resulting on-policy state distributions may differ. Single train seed pilot, not compute-matched or innovation proof.',
        user_authorization='Nov10 user continue: after independent T454 descriptor supports testing representation interference, register one bounded head-only matched mechanism. Not resuming T453 or expanding its rounds.',
        upstream_T453_remote='PENDING_6_OF16_SEGMENTS_VERIFIED_ALL_RAW_LOCAL',
        fresh16=None, reload78=None, six_gate=False, eight_started=False)
    # Lock the entire original scientific configuration apart from the mask.
    for key in ['research_initial', 'initial_success_ids', 'warmup_steps', 'rounds', 'epochs', 'teacher_first',
                'raw_sigma', 'ppo_aux_weight', 'seed', 'training_jobs', 'panel_jobs', 'repair_lr',
                'actor_lr', 'value_lr', 'gamma_native', 'lambda_native', 'target_kl', 'initial_correction', 'own_rows']:
        assert p[key] == base[key], key
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T455_PLAN.md'), str(DOC/'T455_PRIMARY_SOURCES.md'), str(DOC/'T453_FINDINGS.md'),
        str(prior/'protocol.json'), str(prior/'summary.json'), str(prior/'audit.json'),
        str(diagnosis/'protocol.json'), str(diagnosis/'findings.json'), str(diagnosis/'audit.json')}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['constructor_reads'] = sorted(files)
    assert M.used()+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T455_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'],
        global_full=M.used()-M.used(OUT)+p['max_bytes'], max_physics=p['max_physics'],
        worst_physics=p['worst_physics'], max_commands=p['max_commands'], maximum_all_commands=422,
        main_seconds=p['main_seconds'], learning_seconds=p['learning_seconds'], audit_seconds=p['audit_seconds'],
        all_seconds=p['all_seconds'], estimated_main_minutes=[10, 25], hard_cap_NOT_queued_duration=True,
        BC_live=128, BC_shadow=128, PPO_Actor_live_max=8, PPO_Value_live_max=8,
        Actor_trainable=130, Actor_frozen=52736, fresh16_reload78_NOT_reserved=True,
        prior_T453_raw_backup_pending=True, unique_bytes_deleted=0))
    print('T455_REGISTERED', flush=True)


def preflight():
    bind()
    D.preflight()
    p = read(OUT/'protocol.json')
    obj = F.objects(p)
    before = frozen_digest(obj)
    assert obj[0].phash() == p['research_initial']['policy_hash']
    pre = read(OUT/'preflight.json')
    pre.update(Actor_trainable=130, Actor_frozen=52736,
        every_preflight_BC_and_PPO_step_frozen_parameter_and_Adam_hash_guarded=True,
        initial_frozen_parameter_and_Adam_digest=before)
    save(OUT/'preflight.json', pre)


def main():
    bind()
    D.main()


def audit():
    bind()
    D.audit()
    p = read(OUT/'protocol.json')
    parent = F.objects(p)
    expected = frozen_digest(parent)
    for model in read(OUT/'summary.json')['models']:
        obj = make(p)
        P.load(obj, P.checkpoint(model['path'])['joint_state'])
        assert frozen_digest(obj) == expected
    audit = read(OUT/'audit.json')
    audit.update(all_saved_encoder_GRU_parameters_and_mature_Adam_states_exact_parent=True,
        frozen_parameter_and_Adam_digest=expected, Actor_trainable=130, Actor_frozen=52736)
    save(OUT/'audit.json', audit)


def launch():
    bind()
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T455',
            '--why=Bounded frozen representation matched Stage PPO', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, learning_stop=now+p['learning_seconds'],
        native_stop=now+p['main_seconds'], all_stop=now+p['all_seconds'],
        protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T455_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
