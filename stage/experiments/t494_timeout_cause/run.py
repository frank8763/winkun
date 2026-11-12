"""Frozen reproduction of the fourteen T493 raw_ppo_02 timeouts; no learning."""
from pathlib import Path
import copy, fcntl, json, os, shutil, sys, time, traceback
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t494_timeout_cause'
DOC = ROOT / 'stage/docs/review_20261112'
sys.path.insert(0, str(ROOT / 'stage/experiments/t493_wide8_night'))
import t493_runtime as T
read, save, sha, digest, np = T.read, T.save, T.sha, T.digest, T.np

def verify(p, full=False):
    assert sha(HERE / 'protocol.json') == sha(OUT / 'protocol.json')
    assert T.M.used(OUT) <= p['max_bytes']
    assert T.M.used() - T.M.used(OUT) + p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40 * 2**30
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not T.torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0)) <= {14, 15}
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f

def setup():
    T.B.OUT = T.B.ACTIVE = T.M.OUT = OUT
    T.B.verify = T.M.verify = verify
    T.M.setup_runtime()

def detail(a, r):
    dt = a['native_counts'].astype(float) * .002
    t = np.r_[0., np.cumsum(dt)]
    poses = np.vstack([a['poses'], a['next_poses'][-1]])
    d = np.linalg.norm(poses[:, :2] - a['task_goal_xy'], axis=1)
    displacement = np.linalg.norm(a['next_poses'][:, :2] - a['poses'][:, :2], axis=1)
    speed = displacement / dt
    radial = (d[:-1] - d[1:]) / dt
    angle = np.arctan2(a['task_goal_xy'][1] - a['poses'][:, 1], a['task_goal_xy'][0] - a['poses'][:, 0])
    heading = np.abs(np.arctan2(np.sin(angle - a['poses'][:, 2]), np.cos(angle - a['poses'][:, 2]))) * 180 / np.pi
    def segment(mask):
        w = dt[mask]
        avg = lambda v: float(np.average(v[mask], weights=w))
        return dict(seconds=float(w.sum()), speed_m_s=avg(speed), radial_progress_m_s=avg(radial),
                    heading_error_deg=avg(heading), drive_command=avg(a['actions'][:, 0]),
                    steer_abs=avg(np.abs(a['actions'][:, 1])),
                    steer_saturated_fraction=avg((np.abs(a['actions'][:, 1]) >= .98).astype(float)),
                    moving_away_seconds=float(dt[mask & (radial < -.001)].sum()))
    report = dict(id=r['job_id'], source_array_digest=r['array_digest'], seconds=float(t[-1]),
                  remaining_m=float(d[-1]), min_decision_distance_m=float(d.min()),
                  closest_time_s=float(t[d.argmin()]), path_m=float(displacement.sum()),
                  radial_displacement_m=float(d[0] - d[-1]),
                  detour_ratio=float(displacement.sum() / max(1e-12, d[0] - d[-1])),
                  first20=segment(t[:-1] < 20), last20=segment(t[:-1] >= t[-1] - 20),
                  overall=segment(np.ones(len(dt), dtype=bool)),
                  distance_by_seconds={str(k):float(np.interp(k, t, d)) for k in [0, 10, 20, 40, 60, 80, 100, float(t[-1])]},
                  goal_radius_m=.2, timeout=True, collision=False)
    assert r['timeout'] and not r['success'] and not r['collision']
    assert abs(report['remaining_m'] - r['remaining_m']) < 1e-10
    assert report['min_decision_distance_m'] > .2
    return report

def prepare():
    assert not OUT.exists()
    old = ROOT / 'stage/runs/t493_wide8_night'
    assert read(old / 'SEALED.json')['all_scientific_processes_stopped']
    assert read(old / 'raw_ppo_02/audit.json')['state'] == 'PASS'
    p = read(old / 'protocol.json')
    c = read(old / 'raw_ppo_02/summary.json')['result']['candidate']
    rr = [r for r in T.rows(old / 'raw_ppo_02') if r['role'] == 'FULL54_EIGHT']
    failed = [r for r in rr if not r['success']]
    assert len(rr) == 54 and len(failed) == 14 and all(r['timeout'] and not r['collision'] for r in failed)
    jobs = {j['id']: j for j in p['wide8_jobs']}
    p.update(id='T494_FROZEN_FOURTEEN_TIMEOUT_CAUSE', out=str(OUT), initial=c['model'],
             cached={r['job_id']:r for r in failed}, probe_jobs=[jobs[r['job_id']] for r in failed],
             max_bytes=16*2**20, max_physics=1493352, max_commands=32,
             main_seconds=1800, learning_seconds=1, audit_seconds=300,
             absolute_learning_stop=time.time(), absolute_native_stop=time.time()+2100,
             absolute_all_stop=time.time()+2700, original_Actor_Adam=0, original_Value_Adam=0,
             authorization='Nov12 user explicitly asks determine all14 timeout causes and modify; this is NEW bounded frozen diagnosis, not restart or extension of T493.',
             scientific_scope='14 deterministic histories plus14 exact native repeats, all hashes must match original compact evaluation; descriptive cause metrics, no causal intervention or navigation improvement claim.')
    p.pop('initial_state', None); p.pop('original_cases', None)
    p['protected_files'] = dict(p['protected_files']);p['protected_files'][c['model']['path']] = c['model']['sha256']
    p['inputs'] = dict(p['inputs'])
    for f in [HERE/'run.py', old/'raw_ppo_02/summary.json', old/'raw_ppo_02/audit.json', old/'raw_ppo_02/cases.jsonl', Path(c['model']['path'])]:p['inputs'][str(f)]=sha(f)
    assert T.M.used()+p['max_bytes']+2**20 < 2**30
    OUT.mkdir();DOC.mkdir(exist_ok=True)
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    verify(p,True)
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,source14_timeouts_verified=True,shape_contract='original full-array digest required on native reproduction',epoch=time.time()))
    save(DOC/'T494_BUDGET.json',dict(max_physics=p['max_physics'],max_commands=32,own_bytes=p['max_bytes'],global_reserved=T.M.used()-T.M.used(OUT)+p['max_bytes'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],quota_cores=1,GPU=False))
    print('T494_REGISTERED_ZERO_NATIVE_ZERO_ADAM',flush=True)

def main():
    p=read(OUT/'protocol.json');setup();verify(p,True);T.E.N.offline();T.E.L.install_stop_handlers();rt=T.B.Runtime(p)
    state,error='FAILED',None
    try:
        rt.pool=T.Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        values=rt.request(p['probe_jobs'],p['initial'],'FROZEN14_TIMEOUT',full=True)
        diagnosis=[]
        for a,r in values:
            prev=p['cached'][r['job_id']]
            assert r['array_digest']==prev['array_digest'] and r['array_hashes']==prev['array_hashes'] and r['integration_sha256']==prev['integration_sha256']
            diagnosis.append(detail(a,r))
        save(OUT/'diagnosis.json',diagnosis)
        rt.request(p['probe_jobs'],p['initial'],'EXACT14_TIMEOUT',full=True,references=[r for a,r in values])
        state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,Actor_Adam=0,Value_Adam=0))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();T.stopped(OUT)
    rr,rep=T.rows(OUT),T.rows(OUT,'replays.jsonl');assert len(rr)==len(rep)==14
    details=[]
    for r in rr:
        assert sha(r['path'])==r['sha256'];a=T.E.N.S.arrays(r['path'])
        assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        T.M.audit_full_history(p,r,a);details.append(detail(a,r));old=p['cached'][r['job_id']]
        assert r['array_hashes']==old['array_hashes'] and r['array_digest']==old['array_digest'] and r['integration_sha256']==old['integration_sha256']
    assert details==read(OUT/'diagnosis.json')
    by={r['path']:r for r in rr}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rr+rep)==p['max_physics'] and s['commands']==32
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,all14_original_full_digests_and14_exact_repeats=True,physics=s['physics'],commands=s['commands']))
    save(DOC/'T494_DIAGNOSIS.json',dict(state='FROZEN_REPRODUCTION_AUDIT_PASS',rows=details,causal_intervention_tested=False))
    print('T494_DIAGNOSIS_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
