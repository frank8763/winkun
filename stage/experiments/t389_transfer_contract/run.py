"""Frozen-data teacher/student interface and prefix-drift audit; zero native/Adam."""
from pathlib import Path
import importlib.util
import json
import os
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t389_transfer_contract'
DOC = ROOT / 'stage/docs/review_20261106'
spec = importlib.util.spec_from_file_location('stage_t389_saved_contract', ROOT / 'stage/experiments/t387_full_horizon_night/night_common.py')
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
np, torch, read, save, sha = N.np, N.torch, N.read, N.save, N.sha
PREVIOUS = ROOT / 'stage/runs/t387_full_horizon_night'


def prepare():
    assert not OUT.exists()
    DOC.mkdir(parents=True, exist_ok=True)
    paths = [HERE / 'run.py', DOC / 'T389_PLAN.md',
             PREVIOUS / 'protocol.json', PREVIOUS / 'student/protocol.json',
             PREVIOUS / 'student/initial.pt.gz', PREVIOUS / 'student/candidate.pt.gz',
             PREVIOUS / 'student/student_centre.npz', PREVIOUS / 'phase_warp/mixed.npz',
             PREVIOUS / 'student/audit.json', PREVIOUS / 'phase_warp/audit.json',
             ROOT / 'stage/experiments/t387_full_horizon_night/night_common.py']
    p = dict(id='T389_SAVED_TEACHER_TRANSFER_CONTRACT', inputs={str(f): sha(f) for f in paths},
             source_protocol=str(PREVIOUS / 'student/protocol.json'),
             native=0, geometry=0, optimizer=0, backward=0, max_seconds=300,
             final_bytes=256*1024, sequence_step_location_tolerance=1e-10,
             action_deviation_thresholds=[.01, .05, .1],
             position_deviation_m=[.01, .05, .1],
             source_rule='Fixed T388D actual successful mixed path and T387 actual failed centre; no fitted or selected new model')
    used=N.K.S.used(ROOT/'stage/runs')
    assert used+p['final_bytes']<=1024**3
    assert N.shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    save(OUT/'preflight.json', dict(state='PASS_METADATA_ONLY', native=0, geometry=0, optimizer=0, backward=0,
                                  complete_reservation=used+p['final_bytes'], global_bytes=1024**3))
    print('T389_REGISTERED_ZERO_NATIVE_ZERO_OPTIMIZER')


def model(base, path):
    q=N.R.actor(base)
    q.load_learned(N.S.read_model(path))
    q.eval().requires_grad_(False)
    return q


def first_above(values, threshold):
    ids=np.flatnonzero(values>threshold)
    return None if not len(ids) else int(ids[0])


def metrics(error):
    return dict(rmse=np.sqrt(np.square(error).mean(0)).tolist(),
                max_abs=np.abs(error).max(0).tolist(),
                p95_abs=np.quantile(np.abs(error),.95,axis=0).tolist())


def forward(q, arrays):
    x=torch.tensor(arrays['states'])
    seq=q.sequence(x,q.base_actions(x)).detach().numpy()
    q.reset_noise()
    hidden=None
    actions=[]
    locations=[]
    for row in x:
        action, location, hidden, _=q.step(row[None],hidden)
        actions.append(action[0].detach().numpy().copy())
        locations.append(location[0].detach().numpy().copy())
    locations=np.asarray(locations)
    return seq, locations, np.asarray(actions)


def main():
    N.offline()
    p=read(OUT/'protocol.json')
    assert not (OUT/'summary.json').exists()
    for f,h in p['inputs'].items():assert sha(f)==h, f
    for folder in ['student','phase_warp']:
        s=read(PREVIOUS/folder/'summary.json')
        assert s['state']=='COMPLETE' and s['all_workers_stopped']
        assert read(PREVIOUS/folder/'audit.json')['state']=='PASS'
        assert not Path('/proc',str(s['pid'])).exists()
    assert read(PREVIOUS/'phase_warp/summary.json')['teacher_safe_GOAL']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.optim.Adam.step=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('Zero optimizer audit'))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('T389 bounded audit deadline')))
    signal.alarm(p['max_seconds'])
    start=time.time()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    try:
        base=N.science(read(p['source_protocol']))
        teacher=N.S.arrays(PREVIOUS/'phase_warp/mixed.npz')
        student=N.S.arrays(PREVIOUS/'student/student_centre.npz')
        assert teacher['end_flags'][-1,1] and not teacher['end_flags'][:,0].any()
        assert student['end_flags'][-1,0] and not student['end_flags'][:,1].any()
        assert N.equal(teacher['states'][0],student['states'][0])
        n=min(len(teacher['states']),len(student['states']))
        assert N.equal(teacher['native_counts'][:n-1],student['native_counts'][:n-1])
        init=model(base,PREVIOUS/'student/initial.pt.gz')
        final=model(base,PREVIOUS/'student/candidate.pt.gz')
        report=dict(state='PASS', native=0, geometry=0, optimizer=0, backward=0,
                    teacher_holds=len(teacher['states']),student_holds=len(student['states']),
                    teacher_success=True,student_success=False,
                    initial_observation_exact=True,initial_policy=init.phash(),final_policy=final.phash(),
                    teacher_fit={},same_input_stream_checks={})
        saved={}
        with torch.inference_mode():
            for name,q in [('initial',init),('final',final)]:
                seq,stream,actions=forward(q,teacher)
                error=float(np.max(np.abs(seq-stream)))
                assert error<=p['sequence_step_location_tolerance']
                err=actions.astype(np.float64)-teacher['actions'].astype(np.float64)
                report['teacher_fit'][name]=dict(prefix20=metrics(err[:20]),suffix=metrics(err[20:]),
                                               sequence_step_max_error=error)
                saved[name+'_teacher_actions']=actions
                saved[name+'_teacher_locations']=stream
            seq,stream,actions=forward(final,student)
            assert N.equal(actions,student['actions'])
            assert np.max(np.abs(stream-student['locations']))<=p['sequence_step_location_tolerance']
            report['same_input_stream_checks']=dict(final_saved_native_actions_exact=True,
                final_saved_native_locations_max_error=float(np.max(np.abs(stream-student['locations']))),
                final_sequence_vs_stream_max_error=float(np.max(np.abs(seq-stream))))
        dpos=np.linalg.norm(teacher['poses'][:n,:2]-student['poses'][:n,:2],axis=1)
        angle=teacher['poses'][:n,2]-student['poses'][:n,2]
        angle=np.arctan2(np.sin(angle),np.cos(angle))
        command=np.max(np.abs(teacher['actions'][:n].astype(float)-student['actions'][:n].astype(float)),axis=1)
        report['first_same_clock_position_deviation_holds']={str(x):first_above(dpos,x) for x in p['position_deviation_m']}
        report['first_same_clock_action_deviation_holds']={str(x):first_above(command,x) for x in p['action_deviation_thresholds']}
        report['prefix20_boundary']=dict(clock_s=4.,position_gap_m=float(dpos[20]),yaw_gap_degrees=float(np.rad2deg(abs(angle[20]))),
                                         final_action_error_on_teacher_prefix20=report['teacher_fit']['final']['prefix20'])
        report['actual_initial_action_pair']=dict(teacher=teacher['actions'][0].tolist(),student=student['actions'][0].tolist())
        saved.update(position_gap_m=dpos,yaw_gap_rad=angle,action_gap=command)
        N.write_npz(OUT/'diagnostic_arrays.npz',saved,96*1024)
        report.update(elapsed_s=time.time()-start,all_workers_stopped=True,pid=os.getpid(),
                      hypotheses_not_causal_proof=True,source_hashes=p['inputs'])
        for f,h in p['inputs'].items():assert sha(f)==h, f
        save(OUT/'summary.json',report)
        save(DOC/'T389_FINDINGS.json',report)
        N.capacity(OUT,p['final_bytes'])
        print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes']},ensure_ascii=False))
    except BaseException:
        save(OUT/'failure.json',dict(error=traceback.format_exc(),native=0,optimizer=0))
        raise
    finally:signal.alarm(0)


def launch():
    assert not (OUT/'launch.json').exists()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    main()


if __name__=='__main__':
    import sys
    globals()[sys.argv[1]]()
