"""Frozen-policy paired initial goal-history control; zero learning.

Only the four reset history frames differ. Their physical poses/timestamps,
current goal, scene, integration state, and all subsequent native steps match.
Deterministic zero logp/means are diagnostic placeholders, never RL samples.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t309_sequence_goal_value'))
from sequence_value import *
import argparse,contextlib,multiprocessing as mp,signal,subprocess
sys.path.insert(0,str(ROOT/'stage/experiments/t280_local_cost'))
import t280_worker as W
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t310_initial_history_recovery'
DOC=ROOT/'stage/docs/distance_20261103'
HISTORY=np.array([72,73,84,85,96,97,108,109])
OTHER=np.setdiff1d(np.arange(120),HISTORY)

def verify(p):
    assert (HERE/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    for f,h in p['input_hashes'].items():assert C.sha(f)==h,f
    assert S.used(ROOT/'stage/runs')<=p['global_bytes'] and S.used(OUT)<=p['final_bytes']
    assert shutil.disk_usage(ROOT).free>=p['min_free_bytes']

def produce(j,q,native,create,witness,deadline,counts,total):
    original=R.task_env;holder={};history=[]
    def factory(create,job):
        env,obs,info=original(create,job);holder['env']=env
        initial=witness.get(env);oldx=T.L.packet(env,obs,0,job['case']['step_cap'],np.zeros(2,np.float32),0)
        integration=R.arrays_hash(R.integration(env));goal=env.goal_xy.copy()
        if j['context']=='aligned':obs=witness.alternate(env,goal,initial)
        x=T.L.packet(env,obs,0,job['case']['step_cap'],np.zeros(2,np.float32),0)
        assert integration==R.arrays_hash(R.integration(env))
        assert np.array_equal(x[OTHER],oldx[OTHER]) and np.array_equal(goal,env.goal_xy)
        holder['initial']=dict(integration_sha256=integration,native_input_sha256=S.digest(oldx),input_sha256=S.digest(x),
            nongoal107_sha256=S.digest(x[H.KEEP]),nonhistory112_sha256=S.digest(x[OTHER]),
            changed_indices=np.flatnonzero(x!=oldx).tolist(),history_goals_before=initial[1].tolist(),
            history_goals_after=witness.get(env)[1].tolist())
        close=env.close
        def closing():
            history.append(witness.get(env));close()
        env.close=closing
        return env,obs,info
    class Recorder:
        def step(self,x):
            history.append(witness.get(holder['env']))
            return q.step(x)
    R.task_env=factory
    try:
        checked_job=dict(j)
        if j['context']=='aligned':checked_job.pop('expected_initial120_sha256',None)
        item=W.generate(checked_job,Recorder(),native,create,deadline,counts,0,total)
        a=item['arrays'];assert len(history)==len(a['states'])+1
        a.update(history_poses=np.asarray([h[0] for h in history]),history_goals=np.asarray([h[1] for h in history]),history_times=np.asarray([h[2] for h in history]))
        item['row'].update(context=j['context'],case_id=j['case_id'],original_job=j,initial_control=holder['initial'],
            policy_hash=S.digest(q.state_dict()),array_hashes={k:S.digest(v) for k,v in a.items()},Actor_updates=0,not_on_policy_training_data=True)
        return item
    finally:
        R.task_env=original
        if holder.get('env') is not None:witness.events.pop(id(holder['env']),None)

def preflight():
    assert not any((OUT/f).exists() for f in ['launch.json','status.json','preflight.json'])
    p=read(HERE/'protocol.json');verify(p);q=B.actor(p);torch.set_num_threads(1)
    # This is a read-only model/input check, not factory initialization or fit.
    checks=[]
    for side in ['LEFT','RIGHT']:
        rows=read(ROOT/'stage/runs/t309_sequence_goal_value/cases.json')
        row=next(r for r in rows if r['side']==side and r['distance_band']==6 and r['split']=='train' and r['arm']=='center')
        a=B.arrays(row['path']);x=a['states'][0]
        assert np.abs(H.rebuild(x,a['poses'][0],a['history_poses'][0],row['goal_xy'],a['history_goals'][0])-x).max()<=2e-6
        with torch.no_grad():one=q.step(torch.tensor(x[None]))[0];two=q.step(torch.tensor(x[None]))[0]
        assert torch.equal(one,two);checks.append(dict(side=side,input_sha256=S.digest(x),mode=one[0].tolist()))
    assert len(p['jobs'])==20 and p['max_commands']==22
    assert p['absolute_deadline_epoch']==1793672260.500127
    # Packed native chain <=192KiB, four actual history poses/goals/times per
    # decision <=54KiB; 256KiB is a conservative per-record bound.
    assert 192*1024+281*4*6*8+8192<=p['episode_bytes']
    C.save(OUT/'preflight.json',dict(state='PASS',native_commands=0,physics_steps=0,optimizer_calls=0,Actor_updates=0,
        checks=checks,full_worstcase_reserved_bytes=S.used(ROOT/'stage/runs')-S.used(OUT)+p['final_bytes'],protocol_sha256=C.sha(HERE/'protocol.json')))
    print('T310 preflight PASS;0 native/optimizer; paired reset-history control only')

def main():
    assert not (OUT/'status.json').exists();p=read(HERE/'protocol.json');verify(p)
    assert read(OUT/'preflight.json')['state']=='PASS'
    start=time.time();deadline=p['absolute_deadline_epoch'];native_deadline=p['native_deadline_epoch']
    meter=None;witness=None;log=None;rows=[];replays=[];state='FAILED';active=None
    counts=mp.get_context('spawn').Array('q',1);last_status=0
    def status(force=False):
        nonlocal last_status
        if not force and time.time()-last_status<10:return
        last_status=time.time();C.save(OUT/'status.json',dict(state='PAIRED_NATIVE_CONTROL',pid=os.getpid(),main_epoch=start,updated_epoch=last_status,
            absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,commands=meter.commands,physics_steps=counts[0],
            cases=len(rows),replays=len(replays),active=active,optimizer_calls=0,Actor_updates=0))
    def alarm(*_):raise TimeoutError('T310 original absolute deadline')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,int(deadline-time.time())-5))
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=B.actor(p)
        meter=Budget(ROOT,OUT,p,start_epoch=p['original_main_epoch']);status(True)
        folder=OUT/'native';folder.mkdir();log=LimitedLog(folder/'native.log')
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):native,create=S.native_setup(folder)
        spec=importlib.util.spec_from_file_location('t310_witness',ROOT/'stage/experiments/t295_hindsight_support/run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from training.e2e_phase10_lidar_v1.phase10_scene_e2e_env import Phase10SceneE2ELidarEnv as Base
        witness=module.Witness(Base);C.save(OUT/'workers.json',dict(parent=os.getpid(),workers=[],native_in_parent=True,CPU_affinity=sorted(os.sched_getaffinity(0))))
        by={r['id']:r for r in read(ROOT/'stage/runs/t310_initial_history_control/cases.json')}
        schedule=[(False,j) for j in p['jobs'] if j['id'] not in by]+[(True,j) for j in p['jobs'] if j['id']!=p['omitted_replay_id']]
        assert len(schedule)==22
        for replay,job in schedule:
            assert time.time()<native_deadline;meter.begin(job['case']['step_cap']);active=job['id'];status(True)
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):item=produce(job,q,native,create,witness,native_deadline,counts,meter.physics)
            r=item['row'];meter.physics=counts[0];complete=r['complete'] and not r['error']
            if not replay or not complete:
                data=B.packed(item['arrays']);buf=io.BytesIO();np.savez_compressed(buf,**data);blob=buf.getvalue();assert len(blob)<=p['episode_bytes']
                meter.charge_payload(len(blob));f=OUT/'episodes'/(r['id']+('_partial_replay' if replay else '')+'.npz');assert not f.exists();f.write_bytes(blob)
                r.update(path=str(f),sha256=C.sha(f));rows.append(r);C.save(OUT/'cases.json',rows)
            if replay and complete:
                ref=by[r['id']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'] and r['initial_control']==ref['initial_control']
                old=B.arrays(ref['path']);assert all(np.array_equal(v,item['arrays'][k]) for k,v in old.items())
                r['original_id']=ref['id'];replays.append(r);C.save(OUT/'replays.json',replays)
            elif not replay:by[r['id']]=r
            assert complete,'Partial/invalid raw preserved; no continuation'
            if job['context']=='native':
                ref=p['historical_references'][job['case_id']]
                assert {k:r['array_hashes'][k] for k in B.NATIVE_KEYS}==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'],'Historical baseline not exact'
            active=None;status(True);verify(p)
        assert len(rows)==3 and len(replays)==19 and sum(r['physics_steps'] for r in rows+replays)==meter.physics
        assert S.digest(q.state_dict())==p['policy_hash'];state='COMPLETE'
    except BaseException:
        C.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),active=active,physics_steps=counts[0],commands=0 if meter is None else meter.commands))
    finally:
        if witness is not None:witness.close()
        if log is not None:log.close()
        signal.alarm(0)
        if meter is not None:meter.close()
    result=dict(state=state,pid=os.getpid(),commands=0 if meter is None else meter.commands,physics_steps=counts[0],raw_record_physics=sum(r['physics_steps'] for r in rows+replays),
        unique_cases=len(rows),exact_replays=len(replays),original_commands=18,original_physics=239896,aggregate_commands=18+(0 if meter is None else meter.commands),aggregate_physics=239896+counts[0],omitted_replay_id=p['omitted_replay_id'],optimizer_calls=0,Actor_updates=0,all_workers_stopped=True,
        main_epoch=start,absolute_deadline_epoch=deadline,native_deadline_epoch=native_deadline,elapsed_s=time.time()-start,
        current_policy_unchanged=True,old275_evaluated=False,six54_evaluated=False,fresh16_executed=False,reload78_executed=False,seven_eight_entered=False,automatic_next_training=False)
    C.save(OUT/'summary.json',result);C.save(OUT/'status.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','main']);args=parser.parse_args();globals()[args.mode]()
