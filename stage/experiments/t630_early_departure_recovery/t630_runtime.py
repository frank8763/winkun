"""Frozen early-departure recovery qualification; never updates a policy."""
from pathlib import Path
import fcntl
import json
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t630_early_departure_recovery'
DOC=ROOT/'stage/docs/day_20261118/t630'
sys.path.insert(0,str(ROOT/'stage/experiments/t629_offline_night_relay'))
import t629_runtime as R
import t630_native as Native
from early_departure import select
M=R.M
read,save,sha=M.read,M.save,M.sha


def plan():
    p=read(HERE/'protocol.json')
    config=read(p['actor_configuration']);config.update(p)
    return config


def verify(p,*args):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert R.used(OUT)<=p['max_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072<2**30
    assert R.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def setup(p):
    R.ROOT=ROOT;R.HERE=HERE;R.OUT=OUT;R.DOC=DOC
    R.B.ROOT=ROOT;R.B.HERE=HERE;R.B.OUT=OUT;R.B.DOC=DOC;R.B.M=M
    R.B.setup(p);R.B.verify=verify;R.B.R.Base.verify=lambda *a:verify(p)
    Native.Prior.install();R.B.N.worker=Native.worker
    R.B.N.N.inspect=Native.Prior.inspect
    R.B.N.restore(p);R.B.R.Base.envelope=R.envelope


def request(rt,p,jobs,role,full=False,refs=None):
    return rt.request(jobs,p['reference_model'],
                      'FRESH_TRAIN4' if full else role,references=refs)


def main():
    p=plan();verify(p);setup(p)
    assert read(ROOT/'stage/runs/t629_offline_night_relay/SEALED.json')['all_scientific_processes_stopped']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),new_optimizer=0))
    rt=R.B.Runtime(p);error=None;results=[];reset_rows=[]
    try:
        rt.pool=R.B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        # One boundary-crossing start and its canonical parent. New fixed jobs,
        # not a restart or completion claim for the old 72-neighborhood run.
        for job in p['reset_jobs']:
            vv=request(rt,p,[job],'RESET_BOUNDARY',full=True)
            request(rt,p,[job],'EXACT_RESET_BOUNDARY',refs=[vv[0][1]])
            reset_rows.append(vv[0][1])
        for source in p['candidate_sources']:
            job=source['request']['job']
            candidate=R.B.R.E.N.S.arrays(source['path'])
            vv=request(rt,p,[job],'BASELINE_ORIGINAL',full=True)
            baseline,row=vv[0]
            request(rt,p,[job],'EXACT_BASELINE',refs=[row])
            record=dict(id=job['id'],candidate_source=source,baseline=row,
                        departure=None,recovery=None,qualified=False)
            if R.safe(row):
                d=select(candidate,baseline);record['departure']=d
                if d is not None:
                    qjob=dict(job,source_path=source['path'],student_takeover_first=d['first'])
                    qq=request(rt,p,[qjob],'ACTUAL_EARLY_RECOVERY',full=True)
                    request(rt,p,[qjob],'EXACT_EARLY_RECOVERY',refs=[qq[0][1]])
                    record.update(recovery=qq[0][1],qualified=R.safe(qq[0][1]))
            results.append(record)
            save(OUT/'progress.json',dict(results=results,reset_rows=reset_rows))
    except BaseException:
        error=traceback.format_exc()
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),
             physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
             error=error,reset_rows=reset_rows,results=results,new_optimizer=0,
             qualified=sum(r['qualified'] for r in results),no_navigation_promotion=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p)
    s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    g=read(OUT/'worker_generations.json')
    assert all(not Path('/proc',str(i)).exists() for z in g for i in [z['parent'],*z['workers']])
    rows=R.B.read_rows('cases.jsonl');reps=R.B.read_rows('replays.jsonl')
    by={r['path']:r for r in rows}
    for row in rows:
        assert row['full'] and sha(row['path'])==row['sha256']
        a=R.B.R.E.N.S.arrays(row['path'])
        assert all(M.digest(v)==row['array_hashes'][k] for k,v in a.items())
        Native.Prior.inspect(p,[(a,row)])
    for r in reps:
        assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes']
        assert r['integration_sha256']==by[r['reference_path']]['integration_sha256']
    for record in s['results']:
        source=record['candidate_source'];assert sha(source['path'])==source['sha256']
        if record['departure'] is not None:
            a=R.B.R.E.N.S.arrays(source['path'])
            b=R.B.R.E.N.S.arrays(record['baseline']['path'])
            assert select(a,b)==record['departure']
        assert record['qualified']==(record['recovery'] is not None and R.safe(record['recovery']))
    assert sum(r['physics'] for r in rows+reps)==s['physics']<=p['max_physics']
    assert len(rows)+len(reps)+6==s['commands']<=p['max_commands']
    assert len(rows)==len(reps)
    save(OUT/'audit.json',dict(state='PASS',new_optimizer=0,all_raw_sha=True,
         original_history_and_own_hidden_reward_density_reconstructed=True,
         exact_replay_of_every_new_trajectory=True,departure_independently_recomputed=True))


if __name__=='__main__':
    {'main':main,'audit':audit}[sys.argv[1]]()
