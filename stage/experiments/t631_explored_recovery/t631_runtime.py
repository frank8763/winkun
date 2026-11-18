"""Fixed eight recovery explorations, conditional single fit, full49 and exact."""
from pathlib import Path
import fcntl
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t631_explored_recovery'
DOC=ROOT/'stage/docs/day_20261118/t631'
sys.path.insert(0,str(ROOT/'stage/experiments/t630_early_departure_recovery'))
import t630_runtime as O
import t631_native as N
import t631_learning as L
M=L.M;R=O.R;read,save,sha=M.read,M.save,M.sha


def plan():
    p=read(HERE/'protocol.json');cfg=read(p['actor_configuration']);cfg.update(p)
    return cfg


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=p['max_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072<2**30
    assert R.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def setup(p):
    O.HERE=HERE;O.OUT=OUT;O.DOC=DOC;O.verify=verify;O.setup(p)
    R.B.N.worker=N.worker;R.B.N.N.inspect=N.inspect


def anchor_arrays(p):
    arrays=[]
    for row in p['new_anchor_rows']:
        assert sha(row['path'])==row['sha256'] and R.safe(row)
        a=R.B.R.E.N.S.arrays(row['path']);N.inspect(p,[(a,row)]);arrays.append(a)
    return arrays


def fit_pair(p,rt,selected):
    R.gpu_guard(p);M.configure_gpu(p)
    corrections=[(R.B.R.E.N.S.arrays(row['path']),row['request']['job']['student_takeover_first']) for row in selected]
    anchors=anchor_arrays(p);obj=M.make(p,p['raw_sigma']);shadow=M.make(p,p['raw_sigma'])
    before=M.state(obj);assert M.digest(before)==M.digest(M.state(shadow))
    event=lambda who:(lambda kind,k:R.B.append('optimizer_calls.jsonl',[who,kind,k]))
    apply=lambda target,who:L.fit(target,p,corrections,anchors,lambda:(rt.guard(),R.gpu_guard(p)),event(who))
    try:
        u=apply(obj,'live');v=apply(shadow,'shadow')
        assert u==v and M.digest(M.state(obj))==M.digest(M.state(shadow))
        obj[0].cpu();model=M.export(obj,p,OUT/'proposal.pt.gz')
        save(OUT/'LEARNING_COMPLETE.json',dict(update=u,proposal=model,epoch=time.time()))
        return obj,before,u,model
    except BaseException:
        M.load(obj,before);M.load(shadow,before)
        raise


def main():
    p=plan();verify(p);setup(p)
    assert read(ROOT/'stage/runs/t630_early_departure_recovery/SEALED.json')['all_scientific_processes_stopped']
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists()
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=R.B.Runtime(p);error=None;obj=before=None
    result=dict(exploration=[],selected=[],update=None,proposal=None,handoffs=[],final49=None,
                accepted=False,model=p['reference_model'],retained_state_sha=None)
    try:
        rt.pool=R.B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        for index,job in enumerate(p['recovery_jobs']):
            jobs=[dict(job) for _ in range(4)];seeds=[[p['seed'],index,k] for k in range(4)]
            vv=rt.request(jobs,p['reference_model'],'FRESH_TRAIN4',seeds=seeds)
            rows=[r for a,r in vv]
            rt.request(jobs,p['reference_model'],'EXACT_EXPLORED_RECOVERY',seeds=seeds,references=rows)
            result['exploration']+=rows
            qualified=[r for r in rows if R.safe(r)]
            if qualified:result['selected'].append(qualified[0])
            save(OUT/'progress.json',result)
        if len(result['selected'])==2:
            obj,before,u,model=fit_pair(p,rt,result['selected'])
            result.update(update=u,proposal=model)
            deterministic=[dict(j,explore_after_prefix=False) for j in p['recovery_jobs']]
            vv=rt.request(deterministic,model,'FINAL49');rows=[r for a,r in vv]
            rt.request(deterministic,model,'EXACT_HANDOFF',references=rows)
            result['handoffs']=rows
            vv=rt.request(p['evaluation_jobs'],model,'FINAL49');rows=[r for a,r in vv]
            keys=[r for r in rows if r['job_id'] in p['key_replay_ids']]
            rt.request([r['request']['job'] for r in keys],model,'EXACT_KEY4',references=keys)
            m=R.metrics(p,rows);keep=R.accepted(m) and all(R.safe(r) for r in result['handoffs'])
            result.update(evaluation_rows=rows,final49=m,accepted=keep,
                          model=model if keep else p['reference_model'])
            if not keep:M.load(obj,before)
            result['retained_state_sha']=M.digest(M.state(obj))
    except BaseException:
        error=traceback.format_exc()
        if obj is not None and before is not None:M.load(obj,before)
        save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=3
        events=R.B.read_rows('optimizer_calls.jsonl') if (OUT/'optimizer_calls.jsonl.gz').exists() else []
        started={who:sum(v[0]==who and v[1]=='BEFORE' for v in events) for who in ['live','shadow']}
        completed={who:sum(v[0]==who and v[1]=='AFTER' for v in events) for who in ['live','shadow']}
        assert all(completed[w]<=started[w]<=p['recovery_steps'] for w in started)
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),
             physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,
             error=error,result=result,Actor_updates_started=started,
             Actor_updates_completed=completed,new_Critic=0,no_global_promotion=True))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error


def audit():
    p=plan();verify(p);setup(p);s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    assert not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc',str(i)).exists() for i in [g['parent'],*g['workers']])
    rows=R.B.read_rows('cases.jsonl');reps=R.B.read_rows('replays.jsonl');by={r['path']:r for r in rows}
    for row in rows:
        assert sha(row['path'])==row['sha256'];a=R.B.R.E.N.S.arrays(row['path'])
        if row['full']:
            assert all(M.digest(v)==row['array_hashes'][k] for k,v in a.items());N.inspect(p,[(a,row)])
        else:
            assert all(M.digest(a[k])==row['array_hashes'][k] for k in ['actions','native_counts','end_flags'])
    for row in reps:
        assert row['exact'] and row['array_hashes']==by[row['reference_path']]['array_hashes']
        assert row['integration_sha256']==by[row['reference_path']]['integration_sha256']
    r=s['result'];chosen=[]
    for job in p['recovery_jobs']:
        values=[x for x in r['exploration'] if x['job_id']==job['id']]
        assert len(values)==4
        q=[x for x in values if R.safe(x)]
        if q:chosen.append(q[0])
    assert chosen==r['selected'];updates=0
    if r['update'] is not None:
        R.gpu_guard(dict(p,absolute_learning_stop=p['absolute_all_stop']));M.configure_gpu(p)
        obj=M.make(p,p['raw_sigma']);before=M.state(obj);events=[]
        corrections=[(R.B.R.E.N.S.arrays(x['path']),x['request']['job']['student_takeover_first']) for x in chosen]
        u=L.fit(obj,p,corrections,anchor_arrays(p),lambda:R.gpu_guard(dict(p,absolute_learning_stop=p['absolute_all_stop'])),
                lambda kind,k:events.append(['live',kind,k]))
        assert u==r['update'];updates=u['attempted_optimizer'];assert updates<=p['recovery_steps']
        obj[0].cpu();assert M.digest(M.state(obj))==r['proposal']['joint_hash']
        assert obj[0].phash()==r['proposal']['policy_hash']
        calls=R.B.read_rows('optimizer_calls.jsonl')
        assert [v for v in calls if v[0]=='live']==events
        assert [['live',*v[1:]] for v in calls if v[0]=='shadow']==events
        m=R.metrics(p,r['evaluation_rows']);assert m==r['final49']
        assert r['accepted']==(R.accepted(m) and all(R.safe(x) for x in r['handoffs']))
        if not r['accepted']:M.load(obj,before)
        assert M.digest(M.state(obj))==r['retained_state_sha']
    assert len(rows)+len(reps)+6==s['commands']<=p['max_commands']
    assert sum(x['physics'] for x in rows+reps)==s['physics']<=p['max_physics']
    save(OUT/'audit.json',dict(state='PASS',independent_historical_Actor=updates,new_native=0,
         all_raw_SHA_and_stochastic_own_history_reconstructed=True,all_exact_replays_counted=True,
         successful_actual_actions_not_failed_teacher_means=True))


if __name__=='__main__':{'main':main,'audit':audit}[sys.argv[1]]()
