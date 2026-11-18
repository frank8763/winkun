"""One current-student suffix batch, conditional PG, union-of-successes gate."""
from pathlib import Path
import fcntl
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t633_conditional_recovery_pg'
DOC=ROOT/'stage/docs/day_20261118/t633'
sys.path.insert(0,str(ROOT/'stage/experiments/t631_explored_recovery'))
import t631_runtime as B
import t633_learning as L
M=L.M;R=B.R;N=B.N
read,save,sha=M.read,M.save,M.sha


def plan():
    p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);return c


def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert R.used(OUT)<=p['max_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072<2**30
    assert R.shutil.disk_usage(ROOT).free>=40*2**30
    for f,h in p['inputs'].items():assert sha(f)==h,f


def setup(p):
    B.HERE=HERE;B.OUT=OUT;B.DOC=DOC;B.verify=verify;B.setup(p)


def load(row,p,require_safe=False):
    assert sha(row['path'])==row['sha256']
    if require_safe:assert R.safe(row)
    a=R.B.R.E.N.S.arrays(row['path']);N.inspect(p,[(a,row)]);return a


def fit_pair(p,rt,rows,anchor):
    R.gpu_guard(p);M.configure_gpu(p)
    arrays=[load(r,p) for r in rows];aa=load(anchor,p,True);nominal=load(p['nominal_eight_row'],p,True)
    obj=M.make(p,p['raw_sigma']);shadow=M.make(p,p['raw_sigma']);before=M.state(obj)
    assert M.digest(before)==M.digest(M.state(shadow))
    event=lambda who:(lambda kind,k:R.B.append('optimizer_calls.jsonl',[who,kind,k]))
    apply=lambda target,who:L.fit(target,p,arrays,aa,nominal,lambda:(rt.guard(),R.gpu_guard(p)),event(who))
    try:
        u=apply(obj,'live');v=apply(shadow,'shadow')
        assert u==v and M.digest(M.state(obj))==M.digest(M.state(shadow))
        obj[0].cpu();model=M.export(obj,p,OUT/'proposal.pt.gz')
        save(OUT/'LEARNING_COMPLETE.json',dict(update=u,proposal=model,epoch=time.time()))
        return obj,before,u,model
    except BaseException:
        M.load(obj,before);M.load(shadow,before);raise


def main():
    p=plan();verify(p);setup(p)
    seal=read(ROOT/'stage/runs/t632_taskwise_recovery/SEALED.json')
    assert seal['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(i)).exists() for i in seal['scientific_pids'])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time()))
    rt=R.B.Runtime(p);error=None;obj=before=None
    result=dict(training_rows=[],anchor=None,update=None,proposal=None,handoffs=[],final49=None,
                accepted=False,model=p['reference_model'],retained_state_sha=None,learning_gate=None)
    try:
        rt.pool=R.B.N.Pool(dict(p,out=str(OUT),phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request([p['new_anchor_job']],p['student_initial'],'FRESH_TRAIN4');anchor=vv[0][1]
        rt.request([p['new_anchor_job']],p['student_initial'],'EXACT_NEW_ANCHOR',references=[anchor]);result['anchor']=anchor
        jobs=[dict(p['recovery_job']) for _ in range(4)];seeds=[[p['seed'],k] for k in range(4)]
        vv=rt.request(jobs,p['student_initial'],'FRESH_TRAIN4',seeds=seeds);rows=[r for a,r in vv]
        rt.request(jobs,p['student_initial'],'EXACT_FRESH_SUFFIX',seeds=seeds,references=rows)
        result['training_rows']=rows;gate=R.safe(anchor) and any(R.safe(r) for r in rows)
        result['learning_gate']=dict(anchor_safe=R.safe(anchor),suffix_safe=sum(R.safe(r) for r in rows),passed=gate)
        save(OUT/'progress.json',result)
        if gate:
            obj,before,u,model=fit_pair(p,rt,rows,anchor);result.update(update=u,proposal=model)
            handoff=dict(p['recovery_job'],explore_after_prefix=False)
            vv=rt.request([handoff],model,'FINAL49');hh=[r for a,r in vv]
            rt.request([handoff],model,'EXACT_HANDOFF',references=hh);result['handoffs']=hh
            vv=rt.request(p['evaluation_jobs'],model,'FINAL49');rr=[r for a,r in vv]
            keys=[r for r in rr if r['job_id'] in p['key_replay_ids']]
            rt.request([r['request']['job'] for r in keys],model,'EXACT_KEY4',references=keys)
            m=R.metrics(p,rr);lost=sorted(set(p['student_success_ids'])-set(m['success_ids']))
            keep=R.accepted(m) and not lost and all(R.safe(r) for r in hh)
            result.update(evaluation_rows=rr,final49=m,lost_student_successes=lost,accepted=keep,
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
        started={w:sum(e[0]==w and e[1]=='BEFORE' for e in events) for w in ['live','shadow']}
        completed={w:sum(e[0]==w and e[1]=='AFTER' for e in events) for w in ['live','shadow']}
        assert all(completed[w]<=started[w]<=p['actor_steps'] for w in started)
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),physics=sum(rt.counts),
            commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result,
            Actor_updates_started=started,Actor_updates_completed=completed,new_Critic=0,no_global_promotion=True))
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
        else:assert all(M.digest(a[k])==row['array_hashes'][k] for k in ['actions','native_counts','end_flags'])
    for row in reps:
        assert row['exact'] and row['array_hashes']==by[row['reference_path']]['array_hashes']
        assert row['integration_sha256']==by[row['reference_path']]['integration_sha256']
    r=s['result'];gate=R.safe(r['anchor']) and any(R.safe(x) for x in r['training_rows'])
    assert r['learning_gate']['passed']==gate;updates=0
    if r['update'] is not None:
        assert gate;R.gpu_guard(dict(p,absolute_learning_stop=p['absolute_all_stop']));M.configure_gpu(p)
        obj=M.make(p,p['raw_sigma']);before=M.state(obj);events=[]
        u=L.fit(obj,p,[load(x,p) for x in r['training_rows']],load(r['anchor'],p,True),load(p['nominal_eight_row'],p,True),
                lambda:R.gpu_guard(dict(p,absolute_learning_stop=p['absolute_all_stop'])),
                lambda kind,k:events.append(['live',kind,k]))
        assert u==r['update'];updates=u['attempted_optimizer'];obj[0].cpu()
        assert M.digest(M.state(obj))==r['proposal']['joint_hash'] and obj[0].phash()==r['proposal']['policy_hash']
        calls=R.B.read_rows('optimizer_calls.jsonl')
        assert [v for v in calls if v[0]=='live']==events
        assert [['live',*v[1:]] for v in calls if v[0]=='shadow']==events
        m=R.metrics(p,r['evaluation_rows']);assert m==r['final49']
        lost=sorted(set(p['student_success_ids'])-set(m['success_ids']));assert lost==r['lost_student_successes']
        assert r['accepted']==(R.accepted(m) and not lost and all(R.safe(x) for x in r['handoffs']))
        if not r['accepted']:M.load(obj,before)
        assert M.digest(M.state(obj))==r['retained_state_sha']
    assert sum(x['physics'] for x in rows+reps)==s['physics']<=p['max_physics']
    assert len(rows)+len(reps)+6==s['commands']<=p['max_commands']
    save(OUT/'audit.json',dict(state='PASS',independent_historical_Actor=updates,new_native=0,
         all_raw_SHA_and_exact_replays=True,conditional_suffix_density_and_history_rebuilt=True,
         original_and_student_success_union_gate_reconstructed=True))


if __name__=='__main__':{'main':main,'audit':audit}[sys.argv[1]]()
