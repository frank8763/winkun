"""Qualify a fixed learned teacher from two actual failed student prefixes.

Teacher success is training-data qualification, never student navigation success.
No learning, optimizer, search or online action teacher is deployed to students.
"""
from pathlib import Path
import sys,time,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t598_independent_recurrent_recovery'))
import t598_runtime as Legacy
L=Legacy.L
OUT=ROOT/'stage/runs/t602_actual_state_teacher';DOC=ROOT/'stage/docs/day_20261116/t602'
Base,F,G,E,P,torch,np=Legacy.Base,Legacy.F,Legacy.G,Legacy.E,Legacy.P,L.torch,L.np
read,save,sha,digest=L.read,L.save,L.sha,L.digest


def setup(folder):
    Legacy.OUT=OUT;Legacy.HERE=HERE;Legacy.DOC=DOC;Legacy.setup(folder)


def verify(p,full=False):return Legacy.verify(p,full)


def schedule_probe(p):
    class Dispatched(Exception):pass
    rt=Base.Runtime(p);rt.last=time.time()
    class VirtualPool:
        def request(self,req):
            rt.counts[0]+=sum(r['job']['case']['step_cap'] for r in req)
            raise Dispatched()
    rt.pool=VirtualPool()
    try:
        assert rt.commands==2
        for replay in [False,True]:
            try:rt.request(p['qualification_jobs'],p['teacher_model'],'ZERO_NATIVE',reference=[dict(path='unused')]*2 if replay else None)
            except Dispatched:pass
        rt.commands+=2;assert rt.commands==8==p['max_commands'] and sum(rt.counts)==p['max_physics']
        return dict(state='PASS_REAL_RUNTIME_COUNTER_PATH',commands=rt.commands,virtual_worst_physics=sum(rt.counts),new_native=0,new_workers=0)
    finally:fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()


def main():
    p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True)
    E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None
    result=dict(qualified=[],rows=None,new_Adam=0,new_RL=0,student_success_NOT_measured=True,full407=None,fresh16=None,reload78=None)
    try:
        rt.pool=Legacy.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['qualification_jobs'],p['teacher_model'],'FIXED_TEACHER_ACTUAL_STUDENT_STATES',full=True)
        Legacy.inspect(p,vv);rows=[r for a,r in vv]
        result.update(rows=rows,qualified=[dict(source_method=m,row=r) for m,r in zip(['mean','tail'],rows) if Legacy.R.safe(r)])
        save(OUT/'qualification_before_replay.json',result)
        rt.request(p['qualification_jobs'],p['teacher_model'],'EXACT_TEACHER_QUALIFICATION',reference=rows)
        state='COMPLETE'
    except BaseException:
        error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=__import__('os').getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error


def audit():
    p=dict(read(OUT/'protocol.json'),_root_sha256=sha(OUT/'protocol.json'));setup(OUT);verify(p,True);G.stopped(OUT)
    assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE'
    rows=G.rows(OUT);reps=G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==2
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert int(a['native_counts'].sum())==r['physics'] and all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        Legacy.inspect(p,[(a,r)])
    for old,new in zip(rows,reps):assert new['exact'] and new['reference_path']==old['path'] and new['array_hashes']==old['array_hashes'] and new['integration_sha256']==old['integration_sha256']
    assert rows==s['result']['rows'] and s['result']['qualified']==[dict(source_method=m,row=r) for m,r in zip(['mean','tail'],rows) if Legacy.R.safe(r)]
    assert s['commands']==8 and s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
    save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,all_history_prefix_and_extra_memories_exact=True,all_native_replays_exact=True,source_model_hashes_unchanged=True,physics=s['physics'],commands=s['commands']))


if __name__=='__main__':globals()[sys.argv[1]]()
