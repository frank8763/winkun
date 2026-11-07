"""Post-eviction current model/physics-factory load, cancel before stepping."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t412_closed_expert_likelihood'))
import t412_learning as E

def main():
    out=ROOT/'stage/runs/t413_remote_working_set/runtime_probe'
    assert not out.exists(),'Run this one zero-step post-cleanup probe once'
    out.mkdir();p=E.read(ROOT/'stage/runs/t412_closed_expert_likelihood/protocol.json')
    for f,h in p['inputs'].items():assert E.sha(f)==h,f
    base=E.N.science(p);q=E.safe_initial(base,p);teacher=E.expert(base,p)
    for actor in (q,teacher):actor.eval().requires_grad_(False)
    student_hash=q.phash();teacher_hash=teacher.phash()
    p=dict(p,out=str(out),native_stop_epoch=time.time()+180)
    counts=E.mp.get_context('spawn').RawArray('q',5);pool=None
    def guard(*args):assert sum(counts)==0 and time.time()<p['native_stop_epoch']-5
    try:
        pool=E.Pool(p,counts,guard);ready=pool.ready
        assert len(ready)==4 and all(x['native']==0 for x in ready)
        ids=[x.pid for x in pool.children]
    finally:
        if pool:pool.close()
    assert sum(counts)==0 and all(not Path('/proc',str(pid)).exists() for pid in ids)
    result=dict(state='PASS',student_policy_hash=student_hash,teacher_policy_hash=teacher_hash,
        workers=ids,ready=ready,all_workers_exited=True,commands=4,native_steps=0,optimizer_updates=0,
        scope='Current input SHA/model load and actual native factories; cancel before rollout. Not navigation success.')
    E.save(out/'result.json',result)
    E.save(ROOT/'stage/docs/storage_20261107/RUNTIME_PROBE.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
