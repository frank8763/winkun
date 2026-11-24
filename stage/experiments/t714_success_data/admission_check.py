"""Bounded zero-physics witness of both real worker bindings and manifest joins."""
from pathlib import Path
import importlib.util,time,unittest.mock as mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t714_admission_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
def main():
    start=time.time();C.old_stopped();p=C.read(C.ROOT/'stage/runs/t712_retention_tradeoff/protocol.json');p['stage_cpus']=list(range(8,16));checks=[]
    original=C.os.sched_setaffinity
    def witness(pipe,index,pc,counts):
        assert C.M.N.N.P.frozen is C.A.frozen
        assert C.M.N.N.G.T.Session is (C.Geo.Session if pc.get('layout_test') else C.M.N.N.C.Session)
        checks.append(dict(layout=bool(pc.get('layout_test')),worker=index))
    try:
        with mock.patch.object(C.M.N.N.G.T,'worker',witness):
            for layout in [False,True]:C.worker(None,0,dict(p,layout_test=layout),None)
    finally:C.os.sched_setaffinity=original
    test=C.read(C.OUT/'prepared_test.json');contracts={r['id']:r for r in test['new_initial_contracts']}
    assert len(test['new_jobs'])==72 and len(test['new_exact_jobs'])==24
    for job in test['new_jobs']+test['new_exact_jobs']:assert job['layout_initial_contract']==contracts[job['id']]
    source=C.read(C.DOC/'SOURCE_SELECTION.json');acq=C.read(C.DOC/'ACQUISITION.json')
    assert {r['job_id'] for r in source['rows']}=={j['id'] for j in acq['jobs']} and len(acq['exact_jobs'])==18
    C.save(C.DOC/'ADMISSION_CHECK.json',dict(state='PASS',worker_bindings=checks,test_contracts=72,exact=24,acquisition_jobs=18,learning=0,physics=0,elapsed_seconds=time.time()-start))
    print('T714_ADMISSION_PASS',flush=True)
if __name__=='__main__':main()
