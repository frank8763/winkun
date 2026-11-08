"""At most eight discarded original BC0 steps; no native or new policy.

The pass signal is the original exact Actor AND Adam AFTER0 hashes. Matching
loss or a close parameter vector is explicitly insufficient.
"""
from pathlib import Path
import sys,json,time,copy,gzip,io,hashlib,signal,traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t422_feedback_recovery_night'))
import t422_common as C
import learn as LEARN
E,np,torch=C.E,C.np,C.torch
OUT=ROOT/'stage/runs/t422_feedback_recovery_night'
DOC=ROOT/'stage/docs/day_20261108'
LIMIT_STEPS=8;LIMIT_SECONDS=600

def main():
    start=time.time();deadline=start+LIMIT_SECONDS;DOC.mkdir(exist_ok=True)
    report=DOC/'T425_PROVENANCE.json';assert not report.exists()
    original={str(f):C.sha(f) for f in OUT.rglob('*') if f.is_file()}
    p=C.read(OUT/'protocol.json');C.verify(p)
    rows=[json.loads(line) for line in (OUT/'learning/optimizer_calls.jsonl').read_text().splitlines()]
    expected_live=next(r for r in rows if (r['tag'],r['which'],r['where'],r['k'])==('BC0','live','AFTER',0))['hashes']
    expected_shadow=next(r for r in rows if (r['tag'],r['which'],r['where'],r['k'])==('BC0','shadow','AFTER',0))['hashes']
    assert expected_live==expected_shadow
    before_file=OUT/'learning/blocks/0/before.pt.gz'
    before=torch.load(io.BytesIO(gzip.decompress(before_file.read_bytes())),weights_only=False,map_location='cpu')
    original_trace=C.read(OUT/'learning/blocks/0/result.json')['bc']['fit']['trace'][0]
    E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def prohibited(*a,**k):raise AssertionError('Provenance probe forbids native simulation, model export or new policy storage')
    E.L.Session=prohibited;E.Pool=prohibited;E.export=prohibited;E.N.write_npz=prohibited
    counted=0;rawstep=torch.optim.Adam.step
    def guard():assert time.time()<deadline-5
    def step(opt,*a,**k):
        nonlocal counted
        guard();assert counted<LIMIT_STEPS
        value=rawstep(opt,*a,**k);counted+=1;return value
    torch.optim.Adam.step=step
    def timeout(signum,frame):raise TimeoutError('600 second provenance budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(LIMIT_SECONDS)
    initial_rows=C.read(OUT/'learning/initial_arrays.json')
    initial_values=[(E.N.S.arrays(r['path']),r) for r in initial_rows]
    record=C.read(OUT/'learning/blocks/0/result.json')
    recoveries=[dict(arrays=E.N.S.arrays(r['path']),first=r['first'],source_id=r['source_id']) for r in record['recovery_sources']]
    probes=[];first_state=None
    for name,flush,foreach in [('default',None,None),('flush_denormal_true',True,None),('flush_denormal_false',False,None),('scalar_no_foreach',False,False),('explicit_foreach',False,True)]:
        guard()
        if flush is not None:assert torch.set_flush_denormal(flush)
        q,opt=E.learner(E.N.science(p),p);E.L.load(q,opt,before['state'])
        if foreach is not None:
            for group in opt.param_groups:group['foreach']=foreach
        initial=E.L.statehash(q,opt);g=LEARN.build_groups(q,initial_values,p,recoveries,p['query_sources'])
        events=[];grad=None
        def event(where,k):
            nonlocal grad
            if where=='BEFORE':grad=C.digest([v.grad for v in E.params(q)])
            events.append(dict(where=where,k=k,hashes=E.L.statehash(q,opt)))
        fit=LEARN.fit_bc(q,opt,g,p,guard,event,steps=1)
        state=E.L.state(q,opt)
        if first_state is None:first_state=copy.deepcopy(state)
        diff=[]
        for key,tensor in state['learned'].items():
            other=first_state['learned'][key]
            if not torch.equal(tensor,other):diff.append(dict(name=key,count=int((tensor!=other).sum()),max_abs=float((tensor-other).abs().max())))
        actual=E.L.statehash(q,opt)
        row=dict(name=name,flush_denormal=flush,foreach=foreach,initial_hashes=initial,after_hashes=actual,expected_original_AFTER0=expected_live,exact_learned=actual['learned']==expected_live['learned'],exact_Adam=actual['optimizer']==expected_live['optimizer'],complete_exact=actual==expected_live,events=events,trace0=fit['trace'][0],original_trace0=original_trace,trace_exact=fit['trace'][0]==original_trace,clipped_gradient_sha256=grad,learned_differences_from_default=diff,param_groups=state['optimizer']['param_groups'],step_dtype={str(i):str(s['step'].dtype) for i,s in state['optimizer']['state'].items()},original_pre_hash=before['hashes'])
        probes.append(row)
        C.save(DOC/'T425_PROVENANCE_PROGRESS.json',dict(state='PROBING',discarded_historical_Adam=counted,probes=probes,native=0,new_navigation_training=0))
        print('T425_PROVENANCE_PROBE',name,actual,'exact',row['complete_exact'],flush=True)
        # A confirmed exact loop is repeated once with only the successful
        # environment correction, before asserting a causal repair.
        if row['complete_exact']:
            break
    exact=[r for r in probes if r['complete_exact']]
    if exact:
        chosen=exact[0];q,opt=E.learner(E.N.science(p),p);E.L.load(q,opt,before['state'])
        if chosen['flush_denormal'] is not None:torch.set_flush_denormal(chosen['flush_denormal'])
        if chosen['foreach'] is not None:
            for group in opt.param_groups:group['foreach']=chosen['foreach']
        g=LEARN.build_groups(q,initial_values,p,recoveries,p['query_sources']);fit=LEARN.fit_bc(q,opt,g,p,guard,lambda *_:None,steps=1)
        confirmation=E.L.statehash(q,opt);assert confirmation==expected_live and fit['trace'][0]==original_trace
    else:confirmation=None
    signal.alarm(0);assert original=={str(f):C.sha(f) for f in OUT.rglob('*') if f.is_file()}
    C.save(report,dict(state='EXACT_ONE_STEP_ADAPTER_VERIFIED_FULL_HISTORY_AUDIT_STILL_REQUIRED' if exact else 'FAIL_CAUSE_UNRESOLVED_NO_THRESHOLD_RELAXATION',elapsed_s=time.time()-start,wall_limit_s=LIMIT_SECONDS,step_limit=LIMIT_STEPS,discarded_historical_Adam=counted,native=0,new_navigation_training=0,new_retained_policy=0,original_shadow_AFTER0_matches_live=True,probes=probes,confirmation_hashes=confirmation,all_original_T422_files_SHA_unchanged=True,original_file_count=len(original),no_old_failure_overwritten=True,old_full_training_audit_not_claimed_PASS=True))
    print('T425_PROVENANCE_DONE',len(exact),counted,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        DOC.mkdir(exist_ok=True);C.save(DOC/'T425_PROVENANCE_ERROR.json',dict(state='FAIL_PROBE_ENTRY_OR_CONTRACT',native=0,error=traceback.format_exc()))
        raise
