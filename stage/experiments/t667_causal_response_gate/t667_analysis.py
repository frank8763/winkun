"""Network response along observed causal histories; not physical derivatives."""
from pathlib import Path
import fcntl,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t667_causal_response_gate';DOC=ROOT/'stage/docs/weekend_20261120_23/t667'
os.sched_setaffinity(0,{14})
sys.path.insert(0,str(ROOT/'stage/experiments/t666_failure_full_history'))
import t666_runtime as S
T,Native,read,sha,digest=S.T,S.Native,S.read,S.sha,S.digest
torch,np=T.torch,T.np
from status_io import save


def guard(p,start,limit):
    assert time.time()<min(start+limit,p['absolute_all_stop'])-10
    while True:
        try:S.Wait.Raw.resources();return
        except AssertionError as e:
            if not S.Wait.is_overlap(e):raise
            assert time.time()<min(start+limit,p['absolute_all_stop'])-10
            time.sleep(.25)


def verify(p):
    assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    for f,h in p['inputs'].items():assert sha(f)==h,f
    assert T.R.used(ROOT/'stage/runs')<p['global_bytes'] and T.R.used(OUT)<p['max_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    s=read(Path(p['source_run'])/'SEALED.json');assert s['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])


def inputs(p):
    src=Path(p['source_run']);source=read(src/'protocol.json');assert read(src/'audit.json')['state']=='PASS'
    s=read(src/'summary.json');rows=[r for r in s['rows'] if not r.get('exact')]
    assert len(rows)==8
    T.setup(source);Native.P.frozen=Native.frozen
    pairs=[]
    for pair in source['pairs']:
        vals={}
        for who in ['reference','candidate']:
            row=next(r for r in rows if r['role']==pair['id']+'_'+who)
            assert sha(row['path'])==row['sha256'] and row['full'] and row['seed'] is None
            a=T.R.E.N.S.arrays(row['path']);Native.inspect(source,a,row)
            q=Native.frozen(source,pair[who]['path']);assert q.phash()==row['policy_hash']
            x=torch.tensor(a['states'],dtype=torch.float64)
            with torch.no_grad():
                own=q.sequence(x)
                assert float((own-torch.tensor(a['locations'])).abs().max())<1e-10
                assert np.array_equal(own.clamp(-1,1).float().numpy(),a['actions'])
            vals[who]=dict(row=row,arrays=a,actor=q,x=x)
        a,b=[vals[k]['arrays'] for k in ['reference','candidate']]
        assert np.array_equal(a['states'][0],b['states'][0])
        assert np.array_equal(a['task_goal_xy'],b['task_goal_xy'])
        assert np.array_equal(a['integration_initial'],b['integration_initial']) if 'integration_initial' in a else True
        pairs.append((pair,vals))
    return source,pairs


def compute(p,limit):
    start=time.time();guard(p,start,limit);source,pairs=inputs(p)
    result=[];directions=[];eligible_total=0;numeric_all=True;nonzero_pairs=0;secant_pairs=0
    for pair,v in pairs:
        guard(p,start,limit)
        ar,ac=v['reference']['arrays'],v['candidate']['arrays']
        n=min(len(ar['states']),len(ac['states']))
        tr=np.r_[0,np.cumsum(ar['native_counts'])[:-1]][:n];tc=np.r_[0,np.cumsum(ac['native_counts'])[:-1]][:n]
        assert np.array_equal(tr,tc),'Decision times differ before termination'
        delta=np.linalg.norm(ar['poses'][:n,:2]-ac['poses'][:n,:2],axis=1)
        departure=np.flatnonzero(delta>=.01);end=int(departure[0]) if len(departure) else n-1
        points=sorted(set(np.rint(np.linspace(0,end,8)).astype(int).tolist()))
        xr,xc=v['reference']['x'],v['candidate']['x']
        qr,qc=v['reference']['actor'],v['candidate']['actor']
        with torch.no_grad():
            rr=qr.sequence(xr[:n]).clamp(-1,1);cr=qc.sequence(xr[:n]).clamp(-1,1)
            rc=qr.sequence(xc[:n]).clamp(-1,1);cc=qc.sequence(xc[:n]).clamp(-1,1)
        details=[];eligible=0;secant={k:dict(error2=0.,signal2=0.) for k in ['reference','candidate']}
        for index in points:
            h=xr[:index+1].clone();d=xc[:index+1]-h
            directions.append(dict(id=pair['id'],index=index,raw_history_sha256=digest(h),raw_direction_sha256=digest(d),
                history_path=v['reference']['row']['path'],direction_other_path=v['candidate']['row']['path'],normalization='raw direction; actor.normalized applied exactly once'))
            both_valid=True;reports={}
            for who,q in [('reference',qr),('candidate',qc)]:
                guard(p,start,limit)
                before=q.phash()
                def output(x):return q.sequence(x)[-1].clamp(-1,1)
                value,jvp=torch.autograd.functional.jvp(output,h,d,create_graph=False,strict=False)
                with torch.no_grad():
                    plus=output(h+p['finite_difference_epsilon']*d);minus=output(h-p['finite_difference_epsilon']*d)
                    finite=(plus-minus)/(2*p['finite_difference_epsilon'])
                    endpoint=output(h+d);sec=endpoint-value
                    loc=q.sequence(h)[-1]
                kink=bool((loc.abs()-1).abs().min()<p['clip_kink_tolerance'])
                error=(finite-jvp).abs();allowed=p['numeric_atol']+p['numeric_rtol']*jvp.abs()
                numeric=bool((error<=allowed).all())
                nonzero=bool(jvp.norm()>p['nonzero_tolerance']) and not kink
                both_valid=both_valid and nonzero
                if not kink:numeric_all=numeric_all and numeric
                if nonzero:
                    secant[who]['error2']+=float((jvp-sec).square().sum())
                    secant[who]['signal2']+=float(sec.square().sum())
                reports[who]=dict(command=value.tolist(),jvp=jvp.tolist(),finite_difference=finite.tolist(),
                    secant=sec.tolist(),numeric_pass=numeric,max_numeric_error=float(error.max()),clip_kink=kink,nonzero_differentiable=nonzero,
                    derivative_difference_from_secant=float((jvp-sec).norm()))
                assert q.phash()==before and all(x.grad is None for x in q.parameters())
            if both_valid:eligible+=1
            details.append(dict(index=index,seconds=float(tr[index]*.002),xy_difference_m=float(delta[index]),
                raw_direction_norm=float(d.norm()),responses=reports,
                response_parameter_change_norm=float(np.linalg.norm(np.asarray(reports['candidate']['jvp'])-np.asarray(reports['reference']['jvp'])))))
        residuals={who:(x['error2']/max(x['signal2'],1e-30))**.5 for who,x in secant.items()}
        sec_ok=all(x<p['max_relative_secant_residual'] for x in residuals.values()) and eligible>0
        eligible_total+=eligible;nonzero_pairs+=int(eligible>0);secant_pairs+=int(sec_ok)
        result.append(dict(id=pair['id'],source_models={w:v[w]['row']['policy_hash'] for w in v},
            reference_success=T.Prior.safe(v['reference']['row']),candidate_collision=v['candidate']['row']['collision'],
            first_cm_index=int(departure[0]) if len(departure) else None,first_cm_seconds=float(tr[end]*.002) if len(departure) else None,
            same_reference_history_command_RMS=float((cr-rr).square().mean().sqrt()),
            same_candidate_history_command_RMS=float((cc-rc).square().mean().sqrt()),
            own_history_command_RMS=float((cc-rr).square().mean().sqrt()),eligible_prefixes=eligible,
            relative_secant_residual=residuals,secant_gate=sec_ok,prefixes=details))
    checks=dict(numeric_all=numeric_all,at_least16_nonzero=eligible_total>=16,at_least3_nonzero_pairs=nonzero_pairs>=3,at_least3_secant_pairs=secant_pairs>=3)
    return dict(state='PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,
        total_directions=len(directions),eligible_prefixes=eligible_total,nonzero_pairs=nonzero_pairs,secant_pairs=secant_pairs,
        normalization_applied_once=True,full_prefix_own_carry=True,not_a_physical_derivative=True,
        new_native=0,new_optimizer=0,pairs=result,directions=directions)


def execute(action):
    p=read(OUT/'protocol.json');verify(p);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();result=compute(p,p['main_seconds'] if action=='main' else p['audit_seconds'])
    if action=='main':save(OUT/'RESULTS.json',result)
    else:
        original=read(OUT/'RESULTS.json');assert result==original,'Independent response computation differs'
        save(OUT/'audit.json',dict(state='PASS',results_digest=digest(result),all_input_hashes_and_history_rebuilt=True,
            numeric_gate=result['state'],new_native=0,new_optimizer=0))
    save(OUT/(action+'_receipt.json'),dict(state='COMPLETE',pid=os.getpid(),elapsed_s=time.time()-started,new_native=0,new_optimizer=0))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':execute(sys.argv[1])
