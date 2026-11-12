"""One neural drive-head intervention, followed by full54 native evaluation.

This is a causal parameter-calibration pilot, not new policy-gradient training,
a runtime teacher, a planner, an algorithm novelty claim, or a stable8m claim.
"""
from pathlib import Path
import fcntl, json, math, os, shutil, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t495_drive_head_cause';DOC=ROOT/'stage/docs/review_20261112'
sys.path.insert(0,str(ROOT/'stage/experiments/t493_wide8_night'))
import t493_runtime as T
read,save,sha,digest,np,torch=T.read,T.save,T.sha,T.digest,T.np,T.torch

def verify(p,full=False):
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.M.used(OUT)<=p['max_bytes'] and T.M.used()-T.M.used(OUT)+p['max_bytes']<=2**30
    assert shutil.disk_usage(ROOT).free>=40*2**30
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    assert set(os.sched_getaffinity(0))<={14,15}
    for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f

def setup():
    T.B.OUT=T.B.ACTIVE=T.M.OUT=OUT;T.B.verify=T.M.verify=verify;T.M.setup_runtime()

def proposal(p):
    obj=T.L.make(p);before=T.P.state(*obj)
    # Exactly one scalar in the deployed network is changed. The multiplier,
    # actuator bounds, steering function, encoder, memory and task stay fixed.
    with torch.no_grad():obj[0].head.bias[0].add_(p['drive_logit_increment'])
    after=T.P.state(*obj)
    assert digest(before['value'])==digest(after['value'])
    assert digest(before['actor_opt'])==digest(after['actor_opt'])
    assert digest(before['value_opt'])==digest(after['value_opt'])
    for k in before['actor']:
        if k=='head.bias':
            assert torch.equal(before['actor'][k][1:],after['actor'][k][1:])
            assert torch.allclose(after['actor'][k][0]-before['actor'][k][0],after['actor'][k].new_tensor(p['drive_logit_increment']),rtol=0,atol=1e-15)
        else:assert torch.equal(before['actor'][k],after['actor'][k]),k
    return obj

def prepare():
    assert not OUT.exists()
    prior=ROOT/'stage/runs/t494_timeout_cause';assert read(prior/'audit.json')['state']=='PASS'
    T.stopped(prior);assert (prior/'local_manifest.json').exists()
    diag=read(prior/'diagnosis.json');assert len(diag)==14
    # Qualification is descriptive, not proof of causality. If it fails,
    # refuse registration rather than changing thresholds after evaluation.
    assert all(r['overall']['moving_away_seconds']<1 and r['detour_ratio']<1.1 and r['last20']['heading_error_deg']<20 and r['first20']['speed_m_s']<.04 for r in diag)
    p=read(prior/'protocol.json');p.pop('cached');p.pop('probe_jobs')
    root=ROOT/'stage/runs/t493_wide8_night/raw_ppo_02'
    candidate=read(root/'summary.json')['result']['candidate']
    rows=[r for r in T.rows(root) if r['role']=='FULL54_EIGHT']
    assert candidate['wide8']['success']==40 and not candidate['accepted']
    early=[]
    for r in T.rows(prior):
        a=T.E.N.S.arrays(r['path']);mask=np.r_[0.,np.cumsum(a['native_counts']*.002)[:-1]]<20
        early.append(float(np.median(a['locations'][mask,0])))
    median=float(np.median(early));target=median+p['raw_sigma'][0]
    assert -1.2<median<.5 and target<1.2
    increment=math.atanh(target/1.25)-math.atanh(median/1.25)
    now=time.time()
    p.update(id='T495_SINGLE_DRIVE_HEAD_CAUSAL_CALIBRATION',out=str(OUT),
             current=candidate,initial=candidate['model'],drive_logit_increment=increment,
             early_drive_location_median=median,early_drive_location_target=target,
             original_failure_ids=[r['job_id'] for r in rows if not r['success']],
             max_bytes=52*2**20,max_physics=5760072,max_commands=116,
             main_seconds=3600,learning_seconds=1,audit_seconds=600,
             absolute_learning_stop=now,absolute_native_stop=now+3900,absolute_all_stop=now+4800,
             Actor_Adam=0,Value_Adam=0,
             authorization='Nov12 user asks diagnose14 then modify and run them. New independent pilot from explicitly UNPROMOTED40/54 research model; frozen39 retained.',
             scientific_scope='One predeclared positive drive-head logit shift maps median early drive location up by existing one raw-sigma(.2). No search grid, no new Actor/Value optimizer, no runtime heuristic, no time/radius/geometry/actuator changes.',
             acceptance='All54 valid and no collision, all initial40 success IDs retained and success>40. Full54allpass still only development: under8/fresh/reload gate NOT run here.',
             replay_contract='Full54 native first. If all54safe, new worker processes and independently copied model repeat full54 exactly; otherwise repeat original14 failure IDs exactly. All physical repeats charged; not independent seeds.')
    p['inputs']=dict(p['inputs'])
    for f in [HERE/'run.py',HERE/'driver.py',HERE/'t495_transport.py',prior/'audit.json',prior/'diagnosis.json',prior/'local_manifest.json']:
        p['inputs'][str(f)]=sha(f)
    assert T.M.used()+p['max_bytes']+2**20<2**30
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    verify(p,True);obj=proposal(p);independent=proposal(p);assert digest(T.P.state(*obj))==digest(T.P.state(*independent))
    witness=[]
    for row in T.rows(prior):
        a=T.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);baseline=T.P.frozen(p,p['initial']['path'])
        with torch.no_grad():
            old=baseline.sequence(x);new=obj[0].sequence(x)
            assert torch.equal(old[:,1],new[:,1]) and bool((new[:,0]>=old[:,0]).all())
            h=None;first=[]
            for xx in x[:12]:_,loc,h,_=obj[0].step(xx[None],h);first.append(loc[0])
            error=float((torch.stack(first)-new[:12]).abs().max());assert error<1e-10
        witness.append(dict(id=row['job_id'],steering_function_exact_on_original_history=True,min_drive_increment=float((new[:,0]-old[:,0]).min()),step_sequence_error=error))
    # Serialize/reload actual altered network before any native step.
    model=T.P.export(obj,OUT/'proposal.pt.gz',p['model_bytes'])
    q=T.P.frozen(p,model['path']);assert q.phash()==model['policy_hash'] and digest(T.P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
    save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,exactly_one_neural_parameter_changed=True,drive_logit_increment=increment,median_raw_before=median,median_raw_target=target,model=model,witness=witness,serialized_joint_exact=True))
    save(DOC/'T495_BUDGET.json',dict(max_physics=p['max_physics'],max_commands=p['max_commands'],own_bytes=p['max_bytes'],global_reserved=T.M.used()-T.M.used(OUT)+p['max_bytes'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],Actor_Adam=0,CPU=[14,15],GPU=False))
    print('T495_REGISTERED_PREFLIGHT_PASS',increment,flush=True)

def main():
    p=read(OUT/'protocol.json');setup();verify(p,True);T.E.N.offline();T.E.L.install_stop_handlers();rt=T.B.Runtime(p)
    model=read(OUT/'preflight.json')['model'];assert sha(model['path'])==model['sha256'];state,error,result='FAILED',None,None
    try:
        rt.pool=T.Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard)
        vv=rt.request(p['wide8_jobs'],model,'DRIVE_HEAD_FULL54',full=True);rows=[r for a,r in vv]
        count=T.R.M.tally(rows);ids={r['job_id'] for r in rows if r['success'] and not r['collision']}
        before=set(p['current']['success_ids']);accepted=before<=ids and count['success']>40 and count['contact']==count['invalid']==count['censored']==0
        allpass=count['success']==54
        if allpass:
            rt.pool.close();rt.commands+=2
            dest=OUT/'independent_model.pt.gz';shutil.copyfile(model['path'],dest);assert sha(dest)==model['sha256']
            rt.pool=T.Pool(dict(p,native_stop_epoch=rt.deadline),rt.counts,rt.guard);rt.commands+=2
            rt.request(p['wide8_jobs'],dict(model,path=str(dest)),'INDEPENDENT_FULL54_MODEL_RELOAD',full=True,references=rows)
        else:
            by={r['job_id']:r for r in rows};jj=[j for j in p['wide8_jobs'] if j['id'] in p['original_failure_ids']]
            rt.request(jj,model,'EXACT_ORIGINAL14',full=True,references=[by[j['id']] for j in jj])
        result=dict(model=model,wide8=count,sides={s:sum(s in x for x in ids) for s in ['LEFT','RIGHT']},accepted=accepted,
                    gained_ids=sorted(ids-before),lost_ids=sorted(before-ids),success_ids=sorted(ids),
                    targeted14_goals=sum(x in ids for x in p['original_failure_ids']),
                    eight_development_allpass=allpass,independent54_model_readback_exact=allpass,
                    old275=None,wide6=None,fresh16=None,under8_gate=False,new_Actor_Adam=0,new_Value_Adam=0)
        save(OUT/'result.json',result);state='COMPLETE'
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        if rt.pool:rt.pool.close();rt.commands+=2
        save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,result=result,Actor_Adam=0,Value_Adam=0))
        fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
    assert state=='COMPLETE',error

def audit():
    p=read(OUT/'protocol.json');setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();T.stopped(OUT)
    model=s['result']['model'];assert digest(T.P.state(*proposal(p)))==model['joint_hash'] and sha(model['path'])==model['sha256']
    rr,rep=T.rows(OUT),T.rows(OUT,'replays.jsonl');assert len(rr)==54 and len(rep)==(54 if s['result']['eight_development_allpass'] else 14)
    for r in rr:
        assert sha(r['path'])==r['sha256'];a=T.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());T.M.audit_full_history(p,r,a)
    by={r['path']:r for r in rr}
    for r in rep:
        old=by[r['reference_path']];assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
    assert s['physics']==sum(r['physics'] for r in rr+rep)
    assert s['commands']==len(rr)+len(rep)+4+(4 if s['result']['eight_development_allpass'] else 0)
    ids={r['job_id'] for r in rr if r['success'] and not r['collision']};assert ids==set(s['result']['success_ids']) and T.R.M.tally(rr)==s['result']['wide8']
    save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,all54_full_arrays_density_history_reward_verified=True,original14_exact_or_independent54_readback_exact=True,exact_single_parameter_change=True,physics=s['physics'],commands=s['commands']))
    print('T495_AUDIT_PASS',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
