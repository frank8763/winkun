"""Direct noisy closed-loop validation of two fixed rejected candidates."""
from pathlib import Path
import fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t686_direct_stochastic_validation';DOC=ROOT/'stage/docs/weekend_20261120_23/t686'
import t686_wait as Wait
from status_io import save
sys.path.insert(0,str(ROOT/'stage/experiments/t679_episode_parameter_exploration'))
import t679_calibration as C
sys.path.insert(0,str(ROOT/'stage/experiments/t684_temporal_noise_collection'))
import t684_native as SourceNative
T,M,torch,np=C.T,C.M,C.torch,C.np
read,sha,digest=C.read,C.sha,C.digest
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait

def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')-T.R.used(OUT)+p['max_bytes']<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f

T.verify=verify

def baseline(p):
    rows=[r for r in read(p['baseline_summary'])['rows'] if r['role']=='iid_fresh']
    assert len(rows)==32 and len({(r['job_id'],tuple(r['seed'])) for r in rows})==32
    return rows

def schedule(p):
    jobs=[];seeds=[]
    for i,job in enumerate(p['collection_jobs']):
        for rep in range(4):jobs.append(job);seeds.append([68420261121,i,rep])
    return jobs,seeds

def outcome(r):return int(T.Prior.safe(r))-int(r['collision'])

def result(p,fresh):
    original=baseline(p);prediction=read(p['prediction_results'])['models'];reports={};coexist={}
    for item in p['candidates']:
        name=item['name'];rows=[r for r in fresh if r['policy_hash']==item['model']['policy_hash']]
        assert len(rows)==32;by={k:[r for r in rows if r['job_id']==k] for k in p['collection_ids']}
        oldscreen=read(item['screen']);reports[name]={};qualified=[]
        for key,rr in by.items():
            b=[r for r in original if r['job_id']==key];b.sort(key=lambda r:r['seed'][-1]);rr.sort(key=lambda r:r['seed'][-1])
            assert len(b)==len(rr)==4 and [r['seed'] for r in rr]==[r['seed'] for r in b]
            candidate_safe=sum(T.Prior.safe(r) for r in rr);base_safe=sum(T.Prior.safe(r) for r in b)
            lost=key in oldscreen['lost_ids'];ys=[outcome(r) for r in rr];yb=[outcome(r) for r in b]
            delta=float(np.mean(np.asarray(ys)-np.asarray(yb)));isdelta=prediction[name]['fresh'][key]['trajectory_weights']['ordinary_IS_change']
            sign=lambda x:int(x>1e-12)-int(x<-1e-12)
            meets=lost and candidate_safe>=1 and candidate_safe>=base_safe
            if meets:qualified.append(key)
            reports[name][key]=dict(candidate_safe=candidate_safe,baseline_safe=base_safe,outcomes=ys,baseline_outcomes=yb,
                collision=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),boundary=sum(r['field_boundary'] for r in rr),
                paired_Y_differences=[y-x for x,y in zip(yb,ys)],paired_mean_Y_change=delta,T685_ordinary_IS_change=isdelta,
                direct_and_OPE_signs_agree=sign(delta)==sign(isdelta),known_seedNone_deterministic_failure=lost,
                descriptive_random_safe_coexists_with_deterministic_failure=meets,
                candidate_paths=[r['path'] for r in rr],baseline_paths=[r['path'] for r in b],seeds=[r['seed'] for r in rr])
        coexist[name]=qualified
    observed=all(coexist.values())
    return dict(gate='FINITE_STOCHASTIC_DETERMINISTIC_DIVERGENCE_OBSERVED' if observed else 'FINITE_DIVERGENCE_CRITERION_NOT_MET',
        models=reports,qualifying_known_lost_IDs=coexist,replicates_per_ID=4,independent_innovation_streams=32,
        candidate_seed_pairs_new_but_CRN_across_models_NOT_independent=True,baseline_reused_new_native=0,
        no_population_stochastic_improvement_or_deterministic_recovery=True,no_learning_admission=True,
        no_optimizer_or_new_candidate=True,full407=None,accepted=False)

def collect():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);T.R.E.L.install_stop_handlers();torch.set_num_threads(1)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;fresh=[];report=None
    try:
        save(OUT/'BASELINE_REUSED.json',dict(rows=baseline(p),new_native=0,new_independent_samples=0))
        T.pool(rt);jobs,seeds=schedule(p)
        for item in p['candidates']:
            vv=rt.request(jobs,item['model'],item['name']+'_fresh',True,seeds=seeds);rr=[r for a,r in vv];fresh.extend(rr)
            rt.request(jobs,item['model'],item['name']+'_exact',False,rr,seeds=seeds)
        report=result(p,fresh);save(OUT/'DIRECT_RESULTS.json',report)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,gate=report['gate'] if report else None,result=report,rows=fresh,
            new_Actor=0,new_Value=0,new_optimizer=0,full407=None,retained=p['safe_retained']))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=read(OUT/'protocol.json');verify(p);T.setup(p);torch.set_num_threads(1);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows}
    assert len(rows)==len(by)==128
    models={m['model']['policy_hash']:m['model'] for m in p['candidates']}
    for r in rows:
        guard();assert sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==models[r['policy_hash']]['sha256']
        a=T.R.E.N.S.arrays(r['path'])
        if r['full']:T.N.inspect(p,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    sourcep=read(p['baseline_protocol']);sourcerows=[json.loads(x) for x in gzip.decompress(Path(p['baseline_cases']).read_bytes()).decode().splitlines()]
    sources={r['path']:r for r in sourcerows}
    for r in sourcerows:
        if not r['role'].startswith('iid_'):continue
        guard();assert sha(r['path'])==r['sha256'];a=T.R.E.N.S.arrays(r['path'])
        if r['full']:SourceNative.inspect(sourcep,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=sources[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    reference_pairs={(r['job_id'],tuple(r['seed'])):r for r in baseline(p)}
    for r in rows:
        if not r['full']:continue
        guard();reference=reference_pairs[(r['job_id'],tuple(r['seed']))]
        a=T.R.E.N.S.arrays(r['path']);b=T.R.E.N.S.arrays(reference['path']);length=min(len(a['states']),len(b['states']))
        assert np.array_equal(a['innovations'][:length],b['innovations'][:length])
        for key in ['initial_native_integration','task_goal_xy','scene_sha256']:
            assert np.array_equal(a[key],b[key]),key
        assert np.array_equal(a['states'][0],b['states'][0]) and not a['hidden_before'][0].any()
    assert baseline(p)==read(OUT/'BASELINE_REUSED.json')['rows']
    fresh=[r for r in rows if not r.get('exact')];assert fresh==s['rows'] and len(fresh)==64
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    again=result(p,fresh);assert again==s['result']==read(OUT/'DIRECT_RESULTS.json') and not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',all64fresh64exact_full_arrays_integrals_and_owncarry=True,baseline32full32exact_reaudited_no_new_native=True,
        paired_initial_state_and_shared_innovation_prefix_exact=True,
        candidate_clip_vs_baseline_latent_logp_semantics_separate=True,all_paired_outcomes_and_OPE_comparisons_exact=True,
        new_native=0,new_optimizer=0,no_model_promoted=True,gate=again['gate']))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
