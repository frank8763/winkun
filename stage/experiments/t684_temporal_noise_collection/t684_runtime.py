"""Matched fixed IID and temporal exploration collection, no optimization."""
from pathlib import Path
import copy,fcntl,gzip,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t684_temporal_noise_collection';DOC=ROOT/'stage/docs/weekend_20261120_23/t684'
import t684_wait as Wait
from status_io import save
import t684_native as N
T,M,torch,np=N.T,N.M,N.torch,N.np
read,sha,digest=N.C.read,N.C.sha,N.C.digest
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait

def verify(p,*args):
    assert p['authorization_approved'] and time.time()<p['absolute_all_stop']
    assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
    assert T.R.used(OUT)<p['max_bytes'] and T.R.used(ROOT/'stage/runs')-T.R.used(OUT)+p['max_bytes']<p['global_bytes']
    assert T.R.D.B.shutil.disk_usage(ROOT).free>40*2**30
    Wait.resources();h=read(DOC/'AFFINITY_GUARD.json');assert h['state']=='ACTIVE_RESOURCE_ONLY' and Path('/proc',str(h['pid'])).exists()
    for f,h in p['inputs'].items():assert sha(f)==h,f

T.verify=verify

def setup(p):
    T.setup(p);T.R.N.worker=N.worker;T.R.N.inspect=T.O.N.inspect=T.N.inspect=N.inspect
    assert T.R.N.Pool.__init__.__globals__['worker'] is N.worker

def schedule(p,tau):
    jobs=[];seeds=[]
    for i,job in enumerate(p['collection_jobs']):
        for rep in range(4):jobs.append(dict(job,temporal_tau=float(tau)));seeds.append([68420261121,i,rep])
    return jobs,seeds

def summary_report(p,fresh,stats):
    groups={arm:{key:[r for r in fresh if r['role']==arm+'_fresh' and r['job_id']==key] for key in p['collection_ids']} for arm in ['iid','tau2']}
    report={};safe={}
    for arm,by in groups.items():
        safe[arm]={};report[arm]={}
        for key,rows in by.items():
            assert len(rows)==4 and len({tuple(r['seed']) for r in rows})==4
            outcomes=[int(T.Prior.safe(r))-int(r['collision']) for r in rows];safe[arm][key]=sum(T.Prior.safe(r) for r in rows)
            report[arm][key]=dict(outcomes=outcomes,safe=safe[arm][key],collision=sum(r['collision'] for r in rows),timeout=sum(r['timeout'] for r in rows),
                boundary=sum(r['field_boundary'] for r in rows),mixed_safe_nonsafe=0<safe[arm][key]<4,
                source_paths=[r['path'] for r in rows],statistics=[stats[r['path']] for r in rows])
    mixed=[k for k in p['field_collection_ids'] if 0<safe['tau2'][k]<4]
    gains=[k for k in p['field_collection_ids'] if safe['tau2'][k]>safe['iid'][k]]
    old={arm:sum(safe[arm][k] for k in p['old_collection_ids']) for arm in safe}
    checks=dict(at_least2_field_mixed_safe_nonsafe=len(mixed)>=2,at_least2_field_more_safe_than_IID=len(gains)>=2,
        aggregate_old_not_below_IID=old['tau2']>=old['iid'],all64fresh=len(fresh)==64)
    paired={k:[dict(seed=[68420261121,p['collection_ids'].index(k),rep],iid=report['iid'][k]['outcomes'][rep],tau2=report['tau2'][k]['outcomes'][rep]) for rep in range(4)] for k in p['collection_ids']}
    return dict(gate='TEMPORAL_EXPLORATION_SUPPORT_PASS' if all(checks.values()) else 'SCIENTIFIC_GATE_FAILED',checks=checks,arms=report,old_safe_totals=old,
        field_mixed_tau2=mixed,field_safe_count_gains=gains,paired_outcomes=paired,
        independent_innovation_streams=32,common_random_numbers_NOT64_independent_pairs=True,replicates_per_ID=4,
        no_population_superiority_or_center_learning=True,no_optimizer=True,full407=None,no_candidate_or_promotion=True)

def statistics(q,a):
    with torch.no_grad():mu=q.sequence(torch.tensor(a['states'])).numpy()
    raw=a['raw_actions']-mu;command=a['actions']-np.clip(mu,-1,1).astype(np.float32)
    def lag(v):return [float(np.corrcoef(v[:-1,j],v[1:,j])[0,1]) if len(v)>2 and v[:-1,j].std()>1e-12 and v[1:,j].std()>1e-12 else None for j in range(2)]
    return dict(raw_offset_RMS=float(np.sqrt(np.square(raw).mean())),command_offset_RMS=float(np.sqrt(np.square(command).mean())),
        command_offset_lag1=lag(command),raw_offset_lag1=lag(raw),atom_fraction=float((a['atom']!=0).mean()),
        nonterminal_native_counts=sorted(set(map(int,a['native_counts'][:-1]))),
        actual_seconds=float(a['native_counts'].astype(np.float64).sum()*.002),unconditional_stationary_sigma_NOT_conditional_history_or_survival=True)

def collect():
    p=read(OUT/'protocol.json');verify(p);setup(p);T.R.E.L.install_stop_handlers();torch.set_num_threads(1)
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'science_started.json',dict(pid=os.getpid(),epoch=time.time()));rt=T.Runtime(p);error=None;fresh=[];stats={};report=None
    try:
        T.pool(rt);q=N.BASE_FROZEN(p,p['initial']['path']);base=digest(q.state_dict())
        for arm,tau in [('iid',0.),('tau2',2.)]:
            jobs,seeds=schedule(p,tau);model=dict(p['initial'],policy_hash=N.behavior(p,tau))
            vv=rt.request(jobs,model,arm+'_fresh',True,seeds=seeds);rr=[r for a,r in vv];fresh.extend(rr)
            stats.update({r['path']:statistics(q,a) for a,r in vv})
            rt.request(jobs,model,arm+'_exact',False,rr,seeds=seeds)
        assert digest(q.state_dict())==base
        report=summary_report(p,fresh,stats);save(OUT/'TEMPORAL_RESULTS.json',report)
    except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
    finally:
        T.close(rt);rt.commands+=3
        save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',pid=os.getpid(),error=error,physics=sum(rt.counts),commands=rt.commands,
            elapsed_s=time.time()-rt.start,gate=report['gate'] if report else None,result=report,rows=fresh,statistics=stats,
            new_Actor=0,new_Value=0,new_optimizer=0,center_unchanged=True,full407=None,retained=p['safe_retained']))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    assert error is None,error

def audit():
    p=read(OUT/'protocol.json');verify(p);setup(p);torch.set_num_threads(1);s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists()
    for g in read(OUT/'worker_generations.json'):assert all(not Path('/proc',str(pid)).exists() for pid in [g['parent'],*g['workers']])
    lock=(ROOT/'stage/runs/.stage_writer.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();last=[0.]
    def guard():
        assert time.time()<min(start+p['audit_seconds'],p['absolute_all_stop'])-10
        if time.time()-last[0]>5:verify(p);last[0]=time.time()
    rows=[json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()];by={r['path']:r for r in rows}
    assert len(rows)==len(by)==128
    stats={};q=N.BASE_FROZEN(p,p['initial']['path']);base=digest(q.state_dict())
    for r in rows:
        guard();assert sha(r['path'])==r['sha256'] and r['loaded_model_sha256']==p['initial']['sha256']
        a=T.R.E.N.S.arrays(r['path']);tau=0. if r['role'].startswith('iid_') else 2.
        assert r['request']['job']['temporal_tau']==tau and r['policy_hash']==N.behavior(p,tau)
        if r['full']:N.inspect(p,a,r);stats[r['path']]=statistics(q,a)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=by[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    fresh=[r for r in rows if not r.get('exact')];assert fresh==s['rows'] and stats==s['statistics']
    assert sum(r['physics'] for r in rows)==s['physics']<=p['max_physics'] and len(rows)+6==s['commands']<=p['max_commands']
    result=summary_report(p,fresh,stats);assert result==s['result']==read(OUT/'TEMPORAL_RESULTS.json')
    assert digest(q.state_dict())==base and not (OUT/'optimizer_calls.jsonl').exists()
    save(OUT/'audit.json',dict(state='PASS',all64fresh64exact_full_arrays_integrals_and_owncarry=True,raw_latent_density_and_conditional_clip_fields_separate=True,
        actual_previous_native_clock_verified=True,all_pairs_outcomes_metrics_reconstructed=True,new_native=0,new_optimizer=0,no_model_promoted=True,gate=result['gate']))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':{'collect':collect,'audit':audit}[sys.argv[1]]()
