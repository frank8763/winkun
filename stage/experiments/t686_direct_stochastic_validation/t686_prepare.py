"""Two fixed candidate loaders and unused model-seed pairs, no native steps."""
from pathlib import Path
import ast,copy,gzip,json,sys,time
import t686_runtime as S
ROOT,HERE,OUT,DOC=S.ROOT,S.HERE,S.OUT,S.DOC
T,M,torch,np=S.T,S.M,S.torch,S.np
read,sha,digest=S.read,S.sha,S.digest
from status_io import save

def main():
    start=time.time();assert not OUT.exists();DOC.mkdir(exist_ok=True)
    preceding=read(HERE/'protocol.json')['preflight_seconds'] if (HERE/'protocol.json').exists() else 0
    last=ROOT/'stage/runs/t685_fresh_proxy_validity';rep=read(last/'FINAL_REPORT.json');seal=read(last/'SEALED.json')
    assert rep['state']=='COMPLETE' and rep['audit']['state']=='PASS'
    ids=seal['scientific_pids']+[rep['helper_pid'],read(last/'DRIVER_STARTED.json')['pid']]
    assert all(not Path('/proc',str(pid)).exists() for pid in ids)
    priorroot=ROOT/'stage/runs/t684_temporal_noise_collection';prior=read(priorroot/'protocol.json');p=copy.deepcopy(prior);p.pop('registration_epoch',None)
    candidates=read(last/'protocol.json')['candidates'];auth=read(DOC.parent/'AUTHORIZATION.json');now=time.time()
    p.update(id='T686_FIXED_CANDIDATE_DIRECT_STOCHASTIC_VALIDATION',authorization_approved=False,out=str(OUT),arm='direct_stochastic_validation',
        candidates=candidates,baseline_summary=str(priorroot/'summary.json'),baseline_protocol=str(priorroot/'protocol.json'),
        baseline_cases=str(priorroot/'cases.jsonl.gz'),prediction_results=str(last/'RESULTS.json'),
        max_physics=16*sum(j['case']['step_cap'] for j in prior['collection_jobs']),max_commands=134,
        max_bytes=320*2**20,child_bytes=320*2**20,main_seconds=3000,audit_seconds=300,
        absolute_learning_stop=min(now+3000,auth['absolute_learning_stop']),absolute_native_stop=min(now+3000,auth['absolute_native_stop']),
        absolute_all_stop=min(now+3600,auth['absolute_all_stop']),finalization_seconds=60,new_Actor=0,new_Value=0,new_optimizer=0)
    assert p['max_physics']==4969728 and 64*p['full_bytes']+64*p['compact_bytes']+16*2**20<p['max_bytes']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);T.setup(p)
    assert T.R.N.Pool.__init__.__globals__['worker'] is T.N.worker
    base=M.L.frozen(p,p['initial']['path']);basehash=digest(base.state_dict());baseline=S.baseline(p)
    sourcecases=[json.loads(x) for x in gzip.decompress(Path(p['baseline_cases']).read_bytes()).decode().splitlines()]
    sourceby={r['path']:r for r in sourcecases};sourcefiles=set();sourcep=read(p['baseline_protocol'])
    canonical={j['id']:j for j in p['collection_jobs']}
    for r in sourcecases:
        if not r['role'].startswith('iid_'):continue
        assert sha(r['path'])==r['sha256'];sourcefiles.add(Path(r['path']));a=T.R.E.N.S.arrays(r['path'])
        job=copy.deepcopy(r['request']['job']);assert job.pop('temporal_tau')==0. and job==canonical[r['job_id']]
        if r['full']:S.SourceNative.inspect(sourcep,a,r)
        else:assert all(digest(v)==r['array_hashes'][k] for k,v in a.items())
        if r.get('exact'):
            ref=sourceby[r['reference_path']];assert r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256']
    jobs,seeds=S.schedule(p);assert len(jobs)==len(seeds)==32 and len(set(map(tuple,seeds)))==32
    unused_pairs=[]
    for item in candidates:
        q=M.L.frozen(p,item['model']['path']);assert sha(item['model']['path'])==item['model']['sha256'] and q.phash()==item['model']['policy_hash']
        assert M.L.fixed_hash(q)==M.L.fixed_hash(base)
        folder=Path(item['summary']).parent;oldcases=[json.loads(x) for x in gzip.decompress((folder/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
        assert all(r['seed'] is None for r in oldcases)
        assert read(item['screen'])['lost_ids'] and read(item['summary'])['retained']['policy_hash']==p['initial']['policy_hash']
        before=digest(q.state_dict())
        for i,row in enumerate(baseline[::4]):
            a=T.R.E.N.S.arrays(row['path']);x=torch.tensor(a['states'][:64]);q.sample=True;q.rng=np.random.default_rng([68420261121,i,0]);q.reset_noise();hidden=None
            with torch.no_grad():
                for state in x:action,loc,hidden,unused=q.step(state[None],hidden)
                sequence=q.sequence(x)
            recorded=np.stack([v['locations'] for v in q.tape]);raw=np.stack([v['raw_actions'] for v in q.tape]);atom=np.stack([v['atom'] for v in q.tape])
            assert np.max(np.abs(recorded-sequence.numpy()))<1e-10
            density=M.P.L.density(sequence,torch.tensor(raw),torch.tensor(atom),p['raw_sigma']).numpy()
            assert np.max(np.abs(density-np.stack([v['logp'] for v in q.tape])))<1e-9
            first=copy.deepcopy(q.tape[0]);q.reset_noise();q.rng=np.random.default_rng([68420261121,i,0])
            with torch.no_grad():q.step(x[0][None],None)
            assert all(np.array_equal(v,q.tape[0][k]) for k,v in first.items())
        assert digest(q.state_dict())==before
        unused_pairs.append(dict(candidate=item['name'],known_deterministic_lost=read(item['screen'])['lost_ids'],new_model_seed_pairs=32))
    assert digest(base.state_dict())==basehash
    resources=S.Wait.Raw.resources();assert T.R.used(ROOT/'stage/runs')+p['max_bytes']<p['global_bytes']
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    pre=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',two_fixed_model_loaders_and_frozen_backbone=True,
        all_baseline32full32exact_reaudited_no_new_native=True,candidate_model_seed_pairs_not_consumed=unused_pairs,
        candidate_owncarry_prefix_density_and_episode_reset=True,baseline_raw_vs_candidate_clip_semantics_verified=True,
        canonical_pool_worker_binding=True,shared32_CRN_streams_not_independent64=True,
        first_native_scene_reset_checked_at_registered_run=True,new_native=0,new_optimizer=0,resources=resources)
    save(DOC/'PREFLIGHT.json',pre)
    deps={Path(f) for f in prior['inputs']}|set(HERE.glob('*.py'))|{DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC.parent/'T686_PRIMARY_REVIEW.md'}|sourcefiles
    deps|={last/f for f in ['protocol.json','FINAL_REPORT.json','SEALED.json','DRIVER_STARTED.json','audit.json','RESULTS.json']}
    deps|={priorroot/f for f in ['protocol.json','summary.json','cases.jsonl.gz','audit.json','FINAL_REPORT.json']}
    for item in candidates:deps|={Path(item[k]) for k in ['summary','screen','learning']}|{Path(item['model']['path']),Path(item['summary']).parent/'cases.jsonl.gz'}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):deps.add(Path(f).resolve())
    p['inputs']={str(f):sha(f) for f in sorted(deps)};p['preflight_seconds']=time.time()-start+preceding
    save(HERE/'protocol.json',p);assert (HERE/'protocol.json').stat().st_size<2**21
    save(DOC/'BUDGET.json',{k:p[k] for k in ['max_physics','max_commands','max_bytes','main_seconds','audit_seconds','absolute_native_stop','absolute_all_stop','preflight_seconds']})
    print(json.dumps(pre))

if __name__=='__main__':main()
