"""Independent zero-native, zero-optimizer frozen Stage acceptance audit.

Passing this evidence audit never repairs the separate failed T422 historical
optimizer reconstruction. No simulator or policy optimizer runs in this file.
"""
from pathlib import Path
import hashlib,json,gzip,io,time,sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonl(path):
    p=Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line] if p.exists() else []

def check_manifest(root,out):
    rows=json.loads((Path(out)/'local_manifest.json').read_text())['files']
    for row in rows:
        p=Path(root)/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    return len(rows)

def stopped(out):
    ids=set()
    for name in ['main_started.json','workers.json','summary.json']:
        p=Path(out)/name
        if not p.exists():continue
        row=json.loads(p.read_text())
        ids.update(row[k] for k in ['pid','parent'] if type(row.get(k)) is int)
        ids.update(row.get('workers',[]))
    assert ids
    for pid in ids:assert not Path('/proc',str(pid)).exists(),('Native PID live or reused',pid)
    return sorted(ids)

def snapshot(E,path,expected_sha,policy_hash):
    assert sha(path)==expected_sha
    data=E.torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),weights_only=False,map_location='cpu')
    assert data['policy_hash']==policy_hash
    assert {key:E.digest(value) for key,value in data['state'].items()}==data['hashes']
    return data

def episode(E,arrays,row,step_cap,full):
    """Recompute persisted terminal/count contract and original-clock rewards."""
    np=E.np;n=arrays['native_counts'].astype(np.int64);flags=arrays['end_flags'].astype(bool)
    assert n.ndim==1 and len(n)>0 and np.all((n>0)&(n<=100))
    assert flags.shape==(len(n),3) and flags[-1].any() and not flags[:-1].any()
    assert np.all(flags.sum(1)<=1)
    assert arrays['actions'].shape==(len(n),2) and np.isfinite(arrays['actions']).all() and np.abs(arrays['actions']).max()<=1
    assert int(n.sum())==row['physics']==row['physics_steps']<=step_cap
    assert row['success']==bool(flags[-1,1]) and row['collision']==bool(flags[:,0].any()) and row['timeout']==bool(flags[-1,2])
    assert row['complete'] and not row['administrative_censor']
    if row['timeout']:assert int(n.sum())==step_cap
    if full:
        assert E.digest(arrays)==row['array_digest']
        assert E.digest(arrays['final_native_integration'])==row['integration_sha256']
        assert arrays['states'].shape==arrays['next_states'].shape==(len(n),120)
        assert E.equal(arrays['states'][1:],arrays['next_states'][:-1])
        assert E.equal(arrays['poses'][1:],arrays['next_poses'][:-1])
        assert not arrays['ownership'].any() and not arrays['stochastic'].any()
        elapsed=np.r_[0,np.cumsum(n)]
        assert np.abs(arrays['states'][:,112]-(step_cap-elapsed[:-1])/step_cap).max()<=2e-6
        assert np.abs(arrays['next_states'][:,112]-(step_cap-elapsed[1:])/step_cap).max()<=2e-6
        goal=arrays['task_goal_xy'];distance=np.linalg.norm(arrays['next_poses'][-1,:2]-goal)
        assert abs(float(distance)-row['remaining_m'])<1e-12
        if row['success']:assert distance<=.20000001
        gamma=E.N.K.R.GAMMA;discount=gamma**n;terminal=flags.any(1)
        d0=np.linalg.norm(arrays['poses'][:,:2]-goal,axis=1)
        d1=np.linalg.norm(arrays['next_poses'][:,:2]-goal,axis=1)
        reward=-.000002*(1-discount)/(1-gamma)+.03*d0-discount*.03*d1*(~terminal)
        reward+=gamma**(n-1)*(flags[:,1].astype(float)-flags[:,0]-.5*flags[:,2])
        assert np.abs(reward-arrays['rewards']).max()<2e-12
    else:assert set(arrays)=={'actions','native_counts','end_flags'}
    return int(n.sum())

def main():
    import t424_eval as F
    E=F.E;np=F.np;torch=F.torch
    p=F.read(F.OUT/'protocol.json');F.verify(p)
    summary=F.read(F.OUT/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    assert not (F.OUT/'audit.json').exists()
    absent=stopped(F.OUT);members=check_manifest(F.ROOT,F.OUT)
    assert not list(F.OUT.rglob('partial*.npz'))
    assert not list(F.OUT.rglob('optimizer_calls*'))
    start=time.time();E.N.offline();F.ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<start+p['audit_seconds']-5
    def no_native(*args,**kwargs):raise AssertionError('T424 evidence audit cannot execute native simulation')
    E.L.Session=no_native;E.Pool=no_native;F.Pool=no_native
    q=F.frozen(p);fixed=E.digest(q.state_dict())
    stored=snapshot(E,p['student_start_model'],p['student_start_sha256'],p['expected_policy_hash'])
    assert E.digest(q.learned())==stored['hashes']['learned']
    assert summary['model_path']==p['student_start_model'] and summary['model_sha256']==p['student_start_sha256']
    assert summary['policy_hash']==q.phash()==p['expected_policy_hash']
    assert summary['Actor_updates']==summary['optimizer_steps']==0
    assert summary['preflight_zero_commands']==8
    assert not summary['seven_eight_entered'] and summary['narrow_paused']
    pref=F.read(F.OUT/'preflight.json')
    assert pref['state']=='PASS'
    # Native preflight schema is owned by the registered runner; count only
    # the exact registered zero-step command total recorded by the producer.
    rows=jsonl(F.OUT/'cases.jsonl');replays=jsonl(F.OUT/'replays.jsonl');commands=jsonl(F.OUT/'commands.jsonl')
    result=summary['result']
    old_jobs=p['old275_jobs'];wide_jobs=p['main_jobs'];fresh_jobs=p['fresh_jobs'];reload_jobs=p['reload_jobs']
    pre_jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job']]
    assert len(pre_jobs)==14 and len(wide_jobs)==54 and len(old_jobs)==275 and len(fresh_jobs)==16 and len(reload_jobs)==78
    full_ids={job['id'] for job in reload_jobs};lookup={job['id']:job for job in [*pre_jobs,*wide_jobs,*old_jobs,*fresh_jobs]}
    expected=[];references={};full_count=0;full_hashes={}
    role_map=[('precheck14',pre_jobs),('wide54',wide_jobs),('old275',old_jobs)]
    for role,jobs in role_map:
        chosen=[row for row in rows if row['role']==role]
        assert [row['job_id'] for row in chosen]==[job['id'] for job in jobs],role
        assert E.tally(chosen)==result[role],role
        expected.extend(chosen)
    pre=[row for row in rows if row['role']=='precheck14']
    assert len(p['precheck_history_rows'])==14 and all(row['success'] for row in pre)
    for row,prior in zip(pre,p['precheck_history_rows']):
        for key in ['job_id','policy_hash','array_digest','integration_sha256','physics_steps','success','collision','timeout']:
            assert row[key]==prior[key],('Original accepted BC2 history not exact',row['job_id'],key)
    development=[row for row in rows if row['role'] in ['wide54','old275']]
    development_pass=all(row['success'] and not row['collision'] for row in development)
    fresh=[row for row in rows if row['role']=='fresh16']
    if development_pass:
        for path,hashvalue in p['fresh_ledger'].items():assert sha(path)==hashvalue,path
        assert [row['job_id'] for row in fresh]==[job['id'] for job in fresh_jobs]
        assert E.tally(fresh)==result['fresh16'];expected.extend(fresh)
    else:assert not fresh and result['fresh16'] is None
    assert rows==expected
    for row in rows:
        guard();assert sha(row['path'])==row['sha256']
        a=E.N.S.arrays(row['path']);job=lookup[row['job_id']]
        full=row['role']=='precheck14' or row['job_id'] in full_ids
        assert row['full']==full
        assert row['loaded_model_sha256']==p['student_start_sha256'] and row['policy_hash']==q.phash()
        assert row['seed'] is None and row['teacher'] is False
        assert row['pid'] in absent and type(row['cpu']) is int and 0<=row['cpu']<4
        episode(E,a,row,job['case']['step_cap'],full)
        assert set(a)<=set(row['array_hashes'])
        for key,value in a.items():assert E.digest(value)==row['array_hashes'][key],(row['role'],row['job_id'],key)
        if full:
            full_count+=1;E.audit_array(q,a,p,0,None)
            assert E.digest(q.state_dict())==fixed
            full_hashes[row['path']]={key:E.digest(value) for key,value in a.items()}
            if row['role']!='precheck14':references[row['job_id']]=row
        else:assert row['role']=='old275'
    fresh_pass=bool(development_pass and fresh and all(row['success'] and not row['collision'] for row in fresh))
    if fresh_pass:
        assert [row['job_id'] for row in replays]==[job['id'] for job in reload_jobs]
        assert result['reload78']==dict(exact=78,independent_native_execution=True,independent_sample=False,all_arrays_exact=True)
        for row in replays:
            guard();assert row['role']=='reload78'
            prior=references[row['job_id']]
            assert row['reference_path']==prior['path'] and row['reference_path'] in full_hashes
            for key in ['array_digest','array_hashes','integration_sha256','physics_steps','success','collision','timeout','policy_hash']:
                assert row[key]==prior[key],('Independent native reload',row['job_id'],key)
            assert row['array_hashes']==full_hashes[row['reference_path']]
            assert row['exact'] and row['all_arrays_exact'] and row['independent_native_execution'] and not row['independent_sample']
            assert row['loaded_model_sha256']==p['student_start_sha256'] and row['teacher'] is False and row['seed'] is None
            assert row['complete'] and not row['administrative_censor'] and row['success'] and not row['collision'] and not row['timeout']
            assert row['physics']==row['physics_steps'] and row['pid'] in absent
    else:assert not replays and result['reload78'] is None
    assert result['navigation_gate_pass']==bool(fresh_pass and len(replays)==78)
    assert result['old_training_audit']=='FAIL_RECONSTRUCTION_CAUSE_UNRESOLVED'
    assert result['research_promotion'] is False
    expected_roles=['worker_setup']*4+[row['role'] for row in [*rows,*replays]]
    expected_physics=[0]*4+[row['physics'] for row in [*rows,*replays]]
    assert len(commands)==summary['main_commands']<=p['max_main_commands']
    assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
    assert [row['role'] for row in commands]==expected_roles
    assert [row['physics'] for row in commands]==expected_physics
    assert sum(expected_physics)==summary['physics']<=p['max_physics']
    assert summary['all_commands']==summary['main_commands']+8
    assert summary['elapsed_s']<=p['main_seconds']
    assert len(rows)==343+(16 if development_pass else 0)
    assert full_count==76+(16 if development_pass else 0)
    assert E.digest(q.state_dict())==fixed and sha(p['student_start_model'])==p['student_start_sha256']
    guard();F.verify(p)
    findings=dict(state='PASS_FROZEN_ACCEPTANCE_EVIDENCE_ONLY',native=0,optimizer_steps=0,Actor_updates=0,
        elapsed_s=time.time()-start,native_processes_absent=absent,manifest_members_before_audit=members,
        frozen_model_sha256=p['student_start_sha256'],frozen_policy_hash=q.phash(),frozen_full_state_unchanged=True,
        model_serialized_learned_and_Adam_snapshot_hashes_consistent=True,no_historical_Adam_updates_reconstructed=True,
        precheck14_historical_case_summaries_exact=True,precheck14_is_charged_physical_repeat_not_new_coverage=True,
        full_episodes=full_count,full_history_hidden_deterministic_Actor_actions_native_rewards_original_clock_exact=True,
        persisted_compact_actions_counts_terminal_SHA_and_all_command_physics_ledger_exact=True,
        independent_model_native_reloads=len(replays),native_reload_all_array_digests_match_full_saved_references=True,
        compact_evidence_limitation='Non-reference old275 episodes persist actual actions/counts/flags plus all original array hashes; this zero-native audit does not reconstruct absent full120 observations.',
        old275=result['old275'],wide54=result['wide54'],fresh16=result['fresh16'],reload78=result['reload78'],
        navigation_gate_pass=result['navigation_gate_pass'],research_promotion=False,
        original_T422_training_audit='FAIL_RECONSTRUCTION_CAUSE_UNRESOLVED',
        original_T422_training_audit_not_repaired_by_functional_frozen_acceptance=True,
        physics=summary['physics'],main_commands=summary['main_commands'],preflight_zero_commands=8,all_commands=summary['all_commands'],
        seven_eight_entered=False,narrow_paused=True,missing_is_null_not_failure_rate=True)
    F.save(F.OUT/'audit.json',findings);E.N.manifest(F.OUT)
    print('T424_ZERO_NATIVE_FROZEN_AUDIT_PASS',result['navigation_gate_pass'],flush=True)

if __name__=='__main__':main()
