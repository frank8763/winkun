"""Independent zero-native audit of the fixed terminal-control search.

All optimization proposals are privileged training-time simulations. This audit
does not rerun simulation, optimize a policy, or reinterpret selected controls
as a frozen neural teacher's queried latent action.
"""
from pathlib import Path
import json,time,hashlib,sys
import science as S


def jsonl(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def all_processes_stopped(folder):
    pids=set()
    for name in ['main_started.json','workers.json','summary.json']:
        path=Path(folder)/name
        if not path.exists():continue
        row=S.read(path)
        pids.update(row[key] for key in ['pid','parent'] if type(row.get(key)) is int)
        pids.update(row.get('workers',[]))
    for pid in pids:assert not Path('/proc',str(pid)).exists(),('Live or reused native process',pid)
    return sorted(pids)


def action_controls(theta,nominal,blocks):
    """Independent parameterization: three absolute drives, three offsets."""
    theta=S.np.asarray(theta,S.np.float64)
    nominal=S.np.asarray(nominal)
    assert theta.shape==(6,) and nominal.shape==(100,2)
    assert list(blocks)==[0,20,50,100]
    controls=nominal.copy()
    for block,(first,last) in enumerate(zip(blocks[:-1],blocks[1:])):
        controls[first:last,0]=theta[block]
        controls[first:last,1]=S.np.clip(nominal[first:last,1]+theta[block+3],-1,1)
    return S.np.clip(controls,-1,1).astype(S.np.float32)


def prefix_exact(actual,reference,holds=300):
    """Real t0..60s prefix, excluding trajectory-final-only fields."""
    keys=['states','next_states','actions','raw_actions','locations','native_counts',
          'poses','next_poses','history_poses','history_goals','rewards','end_flags',
          'hidden_before','stochastic','innovations','atom','logp']
    keys += [key for key in ['hold_sha256','future_locations'] if key in actual and key in reference]
    for key in keys:assert S.E.equal(actual[key][:holds],reference[key][:holds]),('Changed original60s prefix',key)
    assert int(actual['native_counts'][:holds].sum())==30000
    assert S.np.all(actual['control_source'][:17]==0)
    assert S.np.all(actual['control_source'][17:holds]==1)
    assert S.np.all(actual['control_source'][holds:]==2)
    assert abs(float(actual['states'][holds,112])-.25)<2e-6
    return keys


def finite_episode_arrays(arrays,goal):
    """Recorded transitions remain causal, finite-horizon and terminal-correct."""
    n=arrays['native_counts'].astype(S.np.int64)
    flags=arrays['end_flags'].astype(bool)
    assert len(n)>0 and S.np.all((n>0)&(n<=100))
    assert flags[-1].any() and not flags[:-1].any() and S.np.all(flags.sum(1)<=1)
    assert arrays['states'].shape==arrays['next_states'].shape==(len(n),120)
    assert S.E.equal(arrays['states'][1:],arrays['next_states'][:-1])
    elapsed=S.np.r_[0,S.np.cumsum(n)]
    assert S.np.max(S.np.abs(arrays['states'][:,112]-(40000-elapsed[:-1])/40000))<2e-6
    assert S.np.max(S.np.abs(arrays['next_states'][:,112]-(40000-elapsed[1:])/40000))<2e-6
    gamma=S.E.N.K.R.GAMMA;discount=gamma**n;terminal=flags.any(1)
    d0=S.np.linalg.norm(arrays['poses'][:,:2]-goal,axis=1)
    d1=S.np.linalg.norm(arrays['next_poses'][:,:2]-goal,axis=1)
    rewards=-.000002*(1-discount)/(1-gamma)+.03*d0-discount*.03*d1*(~terminal)
    rewards+=gamma**(n-1)*(flags[:,1].astype(float)-flags[:,0]-.5*flags[:,2])
    assert S.np.max(S.np.abs(rewards-arrays['rewards']))<2e-12
    assert S.np.isfinite(arrays['actions']).all() and S.np.abs(arrays['actions']).max()<=1
    return n,flags


def native_score(row,actions):
    smooth=float(S.np.square(S.np.diff(actions.astype(S.np.float64),axis=0)).sum())
    if row['success'] and not row['collision']:return [0,float(row['terminal_native']-30000),smooth]
    if not row['collision']:return [1,float(row['remaining_m']),smooth]
    return [2,float(row['remaining_m']),smooth]


def main():
    p=S.read(S.OUT/'protocol.json');S.verify(p)
    summary=S.read(S.OUT/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_sessions_closed'] is True
    assert not (S.OUT/'audit.json').exists()
    absent=all_processes_stopped(S.OUT)
    for row in S.read(S.OUT/'local_manifest.json')['files']:
        assert file_sha(S.ROOT/row['path'])==row['sha256'],row['path']
    assert not list(S.OUT.rglob('partial*.npz'))
    start=time.time()
    S.E.N.offline();S.prohibit_learning();S.torch.set_num_threads(1);S.torch.use_deterministic_algorithms(True)
    def guard():assert time.time()<min(start+p['audit_seconds']-5,p['all_stop_absolute']-5)
    source=S.E.N.S.arrays(p['source_mixed'])
    assert file_sha(p['source_mixed'])==p['source_mixed_sha256']
    assert len(source['states'])==400 and S.np.all(source['native_counts']==100)
    nominal=source['actions'][300:400].copy();goal=source['task_goal_xy']
    assert p['blocks']==[0,20,50,100] and p['prefix_holds']==300 and p['horizon_holds']==100
    assert p['generations']==2 and p['population']==12 and p['elite']==4
    assert summary['Actor_updates']==summary['optimizer_steps']==summary['Actor_Adam_steps']==0
    assert summary['control_search_generations']==2 and summary['control_search_proposals']==24
    assert summary['native_search_passes_completed']==2
    records=jsonl(S.OUT/'branches.jsonl');cem=S.read(S.OUT/'cem.json');snapshots=S.read(S.OUT/'snapshots.json')
    assert len(records)==50 and len(cem)==4 and len(snapshots)==2
    with S.np.load(S.OUT/'predictions.npz',allow_pickle=False) as stored:
        predictions={key:stored[key].copy() for key in stored.files}
    expected_shapes=dict(controls=(25,100,2),poses=(25,100,3),next_poses=(25,100,3),
        native_counts=(25,100),flags=(25,100,3),rewards=(25,100),hold_sha256=(25,100,32),
        final_integration=(25,203),lengths=(25,))
    assert set(predictions)==set(expected_shapes)
    for key,shape in expected_shapes.items():assert predictions[key].shape==shape,key
    assert S.E.equal(predictions['controls'][0],nominal)
    for i,snapshot in enumerate(snapshots):
        assert snapshot['pass_index']==i and snapshot['steps']==30000
        assert snapshot['packet_sha256']==S.digest(source['states'][300])
    for key in ['sha256','packet_sha256','integration_sha256','prefix_payload_sha256']:
        assert snapshots[0][key]==snapshots[1][key],key
    lookup={}
    for pass_index in range(2):
        actual=[row for row in records if row['pass_index']==pass_index]
        assert [row['index'] for row in actual]==list(range(25))
        for index,row in enumerate(actual):
            guard();lookup[pass_index,index]=row
            assert row['kind']==('baseline' if index==0 else 'candidate')
            n=int(predictions['lengths'][index]);assert row['length']==n and 1<=n<=100
            counts=predictions['native_counts'][index,:n].astype(S.np.int64)
            flags=predictions['flags'][index,:n].astype(bool)
            poses=predictions['poses'][index,:n];next_poses=predictions['next_poses'][index,:n]
            assert S.np.all((counts>0)&(counts<=100)) and flags[-1].any() and not flags[:-1].any()
            assert S.np.all(flags.sum(1)<=1)
            assert row['physics']==int(counts.sum())==row['terminal_native']-30000
            assert 0<row['physics']<=10000 and row['terminal_native']<=40000
            assert row['success']==bool(flags[-1,1]) and row['collision']==bool(flags[:,0].any()) and row['timeout']==bool(flags[-1,2])
            assert abs(float(S.np.linalg.norm(next_poses[-1,:2]-goal))-row['remaining_m'])<1e-12
            if row['success']:assert not row['collision'] and row['remaining_m']<=.20000001
            assert S.E.equal(poses[0],source['poses'][300])
            if n>1:assert S.E.equal(next_poses[:-1],poses[1:])
            actions=predictions['controls'][index,:n]
            assert S.np.isfinite(actions).all() and S.np.abs(actions).max()<=1
            assert row['score']==native_score(row,actions)
            # Reconstruct every persisted compact field's full-array hash.
            # Rejected states/laser arrays are only producer-verified across
            # two actual passes; this audit does not invent them from poses.
            for key,target in [('poses','poses'),('next_poses','next_poses'),('native_counts','native_counts'),
                               ('end_flags','flags'),('rewards','rewards'),('hold_sha256','hold_sha256')]:
                reconstructed=S.np.concatenate([source[key][:300],predictions[target][index,:n]])
                assert S.digest(reconstructed)==row['array_hashes'][key],(pass_index,index,key)
            combined_actions=S.np.concatenate([source['actions'][:300],actions])
            assert S.digest(combined_actions)==row['array_hashes']['actions']
            assert S.digest(predictions['final_integration'][index])==row['integration_sha256']==row['array_hashes']['final_native_integration']
            # Interval reward is recomputed with the original native discount.
            gamma=S.E.N.K.R.GAMMA;discount=gamma**counts;terminal=flags.any(1)
            d0=S.np.linalg.norm(poses[:,:2]-goal,axis=1);d1=S.np.linalg.norm(next_poses[:,:2]-goal,axis=1)
            rewards=-.000002*(1-discount)/(1-gamma)+.03*d0-discount*.03*d1*(~terminal)
            rewards+=gamma**(counts-1)*(flags[:,1].astype(float)-flags[:,0]-.5*flags[:,2])
            assert S.np.max(S.np.abs(rewards-predictions['rewards'][index,:n]))<2e-12
            assert S.np.all(predictions['native_counts'][index,n:]==0)
            assert not predictions['flags'][index,n:].any() and not predictions['hold_sha256'][index,n:].any()
            assert S.np.isnan(predictions['poses'][index,n:]).all() and S.np.isnan(predictions['next_poses'][index,n:]).all() and S.np.isnan(predictions['rewards'][index,n:]).all()
            if pass_index:
                prior=lookup[0,index]
                assert {k:v for k,v in row.items() if k!='pass_index'}=={k:v for k,v in prior.items() if k!='pass_index'}
    baseline=lookup[0,0]
    assert baseline['length']==100 and baseline['terminal_native']==40000
    assert baseline['timeout'] and not baseline['success'] and not baseline['collision']
    for key,value in baseline['array_hashes'].items():assert S.digest(source[key])==value,('Original failed suffix differs',key)
    for pass_index in range(2):
        rng=S.np.random.default_rng(p['seed']);mu=S.np.asarray(p['mu']);sd=S.np.asarray(p['sd'])
        for generation in range(2):
            guard();item=cem[pass_index*2+generation]
            assert item['pass_index']==pass_index and item['generation']==generation
            assert item['mu']==mu.tolist() and item['sd']==sd.tolist()
            # Actual sampler consumes12x6 normals, then replaces row0 by mu.
            theta=rng.normal(size=(12,6))*sd+mu;theta[0]=mu
            theta=S.np.clip(theta,S.np.asarray(p['theta_low']),S.np.asarray(p['theta_high']))
            assert theta.tolist()==item['theta'] and S.E.equal(theta[0],mu)
            rows=[lookup[pass_index,1+generation*12+j] for j in range(12)]
            assert [r['score'] for r in rows]==item['scores']
            for j,parameter in enumerate(theta):
                expected=action_controls(parameter,nominal,p['blocks'])
                assert S.E.equal(expected,predictions['controls'][1+generation*12+j]),(pass_index,generation,j)
                assert S.E.equal(expected,S.controls(parameter,nominal,p))
            elite=sorted(range(12),key=lambda j:(rows[j]['score'],j))[:4]
            assert elite==item['elite']
            mu=theta[elite].mean(0);sd=S.np.maximum(theta[elite].std(0),S.np.asarray(p['min_sd']))
            assert mu.tolist()==item['next_mu'] and sd.tolist()==item['next_sd']
    candidates=[lookup[0,index] for index in range(25)]
    best=min(candidates,key=lambda row:(row['score'],row['index']))
    result=summary['result'];assert result['selected_branch']==best and result['selected_index']==best['index']
    selected_path=Path(result['selected_path']);assert selected_path==S.OUT/'selected_mixed.npz'
    assert file_sha(selected_path)==result['selected_sha256']
    selected=S.E.N.S.arrays(selected_path)
    assert S.digest(selected)==result['selected_array_digest']
    ordinary={key:value for key,value in selected.items() if key!='control_source'}
    assert S.digest(ordinary)==best['array_digest']
    assert set(ordinary)==set(best['array_hashes'])
    for key,value in ordinary.items():assert S.digest(value)==best['array_hashes'][key],key
    prefix_keys=prefix_exact(selected,source,300)
    counts,flags=finite_episode_arrays(selected,goal)
    n=best['length'];assert len(counts)==300+n and int(counts.sum())==best['terminal_native']
    assert S.E.equal(selected['actions'][300:],predictions['controls'][best['index'],:n])
    assert S.E.equal(selected['task_goal_xy'],goal)
    assert S.E.equal(selected['scene_sha256'],source['scene_sha256'])
    assert S.np.all(selected['ownership'][:17]==0) and S.np.all(selected['ownership'][17:]==1)
    assert S.np.all(selected['stochastic']==0)
    q=S.E.safe_initial(S.E.N.science(p),p).eval().requires_grad_(False)
    assert q.phash()==p['student_policy_hash'];fixed=S.digest(q.state_dict())
    S.E.audit_array(q,selected,p,0,None)
    assert S.digest(q.state_dict())==fixed and q.phash()==result['student_policy_hash']==p['student_policy_hash']
    assert result['frozen_student_unchanged'] and result['mixed_offpolicy'] and result['labels_actual_executed_bounded_actions_only']
    assert result['valid_recovery_label']==bool(best['success'] and not best['collision'])
    assert all(result[key] is None for key in ['old275','wide54','fresh16','reload78'])
    assert result['no_six_metre_acceptance']
    commands=jsonl(S.OUT/'commands.jsonl')
    assert len(commands)==summary['main_commands']==54
    assert [row['id'] for row in commands]==list(range(1,55))
    expected_roles=['native_setup']
    expected_physics=[0]
    for pass_index in range(2):
        expected_roles.append(f'PASS{pass_index}_PREFIX');expected_physics.append(30000)
        for index in range(25):
            row=lookup[pass_index,index];expected_roles.append(f'PASS{pass_index}_{row["kind"]}{index}');expected_physics.append(row['physics'])
    expected_roles.append('SELECTED_COMPLETE_ORIGIN_REPLAY');expected_physics.append(best['terminal_native'])
    assert [row['role'] for row in commands]==expected_roles
    assert [row['physics'] for row in commands]==expected_physics
    assert sum(expected_physics)==summary['physics']<=600000
    assert summary['preflight_commands']==2 and summary['all_commands']==56
    assert summary['elapsed_s']<=p['main_seconds'] and p['prior_day_actual']['physics']+summary['physics']<=50_000_000
    assert p['prior_day_actual']['commands']+56<=2600 and p['prior_day_actual']['main_wall_s']+summary['elapsed_s']<=20000
    guard();S.verify(p)
    S.save(S.OUT/'audit.json',dict(state='PASS',elapsed_s=time.time()-start,native=0,Actor_updates=0,optimizer_steps=0,
        all_native_processes_absent=absent,original60s_prefix_recorded_fields_exact=prefix_keys,
        native_passes=2,branch_records=50,live_control_proposals=24,exact_native_control_repeats=24,
        original_suffix_control_replayed_exact=True,independent_CEM_samples_elites_ranks_selected_controls_exact=True,
        full_selected_array_SHA_student_hidden_history_reward_and_original_clock_exact=True,
        compact_branch_arrays_reward_and_integer_native_ledger_reconstructed=True,
        rejected_full_array_evidence='Producer compared every full array across two native passes; audit verifies persisted compact reconstruction and matching per-array hashes, not nonexistent full rejected-state files.',
        controls_are_executed_offpolicy_teacher_actions_not_student_latents=True,
        physics=summary['physics'],main_commands=54,preflight_commands=2,all_commands=56,
        valid_recovery_label=result['valid_recovery_label'],no_six_metre_acceptance=True,seven_eight_entered=False))
    S.E.N.manifest(S.OUT)
    print('T421_ZERO_NATIVE_AUDIT_PASS',result['valid_recovery_label'],flush=True)


if __name__=='__main__':main()

