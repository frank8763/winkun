"""Independent T422 evidence and discarded historical-optimizer reconstruction.

Importing this module performs no native simulation, model update or file write.
Audits return their findings; the sole offline driver owns persisted receipts.
"""
from pathlib import Path
import copy
import gzip
import hashlib
import io
import json
import sys
import time

import learn as LEARN
import reference as REFERENCE
import t422_common as COMMON

E=LEARN.E
np,torch=E.np,E.torch


def jsonl(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest(out):
    out=Path(out)
    rows=E.read(out/'local_manifest.json')['files']
    for row in rows:
        path=E.ROOT/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    return len(rows)


def check_native_stopped(out):
    """Native main/workers must be gone; the independent driver may remain."""
    out=Path(out);pids=set()
    for name in ['main_started.json','workers.json']:
        path=out/name
        if not path.exists():continue
        value=E.read(path)
        pids.update(value[key] for key in ['pid','parent'] if type(value.get(key)) is int)
        pids.update(value.get('workers',[]))
    for pid in pids:
        assert not Path('/proc',str(pid)).exists(),('Native PID remains or reused',pid)
    return sorted(pids)


def check_model(q,opt,record):
    """Compare both complete serialized learned and mature Adam states."""
    path=Path(record['path']);assert sha(path)==record['sha256']
    payload=torch.load(io.BytesIO(gzip.decompress(path.read_bytes())),weights_only=False,map_location='cpu')
    actual=E.L.statehash(q,opt)
    assert actual==record['hashes']==payload['hashes']
    assert {key:E.digest(value) for key,value in payload['state'].items()}==actual
    assert q.phash()==record['policy_hash']==payload['policy_hash']
    return actual


def reconstruct_pair(q,opt,fit,expected,guard):
    """Actually recompute live and shadow from the same full pre-update state.

    ``fit(event)`` must execute the registered deterministic historical fit.
    Event records include full learned/Adam hashes before and after each step.
    Both reconstructed trajectories must agree; the returned model is live.
    """
    before=E.L.state(q,opt);results=[];states=[];events=[]
    for branch in ['live','shadow']:
        E.L.load(q,opt,before)
        local=[]
        def event(where,index):
            guard();assert where in ['BEFORE','AFTER']
            local.append(dict(branch=branch,where=where,index=int(index),hashes=E.L.statehash(q,opt)))
        result=fit(event)
        assert result==expected,branch
        results.append(result);states.append(E.L.state(q,opt));events.extend(local)
    assert {key:E.digest(value) for key,value in states[0].items()}=={key:E.digest(value) for key,value in states[1].items()}
    assert [{k:v for k,v in row.items() if k!='branch'} for row in events if row['branch']=='live']==[{k:v for k,v in row.items() if k!='branch'} for row in events if row['branch']=='shadow']
    E.L.load(q,opt,states[0])
    return dict(result=results[0],events=events,before_hashes={key:E.digest(value) for key,value in before.items()},after_hashes=E.L.statehash(q,opt))


def check_episode(arrays,row,step_cap=40000):
    """Native count, original-time terminal priority, and persisted digests."""
    counts=arrays['native_counts'].astype(np.int64);flags=arrays['end_flags'].astype(bool)
    assert len(counts)>0 and np.all((counts>0)&(counts<=100))
    assert flags.shape==(len(counts),3) and flags[-1].any() and not flags[:-1].any()
    assert np.all(flags.sum(1)<=1)
    assert int(counts.sum())==row['physics_steps']==row['physics']
    assert bool(flags[-1,1])==row['success'] and bool(flags[:,0].any())==row['collision'] and bool(flags[-1,2])==row['timeout']
    assert E.digest(arrays)==row['array_digest'] and E.digest(arrays['final_native_integration'])==row['integration_sha256']
    assert E.equal(arrays['states'][1:],arrays['next_states'][:-1])
    assert arrays['states'].shape==arrays['next_states'].shape==(len(counts),120)
    elapsed=np.r_[0,np.cumsum(counts)]
    assert np.max(np.abs(arrays['states'][:,112]-(step_cap-elapsed[:-1])/step_cap))<=2e-6
    assert np.max(np.abs(arrays['next_states'][:,112]-(step_cap-elapsed[1:])/step_cap))<=2e-6
    return int(counts.sum())


def feedback_score(row,arrays):
    smooth=float(np.square(np.diff(arrays['actions'].astype(np.float64),axis=0)).sum())
    if row['success'] and not row['collision']:
        return [0,int(arrays['native_counts'].sum()),smooth]
    return [2 if row['collision'] else 1,float(row['remaining_m']),smooth]


def check_feedback(compact,ref,first,gains,mode):
    """Recompute the controller on every persisted actual recorded pose."""
    counts=compact['native_counts'].astype(np.int64)
    elapsed=np.r_[0,np.cumsum(counts)]*.002
    assert np.all(compact['feedback_progress'][:first]==-1)
    assert np.all(compact['feedback_speed'][:first]==0)
    progress=first
    for index in range(first,len(counts)):
        prior=progress
        action,progress,_=REFERENCE.feedback(ref,compact['poses'][index],
            float(compact['feedback_speed'][index]),float(elapsed[index]),progress,gains,mode)
        independent,independent_progress=feedback_reconstruction(ref,compact['poses'][index],
            float(compact['feedback_speed'][index]),float(elapsed[index]),prior,gains,mode)
        assert E.equal(action,independent) and progress==independent_progress
        assert E.equal(action,compact['actions'][index]),(mode,index,'executed feedback')
        assert progress==int(compact['feedback_progress'][index]),(mode,index,'reference progress')


def feedback_reconstruction(ref,pose,speed,elapsed,progress,gains,mode):
    """Independent reconstruction of the registered feedback equations."""
    pose=np.asarray(pose,np.float64);gains=np.asarray(gains,np.float64)
    n=len(ref['arrays']['actions'])
    def wrap(angle):return (np.asarray(angle)+np.pi)%(2*np.pi)-np.pi
    if mode=='time':
        index=int(np.clip(np.searchsorted(ref['times'],elapsed,side='right')-1,0,n))
    else:
        assert mode=='progress'
        indices=np.arange(progress,min(n,progress+12)+1)
        delta=ref['poses'][indices,:2]-pose[:2]
        costs=np.square(delta).sum(1)+.04*np.square(wrap(ref['poses'][indices,2]-pose[2]))
        index=int(indices[int(np.argmin(costs))])
    delta_goal=np.asarray(ref['goal'],np.float64)-pose[:2]
    distance=float(np.linalg.norm(delta_goal))
    terminal=bool(index==n or (mode=='time' and elapsed>=ref['times'][-1]))
    if terminal:
        target_xy=np.asarray(ref['goal'],np.float64)
        target_yaw=float(np.arctan2(delta_goal[1],delta_goal[0]));target_speed=.025
        base=np.zeros(2,np.float64)
    else:
        target_xy=ref['poses'][index,:2];target_yaw=float(ref['poses'][index,2])
        target_speed=float(ref['refspeed'][index]);base=ref['arrays']['actions'][index].astype(np.float64)
    delta=target_xy-pose[:2];cosine,sine=np.cos(pose[2]),np.sin(pose[2])
    forward=float(cosine*delta[0]+sine*delta[1]);left=float(-sine*delta[0]+cosine*delta[1])
    yaw=float(wrap(target_yaw-pose[2]));ky,kpsi,kx,kv,kyawslow,kendslow=gains
    steer=base[1]+ky*left+kpsi*yaw
    drive=base[0]+kx*forward+kv*(target_speed-speed)-kyawslow*abs(yaw)-kendslow*max(0.,1.-distance/.6)
    return np.clip(np.asarray([drive,steer],np.float64),-1.,1.).astype(np.float32),index


def audit_search(p,rootout):
    """Zero-native search audit; write only its final evidence receipt."""
    rootout=Path(rootout)
    # The public interface accepts either the root or its search child.
    out=rootout if rootout.name=='search' else rootout/'search'
    COMMON.verify(p);summary=E.read(out/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_sessions_closed'] is True
    assert not (out/'audit.json').exists()
    stopped=check_native_stopped(out);manifest_members=check_manifest(out)
    assert not list(out.rglob('partial*.npz'))
    start=time.time();E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def guard():
        assert time.time()<min(start+p.get('search_audit_seconds',p['audit_seconds'])-5,p['all_stop_absolute']-5)
    ref=REFERENCE.load_reference(p['reference_path'],p['reference_sha256'])
    source=E.N.S.arrays(p['source_mixed']);first=p['recovery_prefixes'][0]
    assert first==17 and p['search_modes']==['time','progress']
    assert p['search_population']==12 and p['search_generations']==3 and p['search_elite']==4
    assert E.equal(source['initial_native_integration'],ref['arrays']['initial_native_integration'])
    assert E.equal(source['states'][0],ref['arrays']['states'][0])
    cases=jsonl(out/'cases.jsonl');cem=E.read(out/'cem.json');snapshots=E.read(out/'snapshots.json')
    assert len(cases)==144 and len(cem)==12 and len(snapshots)==4
    compact_keys={'actions','poses','next_poses','native_counts','end_flags','rewards',
        'hold_sha256','feedback_progress','feedback_speed','final_native_integration'}
    lookup={};compacts={}
    for mode in p['search_modes']:
        for repeat in range(2):
            rows=[row for row in cases if row['mode']==mode and row['pass_index']==repeat]
            assert [row['index'] for row in rows]==list(range(36))
            for row in rows:
                guard();index=row['index'];lookup[mode,repeat,index]=row
                path=Path(row['compact_path'])
                assert path==out/'compact'/f'{mode}_{index:03}.npz'
                assert sha(path)==row['compact_sha256'] and row['first']==first
                a=E.N.S.arrays(path);assert set(a)==compact_keys
                counts=a['native_counts'].astype(np.int64);flags=a['end_flags'].astype(bool)
                n=len(counts);assert first<n<=400 and row['length']==n
                assert a['actions'].shape==(n,2) and a['poses'].shape==a['next_poses'].shape==(n,3)
                assert a['end_flags'].shape==(n,3) and a['hold_sha256'].shape==(n,32)
                assert a['feedback_progress'].shape==a['feedback_speed'].shape==(n,)
                assert np.all((counts>0)&(counts<=100)) and np.all(counts[:first]==100)
                assert flags[-1].any() and not flags[:-1].any() and np.all(flags.sum(1)<=1)
                assert int(counts.sum())==row['total_native']==row['physics_steps']
                assert int(counts[first:].sum())==row['physics']<=40000-first*100
                assert row['total_native']<=40000
                assert row['success']==bool(flags[-1,1]) and row['collision']==bool(flags[:,0].any()) and row['timeout']==bool(flags[-1,2])
                assert not row['administrative_censor'] and row['complete']
                assert E.equal(a['poses'][1:],a['next_poses'][:-1])
                assert np.isfinite(a['actions']).all() and np.abs(a['actions']).max()<=1
                assert np.isfinite(a['poses']).all() and np.isfinite(a['next_poses']).all()
                assert np.isfinite(a['feedback_speed']).all()
                remaining=float(np.linalg.norm(a['next_poses'][-1,:2]-ref['goal']))
                assert abs(remaining-row['remaining_m'])<1e-12
                if row['success']:assert remaining<=.20000001 and not row['collision']
                if row['timeout']:assert row['total_native']==40000
                for key,value in a.items():
                    assert E.digest(value)==row['array_hashes'][key],(mode,repeat,index,key)
                assert E.digest(a['final_native_integration'])==row['array_hashes']['final_native_integration']
                for key in ['actions','poses','next_poses','native_counts','end_flags','rewards','hold_sha256']:
                    assert E.equal(a[key][:first],source[key][:first]),(mode,index,key,'original prefix')
                gamma=E.N.K.R.GAMMA;discount=gamma**counts
                d0=np.linalg.norm(a['poses'][:,:2]-ref['goal'],axis=1)
                d1=np.linalg.norm(a['next_poses'][:,:2]-ref['goal'],axis=1)
                rewards=-.000002*(1-discount)/(1-gamma)+.03*d0-discount*.03*d1*(~flags.any(1))
                rewards+=gamma**(counts-1)*(flags[:,1].astype(float)-flags[:,0]-.5*flags[:,2])
                assert np.max(np.abs(rewards-a['rewards']))<2e-12
                assert feedback_score(row,a)==row['score']
                check_feedback(a,ref,first,row['gains'],mode)
                if repeat:
                    previous=lookup[mode,0,index]
                    assert {k:v for k,v in row.items() if k!='pass_index'}=={k:v for k,v in previous.items() if k!='pass_index'}
                else:compacts[mode,index]=a
    for mode in p['search_modes']:
        for repeat in range(2):
            rng=np.random.default_rng(p['search_seed']);mu=np.asarray(p['gain_mu']);sd=np.asarray(p['gain_sd'])
            entries=[item for item in cem if item['mode']==mode and item['pass_index']==repeat]
            assert [item['generation'] for item in entries]==[0,1,2]
            for generation,item in enumerate(entries):
                assert mu.tolist()==item['mu'] and sd.tolist()==item['sd']
                theta=rng.normal(size=(12,6))*sd+mu;theta[0]=mu
                theta=np.clip(theta,p['gain_low'],p['gain_high'])
                assert theta.tolist()==item['theta']
                rows=[lookup[mode,repeat,generation*12+j] for j in range(12)]
                assert [row['gains'] for row in rows]==theta.tolist()
                assert [row['score'] for row in rows]==item['scores']
                elite=sorted(range(12),key=lambda j:(rows[j]['score'],j))[:4]
                assert elite==item['elite']
                mu=theta[elite].mean(0);sd=np.maximum(theta[elite].std(0),p['gain_min_sd'])
                assert mu.tolist()==item['next_mu'] and sd.tolist()==item['next_sd']
    assert [(row['mode'],row['pass_index']) for row in snapshots]==[(mode,repeat) for mode in p['search_modes'] for repeat in range(2)]
    assert all(row['physics']==first*100 for row in snapshots)
    for key in ['hash','prefix_digest']:
        assert len({row[key] for row in snapshots})==1,key
    original=[row for row in cases if row['pass_index']==0]
    best=min(original,key=lambda row:(row['score'],p['search_modes'].index(row['mode']),row['index']))
    result=summary['result'];assert result['best']==best
    assert result['path']==str(out/'selected.npz') and sha(result['path'])==result['sha256']
    selected=E.N.S.arrays(result['path']);assert E.digest(selected)==best['array_digest']
    assert set(selected)==set(best['array_hashes'])
    for key,value in selected.items():assert E.digest(value)==best['array_hashes'][key],key
    assert all(E.equal(selected[key],value) for key,value in compacts[best['mode'],best['index']].items())
    assert E.equal(selected['task_goal_xy'],ref['goal']) and E.equal(selected['scene_sha256'],source['scene_sha256'])
    assert E.equal(selected['initial_native_integration'],source['initial_native_integration'])
    assert np.all(selected['ownership'][:first]==0) and np.all(selected['ownership'][first:]==1)
    assert not selected['stochastic'].any()
    assert np.array_equal(selected['feedback_speed'][first:],(.5*selected['states'][first:,113]).astype(np.float32))
    q=COMMON.frozen(p);fixed=E.digest(q.state_dict())
    assert q.phash()==p['student_policy_hash']
    E.audit_array(q,selected,p,first,None)
    assert E.digest(q.state_dict())==fixed
    assert result['valid_recovery']==bool(best['success'] and not best['collision'])
    assert result['Actor_updates']==summary['Actor_updates']==summary['optimizer_steps']==0
    assert result['no_six_metre_acceptance']
    commands=jsonl(out/'commands.jsonl');roles=['SETUP'];physical=[0]
    for mode in p['search_modes']:
        for repeat in range(2):
            roles.append(mode+'_PREFIX'+str(repeat));physical.append(first*100)
            for index in range(36):
                roles.append(f'{mode}_PASS{repeat}_CANDIDATE{index}');physical.append(lookup[mode,repeat,index]['physics'])
    roles.append('SELECTED_FULL_ORIGIN_REPLAY');physical.append(best['total_native'])
    assert len(commands)==summary['commands']==150<=p['search_commands']
    assert [row['id'] for row in commands]==list(range(1,151))
    assert [row['role'] for row in commands]==roles and [row['physics'] for row in commands]==physical
    assert sum(physical)==summary['physics']<=p['search_physics']
    assert summary['elapsed_s']<=p['search_seconds']
    guard();COMMON.verify(p)
    findings=dict(state='PASS',native=0,Actor_updates=0,optimizer_steps=0,elapsed_s=time.time()-start,
        native_processes_absent=stopped,manifest_members_before_audit=manifest_members,
        unique_controller_proposals=72,exact_native_search_repeats=72,
        CEM_samples_elites_scores_selection_exact=True,actual_feedback_controls_native_counts_reward_pose_chain_exact=True,
        full_selected_student_actual_history_hidden_density_and_original_clock_exact=True,
        compact_array_per_member_SHA_verified=True,
        rejected_full_state_limitation='Full rejected arrays were compared in RAM by the two native producer passes. The independent audit checks their persisted compact arrays and per-array hashes, not absent rejected full120 observations.',
        valid_recovery=result['valid_recovery'],physics=summary['physics'],commands=summary['commands'],
        teacher_feedback_has_no_neural_latent=True,no_six_metre_acceptance=True,seven_eight_entered=False)
    E.save(out/'audit.json',findings);E.N.manifest(out)
    return findings


def joint_rows(rows,p,role,q,expected):
    values=[]
    for suffix,jobs in [('old8',p['old8_jobs']),('left5',p['left5_jobs']),('right',[p['right_job']])]:
        chosen=[row for row in rows if row['role']==role+'_'+suffix]
        assert [row['job_id'] for row in chosen]==[job['id'] for job in jobs]
        values.extend(chosen)
    assert all(row['policy_hash']==q.phash() and row['seed'] is None and not row['teacher'] for row in values)
    ids=sorted(row['job_id'] for row in values if row['success'])
    actual=dict(old8=E.tally(values[:8]),left5=E.tally(values[8:13]),right1=E.tally(values[13:]),
        success_ids=ids,all14=len(ids)==14,policy_hash=q.phash())
    assert actual==expected,(role,'joint check')
    return values,set(ids)


def read_snapshot(record):
    path=Path(record['path']);assert sha(path)==record['sha256']
    payload=torch.load(io.BytesIO(gzip.decompress(path.read_bytes())),weights_only=False,map_location='cpu')
    assert {key:E.digest(value) for key,value in payload['state'].items()}==payload['hashes']==record['hashes']
    assert payload['policy_hash']==record['policy_hash']
    return payload


def audit_learning(p,rootout):
    """Reconstruct every retained/rejected BC/PPO live and shadow trajectory."""
    rootout=Path(rootout);out=rootout if rootout.name=='learning' else rootout/'learning'
    COMMON.verify(p);summary=E.read(out/'summary.json')
    assert summary['state']=='COMPLETE' and summary['all_workers_stopped'] and summary['all_sessions_closed']
    assert not (out/'audit.json').exists()
    stopped=check_native_stopped(out);manifest_members=check_manifest(out)
    assert not list(out.rglob('partial*.npz'))
    search=E.read(COMMON.OUT/'search/summary.json')
    assert search['state']=='COMPLETE' and search['result']['valid_recovery']
    assert E.read(COMMON.OUT/'search/audit.json')['state']=='PASS'
    E.N.offline();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    E.set_out(out);start=time.time()
    def guard():
        seconds=p.get('learning_audit_seconds',p['audit_seconds'])
        assert time.time()<min(start+seconds-5,p['all_stop_absolute']-5)
    rows=jsonl(out/'cases.jsonl');reps=jsonl(out/'replays.jsonl');commands=jsonl(out/'commands.jsonl')
    base=E.N.science(p);q,opt=E.learner(base,p);fixed=E.original_hash(q)
    all_jobs=[*p['old8_jobs'],*p['left5_jobs'],p['right_job'],*p['centres'],*p['main_jobs'],*base['final_old_jobs'],*p['fresh_jobs']]
    jobs={job['id']:job for job in all_jobs}
    for row in rows:
        guard();assert sha(row['path'])==row['sha256']
        a=E.N.S.arrays(row['path']);counts=a['native_counts'].astype(np.int64);flags=a['end_flags'].astype(bool)
        assert int(counts.sum())==row['physics_steps']==row['physics']
        assert flags[-1].any() and not flags[:-1].any() and np.all(flags.sum(1)<=1)
        assert row['success']==bool(flags[-1,1]) and row['collision']==bool(flags[:,0].any()) and row['timeout']==bool(flags[-1,2])
        assert row['complete'] and not row['administrative_censor']
        assert np.all((counts>0)&(counts<=100)) and np.isfinite(a['actions']).all() and np.abs(a['actions']).max()<=1
        if row['full']:check_episode(a,row,jobs[row['job_id']]['case']['step_cap'])
        else:assert set(a)=={'actions','native_counts','end_flags'}
    for row in reps:
        if row['role']=='reload78':
            matches=[old for old in rows if old['job_id']==row['job_id'] and old['role'] in ['wide54','old275','fresh16']]
        else:
            matches=[old for old in rows if old['job_id']==row['job_id'] and old['role']==row['role'].replace('_exact','',1) and old['seed']==row['seed']]
        assert len(matches)==1,(row['role'],row['job_id'],len(matches))
        prior=matches[0]
        for key in ['array_digest','integration_sha256','physics_steps','success','collision','timeout','policy_hash','seed']:
            assert row[key]==prior[key],(row['role'],key)
        assert row['physics']==row['physics_steps'] and row['exact'] and row['full_array_exact'] and not row['independent_sample']
        assert row['reference_path']==prior['path']
    result=summary['result'];expected_rows=[];expected_reps=[]
    expected_commands=[('DIRECT_SETUP',0)]+[('WORKER_SETUP',0)]*4
    def add_cases(values):
        expected_rows.extend(values);expected_commands.extend((row['role'],row['physics']) for row in values)
    def add_replays(values):
        expected_reps.extend(values);expected_commands.extend((row['role'],row['physics']) for row in values)
    initial_rows,initial_ids=joint_rows(rows,p,'initial',q,result['initial']);selected=initial_ids.copy();add_cases(initial_rows)
    assert E.read(out/'initial_arrays.json')==initial_rows
    assert q.phash()==p['initial_expected_policy_hash'] and initial_ids==set(p['initial_expected_success_ids'])
    assert len(p['initial_fixture_rows'])==len(initial_rows)
    for old,new in zip(p['initial_fixture_rows'],initial_rows):
        for key in ['job_id','policy_hash','array_digest','integration_sha256','physics_steps','success','collision','timeout']:
            assert old[key]==new[key],('Initial historical physical repeat',key)
    initial_values=[]
    for row in initial_rows:
        assert row['full'];a=E.N.S.arrays(row['path']);E.audit_array(q,a,p,0,None)
        assert not a['ownership'].any() and not a['stochastic'].any();initial_values.append((a,row))
    recorded_rounds=summary['rounds']
    if recorded_rounds:assert E.read(out/'rounds.json')==recorded_rounds
    else:assert not (out/'rounds.json').exists()
    events=[];recoveries=[];calls=0;consecutive=0;query_commands=0;query_physics=0
    ref=REFERENCE.load_reference(p['reference_path'],p['reference_sha256'])
    source=E.N.S.arrays(p['source_mixed'])
    for rnd,record in enumerate(recorded_rounds):
        guard();assert rnd<p['learning_rounds'] and record['round']==rnd
        assert E.read(out/'blocks'/str(rnd)/'result.json')==record
        assert record['initial_success_ids']==sorted(initial_ids) and record['selected_before']==sorted(selected)
        before=read_snapshot(record['before_model']);check_model(q,opt,record['before_model'])
        assert record['before_model']['path']==str(out/'blocks'/str(rnd)/'before.pt.gz')
        before_state=copy.deepcopy(before['state'])
        entry=record['query'];assert sha(entry['path'])==entry['sha256']
        a=E.N.S.arrays(entry['path']);first=p['recovery_prefixes'][rnd%len(p['recovery_prefixes'])]
        assert first==entry['first'] and 0<=first<len(a['states'])
        assert E.equal(a['task_goal_xy'],ref['goal']) and E.equal(a['initial_native_integration'],ref['arrays']['initial_native_integration'])
        assert E.equal(a['scene_sha256'],source['scene_sha256'])
        assert not a['stochastic'].any() and not a['ownership'][:first].any() and a['ownership'][first:].all()
        assert np.array_equal(a['feedback_speed'][first:],(.5*a['states'][first:,113]).astype(np.float32))
        E.audit_array(q,a,p,first,None)
        check_feedback(a,ref,first,search['result']['best']['gains'],search['result']['best']['mode'])
        if rnd==0:
            assert entry['path']==search['result']['path'] and entry['sha256']==search['result']['sha256']
            assert entry['historical_search_reuse'] and entry['new_native']==0 and entry['exact']
            assert entry['row']==search['result']['best'] and entry['source_id']=='SEARCH_VERIFIED_REUSE'
        else:
            qr=entry['row'];check_episode(a,qr)
            expected_role=f'FEEDBACK_RECOVERY_{rnd}_PREFIX{first}'
            assert qr['role']==entry['source_id']==expected_role and qr['job_id']==p['job']['id']
            assert qr['policy_hash']==q.phash() and qr['first']==first
            assert qr['gains']==search['result']['best']['gains'] and qr['mode']==search['result']['best']['mode']
            assert qr['source_model']==record['before_model']['path'] and qr['source_model_sha256']==record['before_model']['sha256']
            assert qr['student_model_sha256']==record['before_model']['sha256']
            assert E.read(out/'recoveries'/f'{rnd:02}.json')==qr
            expected_commands.append((expected_role,qr['physics']));query_commands+=1;query_physics+=qr['physics']
            if qr['success'] and not qr['collision']:
                exact=E.read(out/'recoveries'/f'{rnd:02}_exact.json')
                assert exact==dict(array_digest=qr['array_digest'],physics=qr['physics'],exact=True,independent_sample=False)
                assert entry['exact'];expected_commands.append((expected_role+'__EXACT',qr['physics']));query_commands+=1;query_physics+=qr['physics']
            else:assert not entry.get('exact',False) and not (out/'recoveries'/f'{rnd:02}_exact.json').exists()
        bc,rl=record['bc'],record['rl']
        if a['end_flags'][-1,1] and not a['end_flags'][:,0].any():
            uid=[q.phash(),first,E.digest(a)];prior=next((old for old in recoveries if old['uid']==uid),None)
            if prior is None:
                assert entry['uid']==uid;recoveries.append(copy.deepcopy(entry))
            else:assert entry['duplicate_of']==prior['path']
            assert record['recovery_sources']==recoveries and bc is not None
            entries=[dict(arrays=E.N.S.arrays(old['path']),first=old['first'],source_id=old['source_id']) for old in recoveries]
            g=LEARN.build_groups(q,initial_values,p,entries,p['query_sources'])
            assert len(g['anchors'])==len(initial_ids) and max(float(E.old_kl(q,item,p)) for item in g['anchors'])>=0
            pair=reconstruct_pair(q,opt,lambda event:LEARN.fit_bc(q,opt,g,p,guard,event,steps=p['bc_steps']),bc['fit'],guard)
            assert pair['before_hashes']==before['hashes'];calls+=p['bc_steps']
            events.extend(dict(tag='BC'+str(rnd),which=row['branch'],where=row['where'],k=row['index'],hashes=row['hashes']) for row in pair['events'])
            check_model(q,opt,bc['proposed_model'])
            values,safe=joint_rows(rows,p,f'BC{rnd}_proposal',q,bc['checks']);add_cases(values)
            keep=initial_ids<=safe and selected<=safe and bool(safe-selected)
            assert keep==bc['accepted']
            if keep:selected=safe
            else:E.L.load(q,opt,before_state)
            assert E.L.statehash(q,opt)==bc['retained_hashes']
        else:
            assert bc is None and rl is None and record['reason']=='FEEDBACK_FAILED_REAL_GOAL_NO_NEW_LABEL'
            assert E.L.statehash(q,opt)==before['hashes']
        eligible=bool(bc is not None and bc['accepted'] and bc['checks']['old8']['success']==8)
        assert (rl is not None)==eligible
        if rl is not None:
            opt=torch.optim.Adam(E.params(q),lr=p['lr'])
            rbefore=read_snapshot(rl['before_model']);check_model(q,opt,rl['before_model'])
            rstate=copy.deepcopy(rbefore['state'])
            assert rl['before_model']['path']==str(out/'blocks'/str(rnd)/'rl_before.pt.gz')
            training=[row for row in rows if row['role']==f'train{rnd}']
            assert training==rl['items'] and len(training)==4
            batch=[]
            for k,row in enumerate(training):
                seed=[p['seed'],rnd,k]
                assert row['seed']==seed and row['full'] and not row['teacher'] and row['job_id']==p['job']['id'] and row['policy_hash']==q.phash()
                arr=E.N.S.arrays(row['path']);E.audit_array(q,arr,p,0,seed)
                assert not arr['ownership'].any() and arr['stochastic'].all();batch.append(arr)
            add_cases(training)
            repeats=[row for row in reps if row['role']==f'train{rnd}_exact']
            assert [row['seed'] for row in repeats]==[[p['seed'],rnd,k] for k in range(4)];add_replays(repeats)
            pair=reconstruct_pair(q,opt,lambda event:LEARN.fit_ppo(q,opt,batch,g,p,rnd,guard,event),rl['fit'],guard)
            assert pair['before_hashes']==rbefore['hashes'];calls+=p['epochs']
            events.extend(dict(tag='RL'+str(rnd),which=row['branch'],where=row['where'],k=row['index'],hashes=row['hashes']) for row in pair['events'])
            check_model(q,opt,rl['proposed_model'])
            values,safe=joint_rows(rows,p,f'RL{rnd}_proposal',q,rl['checks']);add_cases(values)
            keep=bool(rl['fit']['numeric_accept'] and initial_ids<=safe and selected<=safe)
            assert keep==rl['accepted']
            if keep:selected=safe
            else:E.L.load(q,opt,rstate)
            assert E.L.statehash(q,opt)==rl['retained_hashes']
        assert E.original_hash(q)==fixed
        accepted=bool((bc and bc['accepted']) or (rl and rl['accepted']))
        consecutive=0 if accepted else consecutive+1
        assert record['selected_after']==sorted(selected) and record['consecutive_rejections']==consecutive
        if consecutive>=2:assert rnd==len(recorded_rounds)-1 and summary['reason']=='TWO_CONSECUTIVE_REJECTIONS'
    assert events==jsonl(out/'optimizer_calls.jsonl')
    assert summary['updates']==dict(live=calls,shadow=calls)
    check_model(q,opt,result['model'])
    values,safe=joint_rows(rows,p,'final',q,result['final_joint']);add_cases(values)
    assert initial_ids<=safe and selected<=safe
    centres=[row for row in rows if row['role']=='centres']
    assert [row['job_id'] for row in centres]==[job['id'] for job in p['centres']]
    assert E.tally(centres)==result['centres'];add_cases(centres)
    centres_exact=[row for row in reps if row['role']=='centres_exact']
    assert [row['job_id'] for row in centres_exact]==[job['id'] for job in p['centres']];add_replays(centres_exact)
    dev=[row for row in rows if row['role']=='wide54'];old=[row for row in rows if row['role']=='old275'];fresh=[row for row in rows if row['role']=='fresh16'];reloads=[row for row in reps if row['role']=='reload78']
    if result['final_joint']['all14']:
        assert [row['job_id'] for row in dev]==[job['id'] for job in p['main_jobs']]
        assert [row['job_id'] for row in old]==[job['id'] for job in base['final_old_jobs']]
        assert E.tally(dev)==result['wide54'] and E.tally(old)==result['old275'];add_cases(dev);add_cases(old)
    else:assert not dev and not old and result['wide54'] is result['old275'] is None
    full_development=bool(dev and old and all(row['success'] for row in dev+old))
    if full_development:
        assert [row['job_id'] for row in fresh]==[job['id'] for job in p['fresh_jobs']]
        assert E.tally(fresh)==result['fresh16']
        for path,expected in p['fresh_ledger'].items():assert sha(path)==expected
        add_cases(fresh)
    else:assert not fresh and result['fresh16'] is None
    fresh_pass=bool(fresh and all(row['success'] for row in fresh))
    if fresh_pass:
        assert [row['job_id'] for row in reloads]==[job['id'] for job in p['reload_jobs']]
        assert len(reloads)==78 and result['reload78']==dict(exact=78,independent_sample=False,all_arrays_native_exact=True);add_replays(reloads)
    else:assert not reloads and result['reload78'] is None
    assert result['stable_wide_six']==fresh_pass and not summary['seven_eight_entered']
    for row in centres+dev+old+fresh:assert row['policy_hash']==q.phash()
    for row in centres_exact+reloads:
        assert row['policy_hash']==q.phash() and row['loaded_model_sha256']==result['model']['sha256']
    assert rows==expected_rows and reps==expected_reps
    assert len(commands)==summary['commands']==len(expected_commands)<=p['learning_commands']
    assert [row['id'] for row in commands]==list(range(1,len(commands)+1))
    assert [(row['role'],row['physics']) for row in commands]==expected_commands
    complete_physics=sum(row['physics'] for row in rows+reps)+query_physics
    assert complete_physics==sum(row['physics'] for row in commands)==summary['physics']<=p['learning_physics']
    assert len(commands)==5+len(rows)+len(reps)+query_commands
    assert summary['elapsed_s']<=p['learning_main_seconds']
    guard();COMMON.verify(p)
    findings=dict(state='PASS',native=0,new_training=0,elapsed_s=time.time()-start,
        native_processes_absent=stopped,manifest_members_before_audit=manifest_members,
        discarded_historical_Adam=dict(live=calls,shadow=calls,total=2*calls),
        every_live_and_shadow_Actor_Adam_step_trace_and_complete_statehash_reconstructed=True,
        BC_rejection_and_RL_fresh_Adam_rejection_rollback_exact=True,
        actual_student_history_hidden_onpolicy_density_and_native_reward_exact=True,
        teacher_feedback_rebuilt_on_actual_poses_and_executed_only_targets=True,
        source_recovery_success_exact_replay_and_label_deduplication_checked=True,
        original_base_head_adapter_frozen=True,
        compact_acceptance_limitation='Acceptance simulations persist actions/counts/flags and full array digests; audit reconstructs serialized policy states, not absent acceptance full120 files.',
        physics=summary['physics'],commands=summary['commands'],old275=result['old275'],wide54=result['wide54'],fresh16=result['fresh16'],reload78=result['reload78'],
        stable_wide_six=result['stable_wide_six'],seven_eight_entered=False)
    E.save(out/'audit.json',findings);E.N.manifest(out)
    return findings


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['search','learning']
    protocol=E.read(COMMON.OUT/'protocol.json')
    findings=(audit_search if sys.argv[1]=='search' else audit_learning)(protocol,COMMON.OUT)
    print('T422_ZERO_NATIVE_AUDIT_PASS',sys.argv[1],findings['state'],flush=True)
