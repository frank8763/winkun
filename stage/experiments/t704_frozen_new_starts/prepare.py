"""Policy-independent new resets and frozen-loader checks; zero integration."""
from pathlib import Path
import copy,gzip,importlib.util,json,math,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('t704_prepare_controller',HERE/'run.py')
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
R,D,A=K.OUT,K.DOC,K.A
read,save,sha=K.read,K.save,K.sha

def key(native,scene,goal):return A.digest(dict(native=native,scene=scene,goal=goal))

def main():
    start=time.time();bound=read(D/'PREPARATION_BOUND.json');K.old_stopped();assert not R.exists();R.mkdir();(R/'fit').mkdir()
    A.torch.set_num_threads(1);A.torch.use_deterministic_algorithms(True);K.M.N.N.G._real_affinity(0,{15})
    import mujoco
    def forbidden(*a,**kw):raise AssertionError('T704 preparation forbids physical integration or learning')
    mujoco.mj_step=forbidden;A.torch.optim.Adam.step=forbidden;A.torch.optim.LBFGS.step=forbidden
    def guard():
        assert time.time()<bound['absolute_stop']-10;K.C.resources()
    raw=json.loads((ROOT/'stage/runs/t703_projected_local/protocol.json').read_text());p=read(ROOT/'stage/runs/t703_projected_local/protocol.json')
    jobs={j['id']:j for j in p['evaluation_jobs']}
    parents=[jobs[f't502_field_sector{s}_offset{o}'] for s in range(8) for o in [1,2,3]]+[jobs[k] for k in ['dev_task6_05','dev_task6_11','dev_task6_15','hold_task6_03']]
    assert len(parents)==28
    proposals=[]
    for parent in parents:
        for ring,radius,yaw in [('near',.02,.5),('outer',.06,1.5)]:
            for d,angle in enumerate([45,135,225,315]):
                j=copy.deepcopy(parent);j.pop('initial_contract',None)
                identifier=f'T704_{parent["id"]}_{ring}_{angle}';dx=radius*math.cos(math.radians(angle));dy=radius*math.sin(math.radians(angle));dyaw=yaw*(1 if d%2==0 else -1)
                j.update(id=identifier,source_reset_id=identifier,expected_initial120_sha256=None,neighborhood_parent_id=parent['id'],
                    prospective_ring=ring,prospective_direction_deg=angle,prospective_delta=[dx,dy,dyaw],
                    evaluation_cluster=f'field_sector{parent["start_sector"]}' if 'fullfield_start_pose' in parent else 'shared_5m_LEFT_source_snapshot')
                if 'fullfield_start_pose' in j:
                    j['fullfield_start_pose']=[j['fullfield_start_pose'][0]+dx,j['fullfield_start_pose'][1]+dy,j['fullfield_start_pose'][2]+math.radians(dyaw)]
                    if not -math.pi<j['fullfield_start_pose'][2]<=math.pi:
                        j['requested_unwrapped_yaw']=j['fullfield_start_pose'][2]
                        j['fullfield_start_pose'][2]=math.atan2(math.sin(j['fullfield_start_pose'][2]),math.cos(j['fullfield_start_pose'][2]))
                else:
                    j['delta_x_m']+=dx;j['delta_y_m']+=dy;j['delta_yaw_deg']+=dyaw
                proposals.append(j)
    assert len(proposals)==224 and len({j['id'] for j in proposals})==224
    # Lock proposal identities before loading any candidate or executing actions.
    save(D/'PROPOSED_INITIALS.json',dict(jobs=proposals,coordinate_contract='relative actual parent;XY vector lengths .02/.06m,4diagonals,yaw alternating .5/1.5deg',locked_before_model_execution=True))
    history={};coverage=[]
    def remember(k,source):history.setdefault(k,[]).append(source)
    # Complete historical431 payload metadata still carries initial state hashes,
    # even where the stored payload itself is compact.
    for rel in ['t694_full_history_compatibility/evaluation','t696_existing_success_compatibility/evaluation','t698_cross_representation_success/evaluation','t703_projected_local/known_proj8']:
        f=ROOT/'stage/runs'/rel/'cases.jsonl.gz'
        if not f.exists():continue
        count=0
        for line in gzip.decompress(f.read_bytes()).decode().splitlines():
            row=json.loads(line);h=row.get('array_hashes',{})
            if all(k in h for k in ['initial_native_integration','scene_sha256','task_goal_xy']):
                remember(key(h['initial_native_integration'],h['scene_sha256'],h['task_goal_xy']),rel+':'+row['job_id']);count+=1
        coverage.append(dict(path=str(f),sha256=sha(f),rows=count))
    for row in p['protected403_rows']+p['novel_rows']:
        guard();arr=A.Old.Existing.L.T.R.E.N.S.arrays(row['path'])
        remember(key(A.digest(arr['initial_native_integration']),A.digest(arr['scene_sha256']),A.digest(arr['task_goal_xy'])),'403oldplus7source:'+row['id'])
    for c in p['new_initial_contracts']:
        scenehash=A.digest(A.np.frombuffer(bytes.fromhex(c['scene']),dtype=A.np.uint8))
        # Old78 packet/native matching is additionally checked without needing
        # to infer a goal from an ID.
        history.setdefault('native:'+c['native_hash'],[]).append('old78:'+c['id'])
    import t698_native as N
    C,F=N.C,N.F;C.ACTIVE=p;folder=R/'preflight_initials';folder.mkdir()
    module,native,create,w,_=C.C.consistent_reset.setup(F,folder,p,C.C.CANONICAL_SETUP);original=copy.deepcopy(module.ACTIVE)
    valid=[];invalid=[];duplicates=[];contracts=[];seen={}
    try:
        for job in proposals:
            guard();module.ACTIVE=copy.deepcopy(original)
            if 'circle_shift_x_m' in job:module.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m']
            current=None
            try:
                current=C.Session(job,native,create,w,guard,forbidden,None)
                pose=list(current.env._pose()[:3]);packet=current.packet()
                assert not current.env._collision() and not current.done(),'initialcontact_or_done'
                assert A.np.linalg.norm(current.goal-A.np.asarray(pose[:2]))>.2,'initialgoal'
                assert not current.hidden.any() and current.steps==0 and current.env.data.time==0
                if 'fullfield_start_pose' in job:
                    assert max(abs(v) for v in pose[:2])<=7,'initialboundary'
                    assert A.np.max(A.np.abs(A.np.array(pose)-job['fullfield_start_pose']))<1e-10
                nativehash=A.digest(current.initial);scenehash=A.digest(A.np.frombuffer(bytes.fromhex(current.scene),dtype=A.np.uint8));goalhash=A.digest(current.goal)
                combined=key(nativehash,scenehash,goalhash)
                matches=history.get(combined,[])+history.get('native:'+nativehash,[])+seen.get(combined,[])
                if matches:
                    duplicates.append(dict(id=job['id'],matches=matches));continue
                seen[combined]=[job['id']];valid.append(job)
                contracts.append(dict(id=job['id'],pose=pose,packet_hash=A.digest(packet),native_hash=nativehash,scene=current.scene,goal_hash=goalhash,combined_initial_key=combined))
            except AssertionError as error:invalid.append(dict(id=job['id'],reason=str(error),traceback=traceback.format_exc(),not_navigation_failure=True))
            finally:
                if current is not None:current.close()
    finally:w.close()
    assert len(valid)+len(invalid)+len(duplicates)==224
    # No replacements. Fixed valid subset and invalid/duplicate reasons precede
    # all four model executions. One prechosen first-near repeat per parent only.
    exact=[j for j in valid if j['prospective_ring']=='near' and j['prospective_direction_deg']==45]
    assert valid,'All proposed starts invalid or duplicate; reject without policy results'
    initials=dict(planned=224,valid=len(valid),invalid=invalid,duplicates=duplicates,contracts=contracts,
        history_metadata=coverage,history_scope='403old+7source,old431metadata,T69978; not exhaustive all original history',
        parent_count=28,exact_count=len(exact),no_replacement=True,not_independent_tasks=True)
    save(D/'INITIALS.json',initials)
    finals=dict(parent=dict(model=p['initial']),T696=dict(model=p['existing_A']),
        T703=dict(model=read(ROOT/'stage/runs/t703_projected_local/fit/summary.json')['finals']['proj8']['model']),
        refined108=dict(model=read(ROOT/'stage/runs/t701_compact_retention/fit/summary.json')['finals']['replay_refined108']['model']))
    saved_sample=A.Old.Existing.L.T.R.E.N.S.arrays(p['protected403_rows'][0]['path']);x=A.torch.tensor(saved_sample['states'][:8],dtype=A.torch.float64)
    checks={}
    with A.torch.no_grad():
        for name,entry in finals.items():
            guard();m=entry['model'];assert sha(m['path'])==m['sha256'];q=A.frozen(p,m['path']);again=A.frozen(p,m['path'])
            assert q.phash()==m['policy_hash'] and A.digest(q.state_dict())==A.digest(again.state_dict())
            q.sample=again.sample=False;q.rng=again.rng=None;q.reset_noise();again.reset_noise();h=h2=None
            for row in x:
                c,mu,h,_=q.step(row[None],h);c2,mu2,h2,_=again.step(row[None],h2)
                assert A.Old.Existing.same_float32(c.numpy(),c2.numpy()) and A.torch.equal(h,h2)
            checks[name]=dict(sha256=m['sha256'],policy_hash=q.phash(),reload_and_owncarry_batch1_exact=True,
                tensor_bytes=sum(v.numel()*v.element_size() for v in q.state_dict().values()))
    save(R/'fit/summary.json',dict(state='REUSED_FROZEN_NO_FIT',finals=finals))
    save(R/'fit/audit.json',dict(state='PASS',checkpoint_SHA_and_policy_hash_and_reload_exact=checks,new_Actor=0,new_optimizer=0,
        actual_training_audits_reused_not_retrained=True))
    import unittest.mock as mock
    witness=[]
    def check(pipe,index,pc,counts):assert K.M.N.N.P.frozen is A.frozen;witness.append(index)
    with mock.patch.object(K.M.N.N.G.T,'worker',check):K.worker(None,0,dict(p,stage_cpus=list(range(8,16))),None)
    assert witness==[0]
    # Complete full-payload inspector and one deliberate hash corruption.
    row=dict(job_id=p['protected403_rows'][0]['id'],request=dict(model=p['initial']['path']),seed=None,teacher=False,
        physics=int(saved_sample['native_counts'].sum()),array_hashes={k:A.digest(v) for k,v in saved_sample.items()})
    K.M.N.BASE_INSPECT(p,saved_sample,row)
    bad=dict(row,array_hashes=dict(row['array_hashes'],actions='0'*64))
    try:K.M.N.BASE_INSPECT(p,saved_sample,bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupt action hash not rejected')
    test=R/'helper_preflight';test.mkdir();(D/'helper_preflight').mkdir()
    save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));save(test/'status.json',dict(child_pid=None))
    cmd=['systemd-run','--user','--unit=stage-t704-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15',
        '--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',
        f'--property=WorkingDirectory={ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
    assert read(D/'helper_preflight/AFFINITY_GUARD.json')['state']=='ENDED'
    raw.update(new_jobs=valid,new_exact_jobs=exact,new_invalid_initials=invalid,new_duplicate_initials=duplicates,new_initial_contracts=contracts,
        prospective_parent_ids=[j['id'] for j in parents],prospective_all224=proposals)
    save(R/'prepared_protocol.json',raw)
    storage=4*(len(valid)*p['fresh_bytes']+len(exact)*p['compact_bytes']+32*2**20)+32*2**20
    assert storage<4*2**30 and K.used(ROOT/'stage/runs')+4*2**30<8*2**30
    save(D/'PREFLIGHT.json',dict(state='PASS',planned224=True,valid=len(valid),invalid=len(invalid),duplicates=len(duplicates),
        frozen_loader_checks=checks,canonical_worker_loader_binding=True,full_inspector_corruption_red=True,actual_pure_helper_cgroup=True,
        storage_worst_bytes=storage,new_native=0,new_Actor=0,new_optimizer=0,elapsed_seconds=time.time()-start,resource=K.C.resources()))
    print('T704_PREFLIGHT_PASS',len(valid),len(invalid),len(duplicates),storage,flush=True)

if __name__=='__main__':
    start=time.time()
    try:main()
    except BaseException:
        save(D/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),elapsed_seconds=time.time()-start,new_native=0,new_learning=0));raise
