"""New pose tests on the existing Stage map, fixed before any model outcome."""
from pathlib import Path
import copy,gzip,importlib.util,json,math,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t711_prepare_controller',HERE/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A,D,O=C.A,C.DOC,C.OUT


def main():
    start=time.time();bound=C.read(D/'PREPARATION_BOUND.json');C.old_stopped();assert not O.exists();O.mkdir();(O/'fit').mkdir()
    A.torch.set_num_threads(1);os.sched_setaffinity(0,{15})
    import mujoco
    def forbidden(*args,**kw):raise AssertionError('T711 preparation forbids physics or learning')
    mujoco.mj_step=forbidden;A.torch.optim.Adam.step=forbidden;A.torch.optim.LBFGS.step=forbidden
    def guard():assert time.time()<bound['absolute_stop']-10;C.C.resources()
    raw=json.loads((ROOT/'stage/runs/t710_whole_head_replay/protocol.json').read_text());p=C.read(ROOT/'stage/runs/t710_whole_head_replay/protocol.json')
    jobs={j['id']:j for j in p['evaluation_jobs']};parents=[jobs[f't502_field_sector{s}_offset{o}'] for s in range(8) for o in [1,2,3]]
    proposals=[]
    for parent in parents:
        for k,angle in enumerate([22.5,112.5,202.5,292.5]):
            j=copy.deepcopy(parent);j.pop('initial_contract',None);radius=.04;dyaw=.9*(1 if k%2==0 else -1);dx=radius*math.cos(math.radians(angle));dy=radius*math.sin(math.radians(angle))
            j.update(id=f'T711_{parent["id"]}_r04_a{angle}',source_reset_id=f'T711_{parent["id"]}_r04_a{angle}',expected_initial120_sha256=None,neighborhood_parent_id=parent['id'],prospective_ring='fixed_r04_yaw09',prospective_direction_deg=angle,prospective_delta=[dx,dy,dyaw],evaluation_cluster=f'field_sector{parent["start_sector"]}')
            j['fullfield_start_pose']=[parent['fullfield_start_pose'][0]+dx,parent['fullfield_start_pose'][1]+dy,math.atan2(math.sin(parent['fullfield_start_pose'][2]+math.radians(dyaw)),math.cos(parent['fullfield_start_pose'][2]+math.radians(dyaw)))]
            proposals.append(j)
    assert len(proposals)==96
    C.save(D/'PROPOSED_INITIALS.json',dict(jobs=proposals,locked_before_policy_execution=True,same_static_map_not_independent_geometry=True,no_replacements=True))
    historical=set()
    for rel in ['t704_frozen_new_starts/new_refined108','t709_induced_replay/known_induced','t699_unexecuted_continuation/new_replay_108']:
        f=ROOT/'stage/runs'/rel/'cases.jsonl.gz'
        if f.exists():
            for line in gzip.decompress(f.read_bytes()).decode().splitlines():
                row=json.loads(line);h=row.get('array_hashes',{})
                if all(key in h for key in ['initial_native_integration','scene_sha256','task_goal_xy']):historical.add(tuple(h[key] for key in ['initial_native_integration','scene_sha256','task_goal_xy']))
    import t698_native as N
    K,F=N.C,N.F;K.ACTIVE=p;folder=O/'preflight_initials';folder.mkdir()
    module,native,create,w,_=K.C.consistent_reset.setup(F,folder,p,K.C.CANONICAL_SETUP);original=copy.deepcopy(module.ACTIVE)
    valid=[];invalid=[];duplicates=[];contracts=[];seen=set()
    try:
        for job in proposals:
            guard();module.ACTIVE=copy.deepcopy(original);module.ACTIVE['circle_shift_x_m']=job['circle_shift_x_m'];current=None
            try:
                current=K.Session(job,native,create,w,guard,forbidden,None);pose=list(current.env._pose()[:3]);packet=current.packet()
                assert not current.env._collision() and not current.done() and A.np.linalg.norm(current.goal-A.np.asarray(pose[:2]))>.2
                assert not current.hidden.any() and current.steps==0 and current.env.data.time==0
                assert A.np.max(A.np.abs(A.np.array(pose)-job['fullfield_start_pose']))<1e-10 and max(abs(v) for v in pose[:2])<=7
                nh=A.digest(current.initial);sh=A.digest(A.np.frombuffer(bytes.fromhex(current.scene),dtype=A.np.uint8));gh=A.digest(current.goal);combined=(nh,sh,gh)
                if combined in historical or combined in seen:duplicates.append(dict(id=job['id'],reason='exact_initial_scene_goal_hash_duplicate'));continue
                seen.add(combined);valid.append(job);contracts.append(dict(id=job['id'],pose=pose,packet_hash=A.digest(packet),native_hash=nh,scene=current.scene,goal_hash=gh))
            except AssertionError as e:invalid.append(dict(id=job['id'],reason=str(e),not_navigation_failure=True))
            finally:
                if current is not None:current.close()
    finally:w.close()
    assert valid and len(valid)+len(invalid)+len(duplicates)==96
    exact=[j for j in valid if j['prospective_direction_deg']==22.5]
    C.save(D/'INITIALS.json',dict(planned=96,valid=len(valid),invalid=invalid,duplicates=duplicates,contracts=contracts,exact_count=len(exact),same_map=True,parent_count=24,cluster_count=8,no_replacements=True,history_scope='T704joint224,T709known431,T699new78 actual initial hash metadata; not exhaustive history'))
    specs=[('parent','t705_constrained_replay','INITIAL'),('compact','t703_projected_local','proj8'),('B0','t701_compact_retention','replay_refined108'),('constrained','t705_constrained_replay','constrained'),('coverage','t709_induced_replay','induced'),('whole_nominal','t710_whole_head_replay','nominal'),('whole_induced','t710_whole_head_replay','induced')]
    finals={};missing=[];checks=[]
    for name,run,key in specs:
        summary=ROOT/'stage/runs'/run/'fit/summary.json';audit=ROOT/'stage/runs'/run/'fit/audit.json'
        if not summary.exists() or not audit.exists() or C.read(audit).get('state')!='PASS':missing.append(dict(name=name,reason='producer_model_or_fit_audit_missing'));continue
        model=p['initial'] if key=='INITIAL' else C.read(summary)['finals'].get(key,{}).get('model')
        if model is None:missing.append(dict(name=name,reason='producer_final_model_missing'));continue
        assert C.sha(model['path'])==model['sha256'];q=A.frozen(p,model['path']);again=A.frozen(p,model['path']);assert q.phash()==model['policy_hash'] and A.digest(q.state_dict())==A.digest(again.state_dict())
        finals[name]=dict(model=model);checks.append(dict(name=name,summary=str(summary),audit=str(audit),sha256=model['sha256'],policy_hash=model['policy_hash']))
    assert finals
    C.save(O/'fit/summary.json',dict(state='REUSED_FROZEN_NO_FIT',finals=finals,missing_models=missing,source_checks=checks));C.save(O/'fit/audit.json',dict(state='PASS',producer_fit_audits_reused_by_SHA=True,all_frozen_loaders_exact=True,new_learning=0))
    # Same actual worker/inspector/helper admission seam as the main comparison.
    import unittest.mock as mock
    witness=[]
    def check(pipe,index,pc,counts):assert C.M.N.N.P.frozen is A.frozen;witness.append(index)
    with mock.patch.object(C.M.N.N.G.T,'worker',check):C.worker(None,0,dict(p,stage_cpus=list(range(8,16))),None)
    assert witness==[0]
    sample=p['protected403_rows'][0];arr=A.Old.Existing.L.T.R.E.N.S.arrays(sample['path']);row=dict(job_id=sample['id'],request=dict(model=p['initial']['path']),seed=None,teacher=False,physics=int(arr['native_counts'].sum()),array_hashes={k:A.digest(v) for k,v in arr.items()});C.M.N.BASE_INSPECT(p,arr,row)
    try:C.M.N.BASE_INSPECT(p,arr,dict(row,array_hashes=dict(row['array_hashes'],actions='0'*64)))
    except AssertionError:pass
    else:raise AssertionError('Corrupt action payload accepted')
    test=O/'helper_preflight';test.mkdir();(D/'helper_preflight').mkdir();C.save(test/'protocol.json',dict(absolute_all_stop=time.time()+1.5));C.save(test/'status.json',dict(child_pid=None))
    command=['systemd-run','--user','--unit=stage-t711-helper-test','--collect','--wait','--pipe','--property=Type=exec','--property=AllowedCPUs=8 9 10 11 12 13 14 15','--property=CPUQuota=100%','--property=Nice=19','--property=MemoryMax=128M','--property=MemorySwapMax=0','--property=RuntimeMaxSec=10',f'--property=WorkingDirectory={ROOT}','/usr/bin/python3',str(HERE/'run.py'),'helper','preflight']
    v=subprocess.run(command,capture_output=True,text=True);assert v.returncode==0,v.stderr;assert C.read(D/'helper_preflight/AFFINITY_GUARD.json')['state']=='ENDED'
    raw.update(new_jobs=valid,new_exact_jobs=exact,new_invalid_initials=invalid,new_duplicate_initials=duplicates,new_initial_contracts=contracts,prospective_parent_ids=[j['id'] for j in parents],prospective_all96=proposals)
    C.save(O/'prepared_protocol.json',raw)
    storage=len(finals)*(len(valid)*p['fresh_bytes']+len(exact)*p['compact_bytes']+32*2**20)+32*2**20
    assert storage<3*2**30 and C.used(ROOT/'stage/runs')+3*2**30<8*2**30
    assert sum(j['case']['step_cap'] for j in valid+exact)/6000+120<1400
    C.save(D/'PREFLIGHT.json',dict(state='PASS',valid=len(valid),invalid=len(invalid),duplicates=len(duplicates),models=list(finals),missing_models=missing,storage_worst_bytes=storage,worker_binding=True,corrupt_inspector_RED=True,actual_helper_cgroup=True,new_native=0,new_learning=0,elapsed_seconds=time.time()-start,resources=C.C.resources()))
    print('T711_PREFLIGHT_PASS',len(valid),len(finals),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        C.save(D/'PREPARATION_REJECTION.json',dict(error=traceback.format_exc(),new_native=0,new_learning=0));raise
