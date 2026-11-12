"""Register frozen24-task field extension; preserve original best policies."""
from pathlib import Path
import copy,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t502_runtime as R
G,L,P,E,np,read,save,sha,digest=R.G,R.L,R.P,R.E,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/review_20261112'

def clearance(env,point):
    vals=[];pt=np.asarray(point,float)
    for i in range(env.model.ngeom):
        name=env.model.geom(i).name
        if not name.startswith(('real_rect_obs_','real_circle_obs_')):continue
        c=env.model.geom_pos[i,:2];size=env.model.geom_size[i];kind=int(env.model.geom_type[i])
        if kind==6:
            v=np.abs(pt-c)-size[:2];dist=np.linalg.norm(np.maximum(v,0))+min(max(v),0.)
        elif kind==5:dist=np.linalg.norm(pt-c)-size[0]
        else:raise AssertionError(name)
        vals.append(float(dist))
    return min(vals)

def prepare():
    assert OUT.exists() and not (OUT/'protocol.json').exists();prior=ROOT/'stage/runs/t501_joint_command_repair';p=read(prior/'protocol.json');summary=read(prior/'joint/summary.json');result=summary['result']
    assert result['accepted'] and result['old275']['success']==271 and result['wide6']['success']==result['wide8']['success']==54
    assert read(prior/'joint/audit.json')['state']=='PASS';G.stopped(prior/'joint');assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists()
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T501.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    base=next(j for j in p['six_jobs'] if j['id'].endswith('LEFT_13'));radius=6.8;points=[[radius*math.cos(i*math.pi/4),radius*math.sin(i*math.pi/4)] for i in range(8)];jobs=[]
    for i,start in enumerate(points):
        for offset in [1,2,3]:
            goal=points[(i+offset)%8];delta=np.asarray(goal)-np.asarray(start);distance=float(np.linalg.norm(delta));yaw=math.atan2(delta[1],delta[0]);job=copy.deepcopy(base)
            job.update(id=f't502_field_sector{i}_offset{offset}',source_reset_id=f't502_new_sector{i}_offset{offset}',fullfield_start_pose=[*start,yaw],fullfield_goal_xy=goal,start_sector=i,sector_offset=offset,radial_distance_m=None,delta_x_m=0.,delta_y_m=0.,delta_yaw_deg=0.,expected_initial120_sha256=None)
            job.pop('radial_distance_m');job['case']['step_cap']=math.ceil(40000*distance/6);job['case']['distance_m']=distance;assert job['case']['step_cap']<=90000;jobs.append(job)
    for k in ['repair_rows','keep_rows','old275_jobs','six_jobs','wide8_jobs','fit_steps','expanded_initial_joint_hash']:p.pop(k,None)
    now=time.time();physical=2*sum(j['case']['step_cap'] for j in jobs);shape=sum(1352*math.ceil(j['case']['step_cap']/100)+4480+65536 for j in jobs)
    p.update(id='T502_FROZEN_FULLFIELD_SECTOR_TRANSFER_PROBE',out=str(OUT),initial=result['model'],field_jobs=jobs,field_limit_m=7.5,steer_seed=20261112501,
             locked_best={k:result[k] for k in ['old275','wide6','wide8','all383']},max_bytes=40*2**20,full_bytes=1200*1024,
             probe_max_physics=physical,root_max_physics=physical,root_max_commands=100,root_max_main_seconds=2400,
             preflight_commands=48,registration_zero_physical_commands=3,preflight_zero_physical_commands=45,
             absolute_learning_stop=now,absolute_native_stop=now+3600,absolute_all_stop=now+5400,
             Actor_Adam=0,Value_Adam=0,shadow_Actor_Adam=0,audit_discarded_Actor_Adam=0,preflight_discarded_Actor_Adam=0,
             scientific_scope='Frozen best joint T501 singleActor,24new startgoal tasks:8sectors onradius6.8m, clockwise offsets1/2/3 distances5.204/9.617/12.565m. Initial yaw faces finalgoal. Same broad Stage geometry/actuator/nativegoalradius. New zero-velocity episode reset, real histories regenerated, h0zero. Not all possible field states or physical reachability proof.',
             timeout_contract='Exactly original per-distance physical allowance:ceil40000*distance/6 native*.002s. Longer than8m may exceed53334; only Session cap assertion widened to90000, no action/reward/termination change.',
             field_contract='All endpoints static clearance>=.65 and actual native initial collisionfalse, positions inside±7m. Actual rollout leaving±7.5m flagged separately, no action override; nativeGOAL and stricterfieldqualifiedGOAL reported. No global path/reachability planner or waypoint.',
             acceptance='Report all24 native counts+24exact and within-field qualification; no model learning/selection or oldtaskpromotion. Four old failures deferred by explicit user, original old-safe and jointbest retained. Fresh16neverconsumed; allfieldstabilitynotproven.',
             authorization='Nov12 user explicitly retains best old results, pauses oldrepair and starts fullfield progression even though4oldjointfailuresremain. New finite coverage probe, not continuing previous repair budget.')
    p['inputs']=dict(p['inputs']);p['protected_files']=dict(p['protected_files']);p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    for f in [*HERE.glob('*.py'),OUT/'registration_geometry.json',prior/'joint/summary.json',prior/'joint/audit.json',prior/'SEALED.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T501.json']:p['inputs'][str(f)]=sha(f)
    assert G.M.used()+p['max_bytes']+2**20<2**30 and shape+10*2**20<p['max_bytes']
    save(OUT/'LOCKED_BEST.json',dict(current_joint=result['model'],known_joint={k:result[k] for k in ['old275','wide6','wide8','all383']},unresolved_old4=[r['job_id'] for r in G.rows(prior/'joint') if not r['success']],safe_five_metre_original_path=str(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'),safe_five_metre_original_sha256=sha(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'),historical_safe_old275=275,not_a_runtime_selector=True))
    p['inputs'][str(OUT/'LOCKED_BEST.json')]=sha(OUT/'LOCKED_BEST.json')
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'T502_BUDGET.json',dict(max_physics=physical,max_commands=100,max_main_seconds=2400,audit_seconds=240,all_seconds=5400,backup_sync_reserve=1500,own_bytes=p['max_bytes'],shape_bound=shape,global_reserved=G.M.used()-G.M.used(OUT)+p['max_bytes'],native_stop=p['absolute_native_stop'],all_stop=p['absolute_all_stop'],CPU=[14,15],CPUQuota='100%',GPU=False,Adam=0))
    print('T502_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
    p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();E.N.offline();folder=OUT/'preflight_geometry';folder.mkdir();module,native,create,w,_=R.field_setup(folder,p);module.ACTIVE['circle_shift_x_m']=.6;geometry={}
    try:
        for job in p['field_jobs']:
            env,obs,_=E.N.K.R.task_env(create,job)
            try:
                pose=np.asarray(env._pose()[:3]);hp,hg,ht=w.get(env);packet=E.N.K.T.L.packet(env,obs,0,job['case']['step_cap'],np.zeros(2,np.float32),0)
                assert np.max(abs(E.N.K.H.rebuild(packet,pose,hp,env.goal_xy,hg)-packet))<=2e-6
                cs,cg=clearance(env,pose[:2]),clearance(env,env.goal_xy);assert min(cs,cg)>=.65,(job['id'],cs,cg)
                geometry[job['id']]=dict(start_pose=pose.tolist(),goal_xy=env.goal_xy.tolist(),start_clearance=cs,goal_clearance=cg,initial_integration=E.N.K.R.integration(env).tolist(),scene=E.N.K.C.model_hash(env),packet_sha256=digest(packet),history_goals=hg.tolist(),history_poses=hp.tolist(),history_times=ht.tolist(),no_initial_contact=not env._collision(),native=0)
            finally:env.close()
    finally:w.close()
    # Geometry-only addition before launch, no science or budget change; keep
    # explicit registration original rather than overwriting its evidence.
    save(OUT/'registered_protocol_before_geometry.json',p);p['geometry_contract']=geometry
    blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    nativefolder=OUT/'preflight_native';nativefolder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.Pool(dict(p,phase='student',out=str(nativefolder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in [p['field_jobs'][0],p['field_jobs'][2]]]);assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,Adam=0,registration_commands=3,commands=45,geometry_cases=24,clearance_and_actual_reset_history_contract_PASS=True,route_reachability_NOT_proven=True,prior_rejected_zero_physical_cases=15,zero_physical_cancel_with_longer_than8m_cap_PASS=True,all_workers_stopped=True))
    print('T502_PREFLIGHT_PASS',flush=True)

def launch():
    p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists();assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    seconds=int(p['absolute_all_stop']-time.time());assert seconds>4900
    args=['systemd-run','--user','--unit=stage-t502-fullfield-probe','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T502','--why=Frozen full-field new-start navigation probe','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
    subprocess.run(args,check=True);print('T502_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
