"""Register, independently preflight, and launch finite offline night relay."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,io,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t493_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT=T.OUT;DOC=ROOT/'stage/docs/overnight_20261111'
def idle():
    for f in Path('/proc').glob('[0-9]*/cmdline'):
        if int(f.parent.name)==os.getpid():continue
        try:args=f.read_bytes().split(b'\0')
        except OSError:continue
        assert not any(os.fsdecode(a).startswith(str(ROOT/'stage/experiments')+'/') for a in args[1:]),('Other Stage process',f.parent.name)
def prepare():
    idle();assert not OUT.exists()
    reserve=read(ROOT/'stage/runs/t492_cold_remote_reserve/audit.json');assert reserve['state']=='PASS'
    old=read(ROOT/'stage/runs/t491_uncentered_eight/protocol.json');result=read(ROOT/'stage/docs/review_20261111/T491_FINAL_REPORT.json')
    assert result['all_scientific_processes_stopped'] and result['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    initial=result['selected'];assert initial['wide8']['success']==39 and initial['wide8']['contact']==1
    rr=[json.loads(x) for x in (ROOT/'stage/runs/t489_left_budget_recovery/pair/cases.jsonl').read_text().splitlines()]
    state=copy.deepcopy(initial);state['distance_sum']=sum(float(r['remaining_m']) for r in rr);state['eval_rows']=rr
    base=read(ROOT/'stage/runs/t449_offline_joint_recovery/protocol.json')
    keys=['source_protocol','input_mean','input_scale','actor_seed','value_seed','raw_sigma','gamma_native','lambda_native','value_lr','target_kl','bc_lr','actor_lr','old275_jobs','old8_jobs','six_jobs','wide8_jobs']
    p={k:copy.deepcopy(old[k]) for k in keys}
    stop=datetime(2026,11,12,9,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+10800<stop
    p.update(id='T493_WIDE8_OFFLINE_MULTIMECHANISM_NIGHT',out=str(OUT),initial=initial['model'],initial_state=state,seed=20261111493,
        epochs=16,rounds_per_arm=12,modes=['raw_ppo','time_cost','awr','episode_return'],keep_weight=1.,
        anchor_jobs=[j for j in p['wide8_jobs'] if j['id'] in initial['success_ids']],original_cases={r['job_id']:r for r in rr},
        old_states=base['old_states'],old_rows=base['old_rows'],max_bytes=400*2**20,full_bytes=800*1024,compact_bytes=8192,model_bytes=1792*1024,
        root_max_physics=210000000,root_max_commands=6000,root_max_main_seconds=39000,
        max_physics=210000000,max_commands=6000,main_seconds=39000,learning_seconds=1800,audit_seconds=900,
        absolute_learning_stop=stop-3600,absolute_native_stop=stop-1500,absolute_all_stop=stop,
        preflight_commands=22,repair_steps=64,
        user_authorization='Nov11 user explicitly authorizes finite multi-line offline overnight8m, all work stopNov12 09:00; after8allpass continue allunder8 regression with multiple repair lines. Supersedes prior21:30 only for this NEW bounded run.',
        scientific_scope='Only wide8 until54allpass. Four distinct registered objectives, round-robin maximum12roundseach, no noise/architecture/seed hyperparameter grid or old-directory restart. Seeds are fixed fresh sample identifiers per registered round.',
        acceptance='Each branch retains only success-count increase with no per-side count loss/contact increase, or exact successID preservation and distance-sum improvement>.02m. All54 valid. Every lostID reported. Arms independent initialized39, no model routing.',
        deployment='Single120 causal Actor, actual64GRUcarry, native bounded command. No map/planner/waypoints/runtime teacher/router.',
        conditional='8m54allpass then329under8 (275old+54wide6), conditional two64step old-reference+8m-keep repairs ifneeded, full383 perrepair. All383safe then16newwithin-map withheld starts plus full78 independent model/readback array exact. No fresh selection/multiple gate attempts.',
        raw_ppo='Signed finite physicalG-V divided equal-episode per-sideRMS, no mean subtraction.',
        time_cost='SamePPO but subtract .002 per simulated physical second for training targets; native safety/goal criteria unchanged, original rewards retained in raw arrays.',
        awr='True actions weighted by exp(clamp((G-V)/.1,-5,log10)), weights frozen at initialV forbatch, maximum-likelihood executed-action density with exact boundary atoms. Explicit AWR-style, not onpolicy surrogate or original algorithm claim.',
        episode_return='Trajectory initialG0 minus independent otherepisode same-sideG0, normalizedsideRMS; entire trajectory clipped-surrogate score sum scaled by fixed534decisioncap, no length-varying normalization. Different reset context baseline remains own-action-independent; no unbiased-variance claim.',
        preservation='All39 initial successful full histories, equalLEFT/RIGHTtotal, eachhistory halfmean+halfupper20percent perdecision normalizedlocation error; targets refreshed to eachround initialActor on fixedhistory, not runtime teacher or guarantee of current-state coverage.')
    # Disjoint explicit withheld perturbations; not new layouts and no prior
    # navigation outcomes used to choose these points or select models.
    poses=[(.012,.017,.7),(-.012,.017,-.7),(.012,-.017,-.7),(-.012,-.017,.7),(.021,.009,1.3),(-.021,.009,-1.3),(.021,-.009,-1.3),(-.021,-.009,1.3)]
    fresh=[]
    for side,index in [('LEFT',13),('RIGHT',40)]:
        for k,(x,y,yaw) in enumerate(poses):
            j=copy.deepcopy(p['wide8_jobs'][index]);j.update(id=f't493_withheld8_{side}_{k:02d}',delta_x_m=x,delta_y_m=y,delta_yaw_deg=yaw,source_reset_id=f'T493_FRESH_{side}_{k:02d}');fresh.append(j)
    p['fresh_jobs']=fresh;p['withheld_protocol_sha256']=digest(fresh)
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(DOC/'PLAN.md'),str(DOC/'MECHANISMS_AND_LIMITS.md'),str(ROOT/'stage/runs/t492_cold_remote_reserve/audit.json'),str(ROOT/'stage/docs/overnight_20261111/release_receipt_T492.json'),
        str(ROOT/'stage/runs/t489_left_budget_recovery/pair/cases.jsonl'),str(ROOT/'stage/docs/review_20261111/T491_FINAL_REPORT.json'),p['old_rows'],p['old_states'],p['initial']['path']}
    import gzip
    oldrows=[json.loads(x) for x in gzip.decompress(Path(p['old_rows']).read_bytes()).decode().splitlines()]
    assert len(oldrows)==211
    for r in oldrows:assert sha(r['path'])==r['sha256'];files.add(r['path'])
    p['inputs']={f:sha(f) for f in sorted(files)};p['protected_files']=dict(old['protected_files']);p['protected_files'][p['initial']['path']]=p['initial']['sha256']
    assert M.used()+p['max_bytes']+2**20<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    # Whole worst-file reserve, including conditional all78 full histories.
    reserved_full=(39+48*4+78)*p['full_bytes'];reserved_models=(48+2+1)*p['model_bytes'];reserved_compact=(48*54+2*383+329)*p['compact_bytes'];meta=20*2**20
    assert reserved_full+reserved_models+reserved_compact+meta<=p['max_bytes']
    OUT.mkdir();encoded=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(encoded);(OUT/'protocol.json').write_text(encoded)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_physics=210000000,max_commands=6000,max_main_seconds=39000,all_stop=stop,learning_stop=p['absolute_learning_stop'],native_stop=p['absolute_native_stop'],own_bytes=p['max_bytes'],global_reserved=M.used()-M.used(OUT)+p['max_bytes'],full_files_reserved=39+48*4+78,full_bytes=reserved_full,model_bytes=reserved_models,compact_bytes=reserved_compact,metadata_bytes=meta,rounds48_NOT_guaranteed_time_budget_admission=True,finite_no_unbounded_retries=True))
    print('T493_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
    p=read(OUT/'protocol.json');T.verify(p,True);assert not (OUT/'preflight.json').exists();start=time.time();assert hasattr(T.B,'native_setup') and hasattr(T.T,'Session') and hasattr(T,'Pool')
    old=read(ROOT/'stage/runs/t490_balanced_eight/capture/summary.json')['result'];arrays=[E.N.S.arrays(r['path']) for r in old['samples']]
    # Real current39 behavior, not synthetic re-labeled onpolicy trajectories.
    assert all(r['loaded_model_sha256']==p['initial']['sha256'] for r in old['samples'])
    results={}
    for mode in p['modes']:
        pp=dict(p,mechanism=mode);obj=L.make(pp);shadow=L.make(pp)
        a=L.fit(obj,arrays,pp,old['diagnostics'],lambda:None,lambda *_:None,steps=2)
        b=L.fit(shadow,arrays,pp,old['diagnostics'],lambda:None,lambda *_:None,steps=2)
        assert a==b and digest(P.state(*obj))==digest(P.state(*shadow));buf=io.BytesIO();torch.save(P.state(*obj),buf);assert digest(torch.load(io.BytesIO(buf.getvalue()),weights_only=False))==digest(P.state(*obj))
        for z in [obj,shadow]:P.load(z,P.checkpoint(p['initial']['path'])['joint_state']);assert digest(P.state(*z))==p['initial']['joint_hash']
        results[mode]=a
    # Both conditional repair objectives are exercised on actual safe old data
    # and actual successful eight histories, with disposable clones only.
    for mode in ['mean','tail']:
        pp=dict(p,repair_mechanism=mode,repair_steps=1,anchor_rows=old['diagnostics']);groups=T.repair_groups(pp);obj=L.make(pp);shadow=L.make(pp)
        a=T.fit_repair(obj,groups,pp,lambda:None,lambda *_:None);b=T.fit_repair(shadow,groups,pp,lambda:None,lambda *_:None)
        assert a==b and digest(P.state(*obj))==digest(P.state(*shadow))
        for z in [obj,shadow]:P.load(z,P.checkpoint(p['initial']['path'])['joint_state']);assert digest(P.state(*z))==p['initial']['joint_hash']
    E.N.offline();folder=OUT/'preflight_cpu_native';folder.mkdir();module,native,create,w,_=T.B.native_setup(folder/'geometry',p);geometry=[]
    try:
        for j in p['fresh_jobs']:
            s=T.T.Session(j,native,create,w,lambda *_:None,lambda:(_ for _ in ()).throw(AssertionError('Zero-native preflight')))
            x=s.packet();hp,hg,ht=w.get(s.env);assert np.all(hg==s.env.goal_xy) and not s.env._collision()
            assert np.max(np.abs(E.N.K.H.rebuild(x,np.asarray(s.env._pose()[:3]),hp,s.env.goal_xy,hg)-x))<=2e-6
            geometry.append(dict(id=j['id'],input120=digest(x),initial=digest(s.initial),goal=s.env.goal_xy.tolist(),start_collision=False));s.close()
    finally:w.close()
    # Creating a Session temporarily patches task_env. Workers use independent
    # fresh spawn; cancellation checks contract without advancing any physics.
    counts=R.M.mp.get_context('spawn').RawArray('q',3);pool=T.Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
    try:
        vals=pool.request([dict(model=p['initial']['path'],job=j,seed=None,cancel_probe=True) for j in p['wide8_jobs'][:2]])
        assert all(v['state']=='CANCEL_PASS' for v in vals) and sum(counts)==0
    finally:pool.close()
    save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands=22,discarded_Adam=144,formal_cpu_Adam=36,prior_resource_change_and_failed_Adam=108,learning_objectives_live_shadow_exact=results,conditional_repair_both_clones_PASS=True,withheld_geometry16=geometry,elapsed_s=time.time()-start,not_navigation_success=True,resource_software_render=T.SOFTWARE_RENDER_WITNESS,learning_device='cpu',stage_cpus=[14,15]))
    print('T493_PREFLIGHT_PASS',flush=True)
def launch():
    idle();p=read(OUT/'protocol.json');T.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert source['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True,cwd=ROOT).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True,cwd=ROOT).strip()
    now=time.time();seconds=int(p['absolute_all_stop']-now);assert seconds>10800
    with (OUT/'driver_console.log').open('xb') as f:
        args=['systemd-run','--user','--unit=stage-t493-night-20261111','--collect',
            '-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle',
            '-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=5','-p','KillMode=control-group',
            '-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=MUJOCO_GL=egl','-p','Environment=LIBGL_ALWAYS_SOFTWARE=1','-p','Environment=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json','-p','Environment=LP_NUM_THREADS=1','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1',
            '-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver_console.log'),'-p','StandardError=append:'+str(OUT/'driver_console.log'),
            'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T493','--why=Offline CPU-limited eight metre night to09:00',
            'timeout','--signal=TERM','--kill-after=2',str(seconds),'taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')]
        launched=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);assert launched.returncode==0
        pid=int(subprocess.check_output(['systemctl','--user','show','stage-t493-night-20261111.service','--property=MainPID','--value'],text=True).strip());assert pid>0
    save(OUT/'launch.json',dict(pid=pid,unit='stage-t493-night-20261111.service',CPUQuota='100%',CPUWeight=1,CPUSchedulingPolicy='idle',stage_cpus=[14,15],GPU_disabled=True,epoch=now,all_stop=p['absolute_all_stop'],source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),offline_no_LLM_network_install=True))
    print('T493_ACTUALLY_LAUNCHED',pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
