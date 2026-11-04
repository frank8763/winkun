"""One force-limit intervention at the already-tested 10rad/s wheel cap.

Uses the immutable T341 diagnostic engine. Force telemetry is newly measured;
none of this is navigation training or a published real-car specification.
"""
from pathlib import Path
import sys,json,time,io,gzip,copy,contextlib,importlib.util
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stage343_speed_engine',ROOT/'stage/experiments/t341_wheel_speed_transfer/run.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
T,K,np,torch=B.T,B.K,B.np,B.torch
read,save,digest=B.read,B.save,B.digest
OUT=ROOT/'stage/runs/t343_drive_force_limit';DOC=ROOT/'stage/docs/review_20261104'
B.HERE,B.OUT=HERE,OUT
original_config=B.speed_config;original_module=B.module

def force_config(env,factor):
    # Both arms have wheel command max10. Only force-cap120 vs240 differs.
    before=K.R.integration(env).copy();cfg=original_config(env,2.);ids=cfg['actuator_ids']
    original=env.model.actuator_forcerange.copy();enabled=env.model.actuator_forcelimited.copy()
    assert env.model.actuator_forcelimited[ids].all() and np.array_equal(original[ids],np.array([[-120.,120.],[-120.,120.]]))
    env.model.actuator_forcerange[ids]=original[ids]*factor
    other=np.ones(env.model.nu,bool);other[ids]=False
    assert np.array_equal(env.model.actuator_forcerange[other],original[other]) and np.array_equal(env.model.actuator_forcelimited,enabled)
    assert np.array_equal(K.R.integration(env),before)
    cfg.update(factor=factor,wheel_command_multiplier=2.,force_multiplier=factor,
        original_force_range=original[ids].tolist(),force_range=env.model.actuator_forcerange[ids].tolist(),
        force_limited=enabled[ids].astype(int).tolist(),intervention='only rear actuator force range after fixing wheel max10')
    cfg['plant_sha256']=digest(dict(max_wheel=env.max_wheel,force_limits=env.model.actuator_forcerange,force_enabled=env.model.actuator_forcelimited,geometry=cfg['geometry_sha256']))
    return cfg

def force_module(p,folder):
    m,c,native,create,w,reward=original_module(p,folder);base=m.R.Session
    class ForceSession(base):
        def __init__(self,*args,**kwargs):self.force_samples=[];super().__init__(*args,**kwargs)
        def step(self,*args,**kwargs):
            super().step(*args,**kwargs)
            self.force_samples.append(self.env.data.actuator_force[[self.env.id_lr_wheel,self.env.id_rr_wheel]].copy())
        def arrays(self):
            a=super().arrays();a['actuator_force_endpoints']=np.asarray(self.force_samples,np.float64).reshape(-1,2);return a
    m.R.Session=ForceSession
    return m,c,native,create,w,reward

B.speed_config=force_config;B.module=force_module

def prepare():
    assert not OUT.exists();assert read(DOC/'T342_AUDIT.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T342.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/experiments/t341_wheel_speed_transfer/protocol.json'))
    p.update(experiment='T343_REAR_FORCE_LIMIT_INTERVENTION_AT_FIXED_WHEEL10',
        factors=[1.,2.],fixed_wheel_command_multiplier=2.,force_limits=[120.,240.],
        contract='Frozen sameT340 policy and same wide1.17m/80s, wheel max10 in both arms. Only rear actuator force-range120->240, gain/gear/friction/damping/timestep/time/turning unchanged. 2zero-physics geometry+4unique+4exactreplays=10commands176000allphysics/900s/own3.25MiB/global1GiB/free40GiB. Factor1 controls are exact historical T341factor2 reexecution, not new independent data. Direct force sampled at decision endpoints, speed every native step. Straight8s administrative censor,not navigation failure. No RL/Adam/teacher/planner/router/Phase; no old275/six54/fresh/reload7/8. This is a new simulation plant,not physical hardware calibration or original narrow acceptance. No extra torque/speed/seed grid.')
    files=set(p['input_hashes'])|{str(HERE/'run.py'),str(DOC/'T343_PLAN.md'),str(ROOT/'stage/experiments/t341_wheel_speed_transfer/run.py'),str(ROOT/'stage/experiments/t341_wheel_speed_transfer/protocol.json'),str(ROOT/'stage/runs/t341_wheel_speed_transfer/summary.json'),str(ROOT/'stage/runs/t342_balanced_consolidation/preflight.json'),str(DOC/'T342_AUDIT.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T342.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print('T343 registered0native/Adam, fixedwheel10 force120vs240')

def audit():
    p=read(HERE/'protocol.json');B.verify(p);s=read(OUT/'summary.json');assert not (DOC/'T343_AUDIT.json').exists()
    assert s['state']=='COMPLETE' and not Path('/proc',str(s['pid'])).exists() and not Path('/proc',str(read(OUT/'launch.json')['pid'])).exists()
    assert len(s['cases'])==len(s['replays'])==4 and s['commands']==10 and s['optimizer_calls']==0
    old=read(ROOT/'stage/runs/t341_wheel_speed_transfer/summary.json');q=B.actor(p);physics=0;metrics=[]
    for row,rep in zip(s['cases'],s['replays']):
        assert K.C.sha(row['path'])==row['sha256'];a=T.arrays(row['path'])
        assert {k:digest(v) for k,v in a.items()}==row['array_hashes']==rep['array_hashes']
        assert rep['all_arrays_and_integration_exact'] and row['integration_sha256']==rep['integration_sha256']
        assert int(a['native_counts'].astype(np.int64).sum())==len(a['native_speed_mps'])==row['physics_steps'];physics+=row['physics_steps']+rep['physics_steps']
        force=a['actuator_force_endpoints'];assert force.shape==(len(a['actions']),2) and np.max(np.abs(force))<=120*row['factor']+1e-8
        assert row['config']['max_wheel']==10 and row['config']['force_multiplier']==row['factor']
        if row['kind']=='navigation':
            assert row['complete'] and a['end_flags'][-1].any()
            with torch.no_grad():assert np.array_equal(q.step(torch.tensor(a['states']))[0].numpy(),a['actions'])
        else:assert np.array_equal(a['actions'],np.tile(np.array([1.,0.],np.float32),(len(a['actions']),1)))
        if row['factor']==1:
            ref=next(z for z in old['cases'] if z['kind']==row['kind'] and z['factor']==2.)
            assert row['physics_steps']==ref['physics_steps'] and row['integration_sha256']==ref['integration_sha256']
            assert all(row['array_hashes'][k]==v for k,v in ref['array_hashes'].items())
        metrics.append(dict(id=row['id'],measured_force_min=float(force.min()),measured_force_max=float(force.max()),
            force_endpoint_saturation_fraction=float(np.mean(np.abs(force)>=120*row['factor']-1e-8)),
            native_speed_peak_mps=float(a['native_speed_mps'].max()),latter_half_speed_mean=float(np.mean(a['native_speed_mps'][len(a['native_speed_mps'])//2:]))))
    assert physics==s['physics_steps']<=p['max_physics']
    save(DOC/'T343_AUDIT.json',dict(state='PASS',native_added=0,optimizer_added=0,full_raw_and_native_replays_exact=True,
        T341_matched_control_exact=True,force_and_speed_measured_not_just_commands=True,physics_steps=physics,commands=10,metrics=metrics,summary_sha256=K.C.sha(OUT/'summary.json')))
    print('T343 audit PASS0native/Adam')

if __name__=='__main__':
    {'prepare':prepare,'preflight':B.preflight,'run':B.main,'audit':audit}[sys.argv[1]]()
