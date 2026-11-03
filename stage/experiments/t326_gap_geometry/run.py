"""Bounded zero-step geometry inspection before a separately versioned course."""
from pathlib import Path
import contextlib,importlib.util,json,os,signal,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t326_gap_geometry';DOC=ROOT/'stage/docs/distance_20261103'
spec=importlib.util.spec_from_file_location('stage326_recorder',ROOT/'stage/experiments/t317_human_demonstration/recorder.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
K,np=R.K,R.np
def prepare():
    assert not OUT.exists();p=R.read(ROOT/'stage/experiments/t325_human_ui/protocol.json')
    p.update(experiment='T326_GAP_GEOMETRY_ONLY',wall_seconds=120,final_bytes=1024**2,payload_bytes=512*1024,metadata_bytes=256*1024,max_commands=1,max_physics=0)
    files={f for f in p['input_hashes'] if Path(f).suffix=='.py'}|{str(HERE/'run.py'),str(ROOT/'stage/docs/distance_20261103/T326_PLAN.md'),p['job']['exact_snapshot']}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<p['global_bytes'];OUT.mkdir()
    K.C.save(HERE/'protocol.json',p);K.C.save(OUT/'protocol.json',p)
def main():
    assert not (OUT/'status.json').exists();p=R.read(HERE/'protocol.json');start=time.time()
    for f,h in p['input_hashes'].items():assert K.C.sha(f)==h
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120seconds geometry only')));signal.alarm(120)
    K.C.save(OUT/'status.json',dict(state='GEOMETRY_ONLY',pid=os.getpid(),main_epoch=start))
    with (OUT/'factory.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        native,create,witness,reward=R.setup(OUT/'runtime');s=R.Session(p,native,create,witness,reward,lambda:None,lambda:(_ for _ in ()).throw(AssertionError('No physical step')))
        try:
            m,e=s.env.model,s.env;geoms=[]
            before=K.R.integration(e).copy()
            for i in range(m.ngeom):
                body=int(m.geom_bodyid[i]);name=R.mujoco.mj_id2name(m,R.mujoco.mjtObj.mjOBJ_GEOM,i)
                geoms.append(dict(id=i,name=name,body=body,robot=body in e.robot_body_ids,type=int(m.geom_type[i]),size=m.geom_size[i].tolist(),local_pos=m.geom_pos[i].tolist(),world_pos=e.data.geom_xpos[i].tolist(),rotation=e.data.geom_xmat[i].reshape(3,3).tolist(),contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i])))
            v=R.View(s)
            try:(OUT/'original_preview.jpg').write_bytes(v.frame(s))
            finally:v.close()
            assert np.array_equal(before,K.R.integration(e)) and s.steps==0
            K.C.save(OUT/'geometry.json',dict(scene_sha256=s.scene,pose=list(e._pose()),goal_xy=s.goal.tolist(),geoms=geoms,original_integration_sha256=K.R.arrays_hash(before)))
        finally:s.close();witness.close()
    summary=dict(state='COMPLETE',pid=os.getpid(),main_epoch=start,elapsed_s=time.time()-start,commands=1,geometry_commands=1,physics_steps=0,Actor_updates=0,Adam_calls=0,all_workers_stopped=True,workers_spawned=0,all_renderers_closed=True,scene_modified=False)
    K.C.save(OUT/'summary.json',summary);K.C.save(OUT/'status.json',summary);signal.alarm(0);print(json.dumps(summary))
if __name__=='__main__':globals()[sys.argv[1]]()
