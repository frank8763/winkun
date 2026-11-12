"""Zero-native-step field geometry review, only verified Stage dependencies."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_runtime as T
G,np=T.G,T.np
OUT=ROOT/'stage/runs/t502_fullfield_probe'
def main():
    assert not OUT.exists();OUT.mkdir();folder=OUT/'registration_geometry';folder.mkdir()
    p=T.read(ROOT/'stage/runs/t501_joint_command_repair/protocol.json');T.E.N.offline()
    module,native,create,w,_=G.B.native_setup(folder,p);module.ACTIVE['circle_shift_x_m']=.6
    scenes=[]
    try:
        for side in ['LEFT','RIGHT']:
            job=next(j for j in p['six_jobs'] if j['id'].endswith(side+'_13'))
            env,obs,_=T.E.N.K.R.task_env(create,job)
            try:
                s=dict(side=side,pose=list(env._pose()[:3]),goal=env.goal_xy.tolist(),model_timestep=float(env.model.opt.timestep),named_geoms=[dict(name=env.model.geom(i).name,type=int(env.model.geom_type[i]),pos=env.model.geom_pos[i].tolist(),size=env.model.geom_size[i].tolist(),body=int(env.model.geom_bodyid[i])) for i in range(env.model.ngeom) if env.model.geom(i).name],methods=[n for n in dir(env) if any(k in n for k in ['clearance','valid','bound','free','goal']) and not n.startswith('__')],native_steps=0)
                scenes.append(s)
            finally:env.close()
    finally:w.close()
    T.save(OUT/'registration_geometry.json',dict(state='ZERO_NATIVE_GEOMETRY_REVIEW',native=0,Adam=0,scenes=scenes))
    print(json.dumps(scenes,ensure_ascii=False))
if __name__=='__main__':main()
