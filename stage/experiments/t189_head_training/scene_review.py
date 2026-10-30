"""Read-only scene/trajectory review; no native stepping or policy update."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon

def main():
    torch.set_num_threads(1)
    plan=json.loads((HERE/'protocol.json').read_text())
    folder=OUT/'scene_review';folder.mkdir(exist_ok=True)
    _,create=native_setup(folder)
    env,obs,_=create(plan['primary_cases'][-2])
    geoms=[]
    for i in range(env.model.ngeom):
        name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if name.startswith(('real_rect_obs_','real_circle_obs_')):
            geoms.append({'name':name,'type':int(env.model.geom_type[i]),'position':env.data.geom_xpos[i].tolist(),'size':env.model.geom_size[i].tolist(),'rotation':env.data.geom_xmat[i].reshape(3,3).tolist()})
    env.close();save(folder/'scene.json',{'geometry':geoms,'physics_steps':0,'optimizer_updates':0,'role':'Diagnostic drawing only; never supplied to Actor or training reward.'})
    episodes=[json.loads(s) for s in (OUT/'episodes.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,side in zip(axes,['LEFT','RIGHT']):
        for g in geoms:
            xy=np.asarray(g['position'][:2]);size=g['size'];rot=np.asarray(g['rotation'])[:2,:2]
            if g['name'].startswith('real_circle'):
                ax.add_patch(Circle(xy,size[0],fc='#d5d8dc',ec='#777777',lw=.6))
            else:
                points=np.asarray([[-size[0],-size[1]],[size[0],-size[1]],[size[0],size[1]],[-size[0],size[1]]])@rot.T+xy
                ax.add_patch(Polygon(points,fc='#d5d8dc',ec='#777777',lw=.6))
            ax.text(*xy,g['name'].replace('real_','').replace('_obs_',''),fontsize=6,clip_on=True)
        original=[r for r in episodes if r['reset_group']=='original_5m' and r['source'].endswith(side)]
        for row in original:
            with np.load(row['telemetry_path']) as z:a=z['trace']
            ax.plot(a[:,1],a[:,2],color='#df761c' if row['arm']=='trainable_head' else '#2379ad',alpha=.25,lw=.7)
        anchor=plan['anchor_sources'][side]['reset_task'];ax.scatter(*anchor['start_pose'][:2],c='#008844',marker='s',s=50,label='near-goal reset')
        goal=anchor['goal_xy'];ax.add_patch(Circle(goal,.2,color='#339966',alpha=.3));ax.scatter(*goal,c='#008844',marker='*',s=110,label='final goal')
        ax.plot([],[],color='#2379ad',label='frozen direct head');ax.plot([],[],color='#df761c',label='trainable direct head')
        ax.set(xlim=(-2,7.5),ylim=(-5,4),title=f'{side}: actual stochastic original-5m traces',xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'original5m_training_paths.png',dpi=150);plt.close(fig)
    print('Saved read-only diagnostic scene and current trajectories; physics steps0')

if __name__=='__main__':main()
