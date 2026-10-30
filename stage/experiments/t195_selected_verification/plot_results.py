"""Diagnostic actual decision positions; no physics, planning or policy inputs."""
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon

def main():
    geometries=json.loads((REPO/'stage/runs/t189_head_training/scene_review/scene.json').read_text())['geometry']
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,side in zip(axes,['LEFT','RIGHT']):
        for g in geometries:
            xy=np.asarray(g['position'][:2]);size=g['size'];rot=np.asarray(g['rotation'])[:2,:2]
            if g['name'].startswith('real_circle'):patch=Circle(xy,size[0],fc='#e4e5e6',ec='#999999',lw=.6)
            else:
                points=np.asarray([[-size[0],-size[1]],[size[0],-size[1]],[size[0],size[1]],[-size[0],size[1]]])@rot.T+xy
                patch=Polygon(points,fc='#e4e5e6',ec='#999999',lw=.6)
            ax.add_patch(patch)
        case='5m_2029653601_LEFT' if side=='LEFT' else '5m_2029654802_RIGHT'
        with np.load(OUT/(case+'.npz')) as z:t=z['trace']
        ax.plot(t[:,1],t[:,2],color='#bc4444' if side=='LEFT' else '#168159',lw=2,label='Selected neural policy')
        if side=='RIGHT':
            with np.load(REPO/'stage/runs/t192_sac_her/episodes/no_her_b4_w7.npz') as z:xy=z['physical_xy']
            ax.plot(xy[::50,0],xy[::50,1],color='#737373',alpha=.65,lw=1,ls='--',label='SAC exploration success')
            with np.load(REPO/'stage/runs/t193_success_consolidation'/(case+'_frozen_trace.npz')) as z:b=z['trace']
            ax.plot(b[:,1],b[:,2],color='#df9130',lw=1.2,label='Consolidation before bias: timeout')
        goal=[1.5750185651952417,-3.676543648434686] if side=='LEFT' else [5.486694873876159,.043289980823624674]
        ax.add_patch(Circle(goal,.2,color='#168159',alpha=.25));ax.scatter(*goal,c='#168159',marker='*',s=100,label='Final goal (0.20 m radius)')
        ax.scatter(t[0,1],t[0,2],c='#335588',s=35,marker='o',label='Original start')
        title='LEFT: collision; 1.477 m remaining' if side=='LEFT' else 'RIGHT: GOAL at 42.794 s; reset grid 27/27'
        ax.set(xlim=(-1.7,7.1),ylim=(-4.8,3.2),xlabel='x (m)',ylabel='y (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('Actual trajectories in the unchanged Stage scene',fontsize=13)
    fig.text(.5,.015,'Lines show recorded decision positions; terminal outcomes/distances come from native physics logs. Fixed development tasks, no generalization claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.095,1,.96));fig.savefig(OUT/'right5m_milestone.png',dpi=160);plt.close(fig)
    print('Saved diagnostic plot; physics0, optimizer0')

if __name__=='__main__':main()
