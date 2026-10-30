from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT=Path(__file__).resolve().parent
short=json.loads((ROOT/'early_paired/summary.json').read_text())
long=json.loads((ROOT/'horizon_diagnostic/summary.json').read_text())
colors={'T146':'#2563eb','T148_0733':'#dc641f'}
names={'T146':'T146','T148_0733':'T148 early snapshot'}
fig,axes=plt.subplots(1,2,figsize=(11.5,5.6),layout='constrained')
for ax,(case,title) in zip(axes,[('2029653601_LEFT','Canonical left turn'),('2029654802_RIGHT','Canonical right turn')]):
    rows=[r for r in long['cases'] if r['id']==case]
    goal=np.array(rows[0]['goal_xy'])
    for r in rows:
        points=[json.loads(l) for l in Path(r['trace_file']).read_text().splitlines()]
        xy=np.array([r['start_pose'][:2]]+[p['pose'][:2] for p in points])
        prior=next(x for x in short['cases'] if x['id']==case and x['model']==r['model'])
        prior_points=[json.loads(l) for l in Path(prior['trace_file']).read_text().splitlines()]
        p28=np.array(prior_points[-1]['pose'][:2])
        ax.plot(xy[:,0],xy[:,1],color=colors[r['model']],lw=2,label=f"{names[r['model']]}: {r['simulation_duration_s']:.2f}s")
        ax.scatter(*p28,color=colors[r['model']],marker='s',s=45,zorder=5)
        ax.scatter(*xy[-1],color=colors[r['model']],marker='^',s=55,zorder=5)
    ax.add_patch(Circle(goal,.2,fc='#daf0e4',ec='#23845f',lw=1.4))
    ax.scatter(*goal,marker='+',c='#23845f',s=100,label='Goal (radius 0.20 m)')
    ax.scatter(*rows[0]['start_pose'][:2],marker='o',c='#273142',s=45,label='Start')
    ax.set(title=title,xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
    ax.grid(alpha=.2)
    ax.margins(.15)
    ax.legend(fontsize=8,loc='upper right')
fig.suptitle('Frozen policies: late arrivals in the same two canonical tasks',fontsize=14)
fig.supxlabel('Squares: original 28 s deadline. Triangles: eventual native GOAL. Obstacles omitted from this plot.\nExploratory two-task diagnostic; no cross-map or statistical superiority claim.',fontsize=10)
fig.savefig(ROOT/'diagnostic_trajectories.png',dpi=180)
plt.close(fig)
print(ROOT/'diagnostic_trajectories.png')
