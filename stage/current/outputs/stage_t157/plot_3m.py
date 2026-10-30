from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
baseline=json.loads((ROOT.parent/'stage_t156/paired_eval/summary.json').read_text())
search=json.loads((ROOT/'search_rollouts/summary.json').read_text())
ids=['2029653601_LEFT','2029653602_RIGHT','2029654801_LEFT','2029654802_RIGHT']
titles=['Left -17.3 deg','Right +13.6 deg','Left -12.5 deg','Hard right +18.8 deg']
fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout='constrained')
for ax,case,title in zip(axes.flat,ids,titles):
    b=next(x for x in baseline['cases'] if x['id']==case and x['model']=='T156_BC')
    n=next(x for x in search['cases'] if x['id']==case and x['model']=='bias05')
    assert b['task_signature']==n['task_signature']
    for row,color,label in [(b,'#356cc3','T156: BC recovery'),(n,'#d46b20','T157: learned throttle bias')]:
        points=[json.loads(l) for l in Path(row['trace_file']).read_text().splitlines()]
        time=[0]+[x['step']*.002 for x in points]
        dist=[3]+[x['remaining_m'] for x in points]
        ax.plot(time,dist,lw=2,color=color,label=label)
        ax.scatter(time[-1],dist[-1],color=color,s=25)
    ax.axhline(.2,color='#2e8b68',ls='--',lw=1)
    ax.axvline(28,color='#8b93a1',ls=':',lw=1)
    ax.set(title=f'{title}: GOAL at {n["steps"]*.002:.3f} s',xlim=(0,28.5),ylim=(0,3.15))
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.supxlabel('Simulation time (s)')
fig.supylabel('Distance to final goal (m)')
fig.suptitle('Four fixed 3 m training tasks: neural policy reaches every goal without collision',fontsize=12)
fig.text(.5,-.015,'Training-task policy search results, not independent generalization evidence. Goal radius 0.20 m; deadline 28 s.',ha='center',fontsize=9)
fig.savefig(ROOT/'3m_training_tasks.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print(ROOT/'3m_training_tasks.png')
