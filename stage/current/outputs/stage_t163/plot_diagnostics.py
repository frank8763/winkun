from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'summary.json').read_text())
paired=json.loads((ROOT.parent/'stage_t161/frozen_eval/summary.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
colors={'T159':'#2473b5','T161':'#d9652b'}
for column,side in enumerate(['LEFT','RIGHT']):
    ax=axs[0,column];bt=axs[1,column]
    for model in ['T159','T161']:
        row=next(r for r in summary['episodes'] if r['id']==f'{model}_5m_{side}')
        with np.load(row['npz'],allow_pickle=False) as z:
            xy=np.concatenate([z['poses_diagnostic_only'][:,:2],[row['final_pose'][:2]]])
            times=np.r_[z['sample_times_s'],row['duration_s']]
            dist=np.r_[z['states'][:,48]*10,row['remaining_m']]
        color=colors[model]
        ax.plot(xy[:,0],xy[:,1],color=color,label=f"{model}: {row['end']}")
        marker='X' if row['end']=='COLLISION' else 's'
        ax.scatter(*xy[-1],marker=marker,s=60,color=color)
        bt.plot(times,dist,color=color,label=model)
        bt.scatter(times[-1],dist[-1],marker=marker,s=60,color=color)
    reference=next(r for r in paired['cases'] if r['distance_m']==5 and r['side']==side)
    ax.scatter(*reference['start_pose'][:2],s=60,marker='o',c='black',label='Start')
    ax.scatter(*reference['goal_xy'],s=150,marker='*',c='#298351',label='Final goal')
    ax.set(title=f'5 m {side}: fixed known task',xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    bt.axhline(.2,ls='--',color='gray',lw=1,label='Goal radius 0.20 m')
    bt.set(xlabel='Physical time (s)',ylabel='Remaining goal distance (m)',ylim=(0,5.2))
    bt.grid(alpha=.2);bt.legend(fontsize=8)
fig.suptitle('Stage frozen 5 m diagnostics: neither policy passed\nNative event markers; obstacles not drawn; no generalization claim',fontsize=13)
fig.savefig(ROOT/'five_meter_diagnostics.png',dpi=170)
print(ROOT/'five_meter_diagnostics.png')
