"""Descriptive paired behavior diagnostics after the stopped-run audit."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from t271_runtime import *
OUT=ROOT/'stage/runs/t271_frozen_feature_exploration'
DOC=ROOT/'stage/docs/long_distance_20261101'


def main():
    assert json.loads((DOC/'T271_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    dest=DOC/'T271_BEHAVIOR.json';assert not dest.exists()
    rows=list(map(json.loads,(OUT/'exploration.jsonl').read_text().splitlines()))
    geometry={r['id']:r for r in map(json.loads,(OUT/'geometry.jsonl').read_text().splitlines())}
    records=[]
    for row in rows:
        with np.load(row['path']) as z:a={k:z[k].copy() for k in z.files}
        goal=np.asarray(geometry[row['source_reset_id']]['goal_xy']);counts=a['native_counts'].astype(np.int64)
        ds=np.linalg.norm(a['next_poses'][:,:2]-goal,axis=1)
        # Stored means are already bounded Gaussian locations, not raw Actor outputs.
        # Use the same inverse CDF at z=0 to account for native truncation/tails.
        dev=a['actions']-F.AR.sample(torch.tensor(a['means']),torch.zeros_like(torch.tensor(a['means']))).numpy()
        record=dict(id=row['id'],source_reset_id=row['source_reset_id'],mode=row['mode'],seed=row['seed'],
            safe=bool(row['success'] and not row['collision']),collision=row['collision'],timeout=row['timeout'],
            physics_s=row['physics_steps']*.002,remaining_m=row['remaining_m'],minimum_sampled_distance_m=float(ds.min()),
            sampled_chord_length_m=float(np.linalg.norm(a['next_poses'][:,:2]-a['poses'][:,:2],axis=1).sum()),
            zero_latent_command_deviation_RMS=np.sqrt((dev.astype(np.float64)**2).mean(axis=0)).tolist(),
            zero_latent_command_deviation_mean=dev.mean(axis=0).tolist(),
            actual_command_delta_RMS=np.sqrt((np.diff(a['actions'],axis=0).astype(np.float64)**2).mean(axis=0)).tolist())
        if row['mode']=='feature_copula':
            f=a['feature_vectors'];record['adjacent_feature_cosine_mean']=float((f[1:]*f[:-1]).sum(axis=1).mean())
            record['within_block_first_feature_cosine_mean']=float(np.mean([f[k]@f[(k//50)*50] for k in range(len(f))]))
        records.append(record)
    pairs=[]
    for a,b in zip(records[::2],records[1::2]):
        assert a['source_reset_id']==b['source_reset_id'] and a['seed']==b['seed']
        pairs.append(dict(source_reset_id=a['source_reset_id'],seed=a['seed'],AR_safe=a['safe'],feature_safe=b['safe'],
            AR_remaining_m=a['remaining_m'],feature_remaining_m=b['remaining_m'],
            feature_minus_AR_remaining_m=b['remaining_m']-a['remaining_m'],
            interpretation='Remaining distance at different terminal events is descriptive, not a common-time causal metric.'))
    C.save(dest,dict(state='FIXED32_EXPLORATION_DESCRIPTIVE_ANALYSIS',physics_added=0,optimizer_added=0,
        records=records,pairs=pairs,independent_training_seeds=0,deterministic_six_m_gate=False,
        nominal_sigma_does_not_prove_equal_realized_perturbation_energy=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    old={r['source_reset_id']:r for r in map(json.loads,(ROOT/'stage/runs/t268_storage_and_frozen_gate/development.jsonl').read_text().splitlines())}
    for ax,name in zip(axes.flat,geometry):
        goal=np.asarray(geometry[name]['goal_xy']);r=old[name]
        with np.load(r['path']) as z:
            ax.plot(np.cumsum(z['native_counts'].astype(np.int64))*.002,
                np.linalg.norm(z['next_poses'][:,:2]-goal,axis=1),'k--',label='Frozen deterministic',linewidth=1.8)
        seen=set()
        for r in rows:
            if r['source_reset_id']!=name:continue
            color='#2b6cb0' if r['mode']=='conditional_AR' else '#c05621'
            with np.load(r['path']) as z:
                x=np.cumsum(z['native_counts'].astype(np.int64))*.002;y=np.linalg.norm(z['next_poses'][:,:2]-goal,axis=1)
            ax.plot(x,y,color=color,alpha=.6,linewidth=1,label=r['mode'] if r['mode'] not in seen else None)
            ax.plot(x[-1],y[-1],marker='x' if r['collision'] else 'o',color=color,markersize=4)
            seen.add(r['mode'])
        ax.axhline(.2,color='#742a2a',linewidth=.8);ax.set_title(name);ax.grid(alpha=.2)
        ax.set_xlabel('Physical time (s)');ax.set_ylabel('Euclidean distance to true goal (m)')
    axes.flat[0].legend(fontsize=8);fig.suptitle('T271 frozen exploration probe: four known starts, no learning')
    fig.tight_layout();fig.savefig(DOC/'T271_DISTANCE_CURVES.svg');plt.close(fig)
    print(json.dumps(dict(state='COMPLETE',records=len(records),pairs=len(pairs))),flush=True)


if __name__=='__main__':main()
