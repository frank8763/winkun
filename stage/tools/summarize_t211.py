"""Postprocess completed bounded Stage T211; no simulator or optimizer calls."""
from pathlib import Path
import json,hashlib,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'stage/runs/t211_guarded_progress'
DOC=ROOT/'stage/docs/results_20261031_guarded_progress'
def safe(r):return r['complete'] and r['success'] and not r['collision']
def main():
    s=json.loads((OUT/'summary.json').read_text());assert s['all_workers_stopped']
    rows=[json.loads(t) for t in (OUT/'rollouts.jsonl').read_text().splitlines()]
    updates=[json.loads(t) for t in (OUT/'updates.jsonl').read_text().splitlines()] if (OUT/'updates.jsonl').exists() else []
    plan=json.loads((OUT/'protocol.json').read_text())
    mismatch=[p for p,h in plan['input_hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not mismatch
    report={'state':s['state'],'stop_reason':s['stop_reason'],'training_episodes':s['training_episodes'],
        'training_physics_steps':s['training_physics_steps'],'physics_steps':s['physics_steps'],'commands':s['episodes'],
        'actor_accepted_updates':s['actor_optimizer_updates'],'actor_proposals':s['actor_optimizer_proposals'],
        'value_updates':s['value_optimizer_updates'],'stable_5m_gate_passed':s['stable_5m_gate_passed'],
        'final_reload_measured':bool(s['final_saved_reload_old8']),'arms':{},'elapsed_s':s['elapsed_s'],'input_hashes_unchanged':True,
        'analysis_physics_steps':0,'analysis_optimizer_updates':0,'T159_T194_replaced':False,'all_workers_stopped':True}
    for arm in plan['arms']:
        train=[r for r in rows if r['arm']==arm and r['role']=='training'];final=[r for r in s['final_saved_reload_old8'] if r['arm']==arm]
        ar=[u for u in updates if u['arm']==arm]
        accepted=[c for u in ar for step in u['guarded_proposals'] for c in step['attempts'] if c['accept']]
        finalleft=[r for r in final if r['case']['id']==plan['primary_cases'][6]['id']]
        report['arms'][arm]={'training_complete':sum(r['complete'] for r in train),'training_safe_GOAL':sum(safe(r) for r in train),
            'training_contact':sum(r['collision'] for r in train),'training_timeout':sum(r['timeout'] for r in train),
            'final_old8_completed':len(final),'final_old8_safe_GOAL':sum(safe(r) for r in final),
            'final_old7_safe_GOAL':sum(safe(r) for r in final if r['case']['id']!=plan['primary_cases'][6]['id']),
            'final_LEFT':finalleft[0] if finalleft else None,
            'max_accepted_behavior_mean_KL':max([c['training_mean_KL'] for c in accepted],default=None),
            'max_accepted_parent_anchor_KL':max([c['parent_anchor_max_KL'] for c in accepted],default=None),
            'accepted_step_factors':[step['factor'] for u in ar for step in u['guarded_proposals'] if step['accepted']],
            'max_batch_logp_error':max([u['behavior_verification']['batch_logp_max_error'] for u in ar],default=None)}
    report['all_new_runs_MiB']=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())/1024**2
    report['own_MiB']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())/1024**2
    # Derive distance/time from stored genuine evaluation trajectories only.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6.2,3.5),layout='constrained')
        chosen=[r for r in rows if r['id']=='parent_6']+[r for r in s['final_saved_reload_old8'] if r['case']['id']==plan['primary_cases'][6]['id']]
        for r in chosen:
            with np.load(r['trace']) as z:
                t=np.cumsum(z['native_counts'].astype(float))*.002
                d=np.linalg.norm(z['next_poses'][:,:2]-z['goal_xy'],axis=1)
            label='T194 parent' if r['arm']=='parent' else r['arm']
            ax.plot(t,d,label=label);ax.scatter(t[-1],d[-1],marker='x' if r['collision'] else 'o',s=30)
        ax.axhline(.2,color='grey',ls='--',lw=1,label='original goal threshold')
        ax.set(xlabel='Simulation time (s)',ylabel='Goal distance (m)',title='Original LEFT 5 m: stored deterministic evaluations')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(OUT/'left_distance.png',dpi=150);plt.close(fig)
        report['figure']='stage/runs/t211_guarded_progress/left_distance.png'
    except ImportError:report['figure']=None
    DOC.mkdir(exist_ok=True)
    for path in (OUT/'analysis.json',DOC/'analysis.json'):path.write_text(json.dumps(report,indent=2)+'\n')
    (DOC/'summary.json').write_bytes((OUT/'summary.json').read_bytes())
    entries=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.rglob('*')) if p.is_file()]
    (DOC/'local_evidence_manifest.json').write_text(json.dumps({'remote_binary_backup_verified':False,'files':entries},indent=2)+'\n')
    assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())<1024**3
    print(json.dumps(report))
if __name__=='__main__':main()
