"""Read-only terminal-label/data-coverage audit, no fitted model or simulation."""
from pathlib import Path
import sys,collections,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t309_sequence_goal_value'))
from sequence_value import *

def main():
    dest=DOC/'T309_TERMINAL_LABEL_AUDIT.json';assert not dest.exists();rows=read(OUT/'cases.json');items=[r for r in rows if r['split']=='train' and r['arm']=='center'];assert len(items)==32
    groups={};allmass=[]
    for r in items:
        a=B.arrays(r['path']);assert C.sha(r['path'])==r['sha256'];masses=[]
        for k in range(len(a['states'])-2):
            law=future_law(a['native_counts'],a['end_flags'],k,28000,True)
            masses.append(sum(e['mass'] for e in law if e['type'] in ['coordinate_absorbed_goal','collision','task_timeout']))
        assert masses;value=float(np.mean(masses));allmass.append(value)
        key=f"{r['side']}_{r['distance_band']}";g=groups.setdefault(key,dict(n=0,safe=0,contact=0,timeout=0,terminal_category_mass=[]))
        g['n']+=1;g['safe']+=int(r['success'] and not r['collision']);g['contact']+=int(r['collision']);g['timeout']+=int(r['timeout']);g['terminal_category_mass'].append(value)
    for g in groups.values():
        assert g['n']==4;g['mean_terminal_category_mass']=float(np.mean(g['terminal_category_mass']));g['empirical_outcome_entropy_nats']=-sum(v/g['n']*math.log(v/g['n']) for v in [g['safe'],g['contact'],g['timeout']] if v)
    C.save(dest,dict(state='READ_ONLY_DATA_AUDIT',native_steps=0,optimizer_calls=0,Actor_updates=0,train_centers=32,per_task=groups,
        objective_episode_anchor_equal_terminal_category_mass=float(np.mean(allmass)),scientific_control_gate_not_evaluated=True,
        scope='Mass includes legitimate safe-goal absorption coordinate and collision/timeout tokens. A large terminal-label share is NOT a code bug. Factual goal can predict shared outcomes without distinguishing actions; finite-data coverage/objective weighting is only a hypothesis, not a causal root-cause proof. Four innovation draws per task are not four independent training seeds. No target or frozen protocol changed.'))
    print(json.dumps(dict(mean_terminal_mass=float(np.mean(allmass)),groups=groups)))

if __name__=='__main__':main()
