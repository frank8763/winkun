"""Declared posthoc evidence descriptors; no policy queries or new experiment.

Read only the existing 211-task old retention states and two human history
timestamps. No native environment, torch, gradient, fitting, or model selection.
Output lives in docs and is source-backed; T324 archived originals are untouched.
"""
from pathlib import Path
import hashlib,json,signal,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'stage/docs/distance_20261103/T324_POSTHOC_DESCRIPTORS.json'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    assert not DEST.exists();start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('300second read-only descriptors')));signal.alarm(300)
    old=ROOT/'stage/runs/t256_distance_frontier/stable_5m_current_states.npz'
    inputs={str(old):sha(old),str(Path(__file__)):sha(__file__)}
    with np.load(old,allow_pickle=False) as z:
        x=z['states'];off=z['offsets'];task=z['task']
        ranges=[dict(task=int(t),episodes=int((task==t).sum()),
            max_recorded_goal_distance_m=float(x[np.concatenate([np.arange(off[k],off[k+1]) for k in range(len(task)) if task[k]==t]),48].max()*10)) for t in np.unique(task)]
        old_info=dict(episodes=len(task),rows=len(x),current_distance_column=48,
            max_recorded_goal_distance_m=float(x[:,48].max()*10),by_task=ranges,
            limitation='This is the fixed old211 successful retention set, not every historical RL training state or every task in the world.')
    history=[]
    for i in ['01','02']:
        f=ROOT/f'stage/runs/t317_human_demonstration/attempt_{i}.npz';inputs[str(f)]=sha(f)
        with np.load(f,allow_pickle=False) as z:
            ht=z['history_times'];span=ht[:,-1]-ht[:,0];dt=np.diff(ht,axis=1)
            history.append(dict(id=i,rows=len(ht),span_seconds_at_40s=float(span[200]),
                timestamps_at_40s=ht[200].tolist(),history_span_min=float(span.min()),history_span_max=float(span.max()),
                frame_step_median=float(np.median(dt)),decision_native_steps=100,decision_seconds=.2))
    five_required=(5-.2)/(23334*.002);six_required=(6-.2)/(28000*.002)
    report=dict(state='POSTHOC_READ_ONLY_DESCRIPTORS',posthoc=True,native_commands=0,physics_steps=0,
        policy_queries=0,gradient_calls=0,optimizer_calls=0,Actor_updates=0,
        old_retention=old_info,human_history=history,
        straight_line_required_net_radial_speed=dict(five_m_per_s=five_required,six_m_per_s=six_required,
            relative_increase_percent=(six_required/five_required-1)*100,
            limitation='Arithmetic lower bound only; ignores detours/dynamics, does not establish physical feasibility or justify time/threshold changes.'),
        input_sha256=inputs,all_inputs_unchanged=all(sha(f)==h for f,h in inputs.items()),elapsed_seconds=time.monotonic()-start)
    assert report['all_inputs_unchanged']
    blob=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();assert len(blob)<128*1024
    DEST.write_bytes(blob);signal.alarm(0);print(json.dumps(report))
if __name__=='__main__':main()
