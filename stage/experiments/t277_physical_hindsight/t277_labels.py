"""First-hit labels from float64 native XY, preserving original task outcomes."""
from pathlib import Path
import hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    p=Path(path);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(p)
def used(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def fixed_targets(source):
    n=len(source['actions']);start=max(1,n//4)
    selected=np.linspace(start,n-1,8,dtype=np.int64);assert len(np.unique(selected))==8
    return selected


def first_hit_record(xy,flags,initial_xy,goal,ends,chosen):
    xy=np.asarray(xy);flags=np.asarray(flags)
    assert xy.dtype==np.float64 and xy.ndim==2 and xy.shape[1]==2 and flags.dtype==np.uint8
    future=int(ends[chosen]);goal=np.asarray(goal,dtype=np.float64)
    base=dict(chosen_decision=int(chosen),chosen_future_step=future,goal_xy=goal.tolist())
    if flags[future-1]&1:return dict(**base,eligible=False,reason='Chosen endpoint is original CONTACT; no replacement target')
    if np.linalg.norm(np.asarray(initial_xy)-goal)<=.2:return dict(**base,eligible=False,reason='Initial pose already within new radius; no replacement')
    distance=np.linalg.norm(xy-goal,axis=1)
    candidates=np.flatnonzero((distance<=.2)&((flags&1)==0))
    assert len(candidates),'Chosen real noncontact endpoint must be reached by recorded motion'
    first=int(candidates[0])+1;assert first<=future
    assert not np.any((flags[:first]&1)!=0),'A goal after a collision is not a real continuation'
    transition=int(np.searchsorted(ends,first,side='left'));begin=0 if transition==0 else int(ends[transition-1])
    previous=float(np.linalg.norm(np.asarray(initial_xy)-goal)) if first==1 else float(distance[first-2])
    assert previous>.2 and float(distance[first-1])<=.2
    return dict(**base,eligible=True,first_goal_step=first,terminal_transition=transition,
        terminal_hold_begin_step=begin,terminal_hold_actual_steps=first-begin,
        original_hold_end_step=int(ends[transition]),inside_original_hold=bool(first<int(ends[transition])),
        first_before_selected_future=bool(first<future),distance_before_m=previous,distance_at_hit_m=float(distance[first-1]),
        bootstrap_discount=0.,terminal_nextstate_available_at_saved_decision=bool(first==int(ends[transition])),
        original_goal_success_unchanged=True,training_ready_replay=False)
