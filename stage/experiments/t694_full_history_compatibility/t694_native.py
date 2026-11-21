"""Original-start canonical Actor execution, plus finite source identity audit."""
import os
import t694_actor as A
Old=A.L.T.N
ORIGINAL_INSPECT=Old.inspect
C=Old.Base
G,F,P,np=Old.G,Old.F,Old.P,Old.np


def worker(pipe,index,p,counts):
    C.ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=C.Session
    G.B.native_setup=lambda folder,pc:C.C.consistent_reset.setup(F,folder,pc,C.C.CANONICAL_SETUP)
    P.frozen=A.frozen
    G.T.worker(pipe,index,p,counts)


PHYSICAL_KEYS=['states','next_states','actions','native_counts','poses','next_poses','end_flags','rewards',
               'history_poses','history_goals','history_times','task_goal_xy','initial_native_integration',
               'final_native_integration','scene_sha256','hidden_before','last_hidden_after']


def reference(p,key):
    old=[r for r in p['protected_rows'] if r['task_id']==key]
    new=[r for r in p['novel_rows'] if r['id']==key]
    assert len(old)+len(new)<=1
    if old:return old[0]['source_path'],old[0]['sha256']
    if new:return new[0]['path'],new[0]['sha256']
    return None


def inspect(p,a,row):
    ORIGINAL_INSPECT(p,a,row)
    ref=reference(p,row['job_id'])
    if ref:
        path,expected=ref;assert A.sha(path)==expected
        source=A.L.T.R.E.N.S.arrays(path)
        for key in PHYSICAL_KEYS:
            assert np.array_equal(a[key],source[key]),(row['job_id'],key,'physical source mismatch')
        assert A.same_float32(a['actions'],source['actions'])
        # Own actor proposals/densities differ across policies; their hold
        # digests are not cross-source physical-integral witnesses.


def audit_source_hashes(p,row):
    ref=reference(p,row['job_id'])
    if not ref:return
    path,expected=ref;assert A.sha(path)==expected
    source=A.L.T.R.E.N.S.arrays(path)
    for key in PHYSICAL_KEYS:
        assert row['array_hashes'][key]==A.digest(source[key]),(row['job_id'],key)
