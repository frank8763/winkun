"""Saved-trajectory local-basis activation and first divergence, no native/fit."""
from pathlib import Path
import os,sys,time,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'));import t552_actor as L
sys.path.insert(0,str(ROOT/'stage/experiments/t546_functional_residual'));import t546_actor as Kernel
OUT=ROOT/'stage/runs/t558_support_overlap';DOC=ROOT/'stage/docs/day_20261115'
torch,np=L.torch,L.np

def main():
    assert not OUT.exists();torch.set_num_threads(1);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15};start=time.time()
    s=ROOT/'stage/runs/t557_bc_stage_isolation';p0=L.read(s/'protocol.json');p=L.read(p0['actor_configuration']);p.update(p0)
    q=L.frozen(p,p['frozen_base']['path']);raw=[json.loads(v) for v in (s/'cases.jsonl').read_text().splitlines()];selected=['t502_field_sector0_offset2','t502_field_sector1_offset2','t502_field_sector4_offset2']
    # Three fixed tasks span direct inherited-basis success versus legacy
    # success. Inspect original baseline plus both BC policies' actual states.
    report=[];inputs={str(s/'protocol.json'):L.sha(s/'protocol.json')}
    for ident in selected:
        base=p['baseline_rows'][ident]
        # Existing full baseline history from immutable T555 keep dataset.
        old=L.read(ROOT/'stage/runs/t555_adam_direction_control/protocol.json')
        baseline=next(r for r in old['keep_rows'] if r['job_id']==ident);ba=L.G.E.N.S.arrays(baseline['path']);inputs[baseline['path']]=baseline['sha256'];metric=[]
        candidates=[('baseline',baseline,ba)]+[(r['role'],r,L.G.E.N.S.arrays(r['path'])) for r in raw if r['job_id']==ident]
        for role,row,a in candidates:
            assert L.sha(row['path'])==row['sha256'];inputs[row['path']]=row['sha256'];x=torch.tensor(a['states'])
            with torch.no_grad():
                n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);f=torch.cat([h[:,0],n],-1)
                banks=[('repair',q.repair_centers,q.repair_radii,q.repair_weights),('added',q.added_centers,q.added_radii,q.added_weights)]+[('layer'+str(i),z.centers,z.radii,z.weights) for i,z in enumerate(q.layers)]
                comps=[]
                for name,centers,radii,weights in banks:
                    blocks=[Kernel.basis(f[k:k+16],centers,radii) for k in range(0,len(f),16)];phi=torch.cat(blocks);d=phi@weights;active=(phi!=0).any(-1)
                    comps.append(dict(bank=name,centers=len(centers),radius_min=float(radii.min()) if len(radii) else None,radius_max=float(radii.max()) if len(radii) else None,active_decisions=int(active.sum()),active_fraction=float(active.double().mean()),correction_absmax=float(d.abs().max()),correction_RMS=float(d.square().mean().sqrt())))
            count=min(len(a['states']),len(ba['states']));assert np.array_equal(a['native_counts'][:max(0,count-1)],ba['native_counts'][:max(0,count-1)])
            err=np.linalg.norm(a['poses'][:count,:2]-ba['poses'][:count,:2],axis=1);yaw=np.abs(np.arctan2(np.sin(a['poses'][:count,2]-ba['poses'][:count,2]),np.cos(a['poses'][:count,2]-ba['poses'][:count,2])))
            ids=np.flatnonzero((err>.03)|(yaw>np.deg2rad(3)))
            first=int(ids[0]) if len(ids) else None
            metric.append(dict(role=role,decisions=len(a['states']),banks=comps,first_3cm_or_3deg_hold=first,first_deviation_seconds=float(a['native_counts'][:first].sum()*.002) if first is not None else None,maximum_aligned_XY_deviation=float(err.max()),source_sha256=row['sha256'],all_states_ACTUALLY_executed=True))
        report.append(dict(id=ident,results=metric))
        print(ident,[(v['role'],sum(b['active_decisions'] for b in v['banks']),v['first_deviation_seconds']) for v in metric],flush=True)
    out=dict(state='COMPLETE_SAVED_ONLY',pid=os.getpid(),epoch=time.time(),elapsed_s=time.time()-start,new_native=0,new_optimizer=0,report=report,inputs=inputs,source_sha256=L.sha(__file__),limitations='Alignment ends at earliest terminal; describes actual saved states and inherited basis activations, not a causal intervention, global stability proof or new policy test. Activation changes need not explain every failure.')
    OUT.mkdir();L.save(OUT/'diagnostic.json',out);L.save(DOC/'T558_RAW_EVIDENCE.json',dict(original_path=str(OUT/'diagnostic.json'),original_sha256=L.sha(OUT/'diagnostic.json'),original_utf8=(OUT/'diagnostic.json').read_text(),reversible=True,new_native=0,new_optimizer=0));print('T558_COMPLETE',flush=True)
if __name__=='__main__':main()
