"""Zero-native, zero-optimizer parameter-scope counterfactual on fixed histories.

Hybrid modules are analytical fixtures, never exported or native-evaluated.
This attributes fixed-history command drift, not closed-loop performance.
"""
from pathlib import Path
import copy,gzip,json,sys,time
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/day_20261119/t652'
sys.path.insert(0,str(ROOT/'stage/experiments/t651_canonical_outcome_pg'))
import t651_runtime as T
L,torch,np=T.R.L,T.torch,T.np

def main():
    p=T.read(ROOT/'stage/runs/t651_canonical_outcome_pg/protocol.json');s=T.read(ROOT/'stage/runs/t651_canonical_outcome_pg/pg/summary.json')
    assert T.read(ROOT/'stage/runs/t651_canonical_outcome_pg/pg/audit.json')['state']=='PASS'
    T.R.B.setup(p);torch.set_num_threads(1)
    before=L.frozen(p,p['initial']['path']);after=L.frozen(p,s['proposal']['path'])
    original=copy.deepcopy(before.state_dict());newstate=after.state_dict()
    groups=lambda key:'adapters' if key.startswith(('drive_adapter.','steer_adapter.')) else 'core_body'
    models={'initial':before,'all_updated':after}
    for scope in ['adapters','core_body']:
        q=L.frozen(p,p['initial']['path'])
        q.load_state_dict({k:newstate[k] if groups(k)==scope else v for k,v in original.items()});models[scope+'_updated']=q
    delta={}
    for key,v in newstate.items():
        if key in original and v.is_floating_point():delta[key]=dict(max_abs=float((v-original[key]).abs().max()),L2=float((v-original[key]).square().sum().sqrt()),parameters=v.numel())
    manifest=T.read(ROOT/'stage/runs/t646_offline_night/OFFLINE_INPUTS.json');selected={r['job_id']:r for r in T.R.D.selected()}
    joint=json.loads(gzip.decompress((ROOT/'stage/runs/t651_canonical_outcome_pg/pg/JOINT407.json.gz').read_bytes()))
    actual={r['job_id']:r for r in joint['rows']};records=[];start=time.time()
    with torch.no_grad():
        for row in manifest['members']:
            old=selected[row['id']];a=T.R.E.N.S.arrays(row['offline_path']);assert T.sha(row['offline_path'])==row['sha256']
            x=torch.tensor(a['states']);target=torch.tensor(a['locations']).clamp(-1,1)
            values={}
            for name,q in models.items():
                loc=q.sequence(x);err=loc.clamp(-1,1)-target
                values[name]=dict(peak=float(err.abs().max()),RMS=float(err.square().mean().sqrt()),initial_drive_error=float(err[0,0]),initial_steer_error=float(err[0,1]))
            assert values['initial']['peak']<1e-10
            current=actual[row['id']]
            # Hash-covered scene is available even when current raw file is
            # compact; use it to rule out static geometry mismatch per task.
            assert current['array_hashes']['scene_sha256']==old['array_hashes']['scene_sha256'],row['id']
            commands=T.R.E.N.S.arrays(current['path'])['actions'];n=min(len(commands),len(a['actions']))
            difference=np.max(abs(commands[:n]-a['actions'][:n]),axis=1)
            k=np.flatnonzero(difference>=.01)
            record=dict(id=row['id'],closedloop_current_success=T.safe(current),collision=current['collision'],timeout=current['timeout'],remaining_m=current['remaining_m'],
                identical_scene_SHA=True,teacher_forced_response=values,actual_command_first_ge_01=int(k[0]) if len(k) else None,
                actual_command_max_common_prefix=float(difference.max()))
            records.append(record)
    lost=[r for r in records if not r['closedloop_current_success']]
    summaries={}
    for name in ['all_updated','adapters_updated','core_body_updated']:
        summaries[name]=dict(old384_median_RMS=float(np.median([r['teacher_forced_response'][name]['RMS'] for r in records])),
             old384_max_peak=max(r['teacher_forced_response'][name]['peak'] for r in records),
             lost6_median_RMS=float(np.median([r['teacher_forced_response'][name]['RMS'] for r in lost])),
             lost6_max_peak=max(r['teacher_forced_response'][name]['peak'] for r in lost))
    report=dict(state='READ_ONLY_ZERO_NATIVE_ZERO_OPTIMIZER_COMPLETE',seconds=time.time()-start,
        final407=s['final407']['success'],lost_prior=s['final407']['lost_prior'],gained_prior=s['final407']['gained_prior'],
        all384_static_scene_hashes_match_original=True,parameter_changes=delta,scope_summaries=summaries,lost_cases=lost,old384=records,
        limitation='Hybrids evaluated only on original fixed histories; not trained methods or closed-loop outcomes, no geometry mismatch does not prove all dynamics identical',
        new_native=0,new_optimizer=0)
    T.save(DOC/'PARAMETER_SCOPE_ANALYSIS.json',report)
    print('ALL384_SCENES_MATCH',flush=True);print(json.dumps(summaries,indent=2),flush=True)
    for v in lost:print(v['id'],v['collision'],v['remaining_m'],v['teacher_forced_response'],flush=True)

if __name__=='__main__':main()
