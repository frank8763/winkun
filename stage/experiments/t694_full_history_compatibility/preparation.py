"""All real sources/geometry, no coefficient commit or physical/optimizer step."""
from pathlib import Path
import ast,copy,json,os,resource,sys,time,traceback
import t694_actor as A
ROOT=A.ROOT;HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/weekend_20261120_23/t694'
sys.path.insert(0,str(ROOT/'stage/experiments/t693_measure_valued_actor'))
import t693_wait as Wait


def main():
    started=time.time();os.sched_setaffinity(0,{14});Wait.Raw.resources()
    assert not (ROOT/'stage/runs/t694_full_history_compatibility').exists()
    assert not (DOC/'PREPARATION.json').exists(),'Do not repeat completed preparation'
    assert (DOC.parent/'EXACT_COMMAND_RETENTION_PRIMARY_REVIEW.md').exists()
    p=copy.deepcopy(A.read(ROOT/'stage/runs/t691_induced_state_rehearsal/protocol.json'))
    p['protected_rows']=[r for r in A.read(p['reference_contract'])['reference_rows'] if r['origin']=='T501_original_success']
    assert len(p['protected_rows'])==len({r['task_id'] for r in p['protected_rows']})==384
    assert {r['task_id'] for r in p['protected_rows']}==set(p['before_success_ids'])
    p.update(support_fraction=.49,minimum_feature_separation=1e-10,maximum_centers=1405,model_bytes=8*2**20)
    def forbidden(*args,**kw):raise AssertionError('No coefficient commit, optimizer or native in preparation')
    A.torch.optim.Adam.step=forbidden
    import mujoco
    mujoco.mj_step=forbidden
    A.torch.set_num_threads(1);A.torch.use_deterministic_algorithms(True)
    def guard():
        assert time.time()<started+550
        Wait.Raw.resources()
    protected,sources,manifest=A.data(p,guard)
    geometry,info=A.construction(p,protected,sources,guard)
    # Verify new static forward arithmetic at zero learned coefficients only.
    original=A.ORIGINAL(p,p['initial']['path']);q=A.Actor(p,len(geometry['radii']))
    missing,extra=q.load_state_dict(original.state_dict(),strict=False)
    assert set(missing)=={'compat_centers','compat_radii','compat_weights'} and not extra
    q.compat_centers.copy_(geometry['centers']);q.compat_radii.copy_(geometry['radii'])
    assert not q.compat_weights.any()
    sample=p['protected_rows'][0];a=A.L.T.R.E.N.S.arrays(sample['source_path'])
    _,newmu=A.trajectory(q,a,sample['task_id'],a['actions'],guard)
    _,oldmu=A.trajectory(original,a,sample['task_id'],a['actions'],guard)
    assert A.torch.equal(newmu,oldmu)
    # Finite formula checks above did not apply learned coefficients to Actor.
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<4*2**30
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    result=dict(state='GEOMETRY_AND_OWNCARRY_PREPARATION_PASS_NOT_COMPLETE_PREFLIGHT',
                geometry=info,manifest=manifest,zero_expanded_readout_original_batch1_function_exact=True,
                new_native=0,new_optimizer=0,new_Actor=0,new_Value=0,no_supervised_coefficients_committed=True,
                peak_rss_bytes=peak,elapsed_seconds=time.time()-started,
                remaining=['actual loader/export/worker/canonicalfactory and failure entrance',
                           'finite fit/audit/evaluation driver','fullzero-updateentrypreflight',
                           'independentworstreservation','sourceexact and unique launch'])
    (DOC/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (DOC/'PREPARATION_REJECTION.json').write_text(json.dumps(dict(state='PREPARATION_REJECTED_NO_COEFFICIENT_COMMIT',error=traceback.format_exc(),
            new_native=0,new_optimizer=0,new_Actor=0,no_threshold_radius_or_center_rescue=True),indent=2)+'\n')
        raise
