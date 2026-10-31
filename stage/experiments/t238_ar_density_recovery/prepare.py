from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t238_ar_density_recovery'
def main():
    assert not OUT.exists(),'No duplicateT238'
    prior=json.loads((ROOT/'stage/runs/t237_ar_density/summary.json').read_text());assert prior['state']=='PREFLIGHT_FAILED_QUADRATURE_NUMERIC_UNDERFLOW'
    plan=dict(experiment='T238',seed=2026103251,quadrature_nodes=1024,draws=16384,max_physics=0,optimizer_calls=0,wall_seconds=120,final_bytes=512*1024,peak_bytes=512*1024,global_bytes=1024**3,
      gates=dict(normalization_abs_max=1e-7,finite_difference_relative_max=1e-5,reset_marginal_max=1e-10,score_mean_max_standard_errors=6.,roundtrip_latent_max=5e-4),
      contract='Zero-physics mathematical preflight of NEW history-conditional bounded AR1 policy, sigma(.035,.03),rho.98. Stable reflected logCDF/finite positive mass, no density clamp. Native action float32 inversion checked, prior-state mean gradient retained. Fixed1024pointGauss-Legendreconditional normalization over seven conditional-sigma latent range, omitted normal tail<3e-12, continuousfloat64inverseCDF and independent autograd da/dz Jacobian. T237 zero-density action-space quadrature underflow retained asoriginalfailedpreflight; scientificonline policy unchanged; rho0/reset marginal identity; all4current/previousmean finite differences;16384joint conditional draws per2components atfixed mean(.3,.95),priorlatent(.4,-.7),stationary innovations; expectedscore compared6MCSE, native roundtrip and actual perstep likelihood reconstruction. No optimizer/navigation/model export/map/planner/teacher/Phase; no parameter grid. Failing any gate prohibits this data path in subsequent RL; 120s/512KiB/global1GiB/free40GiB.',
      references=['https://www.ijcai.org/proceedings/2019/0382.pdf','https://arxiv.org/html/2312.11091v2','https://arxiv.org/abs/1707.06347'])
    paths=list(HERE.glob('*.py'))+[ROOT/'stage/runs/t237_ar_density/summary.json'];plan['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',physics=0,runs_MiB=used/1024**2)))
if __name__=='__main__':main()
