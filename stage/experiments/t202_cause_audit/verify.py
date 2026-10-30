"""Independent saved-array accounting; no environment steps or optimization."""
import os
os.sched_setaffinity(0,{11})
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t202_cause_audit';DOC=ROOT/'stage/docs/results_20261030_cause_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t200_residual_placement'))
import common as C
import numpy as np
import torch
from replay import GAMMA

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    plan=json.loads((HERE/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETE' and summary['optimizer_updates']==summary['training_physics_steps']==0
    launch=json.loads((OUT/'launch.json').read_text());assert not Path('/proc',str(launch['pid'])).exists()
    assert C.sha(HERE/'protocol.json')==C.sha(OUT/'protocol.json')==launch['protocol_sha256']
    for path,h in plan['input_hashes'].items():assert C.sha(path)==h,path
    rows=[json.loads(s) for s in (OUT/'branches.jsonl').read_text().splitlines()];assert len(rows)==48
    policies={};critics={}
    for arm,path in plan['checkpoints'].items():
        ck=torch.load(path,map_location='cpu',weights_only=False);p=C.Policy(arm).eval().requires_grad_(False);p.load_state_dict(ck['policy']);q=C.Critic().eval().requires_grad_(False);q.load_state_dict(ck['critic']);policies[arm]=p;critics[arm]=q
    max_action=max_logp=0.;cap=plan['case']['step_cap']
    for row in rows:
        path=OUT/(row['id']+'.npz');assert C.sha(path)==row['trace_sha256']
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        arm=row['arm'];absolute=row['anchor_absolute_steps'];p=policies[arm];q=critics[arm]
        assert a['steps'][0]==absolute and np.array_equal(a['steps'],absolute+np.r_[0,np.cumsum(a['native_counts'][:-1])])
        assert int(a['native_counts'].sum())==row['physics_steps'] and absolute+row['physics_steps']<=cap
        assert np.allclose(a['states'][:,112],1-a['steps']/cap,atol=3e-8,rtol=0)
        assert np.array_equal(a['poses'][1:],a['next_poses'][:-1])
        d0=np.linalg.norm(a['poses'][:,:2]-a['goal_xy'],axis=1);d1=np.linalg.norm(a['next_poses'][:,:2]-a['goal_xy'],axis=1)
        assert np.allclose(a['states'][:,48],d0/10,atol=3e-8,rtol=0)
        flags=a['end_flags'].astype(float);terminal=flags.any(axis=1);assert terminal[-1] and not terminal[:-1].any()
        assert not np.any(flags[:,0]*flags[:,1]);assert bool(flags[-1,1])==row['success']
        if row['success']:assert d1[-1]<=.2
        discount=GAMMA**a['native_counts'];events=flags[:,1]-flags[:,0]-.5*flags[:,2]
        rewards=-.000002*(1-discount)/(1-GAMMA)+.03*d0-discount*.03*d1*(~terminal)+terminal*GAMMA**(a['native_counts']-1)*events
        assert np.max(np.abs(rewards-a['rewards']))<1e-12
        weights=GAMMA**(a['steps']-absolute);environment=float(weights@rewards);entropy=float(-plan['alpha']*weights@a['continuation_logp'])
        assert abs(environment-row['environment_MC'])<1e-10 and abs(environment+entropy-row['soft_Q_MC'])<1e-10
        rng=np.random.default_rng(row['noise_seed']);sigma=p.logstd.clamp(-3,0).exp().numpy();actions=[np.asarray(row['first_action'],dtype=np.float32)];lps=[0.]
        with torch.no_grad():
            for state in a['states'][1:]:
                mu=p.latent_mean(torch.tensor(state[None])).numpy()[0];eps=rng.normal(size=2);z=mu+sigma*eps;actions.append(np.tanh(z).astype(np.float32));lps.append(float(np.sum(-.5*eps**2-np.log(sigma)-.5*math.log(2*math.pi)-2*(math.log(2)-z-np.logaddexp(0,-2*z)))))
            qu,qv=q(torch.tensor(a['states'][:1]),torch.tensor(a['actions'][:1]));assert abs(float(torch.minimum(qu,qv))-row['minQ'])<1e-7
        error=float(np.max(np.abs(np.asarray(actions)-a['actions'])));lp_error=float(np.max(np.abs(np.asarray(lps)-a['continuation_logp'])));max_action=max(max_action,error);max_logp=max(max_logp,lp_error)
        assert error<2e-6 and lp_error<1e-5
        snap=C.read_snapshot(OUT/f'{arm}_anchor_{absolute}.npz');assert hashlib.sha256(snap['physics'].tobytes()).hexdigest()==row['anchor_physics_sha256']
        assert np.array_equal(C.observation(snap['obs'],1-absolute/cap),a['states'][0])
    assert summary['physics_steps']==28000+sum(r['physics_steps'] for r in rows)<=plan['max_physics']
    baseline=[]
    for arm in plan['checkpoints']:
        for absolute in plan['anchor_elapsed_steps']:
            rr=[r for r in rows if (r['arm'],r['anchor_absolute_steps'],r['first_action_kind'])==(arm,absolute,'baseline')];v=np.asarray([r['soft_Q_MC'] for r in rr]);assert len(v)==4
            baseline.append({'arm':arm,'anchor_absolute_steps':absolute,'minQ':rr[0]['minQ'],'soft_MC_mean':float(v.mean()),'soft_MC_sample_SD':float(v.std(ddof=1)),'Q_minus_MC_mean':float(rr[0]['minQ']-v.mean()),'safe_GOAL':sum(r['success'] and not r['collision'] for r in rr)})
    old_count=0
    for name in ['left_exploration','timeout_progress','sac_gradient']:
        previous=json.loads((ROOT/f'stage/manifests/{name}_20261030_files.json').read_text())
        for row in previous['files']:
            path=ROOT/row['path'];assert path.stat().st_size==row['bytes'] and C.sha(path)==row['sha256'];old_count+=1
    for name in ['exploration_support.json','coverage.json']:
        x=json.loads((OUT/name).read_text());assert x['physics_steps']==x['optimizer_updates']==0
        for path,h in x['input_hashes'].items():assert C.sha(path)==h
    own=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert own<plan['max_bytes'] and used<1024**3
    result={'state':'PASS','validation_scope':'Accounting and provenance PASS, not a claim that Critic calibration or navigation passed','branches':48,'physics_steps':summary['physics_steps'],'optimizer_updates':0,'training_physics_steps':0,'prior_archived_raw_files_unchanged':old_count,'max_reconstructed_action_error':max_action,'max_reconstructed_log_density_error':max_logp,'baseline_value_witnesses':baseline,'own_bytes_before_verification':own,'all_new_runs_bytes_before_verification':used,'source_protocol_sha256':C.sha(HERE/'protocol.json')}
    C.save(OUT/'verification.json',result);DOC.mkdir(exist_ok=True)
    for name in ['summary.json','offline.json','coverage.json','exploration_support.json','verification.json','contract_audit.json']:
        (DOC/name).write_bytes((OUT/name).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    xx=np.arange(len(baseline));fig,ax=plt.subplots(figsize=(7.5,4.1));ax.scatter(xx-.08,[r['minQ'] for r in baseline],label='Predicted min(Q1,Q2)',color='#c44e52',marker='x',s=70,zorder=3)
    ax.errorbar(xx+.08,[r['soft_MC_mean'] for r in baseline],yerr=[r['soft_MC_sample_SD'] for r in baseline],fmt='o',capsize=4,label='Native soft-Q MC (mean ± sample SD)',color='#4c72b0')
    ax.set_xticks(xx,[f"{r['arm']}\n{r['anchor_absolute_steps']*.002:g} s" for r in baseline]);ax.axhline(0,color='gray',lw=.7);ax.set_ylabel('Normalized soft return');ax.set_title('Frozen policy continuation, fixed LEFT task; n=4 per state-action');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(OUT/'critic_calibration.png',dpi=160);plt.close(fig);(DOC/'critic_calibration.png').write_bytes((OUT/'critic_calibration.png').read_bytes())
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
