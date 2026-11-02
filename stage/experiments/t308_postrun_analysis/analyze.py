"""Read-only failed-gate analysis: no native, optimizer, model or raw edits.

Do not replace T308 registered metrics with these post-hoc diagnostics.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t308_iid_native_goal_value'))
from t308_core import *
import time

def main():
    start=time.time();dest=DOC/'T308_POSTHOC_NUMERICS.json';assert not dest.exists()
    p=read(OUT/'protocol.json');summary=read(OUT/'summary.json');assert summary['state']=='COMPLETE' and not summary['combined_value_control_gate']
    assert read(DOC/'T308_AUDIT.json')['state']=='PASS';torch.set_num_threads(1)
    original=read(OUT/'metrics.json');raw=read(OUT/'cases.json');by={r['id']:r for r in raw};models={}
    for name,item in summary['models'].items():
        assert C.sha(item['path'])==item['sha256'];q=Critic();q.load_state_dict(torch.load(io.BytesIO(gzip.decompress(Path(item['path']).read_bytes())),map_location='cpu',weights_only=False)['model']);models[name]=q.eval().requires_grad_(False)
    canonical=[];max_blind_difference=0.
    with torch.no_grad():
        for r in original['controls']:
            assert time.time()-start<120
            a=[arrays(by[r['family']+'_'+arm]['path']) for arm in ['minus','center','plus']];c=a[1]
            x=torch.tensor(c['states'][:1]);body=torch.tensor(relative(c['poses'][0],r['query_world']),dtype=torch.float32)[None,None]/10.;kind=torch.zeros((1,1),dtype=torch.long)
            result={}
            for name,q in models.items():
                scores=[]
                for item in a:
                    act=torch.tensor(item['actions'][:1]) if name=='action' else torch.zeros(1,2)
                    scores.append(float(q.logits(x,torch.zeros(1,2),act,body,kind)[0,0]))
                result[name]=scores
            assert np.ptp(result['blind'])==0.,'Identical factual inputs require exact canonical blind tie'
            max_blind_difference=max(max_blind_difference,float(np.ptp(r['predicted']['blind'])))
            canonical.append(dict(family=r['family'],side=r['side'],query_endpoint_index=r['query_endpoint_index'],
                original_predicted=r['predicted'],canonical_batch1=result,real=r['real_kernel_occupancy']))
    report={}
    for side in ['LEFT','RIGHT']:
        selected=[r for r in canonical if r['side']==side];pairs=[];correct=0;chosen=[];uniform=[]
        for r in selected:
            pred=np.array(r['canonical_batch1']['action']);real=np.array(r['real']);chosen.append(float(real[pred==pred.max()].mean()));uniform.append(float(real.mean()))
            for i,j in [(0,1),(0,2),(1,2)]:
                if abs(real[i]-real[j])>p['value_margin']:
                    pairs.append((pred[i]-pred[j],real[i]-real[j]));correct+=int(np.sign(pred[i]-pred[j])==np.sign(real[i]-real[j]))
        pairs=np.array(pairs);report[side]=dict(pairs=len(pairs),canonical_sign_correct=correct,canonical_sign_accuracy=correct/len(pairs),
            canonical_top_real_mean=float(np.mean(chosen)),uniform_real_mean=float(np.mean(uniform)),
            predicted_pair_abs_quantiles=np.quantile(np.abs(pairs[:,0]),[0,.5,1]).tolist(),real_pair_abs_quantiles=np.quantile(np.abs(pairs[:,1]),[0,.5,1]).tolist(),
            paired_difference_correlation=float(np.corrcoef(pairs.T)[0,1]),blind_canonical_pair_sign_accuracy=0.,blind_canonical_top_equals_uniform=True)
    protected={
        ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz':'63e63e2b444867b26622a2ab0f0c4227bc38d48a7cd2b5a3b378d9b0266006df',
        ROOT/'stage/runs/t299_conditional_goal_likelihood/conditional_goal_likelihood_final_checkpoint.pt.gz':'b15cddbbf5914ef5187ce221082c0e6cdbe3608b0f8a1ff1177f8fefa95f5751',
        ROOT/'stage/runs/t288_five_metre_presentation/stage_5m_left_1080p.mp4':'eb57d8825106a14baa37812fcbe0d230813643c7911293d8549bf4ced960772c',
        ROOT/'stage/runs/t288_five_metre_presentation/stage_5m_right_1080p.mp4':'fc7fe3ae7b58fedfda179ff7db918625404f61aa8dbf7bafa4ef607cda2d0a68'}
    # A copied expected digest must never be silently corrected after a mismatch.
    checks={str(f):dict(expected=h,actual=C.sha(f)) for f,h in protected.items()}
    output=dict(state='POSTHOC_READ_ONLY_DIAGNOSTIC',native_steps=0,optimizer_calls=0,Actor_updates=0,registered_metrics_unchanged=True,
        all_models_unchanged=True,canonical_batch1_blind_exact_ties=True,original_batch3_blind_max_ulp_scale_spread=max_blind_difference,
        per_side=report,details=canonical,protected_files=checks,elapsed_s=time.time()-start,
        warning='Original batched blind strict-argmax control metrics contain numerical pseudo-ordering of identical inputs. Canonical tie is the right interpretation, not a new threshold or replacement of registered failed gate. Action head remains failed; all original metrics retained.')
    assert all(v['actual']==v['expected'] for v in checks.values()),checks
    C.save(dest,output);print(json.dumps({k:v for k,v in output.items() if k not in ['details','protected_files']}))

if __name__=='__main__':main()
