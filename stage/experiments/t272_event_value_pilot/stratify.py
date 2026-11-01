"""Read-only per-distance/side forecast metrics, including long-horizon failures."""
from event_model import *
import gzip
import io
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'
DEST=ROOT/'stage/docs/long_distance_20261101/T272_STRATIFIED.json'


def main():
    assert not DEST.exists()
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T272_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    p=json.loads((OUT/'protocol.json').read_text());d=dataset(p)
    payload=torch.load(io.BytesIO(gzip.decompress((OUT/'event_checkpoint.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    q=EventModel();q.load_state_dict(payload['model']);q.eval()
    with torch.no_grad():pred=q(d['x']).numpy()
    base=np.broadcast_to(frequency_baseline(d),pred.shape);w=episode_weights(d,'test');y=d['y'].numpy();result=[]
    for dist in [5.25,6.]:
        for side in ['LEFT','RIGHT']:
            ee=[i for i,r in enumerate(p['records']) if r['split']=='test' and r['distance_m']==dist and r['side']==side]
            assert len(ee)==4
            mask=np.isin(d['episode'],ee);ww=w*mask[:,None];ww/=ww.sum(0).clip(min=1e-30)[None,:]
            loss=lambda z:(((z-y)**2)*ww[:,:,None]).sum(0)
            a,b=loss(pred),loss(base)
            result.append(dict(distance_m=dist,side=side,episodes=4,final_Brier=a.tolist(),frequency_Brier=b.tolist(),
                mean_final_Brier=float(a.sum(-1).mean()),mean_frequency_Brier=float(b.sum(-1).mean())))
    save(DEST,dict(state='DESCRIPTIVE_FIXED_HOLDOUT_STRATIFICATION',physics_added=0,optimizer_added=0,groups=result,
        limitation='Four episodes per side/distance, one previously consumed map and one reset family; no independent navigation success or confidence-bound claim.'))
    print(json.dumps(dict(state='COMPLETE',groups=[{k:r[k] for k in ['distance_m','side','episodes','mean_final_Brier','mean_frequency_Brier']} for r in result])),flush=True)


if __name__=='__main__':main()
