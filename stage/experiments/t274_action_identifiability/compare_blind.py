"""Matched logged-event forecast comparison, not causal action identification."""
from blind_model import *
import gzip,io
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t274_action_identifiability'
DOC=ROOT/'stage/docs/long_distance_20261101'
DEST=DOC/'T274_COMPARISON.json'


def read_payload(path):
    return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=False)


def main():
    assert not DEST.exists()
    assert json.loads((DOC/'T274_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    p=json.loads((OUT/'protocol.json').read_text());d=dataset(p)
    parent=ROOT/'stage/runs/t272_event_value_pilot'
    a=read_payload(parent/'event_checkpoint.pt.gz');b=read_payload(OUT/'event_checkpoint.pt.gz')
    assert np.array_equal(a['normalization_mean'],b['normalization_mean']) and np.array_equal(a['normalization_scale'],b['normalization_scale'])
    assert a['optimizer_calls']==b['optimizer_calls']==500
    full=EventModel().eval();full.load_state_dict(a['model']);blind=ActionBlind().eval();blind.load_state_dict(b['model'])
    with torch.no_grad():pa=full(d['x']).numpy();pb=blind(d['x']).numpy()
    sa=json.loads((parent/'summary.json').read_text());sb=json.loads((OUT/'summary.json').read_text())
    assert metrics(pa,d,'test')==sa['report']['final_test'] and metrics(pb,d,'test')==sb['report']['final_test']
    # Masked columns must retain their exact original random values: no hidden
    # current-action pathway, while the same full parameter count is retained.
    torch.manual_seed(p['seed']);initial=ActionBlind()
    assert torch.equal(blind.net[0].weight[:,-2:],initial.net[0].weight[:,-2:])
    w=episode_weights(d,'test');y=d['y'].numpy();per=[]
    def brier(prob,weights):return float((((prob-y)**2)*weights[:,:,None]).sum(0).sum(-1).mean())
    for i,r in enumerate(p['records']):
        if r['split']!='test':continue
        ww=w*(d['episode']==i)[:,None];ww/=ww.sum(0).clip(min=1e-30)[None,:]
        per.append(dict(id=r['id'],side=r['side'],distance_m=r['distance_m'],action_input_Brier=brier(pa,ww),blind_Brier=brier(pb,ww)))
    strata=[]
    for dist in [5.25,6.]:
        for side in ['LEFT','RIGHT']:
            ee=[i for i,r in enumerate(p['records']) if r['split']=='test' and r['distance_m']==dist and r['side']==side]
            ww=w*np.isin(d['episode'],ee)[:,None];ww/=ww.sum(0).clip(min=1e-30)[None,:]
            strata.append(dict(distance_m=dist,side=side,episodes=len(ee),action_input_Brier=brier(pa,ww),blind_Brier=brier(pb,ww)))
    # One deterministic row permutation gives a descriptive input-reliance
    # check. Shuffling actions can be off-distribution; not an intervention.
    ix=np.flatnonzero(~d['train']);rng=np.random.default_rng(2026111202)
    changed=d['x'].clone();changed[ix,-2:]=changed[ix[rng.permutation(len(ix))],-2:]
    with torch.no_grad():permuted=full(changed).numpy();blind_changed=blind(changed).numpy()
    assert np.array_equal(pb,blind_changed)
    report=dict(state='MATCHED_CURRENT_ACTION_INPUT_ABLATION',physics_added=0,optimizer_added=0,
        identical_data_splits_initial_parameter_values_normalization_update_count_and_draw_schedule=True,
        changed_factor='Only current query action coordinates masked at model input. Previous action in causal120 remains.',
        train_episodes=80,test_episodes=16,action_input_test=sa['report']['final_test'],blind_test=sb['report']['final_test'],
        action_input_train=sa['report']['final_train'],blind_train=sb['report']['final_train'],
        per_test_episode=per,strata=strata,blind_better_episode_count=sum(r['blind_Brier']<r['action_input_Brier'] for r in per),
        action_shuffled_full_head_test=metrics(permuted,d,'test'),action_shuffling_OOD_not_causal_evidence=True,
        blind_action_invariance_all_rows_exact=True,blind_current_action_weight_columns_exact_initial=True,
        full_checkpoint_sha256=sha(parent/'event_checkpoint.pt.gz'),blind_checkpoint_sha256=sha(OUT/'event_checkpoint.pt.gz'),
        action_ranking_gate=False,six_m_navigation_gate=False,automatic_Actor_coupling=False,
        limitations=['One trained initialization and previously consumed map/family; no population superiority claim.',
            'Removing current action does not make physical control irrelevant; state already carries previous controls and motion.',
            'Observational prediction of a logged continuation does not identify gradients for a different policy.',
            'No per-action repeated stochastic native outcomes or new safe6m positives were created.'])
    save(DEST,report)
    print(json.dumps(dict(state='COMPLETE',action_Brier=sa['report']['final_test']['mean_Brier_total'],
        blind_Brier=sb['report']['final_test']['mean_Brier_total'],strata=strata,blind_better_episodes=report['blind_better_episode_count'])),flush=True)


if __name__=='__main__':main()
