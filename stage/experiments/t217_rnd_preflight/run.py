"""Archived-ray familiarity/noise/budget preflight, zero MuJoCo imports/steps."""
from pathlib import Path
import os,json,time,hashlib,fcntl,traceback,shutil
import torch
import numpy as np
from model import LidarRND
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t217_rnd_preflight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)
def used(p):return sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
def state_hash(m):
    h=hashlib.sha256()
    for name,x in m.state_dict().items():h.update(name.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def stats(x):return {'n':len(x),'mean':float(x.mean()) if len(x) else None,'median':float(x.median()) if len(x) else None,'q95':float(torch.quantile(x,.95)) if len(x) else None}
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'status.json').exists(),'No duplicate T217'
    plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();updates=0;result={}
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
    save(OUT/'status.json',{'state':'RUNNING','pid':os.getpid(),'physics_steps':0,'Actor_updates':0})
    def guard():
        if time.monotonic()-start>plan['wall_seconds']-5:raise TimeoutError('Preflight wall reserve')
        if used(OUT)+512*1024>=plan['final_bytes'] or used(ROOT/'stage/runs')+512*1024>=1024**3 or shutil.disk_usage(ROOT).free<40*1024**3:raise RuntimeError('Storage reserve')
    try:
        with np.load(plan['retention_targets']) as z:
            rays=torch.tensor(z['states'][:,:48]);offsets=z['offsets'].copy()
        train_parts=[rays[offsets[i]:offsets[i+1]] for i in plan['train_tasks']]
        familiar_ids=[]
        for row in plan['left_failure_rows']:
            if row['round']==plan['familiar_left_round']:
                with np.load(row['trace']) as z:train_parts.append(torch.tensor(z['states'][:,:48]))
                familiar_ids.append(row['id'])
        assert len(familiar_ids)==4
        train=torch.cat(train_parts);held=torch.cat([rays[offsets[i]:offsets[i+1]] for i in plan['heldout_tasks']])
        rnd=LidarRND(train.mean(0),train.std(0,unbiased=False));target_hash=state_hash(rnd.target)
        seen=train[torch.tensor(np.linspace(0,len(train)-1,plan['seen_probe_count'],dtype=int))]
        with torch.no_grad():initial=rnd.error(seen).mean()
        opt=torch.optim.Adam(rnd.predictor.parameters(),lr=plan['lr']);losses=[]
        for step in range(plan['predictor_updates']):
            guard();batch=train[rng.integers(0,len(train),plan['batch_size'])]
            loss=rnd.error(batch).mean();assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward()
            assert all(p.grad is None for p in rnd.target.parameters())
            torch.nn.utils.clip_grad_norm_(rnd.predictor.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1
            if step%50==0:losses.append({'step':step+1,'loss':float(loss.detach())})
        with torch.no_grad():
            seen_error=rnd.error(seen);held_error=rnd.error(held)
            distances=torch.cdist(held,train).min(1).values/(48**.5)
            far=distances>=plan['far_rms_range_threshold'];far_error=held_error[far]
            noise=torch.tensor(rng.normal(0,plan['noise_sigma_normalized'],seen.shape),dtype=seen.dtype)
            noisy=(seen+noise).clamp(0,1);noise_error=rnd.error(noisy)
            increment=(noise_error-seen_error).clamp_min(0)
            seen_median=float(seen_error.median());far_median=float(far_error.median()) if bool(far.any()) else None
            gap=far_median-seen_median if far_median is not None else None
            ratio=far_median/max(seen_median,1e-12) if far_median is not None else None
        gates={'target_frozen':state_hash(rnd.target)==target_hash,
            'seen_error_learns':float(seen_error.mean())<=plan['max_seen_loss_fraction']*float(initial),
            'enough_far_samples':int(far.sum())>=plan['minimum_far_samples'],
            'far_seen_separation':ratio is not None and ratio>=plan['minimum_far_seen_ratio'],
            'noise_sensitivity':gap is not None and gap>0 and float(torch.quantile(increment,.95))<=plan['maximum_noise_increment_fraction_of_far_gap']*gap}
        left=[];scale=max(float(torch.quantile(seen_error,.95)),1e-6)
        for row in plan['left_failure_rows']:
            guard()
            with np.load(row['trace']) as z:
                next_rays=torch.tensor(z['next_states'][:,:48]);counts=torch.tensor(z['native_counts'].astype(np.int64));collision=torch.tensor(z['end_flags'][:,0].astype(bool))
            with torch.no_grad():
                error=rnd.error(next_rays);bonus=rnd.bonus(next_rays,counts,collision,plan['physics_cap'],scale,plan['intrinsic_episode_cap'])
                assert float(bonus.sum())<=plan['intrinsic_episode_cap']+1e-8 and bool((bonus[collision]==0).all())
            left.append({'id':row['id'],'observations':len(next_rays),'RND_error':stats(error),'raw_intrinsic_sum':float(bonus.sum()),
                         'collision_bonus_sum':float(bonus[collision].sum()),'known_navigation_success':row['success'],
                         'predictor_training_familiar_episode':row['id'] in familiar_ids})
        # Explicit malformed/over-budget reward contract, not just a nominal trajectory.
        try:rnd.bonus(seen[:1],torch.tensor([101]),torch.tensor([False]),23334,scale)
        except ValueError:pass
        else:raise AssertionError('Oversized action duration accepted')
        cap_probe=rnd.bonus(seen[:1].repeat(234,1),torch.tensor([100]*233+[34]),torch.zeros(234,dtype=torch.bool),23334,1e-12)
        assert abs(float(cap_probe.sum())-.05)<1e-7
        gates['collision_mask_and_episode_cap']=True
        checkpoint=OUT/'rnd_preflight.pt';guard();torch.save({'RND':rnd.state_dict(),'protocol_sha256':sha(HERE/'protocol.json'),
            'training_scope':'offline archived rays; no navigation Actor updates','gates':gates,'automatic_online_enable':False},checkpoint)
        assert checkpoint.stat().st_size<512*1024
        result={'state':'RND_PREFLIGHT_PASS' if all(gates.values()) else 'RND_PREFLIGHT_REJECT','gates':gates,
                'train_n':len(train),'heldout_n':len(held),'train_tasks':plan['train_tasks'],'heldout_tasks':plan['heldout_tasks'],'familiar_left_episode_ids':familiar_ids,
                'initial_seen_error':float(initial),'seen_error':stats(seen_error),'heldout_error':stats(held_error),'heldout_far_error':stats(far_error),
                'far_seen_median_ratio':ratio,'seen_noise_error':stats(noise_error),'positive_noise_increment':stats(increment),
                'normalization_std_floor':.05,'training_curve':losses,'left_failure_corridor':left,
                'checkpoint':{'path':str(checkpoint),'sha256':sha(checkpoint)},'target_sha256_before_after':target_hash,
                'scope':'Same-scene archived-view sanity checks, not true OOD ground truth, real-noise validation, improved exploration or GOAL. RND checkpoint cannot auto-enable online training.'}
    except BaseException:
        result={'state':'FAILED','traceback':traceback.format_exc()}
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    result.update(physics_steps=0,navigation_Actor_updates=0,navigation_Value_updates=0,RND_predictor_updates=updates,
                  elapsed_s=time.monotonic()-start,own_bytes=used(OUT),all_new_runs_MiB=used(ROOT/'stage/runs')/1024**2,
                  input_hashes_unchanged=True,new_online_navigation_started=False)
    assert used(OUT)<plan['final_bytes'] and used(ROOT/'stage/runs')<1024**3
    save(OUT/'summary.json',result);save(OUT/'status.json',result);print(json.dumps(result))
if __name__=='__main__':main()
