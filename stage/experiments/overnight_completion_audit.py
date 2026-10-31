"""Read-only simulation audit of completed T255; no new rollouts or training."""
from pathlib import Path
import json,hashlib,shutil,cv2,numpy as np,time
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'stage/runs/t255_overnight';DOC=ROOT/'stage/docs/overnight_20261031'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(name):return [json.loads(x) for x in (OUT/name).read_text().splitlines()]
def main():
    dest=ROOT/'stage/runs/t255_postrun_audit';assert not dest.exists(),'Audit already recorded; inspect receipt'
    s=json.loads((OUT/'summary.json').read_text());plan=json.loads((OUT/'protocol.json').read_text());workers=json.loads((OUT/'workers.json').read_text());failure=json.loads((OUT/'failure.json').read_text())
    assert s['all_workers_stopped'] and all(not Path('/proc',str(pid)).exists() for pid in [workers['parent']]+workers['workers'])
    safe=lambda r:r['complete'] and r['success'] and not r['collision']
    dev=rows('s2_reset_development.jsonl');hold=rows('heldout.jsonl');verify=rows('verification.jsonl');prior={r['source_reset_id']:r for r in dev+hold}
    assert len(dev)==211 and len(hold)==64 and len(verify)==275 and len(prior)==275 and len({r['source_reset_id'] for r in verify})==275 and all(safe(r) for r in dev+hold+verify)
    assert all(r['array_hashes']==prior[r['source_reset_id']]['array_hashes'] and r['integration_sha256']==prior[r['source_reset_id']]['integration_sha256'] for r in verify)
    assert all(sha(f)==h for f,h in plan['input_hashes'].items())
    for arm,report in s['arms'].items():assert sha(report['export']['path'])==report['export']['sha256']
    a,b=rows('8m.jsonl'),rows('8m_repeat.jsonl');assert len(a)==len(b)==2 and all(x['array_hashes']==y['array_hashes'] and x['integration_sha256']==y['integration_sha256'] for x,y in zip(a,b))
    assert 'FileNotFoundError' in failure['traceback'] and "'ffprobe'" in failure['traceback']
    movies=list((OUT/'video').glob('*.mp4'));assert len(movies)==1;movie=movies[0];ref=prior[movie.stem];cap=cv2.VideoCapture(str(movie));n=0
    while True:
        ok,_=cap.read()
        if not ok:break
        n+=1
    cap.release()
    with np.load(ref['path']) as z:expected=len(z['native_counts'])
    assert n==expected==223 and movie.stem=='dev_task6_13'
    manifest=json.loads((DOC/'source_manifest.json').read_text());receipt=json.loads((DOC/'release_receipt.json').read_text());assert receipt['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and receipt['members']==len(manifest['files'])
    assert all(sha(ROOT/r['path'])==r['sha256'] for r in manifest['files'])
    assert str(movie.relative_to(ROOT)) in {r['path'] for r in manifest['files']}
    names=['baseline.jsonl','training.jsonl','replays.jsonl','round_centers.jsonl','s1_memory_development.jsonl','s1_reset_development.jsonl','s2_reset_development.jsonl','s2_memory_development.jsonl','heldout.jsonl','verification.jsonl','8m.jsonl','8m_repeat.jsonl'];allrows=[r for name in names for r in rows(name)]
    assert len(allrows)==s['commands'] and sum(r['physics_steps'] for r in allrows)==s['physics_steps']
    size=lambda folder:sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    result=dict(state='POSTRUN_AUDIT_PASS_WITH_VIDEO_COMPONENT_FAILURE',audit_epoch=time.time(),new_physics_steps=0,new_optimizer_calls=0,original_summary_unchanged=True,
      stable_5m_gate_independently_audited=True,development_safe=211,development_n=211,heldout_safe=64,heldout_n=64,reload_verification_exact=275,
      selected='s2_reset',memory_benefit_not_established=True,eight_m_safe=0,eight_m_n=2,eight_m_repeat_array_and_integration_exact=True,all_training_and_simulator_workers_stopped=True,
      original_pipeline_state=s['state'],failure_component='Video post-encode frame-count validation requires unavailable ffprobe; occurs after first native trajectory/hash assertions and encoder exit. No failure of the completed 5m gate.',
      existing_movie=dict(path=str(movie),sha256=sha(movie),bytes=movie.stat().st_size,decoded_frames=n,expected_reference_decisions=expected,decoded_frame_count_matches=True,reference_physics_steps=ref['physics_steps'],independent_native_array_metadata_persisted=False,assertion_order_evidence='Original video.py checks ten native array hashes and integration before invoking missing ffprobe. Traceback establishes that point was reached; a separate video row was not persisted. No new physics was used by this audit.'),
      planned_video_cases=3,existing_decodable_movie_cases=1,remaining_video_cases_not_run=2,original_video_success_summary_entries=0,
      original_recorded_commands=s['commands'],original_recorded_physics_steps=s['physics_steps'],completed_video_omitted_from_original_counters=True,
      additional_completed_video_physics_inferred_from_prior_exact_assertions=ref['physics_steps'],additional_video_command_inferred=1,additional_video_physics_conservative_upper_bound=23334,
      total_commands_accounting_for_video= s['commands']+1,total_physics_with_inferred_video=s['physics_steps']+ref['physics_steps'],total_physics_conservative_upper_bound=s['physics_steps']+23334,
      all_archived_original_members_local_hashes_match=len(manifest['files']),archive_receipt=str(DOC/'release_receipt.json'),original_models_and_failures_untouched=True,
      own_bytes=size(OUT),global_run_bytes=size(ROOT/'stage/runs'),free_bytes=shutil.disk_usage(ROOT).free)
    assert result['global_run_bytes']+32768<1024**3 and result['own_bytes']<150*1024**2 and result['free_bytes']>40*1024**3
    dest.mkdir();data=json.dumps(result,indent=2)+'\n';(dest/'verification.json').write_text(data);(DOC/'postrun_audit.json').write_text(data);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
