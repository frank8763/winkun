from pathlib import Path
import json,hashlib,ast,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t288_five_metre_presentation'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()


def main():
    assert not OUT.exists() and not (HERE/'protocol.json').exists()
    old=json.loads((ROOT/'stage/experiments/t287_time_aligned_credit/protocol.json').read_text());rows=[json.loads(l) for l in (ROOT/'stage/runs/t255_overnight/s2_reset_development.jsonl').read_text().splitlines()]
    references={side:next(r for r in rows if r['source_reset_id']==key) for side,key in [('LEFT','dev_task6_13'),('RIGHT','dev_task7_13')]}
    assert all(r['success'] and not r['collision'] and r['complete'] for r in references.values())
    p=dict(experiment='T288',purpose='User requested two clear downloadable five-metre MuJoCo presentation videos, no training or distance expansion',
        checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],policy_hash=old['policy_hash'],initial_base=old['initial_base'],references=references,
        max_commands=2,max_physics=46668,wall_seconds=1800,movie_bytes=14*1024**2,episode_bytes=384*1024,poster_bytes=512*1024,
        payload_bytes=28*1024**2+2*384*1024+4*512*1024,metadata_bytes=1024**2,final_bytes=32*1024**2,global_bytes=1024**3,min_free_bytes=40*1024**3,
        width=1920,height=1080,fps=25,native_steps_between_frames=20,physics_dt_s=.002,control_hold_native_steps=100,
        intro_pause_frames=20,goal_pause_frames=40,optimizer_calls=0,navigation_training_updates=0,
        contract='Two fixed original5m LEFT/RIGHT center known-success reproductions, safeT255s2reset frozen120singleActorhiddenreset, unchanged originalphysics/action/reward/cap/radius. Native2commands46668steps1800s32MiBglobal1GiBfree40GiB.25fps1920x1080H264yuv420pfaststart, every20nativeframes40ms, .8s intro and1.6s outcome pauses explicit. Actual native fresh full10arrays120statesnextstates+integration hashes mustmatchT255 references. Renderers use copied MjModel and copied MjData only, resized visual buffers and lights on copy; perframe originalintegration unchanged asserted. Overview/followcamera, retrospectivepath andgoal/startmarkers renderer-only no Actoraccess. No frameupscaling/generatedimagery/newseedsearch oroldfailedrunrestart. FullframeOpenCVdecode QC replacesunavailableffprobe. Newexactreproductions notstatisticallyindependentnewgeneralization orlongdistancepass. Sourcescommitted andremoteexactbeforemain; allraw/modelhashes/failurespreserved. Endaudit0native0optimizer andRAMfullremote backup, individualMP4Releaseassetsfullbinaryreadback for userdownload. NoPhase/planner/waypoint/teacher/router.')
    paths=[*HERE.glob('*.py'),ROOT/'stage/runs/t255_overnight/s2_reset_development.jsonl',ROOT/'stage/docs/long_distance_20261101/FIVE_METRE_BASELINE.md',ROOT/'stage/docs/overnight_20261031/postrun_audit.json']
    paths.extend(Path(r['path']) for r in references.values());paths.extend(Path(r['job']['exact_snapshot']) for r in references.values())
    p['input_hashes']=dict(old['input_hashes']);p['input_hashes'].update({str(f):sha(f) for f in paths})
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    assert p['payload_bytes']+p['metadata_bytes']<=p['final_bytes']
    used=sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file());assert used+p['final_bytes']<p['global_bytes'] and shutil.disk_usage(ROOT).free>=p['min_free_bytes']
    OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,indent=2)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    print(json.dumps(dict(state='REGISTERED',global_if_full=used+p['final_bytes'],physics_max=p['max_physics'],own_max=p['final_bytes'])))


if __name__=='__main__':main()
