"""Register one bounded audit of the two remaining Stage failures."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t236_closed_loop_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T236'
    base=json.loads((ROOT/'stage/experiments/t231_retained_behavior_repair/protocol.json').read_text())
    prior=json.loads((ROOT/'stage/runs/t231_retained_behavior_repair/summary.json').read_text());assert prior['all_workers_stopped'] and prior['development_safeGOAL']==209
    failed=[r for r in map(json.loads,(ROOT/'stage/runs/t231_retained_behavior_repair/development.jsonl').read_text().splitlines()) if r['collision']]
    ids=[r['id'] for r in failed];assert ids==['dev_task6_00','dev_task6_24']
    sources=[r for r in map(json.loads,(ROOT/'stage/runs/t232_complementary_sources/episodes.jsonl').read_text().splitlines()) if r['source_reset_id'] in ids]
    assert len(sources)==2 and all(r['complete'] and r['success'] and not r['collision'] for r in sources)
    plan=dict(experiment='T236',checkpoint=prior['export']['path'],complementary_checkpoint=base['complementary_checkpoint'],jobs=[dict(j,arm='T231',seed=0,mode='deterministic') for j in base['development_jobs'] if j['id'] in ids],prior_failures=failed,successful_sources=sources,
      max_commands=4,max_physics=93336,wall_seconds=180,final_bytes=1024**2,peak_bytes=1024**2,global_bytes=1024**3,
      contract='Exactly two fixed T231 failure resets, each independently generated twice; all10 arrays/finalintegration must exactly match both generations and originalT231. Preserve complete failure traces losslessly. Offline query already-retained T221 on T231-induced states and existing verified T221 source states, report action errors and time-aligned pose divergence; temporal association is NOT proof of causal attribution, not DAgger/RL update. No optimizer/model export/controller switch/intervention, no task routing/runtime teacher/map/planner/waypoint/Phase. No source/seed/window search. Native success radius/time/action-hold unchanged. No hidden heldout navigation. 4commands/93336allphysics/180seconds/1MiBown/global1GiB/free40GiB; preserve failures; no automatic next launch.',
      references=['https://proceedings.mlr.press/v15/ross11a.html','https://arxiv.org/abs/1803.03835','https://arxiv.org/abs/1903.11524'])
    paths=list(HERE.glob('*.py'))+[Path(plan[k]) for k in ['checkpoint','complementary_checkpoint']]+[Path(r['path']) for r in sources]+[Path(j['exact_snapshot']) for j in plan['jobs']]+[ROOT/'stage/experiments/t229_valid_start_acceptance/valid_support.py',ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py']
    plan['input_hashes']={str(p):sha(p) for p in paths};used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',runs_MiB=used/1024**2)))
if __name__=='__main__':main()
