from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t227_small_noise_recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T227'
    prev=ROOT/'stage/runs/t225_paired_coverage';s=json.loads((prev/'summary.json').read_text());assert s['state']=='COMPLETE' and s['all_workers_stopped'] and not s['success_source_gate']
    failed=ROOT/'stage/runs/t226_local_exploration';failure=json.loads((failed/'summary.json').read_text());assert failure['state']=='FAILED' and failure['all_workers_stopped'] and failure['commands']==4 and failure['physics_steps']==78268
    original=failed/'LEFT_06_2026103221_candidate_small_sigma.npz';assert original.exists()
    old=json.loads((ROOT/'stage/experiments/t225_paired_coverage/protocol.json').read_text());controls=[r for r in map(json.loads,(prev/'episodes.jsonl').read_text().splitlines()) if r['arm']=='candidate'];assert len(controls)==12
    jobs=[]
    for row in controls:
        j=dict(row['job']);j['id']=j['id']+'_small_sigma';jobs.append(j)
    plan=dict(experiment='T227',candidate_checkpoint=s['export']['path'],candidate_hash=old['candidate_hash'],jobs=jobs,paired_high_sigma_controls=controls,
        original_failed_run=str(failed),original_first_trace=str(original),original_first_trace_sha256=sha(original),prior_failed_commands=4,prior_failed_physics=78268,sigma=[.035,.03],rho=.98,max_commands=24,max_physics=560016,wall_seconds=300,final_bytes=3*1024**2,peak_bytes=3*1024**2,global_bytes=1024**3,
        contract='FrozenT221 only, one-factor nativeexplorationsigma reduced10x (.35,.30)->(.035,.03), same2LEFTreset/sixseeds/AR.98standardGaussianinnovations/sensorpolicy/physicalnative. Existing12highsigmacontrols fromT225 exactlypaired, no repeatcontrolphysics orchangedseeds. New12small-sigmaepisodes eachregeneratedindependentlytwice all10arrays/finalintegrationexact; keepallfailures/successfullNPZ<=128KiB. Zero optimizer/reconstructioncalls (readretainedT221gzipandhashcheck), nonzerohistoryconditionactionnoise noPPOdensityclaim. Fullfixedbudgetcompleted, nobestseedselection/autoextension/restart. RequireatleastonesafeGOALforeachfailedLEFTreset beforeseparateconsolidation; originalTASK/GOAL/scene/timeunchanged, no externalteacher/planner/waypoint/runtimecontrol/Phase. 24commands/560016allphysics/300s/3MiBown/global1GiB/free40GiB. Noautomaticlearning/promotion/nextlaunch.',
        references=['https://arxiv.org/abs/1703.09327','https://arxiv.org/html/2312.11091v2','https://arxiv.org/abs/2306.02437'])
    files=list(HERE.glob('*.py'))+[ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t220_temporal_noise/temporal_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py',Path(plan['candidate_checkpoint'])]+[Path(r['path']) for r in controls]+[Path(j['exact_snapshot']) for j in jobs]
    files += [original,failed/'summary.json',failed/'failure.json']
    plan['recovery_contract']='T226endedFAILEDwithoutupdates afterfirstfile139524bytes exceededincorrect128KiBassertexactly; original file preserved. NewT227 fixedsame12jobs/seeds/sigma only evidenceencoding corrected: encodeinRAMreserve192KiB beforewrite, storelosslessly states+finalnextstate because nextstates[:-1]bitwiseequalstates[1:], fullall10arrayhashes retained and independentlyregenerated; firstsavedT226all10arrays mustmatchfreshfirstepisode. Original4collectedcommands/78268physics separatelyreported, no newindependent samplesclaim fromrepeats, totaloriginal+new<=28commands/638284physics. NeverrestartT226 orcontinuelearning; two completeindependentgenerations ofentirefixedmatrix inthisnewprotocol.'
    plan['input_hashes']={str(f):sha(f) for f in files};size=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert size+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',runs_MiB=size/1024**2)))
if __name__=='__main__':main()
