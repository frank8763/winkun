"""Zero-native reconciliation of the preserved joint best versus field branch."""
from pathlib import Path
import gzip
import hashlib
import json
import lzma
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'stage/docs/day_20261118/joint_baseline_reconciliation'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lines(path):
    b=Path(path).read_bytes()
    if str(path).endswith('.xz'):b=lzma.decompress(b)
    if str(path).endswith('.gz'):b=gzip.decompress(b)
    return [json.loads(x) for x in b.decode().splitlines() if x.strip()]


def safe(r):return bool(r['success'] and not r['collision'] and not r.get('field_detail',{}).get('field_boundary_crossed',False))


def main():
    locked=read(ROOT/'stage/runs/t502_fullfield_probe/LOCKED_BEST.json')
    model=locked['current_joint'];assert sha(model['path'])==model['sha256']
    original=lines(ROOT/'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz')
    assert len(original)==383
    fieldpath=next((ROOT/'stage/runs/t502_fullfield_probe/probe').glob('cases.jsonl*'))
    field=lines(fieldpath);assert len(field)==24
    reference=read(ROOT/'stage/runs/t629_offline_night_relay/profile_0/summary.json')['result']['rows']
    assert len(reference)==407;by={r['job_id']:r for r in reference}
    assert all(r['request']['job']==by[r['job_id']]['request']['job'] for r in original)
    assert all(r['request']['job']==by[r['job_id']]['request']['job'] for r in field)
    prior={r['job_id'] for r in original if safe(r)}
    fields={r['job_id'] for r in field if safe(r)}
    current={r['job_id'] for r in reference if safe(r)}
    assert len(prior)==379 and len(fields)==5
    assert len(prior-current)==81 and len((current-prior)-{r['job_id'] for r in field})==0
    assert fields<=current
    def tallies(rows):
        parts={'old275':[],'wide6':[],'wide8':[],'field24':[]}
        for row in rows:
            id=row['job_id']
            key='field24' if id.startswith('t502_field') else 'wide8' if id.startswith('wide117_8m') else 'wide6' if id.startswith('wide117_80') else 'old275'
            parts[key].append(row)
        return {key:dict(tested=len(rr),safe_success=sum(safe(r) for r in rr)) for key,rr in parts.items()}
    result=dict(state='OFFLINE_EVIDENCE_RECONCILED',native_steps=0,optimizer_updates=0,
        T501_model=model,T501_local_bytes_match_locked_original=True,
        T501_source_audit=read(ROOT/'stage/runs/t501_joint_command_repair/joint/audit.json')['state'],
        T501_remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T501.json')['state'],
        T501_historical_full407=tallies(original+field),
        T608_later_full407=tallies(reference),
        all407_job_definitions_exactly_equal=True,
        T501_original_success_ids=sorted(prior),T501_field_success_ids=sorted(fields),
        T608_additional_field_success_ids=sorted(current-{r['job_id'] for r in original}-fields),
        old_success_ids_lost_in_T608=sorted(prior-current),
        retained_historical_global_successes=len(prior|fields),
        T608_safe_successes=len(current),
        desired_union_ids=sorted(prior|fields|current),
        desired_union_size=len(prior|fields|current),
        union_not_proven_achievable_by_one_current_model=True,
        no_new_native_reproduction_claimed=True,
        no_runtime_routing_or_model_merging_performed=True,
        source_SHA={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'stage/runs/t502_fullfield_probe/LOCKED_BEST.json',
            ROOT/'stage/runs/t501_joint_command_repair/joint/summary.json',
            ROOT/'stage/runs/t501_joint_command_repair/joint/cases.jsonl.t595.xz',
            ROOT/'stage/runs/t629_offline_night_relay/profile_0/summary.json',fieldpath]})
    DOC.mkdir(parents=True,exist_ok=True);(DOC/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['state','T501_local_bytes_match_locked_original','T501_historical_full407','T608_later_full407','desired_union_size','native_steps','optimizer_updates']}))


if __name__=='__main__':main()
