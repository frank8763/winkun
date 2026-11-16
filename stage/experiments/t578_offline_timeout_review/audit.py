"""Independent integer accounting and source-integrity check; no simulation/import chain."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'stage/runs/t578_offline_timeout_review'


def load(p):
    return json.loads(Path(p).read_text())


def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    assert not (OUT/'audit.json').exists()
    protocol, result = load(OUT/'protocol.json'), load(OUT/'summary.json')
    for path, expected in protocol['inputs'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
    for suite, directory in [('reference','t550_offline_obstacle_relay'),
                             ('current','t577_offline_recovery_night')]:
        base = ROOT/'stage/runs'/directory/'final407'
        rr = lines(base/'cases.jsonl')
        model = load(base/'summary.json')['result']['model']
        assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest() == model['sha256']
        assert len(rr) == 407 and len({r['job_id'] for r in rr}) == 407
        for row in rr:
            assert row['model_sha256'] == model['sha256']
            assert row['policy_hash'] == model['policy_hash']
        for name, lo, hi in [('old275',0,275),('six54',275,329),
                             ('eight54',329,383),('field24',383,407)]:
            rows = rr[lo:hi]; expected=result['suites'][name][suite]
            assert expected['n']==len(rows)
            assert expected['success']==sum(bool(r['success'] and not r['collision'] and
                r['field_detail']['field_qualified_success']) for r in rows)
            assert expected['contact']==sum(bool(r['collision']) for r in rows)
            assert expected['timeout']==sum(bool(r['timeout']) for r in rows)
    old=lines(ROOT/'stage/runs/t550_offline_obstacle_relay/final407/cases.jsonl')
    new=lines(ROOT/'stage/runs/t577_offline_recovery_night/final407/cases.jsonl')
    for a,b in zip(old,new):
        assert a['job_id']==b['job_id'] and a['request']['job']==b['request']['job']
        for key in ['initial_native_integration','scene_sha256','task_goal_xy']:
            assert a['array_hashes'][key]==b['array_hashes'][key]
    assert len(result['changed_outcomes'])==sum(a['success']!=b['success'] for a,b in zip(old,new))
    assert len(result['full_trace_descriptions'])==30
    assert result['next_diagnostic_draft']['executed'] is False
    assert result['next_diagnostic_draft']['authorized'] is False
    assert all(result[k]==0 for k in ['new_native','new_optimizer','new_policy_inference'])
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
              for p in OUT.iterdir() if p.is_file()}
    value=dict(state='PASS', new_native=0,new_optimizer=0,new_policy_inference=0,
        raw_integer_counts_recomputed=True, all407_matched_jobs_initials_goals_scenes=True,
        current_and_reference_model_files_and_all_row_hashes_verified=True,
        all_analysis_input_hashes_rechecked=True,
        trajectory_metrics_are_descriptions_not_causal_classification=True,
        no_claim_of_fix_or_complete_navigation=True,files=manifest)
    with (OUT/'audit.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print('T578_OFFLINE_AUDIT_PASS; native=0; optimizer=0; no simulation authorized')


if __name__=='__main__':main()
