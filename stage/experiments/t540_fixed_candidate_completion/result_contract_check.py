"""Read-only real-record regression for the missing-job_id integration bug."""
from pathlib import Path
import copy,json,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t540_runtime as R
sys.path.insert(0,str(ROOT/'stage/experiments/common'));from native_case_record import envelope
def main():
    p=R.configuration(R.read(R.OUT/'protocol.json'));source=ROOT/'stage/runs/t508_frozen_completion/pilot/cases.jsonl';rows=R.G.rows(source.parent);results=[]
    for row in rows:
        if row['job_id'] not in ['t502_field_sector1_offset1','t502_field_sector0_offset2']:continue
        a=R.E.N.S.arrays(row['path']);assert R.sha(row['path'])==row['sha256'];raw={k:v for k,v in row.items() if k not in ['job_id','request','role','path','sha256','field_detail']};assert 'job_id' not in raw
        try:R.F.detail(p,raw,a)
        except KeyError as e:assert e.args==('job_id',)
        else:raise AssertionError('Fixture failed to reproduce missing request ID')
        job=row['request']['job'];fixed=envelope(job,raw,a,p['field_limit_m']);rich=R.F.detail(p,fixed,a)
        assert fixed['field_detail']['field_qualified_success']==rich['field_qualified_success'] and fixed['field_detail']['field_boundary_crossed']==rich['field_boundary_crossed']
        assert raw.get('job_id') is None and row['job_id']==job['id']
        conflicting=copy.deepcopy(raw);conflicting['job_id']='wrong_job'
        try:envelope(job,conflicting,a,p['field_limit_m'])
        except AssertionError:pass
        else:raise AssertionError('Conflicting task identity was accepted')
        wrong=copy.deepcopy(raw);wrong['success']=not wrong['success']
        try:envelope(job,wrong,a,p['field_limit_m'])
        except AssertionError:pass
        else:raise AssertionError('Outcome/array inconsistency was accepted')
        results.append(dict(job_id=job['id'],source_path=row['path'],source_sha256=row['sha256'],original_missing_id_reproduced=True,fixed_envelope_matches_legacy_qualified_metrics=True,conflicting_ID_and_outcome_rejected=True))
    assert len(results)==2
    dest=ROOT/'stage/docs/day_20261113/T540_RESULT_CONTRACT_CHECK.json';assert not dest.exists()
    R.save(dest,dict(state='PASS_REAL_RECORD_REGRESSION_ONLY',native=0,optimizer=0,source_script_sha256=R.sha(__file__),adapter_sha256=R.sha(ROOT/'stage/experiments/common/native_case_record.py'),results=results,current_T540_native_arrays_NOT_recovered=True,new_navigation_results=None,ended_runtime_source_unchanged=True,epoch=time.time()));print('T540_RESULT_ENVELOPE_REGRESSION_PASS_ZERO_NATIVE',flush=True)
if __name__=='__main__':main()
