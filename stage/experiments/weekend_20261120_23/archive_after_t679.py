"""One bounded RAM-only evidence archive, only between scientific jobs."""
from pathlib import Path
import importlib.util,json,os,signal,sys,time,subprocess
ROOT=Path(__file__).resolve().parents[3];DOC=ROOT/'stage/docs/weekend_20261120_23'
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
spec=importlib.util.spec_from_file_location('_weekend_source_t654_transport',ROOT/'stage/experiments/t654_verified_deviation_recovery/transport.py')
Client=importlib.util.module_from_spec(spec);spec.loader.exec_module(Client)

def main():
    start=time.time();auth=json.loads((DOC/'AUTHORIZATION.json').read_text());ledger=json.loads((DOC/'LEDGER.json').read_text())
    assert auth['approved'] and ledger['transport_seconds']+180<auth['max_transport_seconds']
    deadline=min(start+175,auth['absolute_all_stop']-30)
    def stopped():
        for name in ['t665_remaining_verification','t666_failure_full_history','t667_causal_response_gate','t668_observed_departure_recovery','t669_old_task_onpolicy_signal','t670_outcome_gradient_gate','t671_joint_outcome_feasibility','t672_full_suite_onpolicy','t673_all_task_signal_gate','t674_projection_feasibility','t675_progress_measurement','t676_boundary_signal','t677_reference_target_contract','t678_reference_target_pilot','t679_episode_parameter_exploration']:
            folder=ROOT/'stage/runs'/name
            s=json.loads((folder/'SEALED.json').read_text());assert s['all_scientific_processes_stopped']
            assert all(not Path('/proc',str(pid)).exists() for pid in s['scientific_pids'])
        for unit in ['stage-t665','stage-t666','stage-t667','stage-t668','stage-t669','stage-t670','stage-t671','stage-t672','stage-t673','stage-t674','stage-t675','stage-t676','stage-t677','stage-t678','stage-t679','stage-t680']:
            assert subprocess.check_output(['systemctl','--user','show',unit,'--property=MainPID','--value'],text=True).strip()=='0'
    stopped();Client.A.stopped=stopped
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded archive180s')));signal.alarm(180)
    client=Client.client();release=json.loads(client.req('GET','https://api.github.com/repos/frank8763/winkun/releases/tags/stage-eight-only-2026-11-11'))
    source=ROOT/'stage/runs/t665_remaining_verification';files=[]
    verified=set()
    for manifest in (ROOT/'stage/docs/long_distance_20261101').glob('source_manifest_WEEKEND_T665_PRIMARY_EVIDENCE_*.json'):
        receipt=manifest.with_name(manifest.name.replace('source_manifest_','release_receipt_'))
        if receipt.exists() and json.loads(receipt.read_text()).get('state')=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED':
            verified.update(x['path'] for x in json.loads(manifest.read_text())['files'])
    for f in sorted(source.glob('*/episodes/*.npz')):
        if str(f.relative_to(ROOT)) in verified:continue
        if sum(x.stat().st_size for x in files)+f.stat().st_size>8*1024**2:break
        files.append(f)
    assert len(files)>2 and sum(x.stat().st_size for x in files)<=8*1024**2
    r=Client.A.archive('WEEKEND_T665_PRIMARY_EVIDENCE_3',files,release,deadline,client)
    assert r['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    Client.A.stopped=stopped;stopped();signal.alarm(0)
    report=dict(state=r['state'],archive_group=r['group'],members=r['members'],compressed_bytes=r['bytes'],elapsed_s=time.time()-start,deleted=0,scope='One <=8MiB T665 evidence batch only; other raw remains pending')
    (DOC/'T665_FOURTH_RAW_BACKUP.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    start=time.time()
    try:main()
    finally:
        p=DOC/'LEDGER.json';v=json.loads(p.read_text());v['transport_seconds']+=time.time()-start;p.write_text(json.dumps(v,indent=2)+'\n')
