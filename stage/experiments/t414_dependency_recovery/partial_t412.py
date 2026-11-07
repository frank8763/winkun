"""Stopped T412 failure: independent zero-step evidence audit and RAM backup.

Never relabels the FAILED producer, changes its scientific protocol, creates a
navigation worker, or performs any optimizer/backward call.
"""
from pathlib import Path
import hashlib,importlib.util,io,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t412_closed_expert_likelihood'
DOC=ROOT/'stage/docs/day_20261107'
sys.path.insert(0,str(ROOT/'stage/experiments/t412_closed_expert_likelihood'))
import t412_learning as E

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def original_pids_stopped():
    ids=set()
    for name in ['launch.json','driver_started.json','main_started.json','workers.json','summary.json']:
        v=read(OUT/name)
        ids.update(v[k] for k in ['pid','parent','launcher_pid'] if type(v.get(k)) is int)
        ids.update(v.get('workers',[]))
    assert all(not Path('/proc',str(pid)).exists() for pid in ids),('Original PID still live or reused',ids)
    return sorted(ids)

def independent():
    assert not (OUT/'partial_audit.json').exists(),'Existing independent partial audit must not be overwritten'
    absent=original_pids_stopped();s=read(OUT/'summary.json');p=read(OUT/'protocol.json')
    assert s['state']=='FAILED' and s['physics']==253273 and s['commands']==18
    immutable={name:sha(OUT/name) for name in ['summary.json','failure.json','protocol.json']}
    producer_manifest=(OUT/'local_manifest.json').read_text();producer_manifest_sha=sha(OUT/'local_manifest.json')
    for row in json.loads(producer_manifest)['files']:
        assert (ROOT/row['path']).stat().st_size==row['bytes'] and sha(ROOT/row['path'])==row['sha256']
    original_audit_exit=None
    if not (OUT/'audit.json').exists():
        log=DOC/'T412_ORIGINAL_FAILED_AUDIT.log'
        with log.open('xb') as stream:
            result=subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(ROOT/'stage/experiments/t412_closed_expert_likelihood/audit.py')],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        original_audit_exit=result.returncode
        assert original_audit_exit==1,'FAILED producer audit must reject completeness'
    audit=read(OUT/'audit.json');assert audit['state']=='FAILED_PRODUCER_PARTIAL_PRESERVED'
    assert audit['native']==audit['historical_discarded_Adam']==0
    original_pids_stopped();rows=[json.loads(x) for x in (OUT/'cases.jsonl').read_text().splitlines()]
    commands=[json.loads(x) for x in (OUT/'commands.jsonl').read_text().splitlines()]
    assert len(rows)==14 and len(commands)==18
    assert [c['id'] for c in commands]==list(range(1,19))
    assert [c['role'] for c in commands[:4]]==['worker_setup']*4
    assert all(c['physics']==0 for c in commands[:4]) and all(c['error'] is None for c in commands)
    assert sum(c['physics'] for c in commands)==sum(r['physics_steps'] for r in rows)==s['physics']
    assert [r['role'] for r in rows]==[c['role'] for c in commands[4:]]
    assert not (OUT/'replays.jsonl').exists() or not (OUT/'replays.jsonl').read_text().strip()
    assert not (OUT/'optimizer_calls.jsonl').exists() or not (OUT/'optimizer_calls.jsonl').read_text().strip()
    assert s['updates']==dict(live=0,shadow=0) and s['rounds']==[]
    E.torch.set_num_threads(1);E.torch.use_deterministic_algorithms(True)
    base=E.N.science(p);q=E.safe_initial(base,p).eval().requires_grad_(False)
    def forbidden(*args,**kwargs):raise AssertionError('Partial audit cannot optimize or backpropagate')
    E.torch.optim.Adam.step=forbidden;E.torch.Tensor.backward=forbidden
    policy=q.phash();assert policy==s['result']['initial']['policy_hash']
    sampling=OUT/'sampling.pt.gz';sampling_sha=sha(sampling)
    assert {r['loaded_model_sha256'] for r in rows}=={sampling_sha}
    stored=E.torch.load(io.BytesIO(E.gzip.decompress(sampling.read_bytes())),weights_only=False,map_location='cpu')
    assert stored['policy_hash']==policy and stored['hashes']['learned']==E.digest(q.learned())
    assert {k:E.digest(v) for k,v in stored['state'].items()}==stored['hashes']
    assert stored['state']['optimizer']['state']=={}
    checked=[]
    for r in rows:
        assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path'])
        assert int(a['native_counts'].sum())==r['physics_steps']==r['physics']
        assert bool(a['end_flags'][-1,1])==r['success'] and bool(a['end_flags'][:,0].any())==r['collision']
        assert bool(a['end_flags'][-1,2])==r['timeout'] and a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
        assert r['complete'] and not r['administrative_censor'] and r['policy_hash']==policy and r['seed'] is None and r['teacher'] is False
        if r['full']:
            assert E.digest(a)==r['array_digest'] and E.digest(a['final_native_integration'])==r['integration_sha256']
            assert E.equal(a['states'][1:],a['next_states'][:-1]) and not a['ownership'].any()
            E.audit_array(q,a,p,0,None)
            checked.append(dict(role=r['role'],sha256=r['sha256'],decisions=len(a['states']),physics=r['physics_steps'],array_digest=r['array_digest']))
    assert len(checked)==8 and all(r['full'] and r['success'] for r in rows[:8])
    assert E.tally(rows[:8])==s['result']['initial']['old8'] and E.tally(rows[8:13])==s['result']['initial']['left5'] and E.tally(rows[13:])==s['result']['initial']['right1']
    assert not list(OUT.rglob('partial*.npz'))
    for name,digest in immutable.items():assert sha(OUT/name)==digest
    result=dict(state='PARTIAL_EVIDENCE_PASS',producer_state='FAILED',producer_completeness_pass=False,
        epoch=time.time(),original_audit_exit=original_audit_exit,original_failed_audit_state=audit['state'],
        all_original_pids_absent=absent,cases=14,commands=18,physics=253273,exact_replays=0,
        full8_actions_hidden_history_density_rewards_arrays_and_integration_exact=checked,
        sampling_model_sha256=sampling_sha,safe_student_policy_hash=policy,empty_live_Adam_state=True,
        live_updates=0,shadow_updates=0,new_native=0,new_Adam=0,independent_optimizer_reconstruction_steps=0,
        initial_old8=s['result']['initial']['old8'],initial_left5=s['result']['initial']['left5'],initial_right1=s['result']['initial']['right1'],
        original_immutable_sha256=immutable,original_producer_manifest_sha256=producer_manifest_sha,
        original_producer_manifest_utf8=producer_manifest,stable_six=False,seven_eight_entered=False,
        missing_evaluations=dict(main54=None,old275=None,fresh16=None,reload78=None),Phase_untouched=True)
    E.save(OUT/'partial_audit.json',result);E.N.manifest(OUT)
    print(json.dumps(dict(state=result['state'],original_audit_exit=original_audit_exit,physics=253273,commands=18,
        full_payloads=8,model_sha256=sampling_sha,manifest_sha256=sha(OUT/'local_manifest.json'))),flush=True)

def backup():
    original_pids_stopped();s=read(OUT/'summary.json');part=read(OUT/'partial_audit.json')
    assert s['state']=='FAILED' and part['state']=='PARTIAL_EVIDENCE_PASS' and not part['producer_completeness_pass']
    for key,digest in part['original_immutable_sha256'].items():assert sha(OUT/key)==digest
    frozen=sha(OUT/'local_manifest.json')
    spec=importlib.util.spec_from_file_location('stage414_partial_curl_transport',ROOT/'stage/experiments/t377_full_human_observation_transfer/transport.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport);transport.install()
    group='T412-FAILED';receipt=transport.B.DOC/('release_receipt_'+group+'.json')
    def stopped(directory):
        assert directory.resolve()==OUT.resolve();original_pids_stopped()
        assert sha(OUT/'local_manifest.json')==frozen
        assert read(OUT/'summary.json')['state']=='FAILED' and read(OUT/'audit.json')['state']=='FAILED_PRODUCER_PARTIAL_PRESERVED'
        assert read(OUT/'partial_audit.json')['state']=='PARTIAL_EVIDENCE_PASS'
        for row in read(OUT/'local_manifest.json')['files']:
            assert (ROOT/row['path']).stat().st_size==row['bytes'] and sha(ROOT/row['path'])==row['sha256']
    stopped(OUT)
    if not receipt.exists():
        transport.B.require_stopped=stopped;transport.B.GROUPS[group]=[OUT.name];transport.B.main(group)
    proof=read(receipt);assert proof['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED' and proof['group']==group
    manifest=transport.B.DOC/('source_manifest_'+group+'.json');assert sha(manifest)==proof['manifest_sha256']
    for row in read(manifest)['files']:assert sha(ROOT/row['path'])==row['sha256']
    stopped(OUT);print(json.dumps(dict(state='FAILED_T412_PARTIAL_AUDITED_AND_ALL_RAW_REMOTE_VERIFIED',
        original_out_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),
        frozen_local_manifest_sha256=frozen,archive_sha256=proof['sha256'],receipt=str(receipt),members=proof['members'],
        producer_state='FAILED',new_native=0,new_Adam=0)),flush=True)

if __name__=='__main__':
    assert sys.argv[1:] in (['audit'],['backup'])
    independent() if sys.argv[1]=='audit' else backup()
