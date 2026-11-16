"""Zero-native preflight and registration. No optimizer is ever constructed."""
from pathlib import Path
import copy
import datetime
import json
import sys
import time
import t579_runtime as R

ROOT,HERE,OUT,DOC=R.ROOT,R.HERE,R.OUT,R.DOC
read,save,sha,torch,np=R.read,R.save,R.sha,R.torch,R.np


def prepare():
    assert not OUT.exists()
    DOC.mkdir(parents=True,exist_ok=True)
    old=read(ROOT/'stage/runs/t577_offline_recovery_night/protocol.json')
    p=copy.deepcopy(old)
    for k in ['stress_chunks','inputs','protected_files','reference_rows','memory_rows',
              'bootstrap_correction','correction_rows','keep_rows','repair_row','repair_rows',
              'neighbor_rows','baseline_rows','old_rows','final_jobs','imported_rows']:
        p.pop(k,None)
    draft=read(ROOT/'stage/runs/t578_offline_timeout_review/summary.json')['next_diagnostic_draft']
    current=read(ROOT/'stage/runs/t577_offline_recovery_night/final407/summary.json')['result']['model']
    models=[dict(old['neural_teacher'],label='T550'),dict(old['static_model'],label='T576_INITIAL'),
            dict(current,label='T577_FINAL')]
    # Compact checkpoint loader requires the exact static-model metadata dictionary.
    end=datetime.datetime.fromisoformat('2026-11-16T12:00:00+08:00').timestamp()
    native=datetime.datetime.fromisoformat('2026-11-16T11:30:00+08:00').timestamp()
    p.update(id='T579_FROZEN_PROGRESS_ATTRIBUTION',out=str(OUT),jobs=draft['jobs'],models=models,
        max_physics=draft['max_physics'],max_commands=40,max_bytes=26*2**20,child_bytes=26*2**20,
        full_bytes=1280*1024,compact_bytes=12*1024,main_seconds=1800,learning_seconds=0,
        audit_seconds=600,transport_seconds=300,source_seconds=180,
        absolute_learning_stop=time.time(),absolute_native_stop=native,absolute_all_stop=end,
        scientific_scope='18fixed autonomous full traces +18exact; three frozen models,0optimizer/0teacher',
        user_authorization='2026-11-16 user: 下一轮直接开始吧; execute the presented6task/3model bounded diagnostic',
        authorization=dict(new_physics_max=2400000,commands_max=40,own_bytes=26*2**20,all_stop='2026-11-16 12:00+08:00'),
        inputs={},protected_files={},historical_rows=[])
    assert time.time()+1800+600+300+180+60<end and time.time()+1800<native
    assert 6*sum(j['case']['step_cap'] for j in p['jobs'])==p['max_physics']<=2400000
    hist=[]
    for d in ['t550_offline_obstacle_relay','t577_offline_recovery_night']:
        hist+=R.G.rows(ROOT/'stage/runs'/d/'final407')
    p['historical_rows']=[r for r in hist if r['job_id'] in {j['id'] for j in p['jobs']}]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    actual=R.E.N.S.arrays(old['bootstrap_correction']['path'])['states'][:16]
    details=[]
    for m in models:
        assert sha(m['path'])==m['sha256']
        q=R.L.frozen(p,m['path']);q.sample=False;q.reset_noise();assert q.phash()==m['policy_hash']
        q.requires_grad_(False);before=R.digest(q.state_dict());h=None;loc=[]
        with torch.no_grad():
            for x in actual:
                _,v,h,_=q.step(torch.tensor(x[None]),h);loc.append(v[0].numpy().copy())
            err=float(np.max(np.abs(q.sequence(torch.tensor(actual)).numpy()-np.asarray(loc))))
        assert err<1e-9 and before==R.digest(q.state_dict())
        details.append(dict(model=m['label'],policy_hash=q.phash(),sequence_step_error=err,parameters_unchanged=True))
    # Verify the actual native codec's worst-length schema before any worker exists.
    full=R.E.N.S.arrays(ROOT/'stage/runs/t577_offline_recovery_night/rl_04_round/episodes/full_0.npz')
    assert len(full['states'])==838
    temp=DOC/'discarded_codec_check.npz'
    R.E.N.write_npz(temp,full,p['full_bytes']);codec_bytes=temp.stat().st_size
    decoded=R.E.N.S.arrays(temp)
    assert all(R.digest(v)==R.digest(decoded[k]) for k,v in full.items())
    temp.unlink()
    upper=18*p['full_bytes']+3*2**20
    assert upper<=p['max_bytes']
    assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):
            p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),ROOT/'stage/docs/day_20261116/t578/PRIMARY_SOURCES.md',
              ROOT/'stage/runs/t578_offline_timeout_review/summary.json',Path(p['actor_configuration'])]:
        p['inputs'][str(f)]=sha(f)
    for m in models:
        p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256']
    for j in p['jobs']:
        f=j['exact_snapshot'];assert Path(f).exists();p['inputs'][f]=sha(f)
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',models=details,
        codec_838_decisions_roundtrip_exact=True,actual_codec_bytes=codec_bytes,
        full_bytes=p['full_bytes'],storage_upper=upper,own_bytes=p['max_bytes'],
        global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072,new_native=0,new_Adam=0))
    p['inputs'][str(DOC/'PREFLIGHT.json')]=sha(DOC/'PREFLIGHT.json')
    p['inputs'][str(DOC/'PLAN.md')]=sha(DOC/'PLAN.md')
    OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob)
    R.verify(p,True)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics_max=p['max_physics'],commands_max=40,
        unique=18,exact=18,new_Adam=0,main_seconds=1800,audit_seconds=600,own_bytes=p['max_bytes'],
        all_stop='2026-11-16 12:00+08:00',full_worst_admission=True))
    print('T579_PREFLIGHT_AND_REGISTRATION_PASS',flush=True)


if __name__=='__main__':prepare()
