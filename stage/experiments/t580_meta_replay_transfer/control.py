"""Prepare one finite meta-update comparison, with an effective-step control."""
from pathlib import Path
import copy
import datetime
import json
import sys
import time
import t580_runtime as R

ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np


def prepare():
    assert not OUT.exists();DOC.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'stage/runs/t579_frozen_progress_attribution'
    assert read(previous/'audit.json')['state']=='PASS'
    assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T579.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert read(previous/'SEALED.json')['all_scientific_processes_stopped']
    assert all(not Path('/proc',str(i)).exists() for i in read(previous/'SEALED.json')['scientific_pids'])
    source=read(ROOT/'stage/runs/t577_offline_recovery_night/protocol.json')
    report=read(ROOT/'stage/runs/t577_offline_recovery_night/final407/summary.json')['result']
    p=copy.deepcopy(read(previous/'protocol.json'))
    for k in ['historical_rows','models','inputs','protected_files']:p.pop(k,None)
    raw=R.G.rows(previous);rep=R.G.rows(previous,'replays.jsonl')
    repair=[r for r in raw if r['role']=='FROZEN_T550' and r['success'] and not r['collision']]
    assert len(repair)==4 and all(any(x['reference_path']==r['path'] and x['exact'] for x in rep) for r in repair)
    jobs=source['evaluation_jobs']+[j for j in p['jobs'] if j['id'] not in {x['id'] for x in source['evaluation_jobs']}]
    keys=[r['job_id'] for r in repair];assert len(jobs)==37 and len(set(keys))==4
    cap=3*(sum(j['case']['step_cap'] for j in jobs)+sum(j['case']['step_cap'] for j in jobs if j['id'] in keys))
    end=datetime.datetime.fromisoformat('2026-11-16T12:00:00+08:00').timestamp()
    native=datetime.datetime.fromisoformat('2026-11-16T11:30:00+08:00').timestamp()
    learning=datetime.datetime.fromisoformat('2026-11-16T10:45:00+08:00').timestamp()
    p.update(id='T580_MATCHED_META_REPLAY_SUPERVISED_TRANSFER',out=str(OUT),initial=report['model'],
        static_model=source['static_model'],keep_rows=source['reference_rows'],repair_rows=repair,
        arms=['ER','MER','ER_MATCHED_STEP'],blocks=32,inner_steps=8,rows_per_chunk=64,
        batch_seed=202611160580,inner_lr=.01,meta_gamma=.3,
        evaluation_jobs=jobs,key_replay_ids=keys,initial_success_ids=source['reference23_success_ids'],
        max_physics=cap,max_commands=130,max_bytes=26*2**20,child_bytes=26*2**20,
        full_bytes=1280*1024,compact_bytes=12*1024,model_bytes=256*1024,
        main_seconds=3000,learning_seconds=600,audit_seconds=600,transport_seconds=300,source_seconds=180,
        absolute_learning_stop=learning,absolute_native_stop=native,absolute_all_stop=end,
        new_Adam=0,new_RL=0,live_SGD=768,shadow_SGD=768,audit_discarded_SGD=768,
        live_meta_interpolations=32,shadow_meta_interpolations=32,audit_meta_interpolations=32,
        preflight_discarded_SGD=112,inputs={},protected_files={},
        count_contract='Only explicit T580 counts above are operative; inherited configuration is loading compatibility.',
        scope='Three same-data256SGD arms:alpha.01 ordinaryER,MERalpha.01gamma.3,ERalpha.003 firstorderstepcontrol; no new PPO/critic.',
        user_authorization='2026-11-16 user: 检查一下，然后开始下一步的实验，就开始元学习试试',
        independent_budget_not_T579_extension=True)
    assert cap==5885814
    assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+40<end
    assert time.time()+p['main_seconds']<native and time.time()+600<learning
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=L.dataset(p);assert len(data)==27
    initial=L.make(p)[0];before=L.measure(initial,data)
    assert max(x['location_RMS'] for x in before if x['kind']=='keep')==0
    schedule=L.schedule(data,p);results=[];total=0
    for arm in p['arms']:
        a,b=L.make(p),L.make(p);start=L.state(a)
        u=L.fit(a,data,p,arm,lambda:None,lambda *_:None,blocks=2)
        v=L.fit(b,data,p,arm,lambda:None,lambda *_:None,blocks=2)
        assert u==v and digest(L.state(a))==digest(L.state(b));total+=32
        L.load(a,start);assert digest(L.state(a))==digest(start)
        results.append(dict(arm=arm,SGD_discarded=32,live_shadow_exact=True,whole_model_SGD_rollback_exact=True))
    a,b=L.make(p),L.make(p)
    L.fit(a,data,p,'ER',lambda:None,lambda *_:None,blocks=1)
    L.fit(b,data,p,'MER',lambda:None,lambda *_:None,blocks=1,gamma_override=1.)
    err=max(float((a[0].state_dict()[k]-v).abs().max()) for k,v in b[0].state_dict().items())
    assert err<1e-12;total+=16;assert total==112
    temp=DOC/'discarded_checkpoint.pt.gz';m=L.export(a,p,temp)
    restored=L.frozen(p,temp);assert restored.phash()==a[0].phash();size=m['bytes'];temp.unlink()
    upper=12*p['full_bytes']+99*p['compact_bytes']+3*p['model_bytes']+8*2**20
    assert upper<=p['max_bytes'] and R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
    save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE',new_native=0,new_Adam=0,
        discarded_SGD=112,arms=results,gamma1_same_SGD_parameter_max_error=err,
        compact_SGD_checkpoint_readback=True,checkpoint_bytes=size,initial_keep_loss_exact_zero=True,
        actual_history_feature_sequence_match=True,schedule_digest=digest(schedule),
        data_history_count=27,keep_static_history_count=23,verified_T550_new_success_count=4,
        own_bytes=p['max_bytes'],storage_upper=upper,global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072))
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):
            p['inputs'][str(Path(f).resolve())]=sha(f)
    for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',
              ROOT/'stage/docs/day_20261116/CONTINUAL_VS_META.md',Path(p['actor_configuration']),
              previous/'audit.json',previous/'summary.json']+[Path(r['path']) for r in p['keep_rows']+repair]:
        p['inputs'][str(f)]=sha(f)
    for model in [p['initial'],p['static_model'],source['neural_teacher']]:
        assert sha(model['path'])==model['sha256'];p['inputs'][model['path']]=model['sha256'];p['protected_files'][model['path']]=model['sha256']
    for j in jobs:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
    OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n'
    (HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
    save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',new_native_max=cap,max_commands=130,
        actual_worst_commands=127,live_SGD=768,shadow_SGD=768,audit_discarded_SGD=768,
        preflight_discarded_SGD=112,new_Adam=0,new_RL=0,own_bytes=p['max_bytes'],
        all_stop='2026-11-16 12:00+08:00',learning_stop='10:45',native_stop='11:30',
        main_seconds=3000,audit_seconds=600,independent_new_mechanism_budget=True))
    print('T580_ZERO_NATIVE_PREFLIGHT_AND_REGISTRATION_PASS',flush=True)


if __name__=='__main__':prepare()
