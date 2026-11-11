"""Register/launch the unstarted portion of T481, original combined caps."""
from pathlib import Path
import ast,copy,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t484_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT=T.OUT;DOC=ROOT/'stage/docs/review_20261111';OLD=ROOT/'stage/runs/t481_fresh16_rehearsal'

def prepare():
    assert not OUT.exists()
    for f in ['launch.json','driver_started.json']:
        assert not Path('/proc',str(read(OLD/f)['pid'])).exists()
    report=read(DOC/'T481_FINAL_REPORT.json');cap=read(OLD/'capture/summary.json');review=read(DOC/'T484_CAPTURE_AUDIT.json')
    assert report['state']=='BOUNDED_QUEUE_STOPPED' and report['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert review['state']=='PASS' and review['native']==review['historical_discarded_Actor_and_Value_Adam']==0
    assert cap['state']=='COMPLETE' and len(cap['result']['samples'])==16
    old=read(OLD/'protocol.json');p=copy.deepcopy(old);end=read(OLD/'launch.json')['all_stop']
    assert time.time()+5400<end
    previous_physics=report['physics'];previous_commands=report['commands']
    pp=12000000-previous_physics;cc=500-previous_commands
    worst=sum(j['case']['step_cap'] for j in p['wide8_jobs']+p['six_jobs']+p['old275_jobs'])
    assert worst<=pp and 387<=cc and 3600+cap['elapsed_s']<=4500
    p.update(id='T484_T481_SYMBOL_ONLY_UNSTARTED_LEARNING_RECOVERY',out=str(OUT),
        capture_summary_path=str(OLD/'capture/summary.json'),capture_audit_path=str(DOC/'T484_CAPTURE_AUDIT.json'),
        max_bytes=10*2**20,metadata_reserved_bytes=4*2**20,max_physics=pp,root_max_physics=pp,
        max_commands=cc,root_max_commands=cc,main_seconds=3600,root_max_main_seconds=3600,
        learning_seconds=1200,audit_seconds=900,all_seconds=5400,absolute_all_stop=end,
        previous_physics=previous_physics,previous_commands=previous_commands,previous_main_seconds=cap['elapsed_s'],
        preflight_commands=0,remote_cache_required=False,
        user_authorization='Nov11 check; stopped then new independent recovery. Same T481 total12M/500cmd/4500main and original actual15:56:56allstop, no recollection or extra epoch/seed budget.',
        mechanism='Exact same T481 fit/batch/epoch function objects. Re-export existing T480.vtrace for audit interface only. Not new algorithm or performance evidence.')
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(OLD/'protocol.json'),str(OLD/'capture/summary.json'),str(DOC/'T481_FINAL_REPORT.json'),
        str(DOC/'T484_CAPTURE_AUDIT.json'),str(DOC/'T484_CAPTURE_REVIEW_PROVENANCE.json'),str(DOC/'T484_PLAN.md'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T481.json')}
    files|={r['path'] for r in cap['result']['samples']+cap['result']['replay_rows']}
    p['inputs']={f:sha(f) for f in sorted(files)};p['constructor_reads']=sorted(files)
    assert M.used()+p['max_bytes']+512*1024<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert 383*p['compact_bytes']+p['model_bytes']+p['metadata_reserved_bytes']<=p['max_bytes']
    for mode in ['fit','batch','epoch']:assert getattr(L,mode) is getattr(L.F,mode)
    assert L.vtrace is L.F.B.vtrace
    attrs={n.attr for n in ast.walk(ast.parse((HERE/'t484_runtime.py').read_text())) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='L'}
    assert all(hasattr(L,n) for n in attrs)
    assert len(json.dumps(p).encode())<2**21
    OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    save(OUT/'preflight.json',dict(state='PASS_EXACT_FUNCTION_IDENTITY_AND_ORIGINAL_PREFLIGHT_REUSED',native=0,commands=0,
        discarded_Adam=0,original_preflight_sha256=sha(OLD/'preflight.json'),original_eight_discarded_Adam_NOT_repeated=True,
        same_frozen_fit_batch_epoch_objects=True,all_runtime_audit_symbols_present=True,complete_actual_capture_review_PASS=True))
    save(DOC/'T484_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',new_max_physics=pp,worst_physics=worst,
        new_max_commands=cc,worst_commands=387,new_main_seconds=3600,shared_T481_T484_physics=12000000,
        shared_commands=500,shared_main_seconds=4500,new_own_bytes=p['max_bytes'],all_stop=end,
        new_preflight_Adam=0,live_max_Adam=32,shadow_max_Adam=32,audit_historical_discarded_max_Adam=32,
        old_collected16_PLUS16_NOT_REPEATED=True,fresh16=None,reload78=None))
    print('T484_REGISTERED_WITH_REMAINING_ORIGINAL_BUDGET',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True)
    assert not (OUT/'launch.json').exists() and read(OUT/'preflight.json')['native']==0
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+5400<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T484','--why=Recover unstarted T481 learning under same original caps',
            'timeout','--signal=TERM','--kill-after=2','5400','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+5400,original_T481_all_stop=p['absolute_all_stop'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),same_original_T481_combined_budget=True,offline_no_LLM_network_install=True))
    print('T484_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
