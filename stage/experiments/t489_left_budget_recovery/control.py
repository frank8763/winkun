"""Register/launch the unstarted portion of T481, original combined caps."""
from pathlib import Path
import ast,copy,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t489_runtime as T
L,P,E,M,R,torch,np=T.L,T.P,T.E,T.M,T.R,T.torch,T.np
read,save,sha,digest=T.read,T.save,T.sha,T.digest
OUT=T.OUT;DOC=ROOT/'stage/docs/review_20261111';OLD=ROOT/'stage/runs/t481_fresh16_rehearsal'

def prepare():
    assert not OUT.exists()
    previous=ROOT/'stage/runs/t488_left_failure_focus'
    for name in ['launch.json','driver_started.json']:assert not Path('/proc',str(read(previous/name)['pid'])).exists()
    report=read(DOC/'T488_FINAL_REPORT.json');assert report['state']=='BOUNDED_QUEUE_STOPPED'
    reviewed=read(DOC/'T489_DIAGNOSTIC_REVIEW.json');assert reviewed['state']=='PASS_COMPLETE_DIAGNOSTICS_NOT_COMPLETE_T488_TRAINING'
    old=read(previous/'protocol.json');p=copy.deepcopy(old)
    end=min(old['absolute_all_stop'],read(previous/'launch.json')['all_stop']);assert time.time()+2700<end
    cap=read(previous/'capture/summary.json');assert cap['state']=='FAILED'
    previous_physics=report['physics'];previous_commands=report['commands']
    assert previous_physics==397758 and previous_commands==18 and reviewed['training_collected']==0
    p.update(id='T489_T488_REMAINING_BUDGET_RECOVERY',out=str(OUT),prior_diagnostic_rows=reviewed['diagnostic_rows'],
        capture_summary_path=str(OUT/'capture/summary.json'),max_bytes=8*2**20,
        max_physics=5000000-previous_physics,root_max_physics=5000000-previous_physics,
        max_commands=100-previous_commands,root_max_commands=100-previous_commands,
        main_seconds=2100,root_max_main_seconds=2100,learning_seconds=600,audit_seconds=180,
        all_seconds=2700,absolute_all_stop=end,previous_physics=previous_physics,previous_commands=previous_commands,
        previous_main_seconds=cap['elapsed_s'],preflight_commands=0)
    assert cap['elapsed_s']+2100<=2400
    files=set(old['inputs'])|{str(f) for f in HERE.glob('*.py')}
    files|={str(previous/'protocol.json'),str(previous/'capture/summary.json'),str(DOC/'T488_FINAL_REPORT.json'),
        str(DOC/'T489_DIAGNOSTIC_REVIEW.json'),str(DOC/'T489_PLAN.md')}
    files|={r['path'] for r in p['prior_diagnostic_rows']}
    p['inputs']={f:sha(f) for f in sorted(files)}
    assert M.used()+p['max_bytes']+100000<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
    assert L.fit is L.F.fit and L.make is L.F.make
    attrs={n.attr for n in ast.walk(ast.parse((HERE/'t489_runtime.py').read_text())) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='L'}
    assert all(hasattr(L,n) for n in attrs)
    OUT.mkdir()
    encoded=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n'
    (HERE/'protocol.json').write_text(encoded);(OUT/'protocol.json').write_text(encoded)
    save(OUT/'preflight.json',dict(state='PASS_EXACT_T488_FUNCTIONS_AND_REUSED_NUMERICAL_PREFLIGHT',native=0,commands=0,discarded_Adam=0,
        original_preflight_sha256=sha(previous/'preflight.json'),same_fit_make_objects=True,old_diagnostic_complete_review_PASS=True))
    save(DOC/'T489_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',new_max_physics=p['root_max_physics'],new_max_commands=p['root_max_commands'],new_main_seconds=2100,
        shared_max_physics=5000000,shared_max_commands=100,shared_main_seconds=2400,all_stop=end,new_own_bytes=p['max_bytes'],
        diagnostic_4_plus4_NOT_repeated=True,preflight_new_native_or_Adam=0))
    print('T489_REMAINING_BUDGET_REGISTERED_NOT_LAUNCHED',flush=True)

def launch():
    p=read(OUT/'protocol.json');T.verify(p,True)
    assert not (OUT/'launch.json').exists() and read(OUT/'preflight.json')['native']==0
    src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    now=time.time();assert now+2700<p['absolute_all_stop']
    with (OUT/'driver_console.log').open('xb') as f:
        c=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T489','--why=Recover unstarted T481 learning under same original caps',
            'timeout','--signal=TERM','--kill-after=2','2700','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'launch.json',dict(pid=c.pid,epoch=now,all_stop=now+2700,original_T488_all_stop=p['absolute_all_stop'],source_remote=src['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),same_original_T488_combined_budget=True,offline_no_LLM_network_install=True))
    print('T489_ACTUALLY_LAUNCHED',c.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
