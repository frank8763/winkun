from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t339_success_to_control';DOC=ROOT/'stage/docs/review_20261104'

def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert not (DOC/'T339_WRAPPER.json').exists()
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    env=os.environ.copy();env['STAGE_PROCESS_CPU']='0';began=time.time()
    with (DOC/'T339_driver.log').open('xb') as f:
        proc=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=3s',str(p['all_work_seconds'])+'s',
            '/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (DOC/'T339_WRAPPER.json').write_text(json.dumps(dict(pid=proc.pid,launch_epoch=began,native_launch_pending=True,
        all_work_deadline_epoch=began+p['all_work_seconds'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=r['remote_commit']),indent=2)+'\n')
    print(json.dumps(dict(orchestration_pid=proc.pid,native_launch_pending=True)))

if __name__=='__main__':main()
