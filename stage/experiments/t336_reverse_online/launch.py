from pathlib import Path
import json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t336_reverse_online';DOC=ROOT/'stage/docs/overnight_20261103'

def main():
    assert not any((OUT/n).exists() for n in ['launch.json','status.json','summary.json'])
    assert not (DOC/'NIGHT_WRAPPER.json').exists()
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'preflight.json').read_text())['state']=='PASS'
    assert time.time()<p['absolute_training_deadline_epoch']-1800
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=json.loads((ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json').read_text())
    assert r['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    env=os.environ.copy();env['STAGE_PROCESS_CPU']='0'
    with (DOC/'driver.log').open('xb') as f:
        proc=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(int(p['absolute_all_work_deadline_epoch']-time.time()))+'s',
            '/usr/bin/taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'night_driver.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (DOC/'NIGHT_WRAPPER.json').write_text(json.dumps(dict(pid=proc.pid,launch_epoch=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),remote_commit=r['remote_commit'],all_work_deadline_epoch=p['absolute_all_work_deadline_epoch'],native_launch_pending=True),indent=2)+'\n')
    print(json.dumps(dict(orchestration_pid=proc.pid,native_launch_pending=True)))

if __name__=='__main__':main()
