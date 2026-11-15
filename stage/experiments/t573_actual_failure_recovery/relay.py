"""Finite local relay: wait only for the Stage B finalizer, then register once.

No LLM dependency. No overlapping native writer or concurrent source sync.
Science requires source sync; network failure preserves NOT_LAUNCHED status.
"""
from pathlib import Path
import json,sys,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/day_20261115/t573';OUT=ROOT/'stage/runs/t573_actual_failure_recovery';STOP=1794747600.;WAIT_STOP=1794743100.
def read(p):return json.loads(Path(p).read_text())
def status(state,**kw):
 p=DOC/'RELAY_STATUS.json';p.write_text(json.dumps(dict(state=state,epoch=time.time(),pid=os.getpid(),**kw),indent=2,ensure_ascii=False)+'\n')
def run(args,seconds):
 r=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=seconds)
 with (DOC/'RELAY.log').open('a') as f:f.write(r.stdout[-7000:]+'\n')
 assert r.returncode==0,(args[:3],r.returncode,r.stdout[-1800:]);return r

def main():
 assert not (OUT/'launch.json').exists();status('WAITING_FOR_T572R_UNIQUE_FINALIZER',no_new_native=True,no_new_optimizer=True)
 while True:
  assert time.time()<min(WAIT_STOP,STOP-1200),'relay admission deadline'
  s=subprocess.run(['systemctl','--user','show','stage-t572r-primal-feasible-solver.service','-p','ActiveState','-p','SubState'],capture_output=True,text=True,check=True).stdout
  if 'ActiveState=inactive' in s and 'SubState=dead' in s:break
  if 'ActiveState=failed' in s:raise RuntimeError('T572R service failed; no automatic science launch')
  time.sleep(5)
 prior=ROOT/'stage/runs/t572r_primal_feasible_solver';assert read(prior/'bounded/summary.json')['state']=='COMPLETE' and read(prior/'bounded/audit.json')['state']=='PASS';sealed=read(prior/'SEALED.json');assert all(not Path('/proc',str(p)).exists() for p in sealed['scientific_pids']);assert not Path('/proc',str(read(prior/'launch.json')['pid'])).exists();status('REGISTERING_AFTER_PRIOR_STOPPED')
 python=[str(ROOT/'run_stage.sh')];run(python+[str(HERE/'control.py'),'prepare'],60);assert time.time()+1140<STOP;status('SOURCE_COMMIT_AND_REMOTE_SYNC_NO_NATIVE')
 run(['git','add','stage/experiments/t573_actual_failure_recovery','stage/docs/day_20261115/t573/PLAN.md','stage/docs/day_20261115/t573/AUTHORIZATION.json','stage/docs/day_20261115/t573/SOURCES.md','stage/docs/day_20261115/t573/PREFLIGHT.json','stage/docs/day_20261115/t573/PREFLIGHT_BEFORE_RELAY.json','stage/docs/day_20261115/t573/RELAY_SOURCE_BINDING.json','stage/docs/day_20261115/t573/BUDGET.json'],30);run(['git','commit','-m','Register one actual lost-task recovery after independent B-arm completion'],30);run(python+[str(HERE/'transport.py'),'source_sync'],120)
 status('DRIVER_RUNNING',no_optimizer_or_new_RL=True);r=subprocess.run(python+[str(HERE/'driver.py')],cwd=ROOT,timeout=min(1140,STOP-time.time()-5));assert r.returncode==0;report=read(DOC/'FINAL_REPORT.json');status('RELAY_FINISHED',report_state=report['state'],remote=report['remote'],new_optimizer=0,new_RL=0)
if __name__=='__main__':
 try:main()
 except BaseException:status('STOPPED_NO_RESTART',error=traceback.format_exc(),originals_preserved=True);raise
