"""Bounded postprocessing for an existing T207 job; never launches training."""
from pathlib import Path
import json,subprocess,time,os,fcntl,traceback,signal
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t207_parameter_pg';HERE=Path(__file__).resolve().parent;DOCS=ROOT/'stage/docs/results_20261031_parameter_pg';TRANSFER=ROOT/'stage/runs/_transfer_parameter_pg_20261031'
def save(name,value):
 p=TRANSFER/name;t=p.with_suffix('.partial');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)
def run(args,timeout=90):return subprocess.run(args,cwd=ROOT,check=True,timeout=timeout)
def push():
 child=subprocess.Popen(['git','-c','http.lowSpeedLimit=1000','-c','http.lowSpeedTime=15','push','origin','HEAD'],cwd=ROOT,start_new_session=True)
 try:return child.wait(timeout=45)==0
 except subprocess.TimeoutExpired:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=3)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  return False
def commit(message,paths):
 run(['git','add','--',*paths]);changed=subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode;assert changed in [0,1]
 if changed:run(['git','commit','--only','-m',message,'--',*paths])
def main():
 TRANSFER.mkdir(exist_ok=True);lock=(TRANSFER/'finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (TRANSFER/'finish_receipt.json').exists(),'No duplicate finish'
 launch=json.loads((OUT/'launch.json').read_text());limit=launch['started_unix']+1200;save('finish_status.json',{'state':'WAITING_EXISTING_T207','training_pid':launch['pid'],'physics_steps':0,'optimizer_updates':0})
 while not (OUT/'summary.json').exists():
  if (OUT/'failure.json').exists():raise RuntimeError('Existing T207 failed; inspect evidence, never restart')
  if time.time()>limit or not Path(f'/proc/{launch["pid"]}').exists():raise RuntimeError('No completed T207 summary within bounded existing-process wait')
  time.sleep(10)
 summary=json.loads((OUT/'summary.json').read_text());assert summary['all_workers_stopped']
 for _ in range(12):
  proc=Path(f'/proc/{launch["pid"]}/stat')
  if not proc.exists() or proc.read_text().split(') ',1)[1].startswith('Z '):break
  time.sleep(1)
 else:raise RuntimeError('Source writer still live; archive refused')
 run(['./run_stage.sh',str(HERE/'summarize.py')]);metrics=json.loads((DOCS/'metrics.json').read_text())
 status=ROOT/'stage/docs/RESEARCH_STATUS.md';text=status.read_text();head,body=text.split('\n',1);status.write_text(head+f"\n\n**T207已完成，全部训练/模拟器进程退出。** 停止{summary['stop_reason']}，参数更新尝试{summary['parameter_update_attempts']}次，稳定5m门槛={summary['stable_5m_gate_passed']}。先读results_20261031_parameter_pg/findings.md；T159/T194保留，无自动晋升、续训或长距离。以下为历史记录。\n"+body)
 with (ROOT/'AGENTS.md').open('a') as f:f.write(f"\n- T207已完成并停止：{summary['stop_reason']}，稳定5m门槛={summary['stable_5m_gate_passed']}，参数更新尝试{summary['parameter_update_attempts']}；不重启、不续训、不自动晋升。读results_20261031_parameter_pg/findings.md。后处理只汇总与备份，零新增交互。\n")
 paths=['AGENTS.md','stage/docs/RESEARCH_STATUS.md',str(DOCS.relative_to(ROOT)),str(HERE.relative_to(ROOT))]
 commit('Record completed paired parameter PG outcomes and final evaluation',paths)
 if not push():
  save('finish_receipt.json',{'state':'RESULTS_COMPLETE_PUBLICATION_PENDING','reason':'Bounded git push failed or timed out; local code/results preserved. No archive deleted, no training restart.','physics_steps':0,'optimizer_updates':0});save('finish_status.json',json.loads((TRANSFER/'finish_receipt.json').read_text()));return
 # Complete the already prepared earlier frontier backup first, without duplicate package.
 frontier=ROOT/'stage/runs/_transfer_frontier_20261031/package_receipt.json'
 if frontier.exists() and not (ROOT/'stage/docs/results_20261031_frontier/release_receipt.json').exists():run(['./run_stage.sh','stage/experiments/t206_feature_direction/archive.py','publish'],timeout=420)
 run(['./run_stage.sh',str(HERE/'archive.py'),'package']);run(['./run_stage.sh',str(HERE/'archive.py'),'publish'],timeout=420)
 receipt=json.loads((DOCS/'release_receipt.json').read_text());assert receipt['state']=='PUBLISHED_VERIFIED' and receipt['temporary_archive_deleted_after_verification']
 with status.open('a') as f:f.write(f"\nT207已Release核验备份：{receipt['url']}；本地逐成员哈希/远端SHA256及大小一致，未完整读回远端二进制；临时包已删除。\n")
 commit('Verify T207 model and evidence release backup',[str(DOCS.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md','stage/manifests/parameter_pg_20261031_files.json','stage/docs/results_20261031_frontier/release_receipt.json'])
 pushed=push();result={'state':'COMPLETE' if pushed else 'RELEASE_VERIFIED_FINAL_RECEIPT_PUSH_PENDING','release_url':receipt['url'],'physics_steps':0,'optimizer_updates':0,'training_restart':False};save('finish_receipt.json',result);save('finish_status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  TRANSFER.mkdir(exist_ok=True);save('finish_failure.json',{'state':'FAILED','traceback':traceback.format_exc(),'training_restart':False});raise
