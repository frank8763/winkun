"""Post-window metadata and SHA review only; no simulation or optimization."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t606r_offline_retention_recovery';DOC=ROOT/'stage/docs/overnight_20261116/recovery_001'
read=lambda f:json.loads(Path(f).read_text())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
 report=read(DOC/'FINAL_REPORT.json');rows=[];live=shadow=audit=0;failed=[]
 for C in sorted(OUT.iterdir()):
  if not C.is_dir():continue
  s=read(C/'summary.json');u=(s.get('result') or {}).get('update') or {};steps=u.get('attempted_Adam',0);live+=steps
  f=C/'optimizer_calls.jsonl';calls=[json.loads(x) for x in f.read_text().splitlines() if x] if f.exists() else []
  for who in ['live','shadow']:
   assert sum(x[0]==who and x[1]=='BEFORE' for x in calls)==steps
   assert sum(x[0]==who and x[1]=='AFTER' for x in calls)==steps
  shadow+=steps;a=read(C/'audit.json') if (C/'audit.json').exists() else None
  if a:assert a['state']=='PASS';audit+=a['historical_discarded_Adam']
  saved=[];original={}
  f=C/'cases.jsonl'
  if f.exists():
   for line in f.read_text().splitlines():
    r=json.loads(line);assert sha(r['path'])==r['sha256'];saved.append(r);original[r['path']]=r
  f=C/'replays.jsonl'
  if f.exists():
   for line in f.read_text().splitlines():
    r=json.loads(line);ref=original[r['reference_path']];assert r['exact'] and r['array_hashes']==ref['array_hashes'] and r['integration_sha256']==ref['integration_sha256'];saved.append(r)
  if s['state']=='COMPLETE':assert s['physics']==sum(r['physics'] for r in saved)
  else:failed.append(dict(child=C.name,summary_physics=s['physics'],saved_row_count=len(saved),full_array_audit_completed=False,unavailable_trajectory_details_not_navigation_failure_rate=True))
  rows.append(dict(child=C.name,physics=s['physics'],commands=s['commands'],main_seconds=s['elapsed_s'],audit=a['state'] if a else None,attempted_live_Adam=steps,summary_sha256=sha(C/'summary.json')))
 assert all(not Path('/proc',str(pid)).exists() for pid in read(OUT/'SEALED.json')['scientific_pids'])
 best=report['best_improved'];assert sha(best['model']['path'])==best['model']['sha256'];r=read(OUT/'rl_00/summary.json')['result'];assert r['accepted'] and len(r['check']['success_ids'])==33 and not r['lost_prior'] and len(r['gained_prior'])==2 and len(r['handoff'])==3 and all(x['success'] and not x['collision'] for x in r['handoff'])
 review=dict(state='POST_WINDOW_READ_ONLY_REVIEW_NO_NATIVE_NO_OPTIMIZER',pipeline_state='FAILED_SECONDARY_DIAGNOSTIC_SERIALIZER',end_CST=datetime.datetime.fromtimestamp(report['epoch'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),T607_executed=False,baseline_success=31,best_success=33,denominator=49,best_percent=100*33/49,contact=12,timeout=4,invalid=0,censored=0,retained_initial31=True,recovered_task_ids=r['gained_prior'],historical_T55039_remaining_missing=r['check']['lost_initial'],recovery_pass=3,recovery_total=3,full407=None,fresh16=None,reload78=None,live_Adam=live,shadow_Adam=shadow,historical_audit_Adam=audit,discarded_preflight_Adam=12,final_best_lineage_RL_updates=8,child_ledger=rows,partial_failed_diagnostic=failed,total_physics=sum(x['physics'] for x in rows),total_commands=sum(x['commands'] for x in rows),prior_T606_conservative_start_close_commands=4,physics_new_in_review=0,Adam_new_in_review=0,best_model=best['model'],all_scientific_pids_stopped=True,all_completed_saved_payload_file_SHA_verified=True,full_pipeline_audit_not_passed=True,remote='PENDING',review_initial_replay_schema_mistake='First read-only review assumed replays had new saved path; corrected to reference_path and exact arrays. 0 native or optimizer.')
 receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T606R.json'
 if receipt.exists():review['remote']=read(receipt)['state']
 (DOC/'MORNING_REVIEW_20261117.json').write_text(json.dumps(review,indent=2)+'\n')
 text='''# 11月17日早晨复核

有效改进：真实RL候选从T598起点31/49升到33/49（67.35%），保留原31项并恢复field_sector4_offset2及宽八米LEFT13两项历史能力，三恢复3/3。相对历史T55039仍缺6项，不称全局突破或完整六米/八米通过。该候选自身独立审计PASS；原始字节和最佳模型SHA复核通过。完整407/fresh16/reload78均未执行。

原管线于11月16日22:35:20因首个冻结诊断的记录器KeyError extra_hidden_before退出。旧T550没有新T598额外记忆字段，通用记录器假定存在，属于工程覆盖错误，不是诊断导航失败证明。该诊断21681物理步但无完整轨迹落盘，不能计成功率；整管线不是audit PASS。T607因父管线失败而未启动，昨夜未跑满。

学习实际18 live、18 shadow、18丢弃历史重构，另12预检丢弃Adam；最佳最终模型只来自rl00的8次RL更新，不将其它支线和回退更新累加成保留训练。原生合计6751905物理/260命令；另保守计原T606零物理启动4命令。全部进程已停，磁盘仍低于1GiB。08:30后本复核0新物理/0新优化器，不启动或重启任何实验。

下一准备：先修复并用冻结双架构记录合同验证T550/T598路径，保留全部错误原件；随后从已审计33项候选和冻结T550区分对照安排剩余6项恢复，不直接重跑旧目录。新学习需新的有效时间窗口与剩余预算。
'''
 (DOC/'MORNING_REPORT_20261117.md').write_text(text+'\n原始远端备份状态：'+review['remote']+'。\n')
 print('REVIEW_PASS_EXISTING_EVIDENCE',live,shadow,audit,review['total_physics'],review['total_commands'],review['remote'])
if __name__=='__main__':main()
