"""Post-run success-data qualification, metadata/SHA/NumPy only, no policy updates."""
from pathlib import Path
import json,hashlib,numpy as np,sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t612_baseline_teacher_recovery';DOC=ROOT/'stage/docs/day_20261117/t612'
read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
sys.path.insert(0,str(Path(__file__).resolve().parent))
import t612_native as Native
def arrays(f):return Native.E.N.S.arrays(f)
def main():
 p=read(OUT/'protocol.json');s=read(OUT/'summary.json');r=s['rows'][0];assert s['state']=='COMPLETE' and read(OUT/'audit.json')['state']=='PASS';assert r['success'] and not r['collision'] and r['field_detail']['field_qualified_success'];assert sha(r['path'])==r['sha256'];a=arrays(r['path']);old=arrays(ROOT/'stage/runs/t611_early_teacher_recovery/episodes/full_0.npz');k=p['teacher_first']
 for key in ['states','next_states','actions','poses','next_poses','history_poses','history_goals','history_times','native_counts']:assert np.array_equal(a[key][:k],old[key][:k]),key
 assert np.array_equal(a['states'][k],old['states'][k]) and np.array_equal(a['initial_native_integration'],old['initial_native_integration']);assert not a['ownership'][:k].any() and a['ownership'][k:].all();assert np.array_equal(a['actions'][k:],a['teacher_actions'][k:]) and not a['teacher_has_extra_memory']
 record=dict(state='QUALIFIED_SUCCESSFUL_TRAINING_ONLY_RECOVERY_DATA',task=r['job_id'],path=r['path'],sha256=r['sha256'],native_array_digest=r['array_digest'],native_array_hashes=r['array_hashes'],teacher=p['handover_teacher'],candidate_prefix_actor=p['candidate_model'],teacher_start_decision=k,teacher_start_native_steps=int(a['native_counts'][:k].sum()),supervision_mask='ownership==1 (suffix only); full actual120 prefix retained for student BPTT',target_array='teacher_locations; executed labels teacher_actions float32',teacher_carry_recomputed_on_actual_prefix=True,parent_T608_teacher_same_state_failed=True,T550_teacher_same_state_succeeded=True,all_prefix_arrays_between_T611_T612_exact=True,source_is_mixed_teacher_not_onpolicy=True,new_independent_navigation_task=False,independent_native_exact_replay=True,independent_double_actor_history_audit='PASS',success=True,collision=False,seconds=r['physics']/500,new_Actor_Adam=0,no_automatic_training=True)
 (DOC/'QUALIFIED_RECOVERY_DATA.json').write_text(json.dumps(record,indent=2)+'\n')
 (DOC/'FINDINGS.md').write_text('''# T612固定历史教师恢复对照

同候选真实前38决策/7.6秒，T60834研究父教师接管在T611碰撞，历史T550教师接管在T612无碰撞到达（剩0.199902944m）。T611与T612前缀states/nextstates/actions/native_counts/pose/history和接管120输入逐位一致，初始原生积分相同，分别由教师实际前缀独立计算自己的memory。只换固定教师、不扫时刻，提供该状态存在可验证成功纠正行为的直接证据；不等于所有恢复状态可达或该方法已解决泛化。

T612一条mixed＋一次逐数组/积分exact，合计60340物理/6命令/46.1568main秒，0Adam/0RL。独立候选和T550全部memory/history/执行动作/奖励核验及原生exact通过，18原文件远端RAM逐成员SHA读回通过，全部进程退出。T550只有真实64循环memory，不含T598额外32/anchor32，字段按架构真实记录。继承audit的two_order_decomposition_exact仅为通用metrics重算标记，不代表本项执行过T610双顺序分解；原件保留。

QUALIFIED_RECOVERY_DATA锁定teacher/candidate/path/array/SHA和第38决策边界。后续若学习，只把ownership==1的teacher后缀当合格监督标签，完整120前缀用于重新计算学生自己的carry；不能用~ownership掩码误学候选前缀，不能把教师hidden当学生hidden，不把mixed teacher数据作onpolicy PPO经验。训练仍待新有效学习窗口和额度；11:30停学已过，本项只完成零更新诊断。

当前单Actor研究候选仍34/49、全场地13/24。接下来有针对性的学习对照应检验加入这条真实恢复历史，能否在继续学新全场地任务时保住唯一失守八米任务及全部34成功ID；单条拟合过关不当闭环通过，不替换T550部署基线。当前结果是有效纠正数据的获得，不称导航新策略成功/顶刊创新已成立。
''')
 print('QUALIFIED_RECOVERY_DATA_SEALED',k,r['physics']/500,'SECONDS_ZERO_LEARNING')
if __name__=='__main__':main()
