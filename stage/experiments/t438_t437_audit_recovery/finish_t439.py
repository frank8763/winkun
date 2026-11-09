"""Stdlib metadata-only T439 finalization after its driver path error.

No simulator/model/optimizer import. Scientific originals remain unchanged.
"""
from pathlib import Path
import json,time,hashlib,shutil,datetime,zoneinfo
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'stage/runs/t439_representation_rl_control'
DOC=ROOT/'stage/docs/review_20261109'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def rows(f):return [json.loads(x) for x in Path(f).read_text().splitlines()]
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    p=read(OUT/'protocol.json');child=OUT/'representation_rl'
    assert time.time()<p['deadline_all']
    assert not (DOC/'T439_FINAL_REPORT.json').exists()
    s=read(child/'summary.json');a=read(child/'audit.json');error=read(DOC/'T439_DRIVER_ERROR.json')
    assert s['state']=='COMPLETE' and a['state']=='PASS' and s['all_workers_stopped']
    assert '/fresh_rl/summary.json' in error['error'] and 'FileNotFoundError' in error['error']
    pids=[read(OUT/f)['pid'] for f in ['launch.json','driver_started.json','representation_rl/main_started.json']]
    pids+=read(child/'workers.json')['workers'];assert all(not Path('/proc',str(x)).exists() for x in pids)
    before={str(f.relative_to(ROOT)):sha(f) for f in child.rglob('*') if f.is_file()}
    v=s['result'];assert v['updates']==dict(live=32,shadow=32) and v['retained_research_rounds']==0 and not v['accepted'] and not v['full329_executed']
    assert v['initial_model']['sha256']==v['model']['sha256']==sha(v['model']['path'])
    tr=[x for x in rows(child/'cases.jsonl') if x['role'].startswith('TRAIN_')]
    old=[x for x in rows(ROOT/'stage/runs/t437_joint_closed_loop_rl/fresh_rl/cases.jsonl') if x['role'].startswith('TRAIN_')]
    oldkeys={(x['job_id'],x['seed'],x['policy_hash'],x['array_digest']) for x in old}
    matched=sum((x['job_id'],x['seed'],x['policy_hash'],x['array_digest']) in oldkeys for x in tr)
    assert len(tr)==matched==16 and len(rows(child/'replays.jsonl'))==16
    rounds=[]
    for rr in v['rounds']:
        before_ids=set(rr['previous_panel_success_ids']);after_ids=set(rr['proposed_panel_success_ids'])
        rounds.append(dict(round=rr['round'],panel_success=len(after_ids),panel_valid=len(p['panel_ids']),
            fixed=sorted(after_ids-before_ids),lost=sorted(before_ids-after_ids),accepted=rr['accepted'],numeric_accept=rr['numeric_accept']))
    report=dict(state='SCIENCE_AND_AUDIT_COMPLETE_DRIVER_PATH_ERROR_METADATA_RECOVERED',
        main_seconds=s['elapsed_s'],physics=s['physics'],commands=s['commands']+8,
        physical_training_collections=16,historical_exact_overlap_with_T437=16,
        new_independent_training_trajectories=0,exact_native_replays=16,
        sampling_outcomes=dict(success=6,contact=9,timeout=1),live=32,shadow=32,
        independent_discarded_historical_Adam=64,retained_research_rounds=0,final_accepted=False,
        rounds=rounds,wide54=None,old275=None,full329_executed=False,
        full329_unexecuted_reason='No proposal retained; source Actor unchanged.',
        fresh16=None,reload78=None,six_gate=False,eight_started=False,
        final_model_Actor_Adam_exact_initial=True,all_scientific_processes_stopped=True,
        independent_audit='PASS',original_driver_error_preserved=True,
        metadata_recovery_new_native=0,metadata_recovery_new_optimizer=0,
        source_model=v['initial_model'],final_model=v['model'],remote='PENDING',
        whole_T437_T439_science_physics=12246664+s['physics'],
        safeT255_not_promoted=True,old_T422_history_audit_NOT_repaired=True)
    save(DOC/'T439_FINAL_REPORT.json',report)
    save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,epoch=time.time(),
        metadata_only_recovery_after_driver_wrong_child_path=True,
        scientific_children_all_bytes_unchanged=True,original_driver_error_sha256=sha(DOC/'T439_DRIVER_ERROR.json')))
    assert before=={str(f.relative_to(ROOT)):sha(f) for f in child.rglob('*') if f.is_file()}
    text='''# T439验收结论：未通过，开放表示未解决本次负迁移\n\n4轮完整结束，16:01:10–16:15:14，843.979秒（14.07分钟），3,546,420物理步、160科学命令+8零物理预检。32live策略梯度/辅助Adam+32shadow，独立64历史丢弃Adam逐事件重构PASS。所有父/worker/driver退出。\n\n4提案在31任务面板分别28/31、29/31、27/31、28/31；相对初始24/31分别修6丢2、修6丢1、修5丢2、修6丢2。全部数值通过，但都有原成功退化，所以0研究保留、最终Actor+Adam文件字节精确回到初始。**没有执行完整329、54/275全量、fresh16、reload78或八米**，它们是未执行而非0%成功。不得用面板29/31替代完整六米成功率。\n\n相同原始策略/焦点/seed的16次采集全部与T437前4轮数组digest精确重合；物理预算照计，探索终止6成功9接触1超时，不能当16条新独立导航证据。另16次本轮exact也不增加独立样本。配对控制数据一致，只改变可训层范围、均值保持不变。\n\nT437前4轮保留1研究更新（之后4轮均拒，最终测量宽52/54旧271/275）；T439开放encoder/core后0保留。这个固定单种子配对反例说明“只需开放更多已有参数就能解决”不足，不能概括所有表示学习无效或宣称统计显著。新旧行为冲突、有限支持及平均目标仍需处理。\n\n原唯一driver在科学和audit都PASS后，仍读取旧fresh_rl目录导致FileNotFoundError，未完成自动报告/封存/备份。原错误和科学源码不覆盖；本次stdlib零模型/零native/零Adam补元数据与封存，原科学子目录前后所有文件SHA完全一致。原自动收尾不能称无故障。完整备份收据产生后另更新remote字段。\n\n下一步仅机制建议：先把当前被修复/退化任务的真实闭环分布纳入联合数据，保护目标从跨时间均值转成可核验的局部功能变化，并检查实际Adam提案后的影响。扩大快速面板只能发现退化，不能令模型自动学会。需要同数据/预算的普通保持对照，不能通过放宽成功半径、延长时限、runtime路由或追加本轮轮数掩盖失败。此处没有新训练启动。\n'''
    (DOC/'T439_FINDINGS.md').write_text(text)
    print('T439_METADATA_FINALIZED_NO_NEW_NATIVE_OR_OPTIMIZER',flush=True)
if __name__=='__main__':main()
