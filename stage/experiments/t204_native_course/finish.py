"""Finish the one existing T204 job; never launch or extend training."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'stage/runs/t204_native_course'
TRANSFER = ROOT / 'stage/runs/_transfer_native_course_20261031'
DOCS = ROOT / 'stage/docs/results_20261031_native_course'
HERE = Path(__file__).resolve().parent


def save(name, obj):
    path = TRANSFER / name
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def run(args, timeout=180):
    subprocess.run(args, cwd=ROOT, check=True, timeout=timeout)


def git_save(message, paths):
    run(['git', 'add', '--', *paths])
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode
    assert changed in [0, 1]
    if changed:
        run(['git', 'commit', '-m', message])
    run(['git', 'push', 'origin', 'HEAD'])


def main():
    TRANSFER.mkdir(exist_ok=True)
    lock = (TRANSFER / 'finish.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (TRANSFER / 'finish_receipt.json').exists(), 'No duplicate postprocessing'
    plan = json.loads((OUT / 'protocol.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert launch['protocol_sha256'] == hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest()
    save('finish_status.json', {'state': 'WAITING_FOR_EXISTING_T204', 'training_pid': launch['pid'], 'physics_steps': 0, 'optimizer_updates': 0})
    while not (OUT / 'summary.json').exists():
        if (OUT / 'failure.json').exists():
            raise RuntimeError('T204 failure evidence present; preserve it, never restart')
        if time.time() > plan['deadline_unix'] + 300:
            raise RuntimeError('Postprocessor wait bound reached; no training action taken')
        if not Path(f'/proc/{launch["pid"]}').exists():
            raise RuntimeError('T204 exited without completed summary; inspect evidence')
        time.sleep(10)
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['state'] == 'COMPLETE' and summary['all_workers_stopped']
    # Wait until the source log writer has exited before hashing the archive.
    for _ in range(12):
        proc = Path(f'/proc/{launch["pid"]}/stat')
        if not proc.exists() or proc.read_text().split(') ', 1)[1].startswith('Z '):
            break
        time.sleep(1)
    else:
        raise RuntimeError('T204 source writer still live; refuse archive')
    save('finish_status.json', {'state': 'SUMMARIZING', 'physics_steps': 0, 'optimizer_updates': 0})
    run(['./run_stage.sh', str(HERE / 'summarize.py')])
    metrics = json.loads((DOCS / 'metrics.json').read_text())
    rows = []
    for arm, value in metrics['arms'].items():
        rows.append(f"| {arm} | {value['training_safe_GOAL']}/{value['training_episodes']} | {value['original_LEFT_training_safe_GOAL']}/{value['original_LEFT_training_episodes']} | {value['final_old8_safe_GOAL']}/8 | {len(value['novel_success_tasks'])} |")
    lines = [
        '# T203–T204结果（2026-10-31）', '',
        f"T204按协议停止：{metrics['stop_reason']}；完成{metrics['rounds_completed']}轮，{metrics['stats']['training_episodes']}训练回合，{metrics['stats']['actor_updates']}次Actor头部更新。原始LEFT冻结通过={metrics['original_LEFT_passed']}，左右稳定5m门槛通过={metrics['stable_5m_gate_passed']}。T159/T194保留。", '',
        '| 组 | 全部训练安全GOAL | 原始LEFT训练安全GOAL | 最终old8安全GOAL | 父策略失败但本组成功的任务数 |',
        '|---|---:|---:|---:|---:|', *rows, '',
        f"初始成功源距原起点{metrics['source_reset_gap_m']:.6f}m；最终共享成功池距离{metrics['final_pool_gap_m']}m。独特新增能力任务{metrics['unique_new_capability_tasks']}个；课程移近、局部成功与原始5m通过分别报告，不能互相替代。", '',
        f"新增训练物理步{metrics['physics']['training']}、诊断{metrics['physics']['diagnostic']}、重放{metrics['physics']['replay']}，与逐回合记录和统计完全相符。精确重放核验{metrics['replay_checks']}次，全通过={metrics['all_replay_checks_verified']}。稳定性网格完成{metrics['robustness_completed']}/54、安全到达{metrics['robustness_safe_GOAL']}/54；未完成不通过。", '',
        '本轮仅一个配对训练种子、共享自适应课程、同一地图和开发任务，冻结特征骨干而更新动作读出头。部署仍是单网络观测到动作；此pilot不证明从零端到端训练优越性、跨地图泛化或方法原创性。保持损失差异只能结合共同任务与验收描述，不能从两个耦合课程轨迹推导统计显著性。', '',
        '动作覆盖、各固定任务的距离/终止结果、训练计数见metrics.json；具体源与方法限制见protocol_notes.md。Stage/Phase完全隔离，原目标/成功半径/时限/障碍及执行接口保持原契约。', '',
        '下一步依据本轮困难前沿，检查原生低速动作覆盖、随机回报方差、七项门槛插值回滚及新增成功分布，再独立注册有界协议。若稳定5m门槛通过，后续仍须独立留出起点/场景与长距离评估；当前脚本不自动追加训练或晋升模型。', ''
    ]
    (DOCS / 'findings.md').write_text('\n'.join(lines))
    status = ROOT / 'stage/docs/RESEARCH_STATUS.md'
    text = status.read_text(); heading, body = text.split('\n', 1)
    current = f"\n**T203/T204全部完成，所有训练与模拟器工作进程已停止。** T204停止原因{metrics['stop_reason']}，{metrics['rounds_completed']}轮/{metrics['stats']['training_episodes']}训练回合；原始LEFT冻结通过={metrics['original_LEFT_passed']}，稳定5m验收={metrics['stable_5m_gate_passed']}。先读[结果与限制](results_20261031_native_course/findings.md)。T159/T194保留；不重复、不自动追加训练或进入长距离。以下为历史记录。\n"
    status.write_text(heading + '\n' + current + body)
    with (ROOT / 'AGENTS.md').open('a') as f:
        f.write(f"\n- T204已完成，停止{metrics['stop_reason']}，原始LEFT通过={metrics['original_LEFT_passed']}，稳定5m门槛={metrics['stable_5m_gate_passed']}；不重复/不续跑。读results_20261031_native_course/findings.md。后处理只有汇总与备份，零训练/物理步，无自动晋升。\n")
    git_save('Record completed T204 outcomes and exact counters', ['AGENTS.md', 'stage/docs/RESEARCH_STATUS.md', str(DOCS.relative_to(ROOT)), str(HERE.relative_to(ROOT))])
    save('finish_status.json', {'state': 'ARCHIVING', 'physics_steps': 0, 'optimizer_updates': 0})
    run(['./run_stage.sh', str(HERE / 'archive.py'), 'package'])
    run(['./run_stage.sh', str(HERE / 'archive.py'), 'publish'], timeout=600)
    release = json.loads((DOCS / 'release_receipt.json').read_text())
    assert release['state'] == 'PUBLISHED_VERIFIED' and release['temporary_archive_deleted_after_verification']
    with status.open('a') as f:
        f.write(f"\nT203/T204原始数据/失败/模型已Release核验备份：{release['url']}；本地逐成员哈希与远端附件SHA256/字节数相符，未完整读回远端二进制。临时包核验后删除；剩余全部runs {release['remaining_new_runs_bytes']}字节。\n")
    git_save('Verify T203 T204 archive and bounded storage', [str(DOCS.relative_to(ROOT)), 'stage/docs/RESEARCH_STATUS.md', 'stage/manifests/native_course_20261031_files.json'])
    receipt = {'state': 'COMPLETE', 'release_url': release['url'], 'stable_5m_gate_passed': metrics['stable_5m_gate_passed'], 'physics_steps': 0, 'optimizer_updates': 0, 'no_training_restart': True}
    save('finish_receipt.json', receipt); save('finish_status.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        TRANSFER.mkdir(exist_ok=True)
        save('finish_failure.json', {'state': 'FAILED', 'traceback': traceback.format_exc(), 'training_restart': False})
        raise
