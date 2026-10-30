# MK-mini Stage：端到端深度强化学习导航

本仓库归档 MK-mini **Stage** 研究主线，与 Phase 研究线隔离。策略直接从局部传感、最终目标相对状态和车体状态输出连续动作。当前没有已证实的新算法优势或真机泛化结论。

## 本机工作目录与执行

后续在本仓库根目录开发和运行代码。执行 `./run_stage.sh` 只检查依赖和文件哈希；获准继续研究后，新代码放 `stage/experiments/`，结果放 `stage/runs/`，通过 `./run_stage.sh stage/experiments/实验脚本.py` 执行。GitHub 保存版本与模型附件，实际计算由本机或另行配置的计算设备承担。用户已检查迁移并授权继续；当前研究状态见[RESEARCH_STATUS](stage/docs/RESEARCH_STATUS.md)，按有界协议运行。

## 入口

- [最新实验状态](stage/docs/RESEARCH_STATUS.md)
- [本轮长距离诊断与文献边界](stage/docs/results_20261030/analysis.md)

- [迁移完成、清理明细与磁盘释放报告](stage/docs/CLEANUP_REPORT.md)
- [迁移与恢复说明](stage/docs/MIGRATION.md)
- [保留模型与暂停恢复边界](stage/docs/RESTORE_CHECKPOINTS.md)
- [当前实验记录](stage/current/outputs/stage_progress.md)
- [顶刊研究路线与创新边界](stage/current/outputs/stage_t162/research_direction.md)
- [5 米失败诊断](stage/current/outputs/stage_t163/findings.md)
- [模型、数据与来源的 SHA256 清单](stage/manifests/archive_manifest.json)
- [模型和数据 Release](https://github.com/frank8763/winkun/releases/tag/stage-archive-2026-10-30)

## 当前能力和暂停状态

T159 单 Actor 通过六项固定 3/4 米训练任务；这不是独立泛化测试。T161 保住六项旧课程，但两项 5 米仍失败。T164 因用户要求迁移和清理而暂停：两组最终保留检查点分别为新增 170199 与 165498 物理步，未完成原定每组 256000 步，不做最终算法对照结论，不自动恢复训练。

源码和较小的协议/结果进入 Git；必要检查点、训练数据、原始轨迹与日志进入 Release 附件。聊天原文、登录凭据不上传。

## 隔离范围

`stage/runtime/` 中部分模拟器文件的历史名称包含 `phase10`，它们是实际 Stage 运行已验证的共享依赖，不代表引入 Phase 的规划研究、规划器或研究结果。保留源文件原名和哈希。该包覆盖当前 T148–T164、TD3 来源证据及已验证运行依赖，不声称备份了本机所有早期研究分支。
