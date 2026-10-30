# MK-mini Stage：端到端深度强化学习导航

本仓库归档 MK-mini **Stage** 研究主线，与 Phase 研究线隔离。策略直接从局部传感、最终目标相对状态和车体状态输出连续动作。当前没有已证实的新算法优势或真机泛化结论。

## 本机工作目录与执行

后续在本仓库根目录开发和运行代码。执行 `./run_stage.sh` 只检查依赖和文件哈希；获准继续研究后，新代码放 `stage/experiments/`，结果放 `stage/runs/`，通过 `./run_stage.sh stage/experiments/实验脚本.py` 执行。GitHub 保存版本与模型附件，实际计算由本机或另行配置的计算设备承担。用户已检查迁移并授权继续；当前研究状态见[RESEARCH_STATUS](stage/docs/RESEARCH_STATUS.md)，按有界协议运行。

## 入口

- [最新实验状态](stage/docs/RESEARCH_STATUS.md)
- [最新控制稳健性训练与动作头修改（T186–T188）](stage/docs/results_20261030_control_robustness/findings.md)
- [实际课程RL训练与证据边界（T180–T185）](stage/docs/results_20261030_onpolicy_course/findings.md)
- [失败机制复查与实验选择（T179）](stage/docs/results_20261030_mechanism_audit/findings.md)
- [场景、动作范围与持续探索对照（T176–T178）](stage/docs/results_20261030_scene_exploration/findings.md)
- [Critic原因诊断与修复对照（T173–T175）](stage/docs/results_20261030_q_audit/findings.md)
- [本轮长距离诊断与文献边界](stage/docs/results_20261030/analysis.md)

- [迁移完成、清理明细与磁盘释放报告](stage/docs/CLEANUP_REPORT.md)
- [迁移与恢复说明](stage/docs/MIGRATION.md)
- [保留模型与暂停恢复边界](stage/docs/RESTORE_CHECKPOINTS.md)
- [当前实验记录](stage/current/outputs/stage_progress.md)
- [顶刊研究路线与创新边界](stage/current/outputs/stage_t162/research_direction.md)
- [5 米失败诊断](stage/current/outputs/stage_t163/findings.md)
- [模型、数据与来源的 SHA256 清单](stage/manifests/archive_manifest.json)
- [模型和数据 Release](https://github.com/frank8763/winkun/releases/tag/stage-archive-2026-10-30)
- [本轮T165–T172模型与原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-progress-2026-10-30)
- [T173–T175 Critic诊断模型与原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-critic-audit-2026-10-30)
- [T176–T178 场景与探索模型、原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-scene-exploration-2026-10-30)

- [T179 机制复查原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-mechanism-audit-2026-10-30)

- [T180–T185 课程RL模型和原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-onpolicy-course-2026-10-30)

## 当前能力和实验状态

T159 单 Actor 通过六项固定 3/4 米训练任务；这不是独立泛化测试。迁移后T165已严格恢复并补足T164原预算，随后完成T166–T172有界诊断和训练。两项原始5米仍未通过，保留T159。T173–T175定位并修复已存碰撞终止样本的局部价值误差，留出动作排序仍不可靠。T176–T178进一步完成场景组件、原生油门范围与探索相关时间对照：四个新PPO候选均旧6/6、原始5m0/2，无晋升。详见[最新报告](stage/docs/results_20261030_scene_exploration/findings.md)。T179进一步确认小障碍也发生车身碰撞，未发现转向网络普遍饱和；下一步先检验原场景策略行为覆盖，见[最新分析](stage/docs/results_20261030_mechanism_audit/findings.md)。T183进一步完成192回合实际MC/PPO训练：两组旧6/6、原始5m0/2，较强保持组仅恢复一个历史成功课程案例。T184/T185检验了起点连接与推理契约，详见[最新结果](stage/docs/results_20261030_onpolicy_course/findings.md)。T187局部训练提高了小扰动下的到达稳定性，仍未通过原始5m；T188动作头候选仅完成初始化核验，尚未训练。见[最新修改](stage/docs/results_20261030_control_robustness/findings.md)。目前无训练作业，不重复启动已完成目录。

源码和较小的协议/结果进入 Git；必要检查点、训练数据、原始轨迹与日志进入 Release 附件。聊天原文、登录凭据不上传。

## 隔离范围

`stage/runtime/` 中部分模拟器文件的历史名称包含 `phase10`，它们是实际 Stage 运行已验证的共享依赖，不代表引入 Phase 的规划研究、规划器或研究结果。保留源文件原名和哈希。该包覆盖当前 T148–T164、TD3 来源证据及已验证运行依赖，不声称备份了本机所有早期研究分支。
