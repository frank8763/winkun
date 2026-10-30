# MK-mini Stage：端到端深度强化学习导航

本仓库归档 MK-mini **Stage** 研究主线，与 Phase 研究线隔离。策略直接从局部传感、最终目标相对状态和车体状态输出连续动作。当前没有已证实的新算法优势或真机泛化结论。

## 本机工作目录与执行

后续在本仓库根目录开发和运行代码。执行 `./run_stage.sh` 只检查依赖和文件哈希；获准继续研究后，新代码放 `stage/experiments/`，结果放 `stage/runs/`，通过 `./run_stage.sh stage/experiments/实验脚本.py` 执行。GitHub 保存版本与模型附件，实际计算由本机或另行配置的计算设备承担。用户已检查迁移并授权继续；当前研究状态见[RESEARCH_STATUS](stage/docs/RESEARCH_STATUS.md)，按有界协议运行。

## 入口

- [最新实验状态](stage/docs/RESEARCH_STATUS.md)
- [LEFT5m原因审计：真实soft-Q、动作探索与数据断层（T202）](stage/docs/results_20261030_cause_audit/findings.md)
- [LEFT专项、SAC梯度与最终失败证据（T196–T200）](stage/docs/results_20261030_sac_gradient/findings.md)
- [右侧原始5米首次确定性通过与复核（T193–T195）](stage/docs/results_20261030_success_consolidation/findings.md)
- [目标重标记审计与SAC/HER对照（T191–T192）](stage/docs/results_20261030_her_contract/findings.md)
- [动作头训练与起点课程（T189–T190）](stage/docs/results_20261030_head_curriculum/findings.md)
- [局部稳健性训练与动作头修改（T186–T188）](stage/docs/results_20261030_control_robustness/findings.md)
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

- [T186–T188 局部控制模型、状态快照与动作头审计 Release](https://github.com/frank8763/winkun/releases/tag/stage-control-robustness-2026-10-30)

- [T193–T195 右5米研究模型、整合与复核原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-success-consolidation-2026-10-30)

- [T196 LEFT探索模型与全部真实轨迹 Release](https://github.com/frank8763/winkun/releases/tag/stage-left-exploration-2026-10-30)
- [T197–T198 超时信号与执行器诊断 Release](https://github.com/frank8763/winkun/releases/tag/stage-timeout-progress-2026-10-30)
- [T199–T200 SAC梯度、匹配容量训练与最终模型 Release](https://github.com/frank8763/winkun/releases/tag/stage-sac-gradient-2026-10-30)

- [T202 LEFT失败原因审计原始证据 Release](https://github.com/frank8763/winkun/releases/tag/stage-cause-audit-2026-10-30)

## 当前能力和实验状态

当前T194研究候选用一个113→2神经Actor完成原始RIGHT5米：**42.794秒、无碰撞**；旧六项3/4米仍6/6。保存模型T195精确复现，RIGHT起点±3cm/±2°网格27/27。原始LEFT5米仍碰撞，左右合计1/2，尚未进入更长距离。它是固定训练任务与同图开发网格的结果，不是跨地图或真机泛化。

模型来源是T192真实SAC探索成功、T193监督策略整合及T194有界神经参数校准；不能称从零纯RL或已证明新算法优势。最终部署仍由单网络按当前观测直接生成原生命令，无规划器、路点或运行时教师。正式保留T159，新候选无自动晋升；详情、哈希和所有失败见[最新报告](stage/docs/results_20261030_success_consolidation/findings.md)。

T196–T200的LEFT专项也已全部结束，无运行作业。三组配对训练未产生LEFT到达增益，RIGHT与旧六项保留；接触、执行器状态、均值梯度与熵尺度证据见最新报告。T202进一步完成冻结原因审计；下一阶段先验证成功源邻域和实际动作覆盖，熵尺度对照不自动启动。所有已结束目录不重复。历史阶段按上述入口查看，实验失败与证据边界保留。

源码和较小的协议/结果进入 Git；必要检查点、训练数据、原始轨迹与日志进入 Release 附件。聊天原文、登录凭据不上传。

## 隔离范围

`stage/runtime/` 中部分模拟器文件的历史名称包含 `phase10`，它们是实际 Stage 运行已验证的共享依赖，不代表引入 Phase 的规划研究、规划器或研究结果。保留源文件原名和哈希。该包覆盖当前 T148–T164、TD3 来源证据及已验证运行依赖，不声称备份了本机所有早期研究分支。
