# MK-mini Stage 工作仓库

- 本仓库是以后 Stage 代码开发、执行与实验输出的主目录，远端为 https://github.com/frank8763/winkun.git。
- 只处理 Stage，与 Phase 研究线完全隔离。不得读取或使用 Phase 聊天、研究模型或结果，不加入外部规划器、人工路点或运行时教师切换。
- `stage/runtime/` 的部分历史文件名含 phase10，是已经核验的 Stage 模拟器依赖，不代表 Phase 研究线。
- **2026-10-30 用户已检查迁移并明确授权继续 Stage 实验**。按有界协议推进，不自动无限扩展训练；保持 Phase 隔离和磁盘约束。新状态先读 `stage/docs/RESEARCH_STATUS.md`。
- 先读 `stage/docs/CLEANUP_REPORT.md`、`stage/docs/MIGRATION.md` 和 `stage/current/outputs/stage_progress.md`。
- 当前保留 Actor 为 T159，六项固定3/4米训练任务通过，不代表泛化。5米尚未通过。T164旧暂停预算已由T165严格恢复补足，见最新研究状态。
- `stage/current/`、`stage/legacy/` 是带哈希的研究证据快照；保留原始文件和失败记录。T145/T146原始文件只读。
- 新代码写 `stage/experiments/`；新结果写 `stage/runs/`，用 STAGE_RUN_ROOT 定位。不要再向家目录生成散落实验目录，不自动保存每个回合的大断点；按研究协议保留必要恢复点和验收模型。
- 模型、数据、长轨迹不进入普通Git历史；必要版本以Release附件保存并核验。Git中保留代码、协议、汇总及SHA256清单。
- 通过仓库根目录 `./run_stage.sh` 检查本机运行依赖。此命令只核验文件/哈希和包导入，不步进模拟器，不推进实验。
- 原历史脚本保留旧绝对路径和哈希，部分依赖仍从本机只读旧路径加载；这是已记录的本机兼容层，不声称完成了跨机器路径重构。后续跨机器恢复依迁移文档处理。
- 历史聊天只取得十条可见Stage聊天的最近回合，未读取全部全文，不得声称读全。

- 本轮 T165 恢复 T164 的回合边界状态，补足原256k预算；新增运行目录总量上限1GiB，每组仅保留latest/previous两个断点。不得改写历史快照或重复启动。
- T165–T172 已完成（T170按预声明课程门槛提前停止）；当前没有运行中的训练或评估。先读最新 RESEARCH_STATUS.md 和 results_20261030/analysis.md，不得重复启动这些目录。T159仍为保留Actor；T171近目标课程成功不能当作原始5m通过。T172匹配起点诊断T159为12/24、T171为5/24，未见新增成功能力。
- T173–T175已完成，Actor全程冻结。终止均衡修正了已存碰撞终止误差，但MC/配对优势在留出状态仍未达门槛，禁止据此自动进行Actor更新或晋升。读results_20261030_q_audit/findings.md与next_protocol.md；下一阶段数据覆盖草案尚未启动。全部新runs仍1GiB上限，不重复启动已完成目录。
- T176–T178已全部完成。先读stage/docs/results_20261030_scene_exploration/findings.md。T177/T178为独立Critic-free MC/PPO协议（不使用未过门槛的T174/T175价值梯度）；四组均旧6/6、原始5m0/2、前沿0/4，全部训练到达0。T159保留，不重复启动、不自动晋升。后续障碍尺寸课程仅为有界草案，尚未启动；新runs总量仍上限1GiB。

- T179（stage/runs/t179_mechanism_audit及其contact_recheck）已全部完成，只有冻结诊断与离线梯度计算，优化器更新0。先读results_20261030_mechanism_audit/findings.md；原尺寸课程草案未启动并已被新诊断建议替代。T180最多40回合的参数敏感性草案尚未启动，无自动晋升。不得重复T179。

- T180的40个固定参数敏感性案例已完成，无5m成功，不重复启动。

- T180–T185全部完成，无运行进程。读stage/docs/results_20261030_onpolicy_course/findings.md。T182/T182b中断证据保留，T182c复用9例只续跑剩余15例。T183两组各96回合，原5m均0/2、旧六项均6/6；课程12/24的新增一例是历史T159已成功案例的恢复，不是历史模型之外的新能力。T159保留，T183 keep10仅研究候选，不重复任何已结束目录，无自动晋升。T186完整状态成功源扰动预检为草案，尚未启动。
