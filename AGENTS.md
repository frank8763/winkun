# MK-mini Stage 工作仓库

- 本仓库是以后 Stage 代码开发、执行与实验输出的主目录，远端为 https://github.com/frank8763/winkun.git。
- 只处理 Stage，与 Phase 研究线完全隔离。不得读取或使用 Phase 聊天、研究模型或结果，不加入外部规划器、人工路点或运行时教师切换。
- `stage/runtime/` 的部分历史文件名含 phase10，是已经核验的 Stage 模拟器依赖，不代表 Phase 研究线。
- **2026-10-30 用户已检查迁移并明确授权继续 Stage 实验**。按有界协议推进，不自动无限扩展训练；保持 Phase 隔离和磁盘约束。新状态先读 `stage/docs/RESEARCH_STATUS.md`。
- 先读 `stage/docs/CLEANUP_REPORT.md`、`stage/docs/MIGRATION.md` 和 `stage/current/outputs/stage_progress.md`。
- 当前保留 Actor 为 T159，六项固定3/4米训练任务通过，不代表泛化。5米尚未通过。T164暂停在170199/165498新增物理步，未完成原预算。
- `stage/current/`、`stage/legacy/` 是带哈希的研究证据快照；保留原始文件和失败记录。T145/T146原始文件只读。
- 新代码写 `stage/experiments/`；新结果写 `stage/runs/`，用 STAGE_RUN_ROOT 定位。不要再向家目录生成散落实验目录，不自动保存每个回合的大断点；按研究协议保留必要恢复点和验收模型。
- 模型、数据、长轨迹不进入普通Git历史；必要版本以Release附件保存并核验。Git中保留代码、协议、汇总及SHA256清单。
- 通过仓库根目录 `./run_stage.sh` 检查本机运行依赖。此命令只核验文件/哈希和包导入，不步进模拟器，不推进实验。
- 原历史脚本保留旧绝对路径和哈希，部分依赖仍从本机只读旧路径加载；这是已记录的本机兼容层，不声称完成了跨机器路径重构。后续跨机器恢复依迁移文档处理。
- 历史聊天只取得十条可见Stage聊天的最近回合，未读取全部全文，不得声称读全。

- 本轮 T165 恢复 T164 的回合边界状态，补足原256k预算；新增运行目录总量上限1GiB，每组仅保留latest/previous两个断点。不得改写历史快照或重复启动。
- T165–T172 已完成（T170按预声明课程门槛提前停止）；当前没有运行中的训练或评估。先读最新 RESEARCH_STATUS.md 和 results_20261030/analysis.md，不得重复启动这些目录。T159仍为保留Actor；T171近目标课程成功不能当作原始5m通过。T172匹配起点诊断T159为12/24、T171为5/24，未见新增成功能力。
