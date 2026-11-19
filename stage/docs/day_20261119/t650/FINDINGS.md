# 基线场景混淆：已证实的问题与冻结修复验证

2026-11-19用户要求检查并继续下一步。T649全部科学已结束，shared/bc/pg审计通过、归档全远端逐SHA通过；两臂均第一提议KL超限且完整回退，实际各1次提议、0次保留。不能据此否定PG/BC方法。

## 快速复现和根因

命令 `./run_stage.sh stage/experiments/t650_reset_contract_repair/initial_contract.py self_check` 在不到1秒读取原始NPZ，输出 `RED_REPRODUCED_SAME_JOB_POLICY_DIFFERENT_INITIAL scene_sha256`。T636成功原轨迹与T649 BC exact同wide117_80_dev6_LEFT_18、同T501策略hash4a690…、相同初始203维积分/位姿/目标、相同任务JSON，但scene_sha与120观测不同。初始列19/20/21激光及63/67/79/91/103历史特征不同，最大0.4156667；第一次动作就相差0.0445155。T636 GOAL28984步，T649 CONTACT17227步。不是训练造成这两条差异，因为T649完整回退后的参数hash与T501相同。

定位：T630 reset_contract.field_setup（继承T502模式）先调用F.BASE_SETUP，它本来安装了t344的wide-aware factory；然后重置K.R.task_env为新包装器，而包装器调用导入时捕获的F.BASE_TASK（t203裸factory），绕过刚安装的wide工厂。T637–T649使用该路径，非fullfield旧wide任务未执行m.widen里的real_circle_obs_9 x偏移；该基线不再是原wide6场景。调用链还绕过t477径向目标包装，旧wide8目标是否保持8米也需在修复初始目标合同中一起验证。修复实际调用初始化之后安装的canonical wide/radial task factory。

最小回归preflight直接调用旧/新setup入口（非只测试独立helper）：旧返回narrow，新捕获canonical后返回wide_radial；无环境步进/优化器。t650_native从原t477.native_setup捕获canonical函数，复制原全场地姿态/目标重置，非fullfield使用其安装的wide+radial工厂。每任务首物理步前强制检查参考scene、120packet、203积分、goal和零hidden，错即0步退出。历史共享源码全保留，只在新模块修复。

## 历史结论修订边界

T637–T649中的raw、内部exact和ActorAdam重构仍是这些实际场景的真实记录，不能删除或改为假通过；但与T501/T636原wide任务比较的“旧任务遗忘/保持率”存在场景混淆，相关结论撤回待重评。T646的40/48邻域画像也不能直接当原wide基线鲁棒性证据。T648命令约束合格但旧6LEFT碰撞，不足以独立证明原wide闭环保护失败。

T502原field24本身通过裸factory创建，实际记录几何与旧wide6不同；本次保留其历史实际geometry作为legacy场地任务，不以同ID静默改成wide。后续如果需要全场地统一更宽障碍布局，应新建任务ID/几何协议，与旧field24分开，并重新建立基线。T501 original旧275/wide6/wide8及T636原六项复现仍成立；本次只恢复一致性，不宣布新增训练能力/全场地通过。

## 有限验证，沿用当前余额

新T650不训练：冻结原T501，6个T636原始案例＋6次exact，最多466672物理步、18命令、900main/300audit、64MiB新记录。最新用户“检查，然后继续下一步”的必要基线修复，使用已批准T649剩余预算，0新增Actor/Critic/optimizer，不重启旧任务。原T649实耗6780012步258命令2124.778秒，合计最坏7246684步276命令3024.778main，均在原31M/1020/6000内；两run物理记录合计占用在原512MiB内。native仍12:38:57前、all13:08:57前，CPU四物理核/Phase隔离/global10GiB/free40GiB不变。

6个模型原轨迹每数组/积分必须逐位匹配T636，6fresh/6exact不作为新独立导航成功或新seed统计；通过后再考虑修复后的学习新协议。本次不新增优化器额度、不延长窗口、不扫参数。
