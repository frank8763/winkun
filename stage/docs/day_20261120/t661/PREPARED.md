# T661预检完成，待独立窗口与计算额度

2026-11-20。用户要求启动后已完成具体代码、真实保存数据零更新预检及资源回归。尚未登记stage/runs/t661_value_state_retention、未启动学习或仿真，旧09:00硬截止不延长。

命令：`./run_stage.sh stage/experiments/t661_value_state_retention/prepare.py`，输出PASS_ZERO_NATIVE_ZERO_OPTIMIZER；额外`preflight_resource.py`同样PASS。详PREFLIGHT.json及RESOURCE_REGRESSION.json。

真实第二批64条完整轨迹中29条安全成功、35条失败。零Value对照的35条失败优势全零；载入第一批Value后35条均有非零优势，且各有负优势决策。全批Value预测范围[-0.1545641,0.2242726]，不裁剪为概率，不声称已校准或已降低方差；失败回合的每个决策并非全部负优势。两种基线的策略梯度逐参数最大差0.0019217008142260626。基线、Actor/Adam分离和detach已实际核对；没有optimizer.step或原生step。

完整387参照、64历史/密度、不同块随机种子、实际runtime全局绑定、canonical native worker、affinity配置loader、transport安装与有限chunk覆盖已检查。CPU暂停恢复回归只操作新建自有惰性进程，不操作Phase进程；外cgroup拒绝、其他合同错误传播、截止不延均通过。GPU短暂占用会在本阶段硬截止内让出等待，非资源合同错误仍传播。

首个预检在任何科学操作前发现泛名learning.py与历史learning模块循环导入，已用唯一t661_learning模块名修正，完整错误保留ENTRY_FAILED_001.log。随后实际入口及完整预检通过；没有对旧模块或历史结果作修改。

需核定：从正式登记起2小时硬停，12,037,198物理步、433命令、5400秒主管线、Actor与影子/独立各≤4次、Value与影子/独立各≤16次、0额外优化器预检，新记录≤192MiB；总10GiB/空闲40GiB/Phase资源隔离不变。代码与明细见PLAN/BUDGET。仅本候选，原历史对照复用，不隐含授权更多臂、更多采样或后续学习。
