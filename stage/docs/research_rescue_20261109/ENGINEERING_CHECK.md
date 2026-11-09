# 物理时间GAE参考实现：工程检查

2026-11-09，Stage only。状态为`REFERENCE_CONTRACTS_PASS`，**不是科学实验注册、导航验收或方法收益**。

代码：`stage/experiments/joint_recurrent_actor_critic/physical_gae.py`。
命令：`taskset -c 0 ./run_stage.sh stage/experiments/joint_recurrent_actor_critic/test_physical_gae.py`。
结果：exit0，9项unittest全部通过，框架报告0.026秒。仅固定合成NumPy数组，无模型、Torch前向、车辆/仿真、优化器、安装或stage/runs写入。

- lambda=1时，与独立向前展开的MC回报一致，覆盖可变hold和随机固定合成数据；不是把实现的后向递推再抄一遍。
- lambda=0时为单步TD。
- 真正任务终点不bootstrap，rollout截断使用实际最终观测V。
- trace在reset处断开，下一回合奖励不泄漏；不同环境列独立。
- gamma/lambda按实际原生步数衰减；不直接按决策计相同折扣。
- 原输入数组不改变；非有限、非法mask、非正/非整数hold与形状错误拒绝。

未验证：实际reward汇总合同接线、critic拟合/校准、recurrent learner、GPU数值、并行环境、Actor更新、吞吐与导航收益。代码显式区分bootstrap与trace连接，仍要求调用者正确提供final observation、终止与删失标签。下一运行不能因为这些合成检查通过就跳过真实接口零步合同和独立有限协议。

本次研究时只读容量扫描：runs逻辑1,030,466,072B，小于1,073,741,824B；磁盘可用242,665,181,184B，高于40GiB。这是当时测量，不是新训练完整储存预留或永久保证。仅剩约41.27MiB逻辑余量，不能直接容纳拟议大批采样；没有删除模型/失败/唯一证据。
