# T570实际加速准备：冻结完整history并发检查

用户明确要求现在开始加速并寻求持续学习，授权CPU多核和可选GPU。原16:30学习/17:00原生截止均不改；本项零仿真、零优化器和权重修改，17:30内结束。正式学习仍需新有效窗口。

用同T562、T567R已审两条真实完整history，在CPU14/15各独立进程上比较串行两job与并发同两job，5次固定配对，加4次预热，共24次前反向，0Adam/0RL。每个job在两个进程中预测、损失和梯度hash逐位一致；模型前后hash不变。计2worker启动/停止共4命令，240秒，记录256KiB；CPUQuota200%，OMP/MKL/OpenBLAS各1，GPU未启用，Nice19。两物理核仍非系统级独占，不查看Phase。GPU可用不代表独占，先不启用共享GPU训练。

仅验证真实网络计算并发和资源隔离，不是原生物理吞吐、GPU速度、独立重复训练或导航提升。审计只独立核持久化结果/源模型SHA与计账，未重算24梯度；必须明示这个范围。原始小文件RAM全远端逐成员SHA，代码源树同步，不生成大checkpoint。所有19M/600/12300及global1GiB/free40限保持，不重启旧目录。

来源：MuJoCo官方独立状态并行 https://mujoco.readthedocs.io/en/latest/programming/simulation.html ；PyTorch官方数值与batch差异 https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html 。此次同job同shape比较不冒充跨batch或跨GPU一致。
