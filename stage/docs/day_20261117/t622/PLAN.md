# T622 GPU回报预测能力对照：已批准，实际启动查launch

T620新Critic第2步即推动Actor，24次Actor后两组共同损伤八米LEFT13；这不证明唯一原因是Critic错误，但损失下降没有证明动作梯度可靠。先冻结Actor，判断价值表示是否学到了可留出的回报信息。

57条真实完整历史（54条既有源PASS prior+T620 A的3新轨迹；在本轮全部为历史复用），按完整task ID拆为39训练轨迹/18留出轨迹，30训练ID/9留出ID。两关键退化任务所有轨迹仅留出，不把同任务另一policy/seed放训练。完整实际历史BPTT，终止含有限任务timeout不bootstrap，按原物理gamma精确递推实际行为MC回报。

A输入当前实际动作，B把当前动作输入置零但保留真实观测与前次实际动作历史；同双Critic结构/初值/batch/Adam，每组256次Critic更新，另256影子及256独立GPU历史重构。总新增512live/512shadow/512audit Critic，Actor0、native0、额外Adam预检0。固定次数，不挑最好中间点，不扫lr/seed/轮数。全轨迹无标签泄露到输入。

MC标签对应原记录行为及其续行，不是当前策略Q；留出MSE改善或A优于B仍不证明动作因果排序/Actor梯度可靠，不自动接Actor学习、不宣称导航改进或全局泛化。报告初末逐轨迹MSE与配对初始回报预测；两臂无论好坏完整保留，T608权重SHA不变。

使用GPU float64，不用混合精度；CPU8/10/12/14四独立物理核可用、线程实测选2，总CPU400%低优先，GPU单科学进程。共享Stage writer/GPU锁，启动前查其他compute，运行发现其他GPU作业即停止让出，Phase研究数据不读取。独立GPU审计逐位重建Critic/Adam/trace，而非放宽CPU/GPU误差当审核。

17:30停学、18:00原生（本轮0）、18:30全停不延长。各main300s/audit300s，尾300s，实际完整余时准入；无native费用、独立输出2MiB、global1GiB/free40GiB不变；日原物理/命令/主管线/存储不扩。GPU加速授权来自用户“充分利用cpu和gpu；加快速度推进实验”，新增优化器额度已获用户明确批准。

方案参考[TD3延迟策略更新](https://spinningup.openai.com/en/latest/algorithms/td3.html)及[RLPD作者实现](https://github.com/ikostrikov/rlpd)。本轮MC监督预测诊断不是这两算法的完整复现或原创算法。
