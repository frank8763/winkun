# T687：完整原Actor梯度通过，固定全范围首步不准入

COMPLETE / FULL_ACTOR_GRADIENT_CONTRACT_PASS / 独立audit PASS，**full_scope_pilot_numerically_admitted=false**。0物理/0原生命令/0Actor/Value/optimizer，无新candidate或checkpoint；46.761752main+45.695163audit秒。driver817896/helper817901/science817902,817966及unit全核实退出。精确GitHub树8a9c8b33d0ef37ade94a0173984c7f88e8dd4dea/commit7ede4e6477d95d206ed63161cad7cf116eb4bfe3后唯一运行；source300按commit去重结算，全部raw保留。

同T501现有64772参数/120输入/GRU64，全部1724完整历史重建初始location与clipGaussian密度；387成功命令参考同原语义。完整q.sequence BPTT替代冻结特征，新增范围梯度范数.1165642、全联合梯度范数.1178496；原11906adapter PG和联合切片分别误差2.41e-17/2.21e-17。全/core/encoder方向数值误差5.09e-11/1.03e-11/2.98e-11，独立复算所有gradient/数组/逐407指标精确。可证明完整梯度实现，不证明这些梯度估计真实期望可靠或有用。

只解析构造同T678 fixed lr3e-5/eps1e-8/normclip1的freshAdam第一步内存位移，表示误差2.78e-17，随后全state精确恢复，未调用optimizer。原68ID面板最大任务episode均值latentGaussian KL **.03302765295 > .01**，最坏field6_offset3；全407中101项超过.01。不能把语义PASS叫学习准入，更不能叫导航成功。固定fullscope提议停止，不缩lr或改阈值/seed救门；此次不是全参数学习效果测试，不能据此证明所有全参数学习无效、网络容量不足或原adapter限制已被证实。

T501继续保留。下一停止继续OPE诊断/恢复T678旧候选；独立检索审查原adapter范围内首Adam坐标归一化与原始联合梯度方向的区别，先核查T627/T628等SGD历史。仅有明确单变量、真实反例和可区分控制才另注册实际学习，无新job已经开始。
