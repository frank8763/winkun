# T323候选：人控可达状态的固定剩余时域完成检验（草案，未注册/未启动）

T322还在完成原协议native诊断；不并发新作业，不从失守旧任务的候选续训。下一低成本问题：safeT255若从人控真实绕行可达的40s状态出发，能否在原56s总期限中完成同一个LEFT6目标？本质是训练状态初始化可用性检验，不是原起点自主六米，也不是延长原cap。

两条T317各仅一个固定40s边界（200×100native），不扫描前缀时刻/速度/模型。到该状态前的human动作只是离线重建训练初态，所有20000重建物理计费；核对前缀12数组和200边界120/真实history，之后只冻结safeT255每个新真实120状态直接输出动作，hiddenreset。原goal、原生碰撞/.20m、elapsed=20000、cap28000、previous_action/hold和传感器历史均保留，剩16s，不把时钟重置到零。整条重建+suffix每源<=28000步，2源及各一次精确重生总4命令/112000物理/300秒；0优化器/RL，不用T322失败策略，不新增噪声。完整记录两个行为的边界、sourcehash和policyhash，人工前缀不造expert/onpolicy logp。

如果出现完整human-derived成功路径，只能成为新的训练参考，并与人工原记录/自主policy结果分开；单Actor仍须从原起点学会再完整275+54+16+78验收，部署禁止前缀回放/教师接管/路由。若失败，本固定边界终止，不靠追加prefix/seed/噪声网格凑成功；据接触/时间耗尽区别再立机制。不是新增一个“六米通过率”。

区别旧负结果：T183/T204径向起点课程不是T317人控完整动力学历史；T278固定目标余1m来自旧policy末段/新reset，也不是这两个40s可达状态；T320全程人为drive=.999而本候选后缀为冻结Actor反馈，两者不能混称相同实验。以上区别不保证本候选有效。

2026-11-03检索一手来源：

- [Backplay arXiv v5](https://arxiv.org/pdf/1807.06919)，读3节以及Figure2/4节负面讨论。采用示范状态用于训练起点的思路；文中指出不佳示范可使反向课程更慢或锁入次优区，不能假设near-goal就容易。其maze有A*示范且游戏状态不同，本文不引入其规划器，也不声称复现完整Backplay。
- [RFCL作者项目](https://reverseforward-cl.github.io/)、[作者代码](https://github.com/StoneT2000/rfcl/blob/main/rfcl/envs/wrappers/curriculum.py)，读reset/set_env_state/动态timelimit逻辑；完整paper OpenReview浏览挑战、arXivPDF过大未完整读取。作者需要environment states，且有自己的动态时限。本项目采用真实native前缀精确重建并保留原剩余时域，**没有照搬动态时限或声称其理论保证**。无第三方代码复制/依赖安装。

这只是有辨别力的前置检验；标准示范reset不是原创算法/安全证书。当前没有新launch，必须T322完全结束/audit/备份、实际容量核算、零步预检与源完整远端后才可唯一执行。
