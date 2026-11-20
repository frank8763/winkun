# T670：旧任务结果梯度与可行方向审查

T669四ID各8当前策略自主轨迹均有安全成功和失败，已独立auditPASS、所有科学PID退出。此项0物理0原生命令0优化器，CPU14单线程、600秒计算+600独立重算、30分钟总硬停、16MiB。模型11906个动作适配参数的原值最终逐hash保持。

每旧ID y=+1安全GOAL/-1碰撞/0其余，固定LOO_i=(sum(y)-y_i)/(n-1)，score梯度为mean_i[(y_i-LOO_i)sum_t grad log pi(a_t|完整实际因果history)]/原score_scale。真实clipped Gaussian含边界atom密度重建<1e-9，所有history无teacher/ownership、初始policy sigma完全相同。全同标签没有样本变化不能当可靠零风险约束。初始梯度为声明当前随机目标的一阶有限样本估计，不保证确定性mean控制。

新目标score取此前T663 low首块24field各2样本，同行为policy/sigma64+exact核验后复用，不重计独立采样；原其余8anchor不用作newfield。以(24field+4反例)任务均衡平均结果梯度加原replay10×387名义保持构造软loss梯度。全骨干冻结使真实causal特征缓存对适配参数精确，不能因此声称环境反馈被微分。

在0优化器条件下计算fresh Adam第1步的解析位移(-lr*g_clip/(abs(g_clip)+eps))，明确未发生真实Adam调用。按4个归一化旧结果score半空间g_j dot d≥0，对此位移求欧氏最近投影；枚举16active sets、KKT/primal/dual/complementarity核对。已有GEM/PCPO/Adam-NSCL近邻，不声称新算法。投影发生在真实参数位移空间而非投影后再被Adam预条件；若后来实训仍需每次Adam实际after-before提取及非线性采样目标复验，并完整恢复Adam状态。

门槛预注册：4旧梯度非零、参数方向数值差分abs1e-5+rel1e-3通过、KKT可行、软位移确有旧冲突、投影改变方向且非零、新场地梯度dot投影>1e-12。另测两方向的非线性逐决策ratio采样surrogate，不将它当完整trajectory importance或实际到达率。临时写入参数只做函数评估，finally逐参数恢复；0优化器调用。结果门槛失败不调阈值救。

主计算保存所有raw梯度/方向/Gram与逐ID支持，独立重算梯度数组及结果字典完全一致。审计PASS与科学门槛PASS分开。PASS仅许可下一一次同数据同初始化同优化器的软平均结果vs逐旧任务结果投影学习pilot；原384全保且新增、同候选407/独立audit仍唯一保留门。反例是开发集合不是新留出，没有普遍无遗忘或稳定性保证。
