# T253条件草案：训练动作分布的零latent读取

2026-10-31查阅[scipy.stats.truncnorm一手官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html)，结合已读ARP/PPO。截断分布的中位数、均值与模式不同，SciPy提供cdf/ppf/median等接口。没有论文保证确定性median比mode更好。

代码核对：近期RL训练动作由physical_mean(center(raw))配小sigma(.035,.03)，经截断Gaussian inverseCDF(Phi(ARlatent))取得；冻结评估则直接裁剪raw。latent/innovation全部置0的原训练计算路径为分布median，不等于裁剪mode。模式读取本身是合法选择，不能叫数学错误或全部失败根因，但应测试部署分布选择。

只读当前T23928588状态得到zero-latent与clippedmode油门差q50=.00443/q99=.02104/max=.02360，64.7%大于.001；转向几乎相同（q99约3e-8，max.00497）。这是数值差异，不是导航收益。没有按哪个量更好而扫描分位数/人为减油门。

若T250未过门槛且仍未使用64导航留出，固定选择早已保留T239210/211做一次单因素读取对照：先独立math.erfc正态CDF+100步动作域二分的固定网格median一致性<1e-7、四clippedmode原控制位一致，再latent0原八中心+四临界双次精确。任何失败就停，不换mean/quantile/sigma/temperature/模型/频率继续搜索。全12安全才完整211，再全过才64原留出。原100步控制/目标/碰撞/总时间不变，单神经Actor前向加原训练分布解析逆变换，无规划器/教师/路由/新控制规则。零权重/优化器更新，不伪报新增RL或创新成立。303命令/7.070202M物理步/1100秒/7MiB/global1GiB；草案不自动启动，须查前一作业原始完成状态再注册。

本机mk_mujoco环境没有SciPy，预检查出后没有安装新依赖。官方SciPy文档作为截断分布/CDF中位数定义来源；实际独立数值参考用标准库math.erfc与动作域二分，和Torch反CDF算法不同，门槛不变。
