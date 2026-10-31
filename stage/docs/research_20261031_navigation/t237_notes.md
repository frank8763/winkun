# T237：相关探索必须作为条件策略验证

2026-10-31查阅[ARP一手论文](https://www.ijcai.org/proceedings/2019/0382.pdf)、[Colored Noise PPO](https://arxiv.org/html/2312.11091v2)及[PPO](https://arxiv.org/abs/1707.06347)。只借鉴时间相关探索和明确条件策略，不宣称论文复现或新算法成立。

定义z_t=Phi^-1(F_trunc(a_t;mu_theta(s_t),sigma))。条件密度为Normal(z_t;rho*z_(t-1),sqrt(1-rho²))乘截断正态密度/标准正态密度。训练时历史状态/动作固定，z_(t-1)也必须按当前参数反求并保留梯度。每回合reset使用rho0；sigma(.035,.03)/rho.98锁定。旧T225/T227/T230占位0 logp永远不用于此更新。

采用反射和log_ndtr计算极小截断质量，不做静默CDF夹取；若原生float32动作不能反求有限概率则拒绝。T237只零物理数学门槛：固定高阶求积归一、reset/白噪声恒等、前后均值四分量有限差分、Monte Carlo score零均值、实际float32动作反求与正负极端尾部。未全过不进入后续RL。蒙特卡洛sanity不是算法性能/安全保证。
