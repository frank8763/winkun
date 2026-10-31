# T238：仅修复预检求积域的数值问题

T237在动作区间[-1,1]求积触及均值之外约65个标准差的极小密度点，前向CDF在float64下为0，按强断言退出。原源码/协议/log和失败后处理summary保留，零物理/优化器。未据此放宽在线密度门槛或启动RL。

T238科学策略与T237一致，仍基于[ARP](https://www.ijcai.org/proceedings/2019/0382.pdf)/[PPO](https://arxiv.org/abs/1707.06347)。重新查阅[torch.special.log_ndtr官方文档](https://docs.pytorch.org/docs/stable/special.html#torch.special.log_ndtr)。仅将求积变量改为条件latent中心±7σ，遗漏标准正态尾质量小于3e-12，低于原1e-7门槛。用连续float64逆CDF的autograd da/dz与独立前向CDF密度交叉核验，在线仍float32且不能反求则拒绝。1024节点/16384MC样本/四分量有限差分/门槛/种子不变，不新增导航/参数调优。
