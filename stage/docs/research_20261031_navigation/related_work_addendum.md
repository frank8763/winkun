# 新增检索与适用边界（2026-10-31）

T210注册前补查[Engstrom等ICLR2020](https://arxiv.org/abs/2005.12729)摘要、[Spinning Up PPO](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)实现页面；运行期间核对其独立Value拟合和KL提前停止片段。其实现是参考，不是本项目直接复制的实现，也没有将PPO推导当作本机稳定性保证。

进一步找到[CLI-PPO，Sensors，2026-09-25](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667403/)，查看方法/动作定义、表8/9、消融及数据声明。该研究已组合PPO、LSTM、ICM和课程，因此这类组合不能直接称本项目创新。其全向三维速度指令与本车油门/舵角不同；高障碍数测试中无LSTM版本反而更好，进一步支持分别消融。没有获得作者源码，不声称复现或可靠迁移其数字。

T210首步动作分布变化偏大后，再查[TRPO，ICML2015](https://proceedings.mlr.press/v37/schulman15.html)原始页面。借鉴的是“在接受更新前检查策略变化”的原则，尚未实现或复现完整TRPO；本项目截断高斯的解析KL另做数值积分核验。离线参数插值不是导航训练/成功，不把有限状态KL门槛包装成全局安全保证。
