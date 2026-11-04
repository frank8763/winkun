# T337：动作边界机制的零物理数值合同

用户要求复盘/参考论文并给下一方案。只读T336两臂首轮各4条既有full训练数组（fixed674、reverse152决策），0native/环境创建、0Actor/V/Q拟合、0优化器；不重构旧SAC更新，不追加T336预算。

比较已记录tanh-Normal与两个未训练数值候选：正确边界质量的CAPG clipped Gaussian、mode保持Beta。目标初始实际动作标准差drive.035/steer.03，固定seed20261104337、每状态512离线抽样；这些是分布样本，不是仿真导航样本。Gaussian用clip解析一二阶矩、Beta解析方差各固定48轮二分求唯一初始尺度，不做训练参数搜索。固定256点Gauss-Legendre积分检查CAPG质量归一化与score期望，自动微分和中心差分检查边界logp梯度。

记录初始确定性mode误差、真实动作变化、CAPG边界质量、初始loc残差Jacobian，与原tanh映射比较；不是整个神经网络/优化器梯度效益或安全证明。预检只检查原件SHA/数组shape/样本数/资源，0随机/梯度。主数学assert：初始mode误差<=1e-12；解析标准差误差<=1e-8；CAPG质量误差<=1e-8；score期望绝对值<=1e-7；有限差分相对误差<=1e-5；所有logp/导数有限。有限抽样方差仅描述，不以MC波动挑seed或修改阈值。

own256KiB/global1GiB/free>=40GiB；主180秒与独立audit180秒。主/审计均独立同seed重算，不称两份独立数据；所有输入/源码SHA、原始T336模型/safeT255只读。launch前本地干净commit/完整远端根树与正式预检；main单writer锁，不写其他runs/源同步。输出只小JSON，不复制模型/训练数组。结束所有父/timeout/inhibitor退，唯一audit精确结果一致，再RAM全原件远端逐成员SHA及源码同步。失败保留，不通过不接原生训练。任何新的CAPG-PPO导航都须另外完整有界协议，当前仅数值原型。
