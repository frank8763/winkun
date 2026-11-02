# T296：同一AR概率分布的对数尾部独立复核

2026-11-02注册，Stage/Phase完全隔离。仅用已结束/审计/完整远端读回的T295数据，不追加物理轨迹/目标/seed/噪声。0native/Actor/optimizer/Critic，main300秒、独立audit300秒、own1MiB/global1GiB/free40GiB。先零步预检PASS/代码提交/远端精确根树同步才唯一launch，不恢复T295目录。

问题是T295旧实现CDF取exp后在新goal均值下被拒绝，尚不等于概率支持为零。新实现对截断高斯的lower/upper interval做反射和logdiff，选择较小尾部的logCDF，固定12步Newton反演logCDF，再按原rho=.98/sigma=(.035,.03)计算同一AR条件密度，当前及previous mean依赖保持；不改变策略、动作、采样器、reward/阈值，不截尾CDF/概率/权重。没有SciPy依赖安装。

思想依据：[SciPy官方ndtri_exp说明](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.special.ndtri_exp.html)指出在极端概率下inverse-logCDF较ndtri(exp(x))保留精度；这里用现有Torch实现Newton，独立mpmath90位参考。14极端标量预检：z误差3.13e-12、log marginal误差1.37e-12，AR当前及前均值有限差分误差9.56e-8，均通过。此验证不等于HPG原文结论在本任务无偏。

8条原mean必须逐位匹配，原logp对历史误差≤1e-8；新goal logp/prefix/末权重均不裁剪，log稳定归一及ESS。目标、样本、两组每四条、ESS≥2/maxweight≤.75/每条末abslog≤20全沿用T295，不能放宽。辅助goal由历史数据选取、已消费开发集；有限四条仅支持筛查，不证明HPG普遍失败或目标不可达，不是原六米目标的无偏估计。冻结T255六米完整依然未通过，固定动作辅助GOAL不能替代自主验收。

结束后独立重算全部原新密度/累计比率/数组/输入和Actor SHA、90位数值/梯度合同与门槛，再RAM全原文件远端逐成员SHA/源同步。筛查未过不据此更新Actor；即使过也须另立真实训练/完整旧275+六米54+16fresh+78独立重载。HPG/数值稳定算法为已有思想，不声称原创、安全证书或六米已解决。
