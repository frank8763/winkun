# T230：所有剩余有效失败起点的一次固定矩阵自身探索

T229已完成全216起态零步有效性，5个4mRIGHT初态重叠单列，不计导航策略失败；211个有效已全部覆盖（复用T228原实测+75新rollout，其中4中心严格复现）。3m108/108、4mRIGHT22/22、5mRIGHT27/27；剩4mLEFT3个、5mLEFT4个碰撞，204/211。没有悄删有效失败，原T228严格门槛仍未过，64留出未运行。

注册前重新检索并读DART正式摘要/作者实现说明（https://proceedings.mlr.press/v78/laskey17a.html ; https://github.com/BerkeleyAutomation/DART），Data Quality in Imitation Learning原摘要和检索可访问理论片段（https://arxiv.org/abs/2306.02437 ; https://papers.neurips.cc/paper_files/paper/2023/file/fe692980c5d9732cf153ce27947653a7-Paper-Conference.pdf），以及前轮Colored Noise PPO方法。动机是补充恢复成功源，但状态多样性、噪声动作不保证闭环改善；T228左侧仍失败就是项目反证。不宣称SIL/DART复现或算法贡献，不引入地图/路点/运行时教师/Phase。

对全部7个有效碰撞起点（不只挑一例），冻结同一个已保留T228Actor，以(.035,.03)/AR.98和预给六种子2026103241–46各执行一次，共42条新源，每条独立完整再生成一次、10数组/最终积分精确。所有成功失败保存，不bestseed、不看结果扩矩阵；每个失败reset至少一安全GOAL才另立学习协议，不能放宽门槛。低sigma的截断高斯CDF差/动作有限边界加入运行时检查，若下溢则拒绝不能静默反向动作；前T227实际数据扫描mass最小2.45e-7、零zero-mass，不改写其结果。源科学条件为同冻结T228小幅探索，非新的RL训练或确定性策略通过。

实际锁84命令/1960056所有物理步/600秒/12MiB；零优化器、无模型导出、global1GiB/free40GiB。所有42条完整执行，不首次成功提前挑样本；新源必须经过独立生成核验。T229原预算/旧重放另计，不把重复当新增独立训练。所有GOAL/.2m半径/终点/场景/时限不变；失败保留、结束退出，不自动晋升/长距离/下一训练。T159/T194/T221/T228保留。
