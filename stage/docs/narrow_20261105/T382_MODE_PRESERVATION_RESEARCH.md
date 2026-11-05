# T382：可行序列均值失效与最小后续检验

2026-11-05；研究笔记，未实现、注册或执行下一实验。Stage only，0新增native/physics/Actor/optimizer；只新增本笔记，不改T382原件。

## 已有结果与能成立的解释

T382把人控**动作**先验、4秒原生预测、每次仅执行0.2秒及可行候选筛选联合使用。真实冻结Actor前缀8秒后，接受54个教师hold，实际到18.8秒、余距3.9122163888米，实际CONTACT/GOAL/timeout均false。第55次重规划（索引54）拒绝均值并按协议停止，未执行危险均值；这是未完成的诊断停止，不能写成实际碰撞或完整任务超时。两pass逐位一致属于1次混合轨迹及其精确重复；独立审计PASS，合计775命令、1,356,568物理步、1790.470秒，Actor/优化器更新0。[原summary](../../runs/t382_receding_human_teacher/summary.json)、[原audit](../../runs/t382_receding_human_teacher/audit.json)、[注册协议](T382_PLAN.md)。

只读检查[原predictions.npz](../../runs/t382_receding_human_teacher/predictions.npz)索引54：

| 输出 | cost | 可行权重 | 原生预测 |
|---|---:|---:|---|
| 候选0 | 34.6646218959 | 0.4041131413 | 完整4秒，无CONTACT |
| 候选1 | 34.7568668074 | 0.3685034154 | 完整4秒，无CONTACT |
| 候选4 | 35.2396790173 | 0.2273834433 | 完整4秒，无CONTACT |
| 三者加权完整控制序列 | 1034.5141735236 | — | 1952步，即3.904秒CONTACT |

同一actual snapshot SHA256为`38f21078b734ddb4a6115066ed4ff421c9b5581c11f8dc83daa9b20ae38683a7`。这给出了**可行控制序列对本次加权操作不闭合的原生反例**，符合非凸可行约束下均值可能失效的机制。不能进一步断言三条轨迹属于不同mode/同伦类：非线性动力学、车体姿态与接触约束即使在同一通道也足以使控制均值失效。没有证明场景无解，也没有证明继续采用某候选必能GOAL。[原预测数组](../../runs/t382_receding_human_teacher/predictions.npz)、[实际轨迹](../../runs/t382_receding_human_teacher/mixed.npz)。

## 两项一手方法与作者实现

- **SVG-MPPI，Honda等，ICRA 2024**：v3，2024-03-01。Fig.1/§III–IV讨论单高斯拟合多峰分布时可能产生碰撞均值；用修改后的SVGD输运guide，选择最低cost guide作为nominal，再估计协方差并执行MPPI。§V–VI同时报告仍有碰撞及平坦区域梯度失效的限制；它不是可行性保证。[全文PDF](https://arxiv.org/pdf/2309.11040v3)。
- **SV-MPC，Lambert等，CoRL 2020/PMLR 2021**：arXiv v4，2021-04-12；§4–5、Appendix B/J保留粒子后验，并允许选最高权重整粒子（Eq.24）或按权重抽取粒子；导航实验采用最佳粒子。对非可微cost可用Monte Carlo的score梯度（Eq.20），并非必须对碰撞模拟器直接反传。该论文支持“保留候选身份再选择”，不证明T382已经存在多个mode，也不保证任意选中的粒子安全。[v4全文](https://arxiv.org/pdf/2011.07641v4)、[正式全文](https://proceedings.mlr.press/v155/lambert21a/lambert21a.pdf)。

核对SVG-MPPI作者仓库commit `f4fb7b74f12d4568790213aab26dd6f0538d1925`（MIT）：`stein_variational_guided_mppi.cpp`第54–131、161–218行确实有额外采样估计梯度、最低cost guide选择，**最后仍加权平均控制**。当前yaml为8000主样本、1guide、10次输运、每guide梯度100样本；不能把“一guide”误读成与T382五个总候选等预算。[作者源码](https://github.com/kohonda/proj-svg_mppi/blob/f4fb7b74f12d4568790213aab26dd6f0538d1925/src/mppi_controller/src/stein_variational_guided_mppi.cpp#L54)、[配置](https://github.com/kohonda/proj-svg_mppi/blob/f4fb7b74f12d4568790213aab26dd6f0538d1925/src/mppi_controller/config/mppi_controller.yaml#L91)、[许可证](https://github.com/kohonda/proj-svg_mppi/blob/f4fb7b74f12d4568790213aab26dd6f0538d1925/LICENSE)。未安装、运行或移植作者代码；本次未确认到SV-MPC原论文的独立原作者代码版本，不能把SVG仓库里的比较实现称为该论文原实现。

适用边界：SVG论文用参考路径/速度waypoints、LiDAR costmap、bicycle预测模型和转向控制；SV-MPC导航为holonomic point robot。它们与Stage含轮胎车体、双动作、0.002秒原生步、0.2秒hold、120维因果局部Actor输入不等价。只借用选择机制；不引入论文的地图、路径或部署规划器。[SVG §V](https://arxiv.org/pdf/2309.11040v3)、[SV-MPC §6/Appendix J](https://proceedings.mlr.press/v155/lambert21a/lambert21a.pdf)。

## 建议下一项：固定状态、固定样本，只改变输出选择

另立有界协议，恢复**T382索引54的完全相同停止snapshot**（含203slot积分、Python状态、history、previous动作、Actor hidden及真实clock），复用原已评估五条完整控制，禁止重采样或调整cost、噪声、时域。预声明在完整4秒无接触或真GOAL的候选中，按`argmin(cost, candidate_id)`选择**整条序列**，这里确定为候选0；不做任何序列/动作混合。原均值失败作为既有对照，不把危险均值真实执行一遍。

从该snapshot用同一native检查完整4秒所选序列，并先核对其原预测trace/积分；通过后恢复同一actual snapshot，仅实际执行首0.2秒，核对实际record、hidden/history与预测首hold。任何不一致/接触即停。该局部检验通过只说明“避免本次聚合拒绝并精确执行一个已验证hold”；因为候选0本来已经通过预测，不能把这一近似重放当作新导航能力或独立安全统计。完整4秒也不是递归可行性或全程GOAL证书。

若随后检验连续重规划，必须在新注册中预锁所选整序列的移位/补尾规则、最大replan数、actual终止与诊断停止、全部恢复/预测/执行/精确重复计费、墙钟及最坏保存字节，先通过global1GiB/free40GiB容量合同；不得在已结束T382里临时fallback或续预算。保持全原生输出验证这一层；失败后再定位“无可行样本、输出选择、短视代价、状态恢复”中的哪一项，避免一次加入聚类、协方差自适应和新梯度而失去归因。

暂缓DBSCAN：只有5条总样本、3条可行序列，不能可靠识别控制/轨迹空间的密度簇；新距离、epsilon、minPts会引入额外选择，且簇内均值同样不自动可行。暂缓SVGD：两篇论文已说明可用采样score梯度，不应声称native不可微就不能用；但那需要另计梯度样本、核/带宽与采样分布合同，超出当前“选择还是平均”的最小单因素问题。以上是本项目证据下的取舍，不是两方法普遍无效的结论。

本阶段仍仅为训练期特权教师研究，部署保持单Actor；教师预测或混合局部通过不能报告Actor自主六米稳定。未产生新训练、成功教师完整GOAL、创新性或泛化证据。

## 查阅范围与追溯

检索/访问日期2026-11-05；限上述两篇原始全文PDF的相关方法、算法、实验及限制段，另读作者源码/配置/许可证指定范围；未逐字读完所有附录、复现论文或穷尽最新文献。arXiv v4部分网页提取超时后用PMLR正式全文核对Eq.20/24与Appendix B/J。未采用搜索结果中的二手解读或同名但含图搜索/waypoint的SVG项目。局部证据仅T382协议、summary/audit及两原NPZ的只读核对，未读Phase资料；Stage历史聊天仍只有既有迁移报告所述范围。

输入SHA256：`summary.json dd80a35a625bcccca1fcedf389740d9d068ff4ff020cff1db29e36654d8bc708`；`audit.json b6b2c77992f6a6220d5a444d744d01e7fd33e35ed38e3b8e06182b8a63b8f23d`；`predictions.npz 7e98e75ba745a8a565f03f17c034e6711bc1759ae00c38b99a92eadb6995aa57`；`mixed.npz 596516e1f042557c5cddd0cf13a9001b39eca56fab74b044fc78070ca37fcd0f`。
