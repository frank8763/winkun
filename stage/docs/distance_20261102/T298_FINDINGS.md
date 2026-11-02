# T298结果：六米2/54，伴随旧能力退化，未进入RL阶段

T298于2026-11-02结束，COMPLETE/stop SUPERVISED_FIT_SCREEN_FAILED，实际947.435秒。单一固定500步自身轨迹辅助目标监督学习完成，每步shadow精确匹配，共500 live+500 shadow监督Adam；预检另8次丢弃clone，独立审计另1000次丢弃重构。新六米RL采集/优化器更新全部0：监督筛查、旧20保护和辅助中心条件未全部通过，不能称本轮新增RL收益。

| 原生验收 | 有效完整 | 安全GOAL | 接触 | 超时 | 无效/删失/缺测 |
|---|---:|---:|---:|---:|---:|
| 旧20保护筛查 | 20 | 18 | 2 | 0 | 0 |
| 两固定辅助目标中心自主执行 | 2 | 1 | 1 | 0 | 0 |
| 完整旧275回归 | 275 | 267 | 5 | 3 | 0 |
| 原六米LEFT开发 | 27 | 0 | 27 | 0 | 0 |
| 原六米RIGHT开发 | 27 | 2 | 0 | 25 | 0 |

六米完整2/54=3.7037%，成功为dev6_RIGHT_18（55.644s）及dev6_RIGHT_21（55.322s），两者都接近原56s时限。RIGHT_18此前已知；限定Stage T256–T296含明确dev6_RIGHT_21/source_reset_id的原生JSONL扫描，8条历史匹配均未安全成功，本轮RIGHT_21是一条新开发成功记录；不称跨seed/地图泛化或稳定扩距。每个本轮验收仅执行一次，16fresh/78独立native读回均**未执行**，不是失败0%。七八米未进入，无新合格六米视频。

旧275=267成功+5接触+3超时，失败全部在已知5m开发家族：LEFT dev_task6_00/06/09/12/24接触，RIGHT dev_task7_02/05/11超时；其余任务和原64家族回归保留。safeT255本体未改变，仍保留原211有效开发+64留出全过的五米基线。研究候选不晋升/不替换。

## 学习信号与本轮实现缺陷

全辅助数据等轨迹动作MSE .2465345562→.1240077019，下降49.6997%，门槛为至少50%，未放宽。即使舍弃这项误差门槛，旧20仅18/20、辅助中心仅1/2也使组合门槛失败。辅助中心左GOAL/右CONTACT；没有同时执行safeT255在新辅助goal下的自主原生控制，所以不能把左成功单独归因这500步。

这轮辅助loss采用`mode(raw)=tanh(atanh(clamp(raw)))`后的动作MSE，暴露了一个应当修正的优化路径问题。初始LEFT辅助数据约99%的drive原始输出在硬裁剪区，84.35%–96.28%的辅助动作平方误差落在零输出梯度位置；500步后LEFT仍81.82%–89.32%误差处于零梯度区。RIGHT多数输出未饱和，误差下降更多。独立后验T298_POSTHOC_SATURATION.json用真实数据/合成有限差分证明硬裁剪区对raw的局部MSE导数为0；相同已有条件AR概率似然（T296对数尾部实现，保留previous-mean导数）在这些坐标有有限非零输出梯度。后验不改变本轮门槛，也不是全参数可学性/闭环收益证明。

这是**本轮新增辅助监督路径**的缺陷，不能倒推为过去所有6m真实策略梯度失败的唯一原因；原RL使用的是平滑center下的条件概率梯度。五米冻结策略本身失败、长目标条件化不泛化、HPG样本权重重叠不足和新学习引起的旧闭环退化仍是不同问题。不能只说地图难或灾难性遗忘。

文献核对：[GCSL论文](https://arxiv.org/abs/1912.06088)、[作者训练loss](https://raw.githubusercontent.com/dibyaghosh/gcsl/master/gcsl/algo/gcsl.py)调用policy条件NLL。[公开网络实现](https://raw.githubusercontent.com/dibyaghosh/gcsl/master/gcsl/algo/networks.py)提供离散和分维离散随机策略头，本项目的连续截断AR不能冒充直接复现。T298仅借鉴自身轨迹目标重标记思想，裁剪MSE实现及约束是工程适配，尚无新算法贡献证明。

## 独立审计、备份及资源

T298_AUDIT.json精确重构500 live+500shadow监督Adam、五个中期Actor/Adam哈希、最终Actor和fresh RL Adam；全部源/351原生payload/派生reward及命令物理核账通过。真实351命令/5364680全部物理步，无额外训练/replay。新模型checkpoint SHA `6e332f6c65a980fe738f66bfb21a3f2b0e930b93e7808350374c8b2eae49ab95`，Actor SHA `f7af27af0a92c57e5b17e44dd89a8ceb54b171a1bbbb819470d3193b5296676e`，fresh RL Adam SHA `881489e608a38e54726d0751cd52ed648891a06e8603b81357fc9935f73c3d76`。监督Adam不作为RL续训状态，最终另保存其哈希、精确源码及可独立重构记录，符合预注册存储方案。

所有父/四worker/timeout/inhibitor在截止前退出。374原始文件已RAM归档，远端完整二进制读回逐成员SHA通过；收据long_distance_20261101/release_receipt_T298.json。safeT255和两1080p用户视频SHA再次完全一致。global逻辑约1010.84MiB<1GiB/free约209GiB，原模型/失败/数据完整保留。T297的69旧日志无损编码独立SHA验证，释放7772236B，无唯一字节删除。

原注册协议尾部遗留“2RLpair reconstruction”文字，明确数值字段及最终cap是**最多1live+1shadow RL**；本轮实际0，独立重构仅1000监督调用。原协议不覆盖，差异在此说明。预检metadata拒绝与启动前同own内的努力缩减证据保留，不是训练中改协议/阈值。
