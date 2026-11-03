# T309注册前文献/作者实现与反证核查

检索日期2026-11-03，Stage only。用户持续要求每项新实验查一手论文、实现与相反证据。本轮未下载大仓库或运行作者代码，以下范围不等于完整复现。

1. [Three Steps at a Time](https://arxiv.org/html/2608.30640)，2026-08-31v1预印本。核对§4、§5.2/5.3、附录E.1：三动作条件表示/DQC单动作expectile.95的依据；原论文单动作蒸馏展示包括H5，本试验H3→H1是Stage固定工程选项，并非原成绩复制。论文也发现预测改善不保证策略提取成功，在线收益与执行方式可能纠缠，不能据此保证单动作Stage必然获益。
2. [作者总仓库README](https://raw.githubusercontent.com/M-Korniak/action-chunked-contrastive-rl/main/README.md)确认在线JAXGCRL与离线OGBench/FQL/DQC是两个子模块；总仓库MIT、子模块另有许可证。只核对README与论文，不复制作者源代码，不把PyTorch小车实现称完整JAX/OGBench复现。
3. [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969)，NeurIPS2025，2026修订。摘要作为序列价值与策略提取近邻；本项不部署连续开环chunk，不引入其离线先验/机器人数据。
4. [TD InfoNCE](https://arxiv.org/html/2310.20141)及[作者页面](https://chongyi-zheng.github.io/td_infonce/)已在方法审阅时核对。可支持长程价值传播，但本项暂不加TD，避免与新数据/表示/蒸馏一同堆变量。新obs的Markov性、终止和目标历史不能借其理论直接假定。
5. [IJRR SRU论文](https://journals.sagepub.com/doi/10.1177/02783649251401926)为长距无地图RL的架构近邻；当前不加SRU/记忆。T255记忆臂210/211、reset选中211/211，尚无本项目记忆获益证据。

Stage反证：T304原回报三步小回归头与T308单动作对比值均未过真实控制筛选。T309新增多真实目标、动作表示和DQC，不能把两历史失败抹掉，也不能将整体收益单归因H3/DQC。部署单120hidden-reset Actor、原生两动作、原0.2s/2ms时标不变；没有外部地图、规划器、路点、运行时教师或策略路由。

候选未来创新问题是有限终止/接触/goal-dependent历史下的可控价值转移，不是“使用DQC”本身。需后续独立算法机制、消融/多训练seed/跨任务布局结果，当前无原创/安全证明或论文效果声明。
