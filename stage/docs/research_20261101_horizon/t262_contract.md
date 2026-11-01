# T262零步事件数据契约结果

固定复用T261所锁定的T259最初48条轨迹，不产生新独立导航样本。原始native end_flags顺序是[CONTACT,GOAL,TIMEOUT]。第一版原型读反前两列，在合成GOAL边界断言即拒绝，尚未处理真实轨迹；原源码/协议/拒绝收据保留。恢复适配器只按正确列顺序转换，并逐条与48个原始metadata核验，独立子目录输出。

恢复后合成与原始合同通过：真实2ms物理时间，原cap以内查询，到达/接触互斥，真实超时为时限内无事件，行政删失未知后缀屏蔽，概率质量守恒。48条轨迹共11415个相关决策行，5/10/20/40/56秒查询产生31838个有效标签（GOAL281、CONTACT3695、观测到该时刻仍无事件27862），25237个未观测或超原剩余cap的查询屏蔽。数字是相关标签数，不是独立样本量。标签NPZ仅2772B，物理/优化器新增0。

没有训练任何Critic或Actor，没有新增导航成功；当前T259源码和协议hash保持。T259紧凑文件不含120维状态，所以critic_training_data_ready=false。当前完整结果在stage/runs/t262_event_contract/flag_order_recovery/summary.json，失败与聚合状态均保留。所有来源与候选研究边界见本目录review.md/candidate_event_method.md。

原始失败/协议/标签/恢复结果共5文件已完整远端读回、逐成员SHA核验：[Release](https://github.com/frank8763/winkun/releases/tag/stage-distance-2026-11-01)，压缩包5601B，SHA256 `941b2ee31491d02c6c8f406ee8776c174df6845ded31255410e248996cdcff55`。本地原件保留。
