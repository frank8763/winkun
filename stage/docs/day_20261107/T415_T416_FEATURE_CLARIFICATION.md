# T415/T416冻结特征的准确解释（核验更正）

2026-11-07，0native/0Adam。原始源、协议、metrics及远端证据不覆盖。

实际GRU为`GRU(32,16), batch_first=False`。`core(encoder(normalized)[:,None])`的输入N×1×32代表N个时间点、batch1，确实沿完整轨迹carry hidden；输出N×1×16拼接原120字段得到136维因果特征。此前仅凭输入形状将其误读为每行zero-init length1是不正确的，已明确撤回。该误读没有改变任何实验源、数据或训练。

非零输出克隆（仅在内存将extra最后层权重设为0.02，0优化器/0原生）在实际旧完整轨迹上核验sequence与逐step carry，location最大误差3.3306690738754696e-16。证据T417_PRETRAIN_SEQUENCE_DIAGNOSTIC.json。正式T417预检还将在已有合成更新后的克隆上直接比较每时刻hidden及location，不增加既定8次合成Adam。

T415/T416两轮使用同一冻结T255表示，完整因果history得到重建；这仍不证明Markov充分性、同状态动作排序或六米稳定。科学结论不变：T416比单步传播更多终点信号，但没有自动接入Actor。下一表示适配试验研究可训练参数集合，不将其解释为修复记忆实现bug。
