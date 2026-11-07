# T415：原生物理时间IQL价值学习pilot

这是新机制的有界离线价值试验，不是T414续训／加BC轮数。T414已结束且原始备份完成；本轮Actor完全冻结，0新导航、0Actor更新、0七八米验收。单Actor部署约束不改，Phase完全隔离、窄场景暂停。

22条既有完整轨迹：T366的五条LEFT成功、T405完整RIGHT成功、T414的四条真实混合控制成功、T410的12条失败学生轨迹。只使用每次物理转移真正执行的动作，未执行的教师查询标签绝不当转移。T366目标位于已审计cases.json而非NPZ，本注册前缺字段0优化器／0物理错误和旧源码保留；只适配同一真实目标元数据并锁SHA，没有生成新目标。

14train（LEFT偶数姿态三条、RIGHT一条、真实混合偶数两条、失败前两轮八条）；8validation（LEFT奇数姿态两条、混合奇数两条、失败最后轮四条）。按整轨迹划分，精确转移payload重复跨集合拒绝；共享前缀／相近状态可能存在，不称未见地图或完全独立状态。3231训练决策、1963验证决策；原生重放不追加样本。

状态为当前120因果输入normalized与冻结16维循环状态，history从该轨迹真实观测从t0重建，不取过去Actor的缓存hidden作新hidden。因输出残差不修改冻结encoder/core，价值特征在本轮完全固定；实际80秒remaining字段index112与累计nativecount逐条断言，reward由原goal/pose仅用于独立核验后按既有公式精确重建，不进入模型特征。nextstate是真实最终观测，跨episode不连接；GOAL/CONTACT与固有80秒任务cap都terminal，行政censor不入库。SMDP折扣为原gamma_native**实际hold，非每条任意.99，末缩短hold照实计。

IQL twin-Q和V各128/128小头，Q输入136+实际action2，V输入136，固定seed20261107415／lr3e-4／expectile.7／target Polyak.005／batch256／3000迭代，liveQ3000+V3000；独立审计重建同样6000丢弃Adam。预检2迭代×2克隆各Q/V，合8丢弃headAdam，0Actor。V拟合targetminQ的上expectile；Q目标R+terminalmask*physicaldiscount*V(next)。不采当前策略的域外动作，不训练Actor。仅诊断计算权重exp(3A)cap100、ESS／饱和，不能把它当已验证的动作选择权重。

保存train/validation的行为MC误差、Bellman误差、每条起点值和权重统计；MC是原行为回报而IQL估计隐式改进策略值，两者误差只是描述性诊断，非校准／动作排序／安全证明。heldout仅根据原轨迹固定计算，不以它调参/换seed/重训。无论这些数值如何，本轮不自动接Actor或宣布六米通过；必须另立可证伪的动作价值验证和匹配Actor提取对照，避免把批量BC再包装成IQL收益。

预算own6MiB（模型含Adam≤4MiB）、globalruns1GiB/free40GiB；全部数据复用不复制大轨迹，特征仅RAM。main900s、audit900s、backup300s、all2400s；今日19:30停止学习、20:00全部结束更严，先预检、提交精确远端源码后唯一launch，结束全部子进程退出→audit逐state/target/Adam/metrics/events精确→必要原件RAM归档远端逐成员SHA。已结束目录不重启、不增加迭代网格。Stage历史不批量恢复，遵守存储规则。

IQL／AWAC为现有标准方法，见同目录T415_IQL_SOURCE_REVIEW.md与一手引用；不称原创、安全证明或全局已解决。本pilot小网络批量CPU拟合，无需新装GPU引擎；本机CUDA已实测可用，后续大训练需另核GPU吞吐与一致性，不伪称本轮CPU小头已利用GPU导航并行。
