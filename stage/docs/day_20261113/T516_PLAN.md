# T516：冻结父策略的非线性恢复兼容性，离线单轮

用户11月13日授权下一步开始。Stage only，与Phase聊天/研究/结果/模型/进程完全隔离。T515已结束：20候选2GOAL/1CONTACT/17TIMEOUT、20exact、128真实headAdam+128shadow及128独立历史重构PASS，完整47原件远端SHA，全部回退，原接受field11/old8模型不动。

本轮不仿真，不再修改全局共享原head。冻结接受T507父Actor全部参数和原成熟Adam，新增184→64tanh→32tanh→2非线性logit修正13986参数，末层零权重零bias，其他确定性seed20261113516。最终location=1.25tanh(parent_logits+new_net([真实hidden64,normalized120]))，修正logit不加旧±.3或T511±.1位置幅度限制，输出仍按原Actor范围、原float32动作边界。所有20完整历史的零初始函数必须逐位等于父策略。新网络freshAdam0/lr.001，原父Adam留在原protected文件不可变；Value/ValueAdam冻结。部署候选仍单120因果Actor、64真实GRUcarry，所有模块每步连续求值，不读taskID/地图/路点/运行时教师或路由。范围解除不证明可同时拟合或安全。

固定T514真实成功恢复的59个ownership决策，按本地index//5奇偶时间块分30训练标签、29验证标签；只训练30，验证不入loss/梯度/步数选择，预检用验证标签篡改证明训练梯度不变。真实全序列hidden和原120历史输入照常保留，其中自然包含先前教师动作；这是同轨迹blocked-label插值检查，不是独立任务/未见导航留出或DAgger保证。前166个学生前缀仅保持其原未裁剪location，不作教师标签。原19个自身成功完整history原location保持，old8/field11等权。

固定512live新网络Adam+512独立shadow，另独立512丢弃历史审计，预检2+2=4另计。loss=.5恢复30+.1同任务原前缀+.4原19保持；双轴sigma归一，半物理时间mean半最差20%决策风险，teacher边界仍原feasible action hinge。无新PG/Value/优化器网格/种子网格/早停调参，不重启结束目录。

最终联合离线门槛预先锁定：训练与验证各轴实际执行动作RMSE≤.05；19个旧episode及前缀分别各轴unclipped-location RMSE≤.002且最大绝对偏差≤.01。全满足才可提议下一独立有界原生检查；任一失败本轮止步，不放宽、不追加同机制步数。即使通过也只是有限存储history上的兼容，不保证闭环、不替换最佳模型；候选文件保留，内存ActorAdam回到零新头。old275/six54/eight54/field24/fresh16/reload78本轮全未执行null。原最佳field11/24不变。

0native/0原生命令，禁止创建原生环境和mj_step；main1200秒/audit1200/backup1200，prepare锁science3600/all5400秒，own8MiB/global1GiB/free40。CPU14/15总1核quota idle/nice19/GPU禁用，不查看调整Phase资源。唯一writer，active不改协议源/外写其他runs/并发audit或source。预检包括held-label梯度隔离、真实完整history缓存与sequence一致、父初始逐位、父参数/原Adam不变、live-shadow/保存读回/整状态回退。源码提交/精确远端根树后唯一后台启动。结束独立全部Adam/指标重构和源模型SHA，完整RAM远端二进制逐成员SHA归档后报告和源码同步；离线数据成功不冒称导航进步。
