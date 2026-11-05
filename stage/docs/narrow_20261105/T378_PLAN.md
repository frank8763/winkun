# T378：固定模型的人控与保持梯度诊断

2026-11-05，Stage only。用户授权下一步开始。依据[文献与方法重设](RESEARCH_RESET_20261105.md)及[NEXT_AFTER_T377](NEXT_AFTER_T377.md)，先验证保持项是否与人控学习相冲突。0原生命令、0物理、0优化器、0新训练。T377已结束，不恢复其300步预算，不改变或晋升模型；Phase隔离。

两固定点是T377初始fallback与最终candidate，原始输入/参考由已锁T377协议解析，485依赖SHA逐一检查。保持参考均为原始初始化父策略输出，不用最终模型重建新参考。每点完整历史求：两条真人各0.5mean、全部211旧参考mean、全部14事实成功history的max平方误差，共227次autograd；两点454，独立审计再454。原300步old索引频率和factual轮转频率只用于在固定模型上计算平均梯度，不重构训练中沿参数路径的实际平均。额外报告固定模型最后schedule条目的方向，明确也不是历史最后一步梯度。

保存每条梯度的loss/范数/原向量SHA及频次，保存两人控、加权old/factual、末schedule梯度原数组。计算范数、夹角、人控与保持内积、人控向保持梯度span投影残差（lstsq rcond1e-12）、人控对总下降方向的点积。零向量余弦记null。scalar_GD_human_descent_ratio=gh·(gh+gp)/||gh||²；<0指无穷小标量GD方向增加人控loss，0至1为方向削弱，>1为同向促进。此推断不等于有动量/自适应缩放的Adam实际效果、唯一导航因果、安全或可学性证明。全模型SHA/所有参数.grad仍None须保持。

两路真人有效标签295+277，第二路前20learner动作不作标签但全297观测history参与计算。按真人转向变化>=0.05分组、四等分时间片记录drive/steer误差；是同一路监督数据的描述，不是新专家查询或验证集。原始两路RMSE必须精确匹配T377已审计initial/final指标。既有0.70m/80s与原0.57m/56s成绩不混用。

预检仅AST/依赖与容量校验、一次二维合成autograd、同向/反向/零span代数。禁止Adam/SGD步进、native setup及multiprocessing入口。预检不执行导航或实际模型梯度。正式main最多600秒，独立audit600秒，含归档全部工作1800秒，自launch起算，更早外层timeout从严；不延长、不重启结束目录。两点全部分析后再解释，不根据好看的中间结果增删数据。

own640KiB，最坏556KiB：协议16、预检8、launch8、workers4、status8、summary12、results192、原梯度96、audit16、两console各32、failure32等另32预留；global1GiB/free40GiB。原NPZ/模型就地只读不复制、无额外checkpoint，不删唯一资料。实际完整预留由prepare写BUDGET。

代码/协议/输入哈希与真实零步预检通过，干净提交、精确远端根树后唯一launch。main只写本run；结束确认原进程与外层链退出后独立复算454次全部梯度SHA、loss、方向和保存原向量逐位精确。完整RAM归档远端二进制读回逐成员SHA后才称备份完成，最后同步结果源码。发生错误保留partial/错误，不放宽阈值或另编号续该诊断。

结论只用于选择下一独立有界学习对照；本协议不实施B的纯人控拟合，不做old275/six54/fresh16/reload78（null）、不进7/8，不录新导航视频。六米稳定状态保持未通过。
