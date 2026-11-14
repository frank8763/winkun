# T545：只增加非干预前缀保持

授权：2026-11-14用户“查清楚，然后思考之后再继续实验”，沿已确认北京时间12:00全部停止，11:00学习/11:30原生及更严相对截止。不重启T544、不续旧预算或扫权重。

## 已确认与仍待检验

T544已完整32初验+32exact、独立auditPASS、全原始远端逐SHA，16live+16virtual、shadow及独立重构分列；候选旧7/8、field11/24拒绝整ActorAdam回退。posthoc0native0optimizer只读取保存数据/模型：完全相同native初态/goal，新模型第0步执行动作由[.902151,-.168723]变[.883117,-.196810]；相同原学生前34s历史上的动作location RMSE=.037482/max=.145728。恢复段执行动作MSE .0658816→.0259991约降60.54%，但自主旧左32.438s已碰撞，早于教师34s接管。encoder/GRU/head/两adapter全改；失败任务不在7旧+12场地成功保持集合。丢失field2off2仍被MIR选13/16次，故不是简单漏采样。证据支持前缀干扰缺口，不证明唯一因果或加软约束一定成功。

## 唯一科学改变

初始模型、fresh全ActorAdam lr3e-5、16真实步/16虚拟MIR、16shadow+16虚拟、固定19真实成功history、MIR配额2旧3场地、.5恢复/.25旧/.25场地、同一T544已验证教师执行动作标签与全部验收规则保持一致。

仅把恢复函数替换为：R_new=R_T544+1.0*P，P为该教师成功history中owner0真实学生前170决策的平均Gaussian均值KL(.5*sum(((location-current_saved_location)/sigma)^2))。sigma仍[.2,.2]。真实恢复项原系数.5，因此实际总目标仅新增.5P；虚拟MIR提案使用同一个新R_new。该段是实际接到安全GOAL教师的未接管前缀，保留其局部驾驶行为，不给原超时整轨迹伪造成功标签。原h0完整GRU因果递推，不单帧zero-init。

P初始=0且梯度=0，不能保护第一次恢复步，也不是硬安全约束；明确检验后续恢复力是否减少早期偏移。训练日志把末段误差与前缀KL分开，原先恢复项没有缩权、不扩16步、不调整学习率/噪声/teacher/seed或网络。lambda0必须与旧T544真实损失和梯度一致，未来输入扰动不能改变前缀输出，virtual/shadow/整Adam回退及模型读回必须通过。

## 数据与验收

全部使用T543已audit/远端逐SHA的19成功history与T544已完整原生成功+exact+独立audit的教师history；0新教师/训练仿真，不重复20历史重建。最终纯学生从原始起点old8+全field24，各一次独立原生exact。旧8/8、原研究12个成功ID全保、field CONTACT≤9、无invalid/censored才保留研究候选；未过整ActorAdam回退，接受T507仍保持。不能用loss或教师GOAL替代自主门槛；old275/完整六54/八54/fresh16/78未安排=null，不称全场地稳定。

## 有界资源

新增64导航回合/68命令，物理最坏3,201,204，0新工程物理/命令；沿用已验证同Actor/原生worker接口。前置零原生合成8丢弃Adam。今日连T543/T544主运行及两工程probe最坏5,789,300物理/210命令，在6.5M/240内。main3600s/learning1800s/audit900s/backup600s/source180s，先完整余时准入，12:00绝不延长。own8MiB/global1GiB/free40GiB；CPU14/15合计一核idle/GPUoff/实际Mesa；不碰Phase。

先源码远端精确树后唯一启动，active单writer、协议科学源不改。结束唯一独立历史优化器重构、全原始RAM远端逐成员SHA和source根树；失败原件保留，不继续同权重或换超参重试。依据IWR/DAgger/MIR的一手资料在T545_PREFIX_PRESERVATION_SOURCES；标准软保持不是原创、防遗忘或安全证书。
