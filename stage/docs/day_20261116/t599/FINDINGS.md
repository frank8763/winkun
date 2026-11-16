# T598 检查结论及独立部分证据恢复

2026-11-16。原T598目录与FAILED摘要、原源、原预算、错误、模型均不修改；未恢复训练，未补原生仿真。T599只消费原已授权、原driver尚未执行的128次历史Adam审计额度。

## 实际结果

- T598主管线783.906663秒，1,702,836物理步，计费命令63；128 live监督Adam、128 shadow均完成。4次预检另外计数。0新RL/critic。
- 三个自主恢复：sector1_offset2接触、sector2_offset2安全GOAL、sector4_offset2超时；1/3。相同原前缀上的冻结T550历史控制为0/3。新增的是一个已知任务的恢复状态能力，不是新增原起点任务/完整六米通过。
- 完整49项原起点均已执行：31成功、13接触、5超时，成功率63.27%。相对T550原39项成功，丢失8、无新增原起点成功。
- 场地24项12/24=50%，原T55015/24；原八中心7/8（dev_task6_13超时），原T5508/8。
- 8个失守ID：dev_task6_13、dev_task6_20、sector0_offset2、sector1_offset2、sector4_offset2、wide117_8m_dev6_LEFT_00、wide117_8m_dev6_LEFT_13、wide117_8m_dev6_LEFT_17。sector名称完整前缀为t502_field_。
- 接管3精确重放和关键2精确重放已完成且逐数组/积分一致。另两关键重放sector4_offset2及wide117_8m_dev6_RIGHT_13未执行，不计导航失败。
- 候选不满足原39保持门，不能保留/晋升或继续接RL。原错误处理记录整ActorAdam回退，T550原字节SHA不变；没有保存回退后joint hash，因此不声称独立验证了进程退出前的完整回退状态。

## 我的计数错误及其影响

继承Runtime在初始化已记2条启动命令，T598 main又在唯一worker pool前加2，重复收费2。注册预算采用52案例+7重放+4实际pool生命周期=63，但实际计数路径为52+7+6=65。程序因此在预算63处拒绝最后2次重放；实际只做52案例+5重放，计数63，未突破已批准上限。

这是注册前没有核对实际计数路径的工程错误，不是训练算法或场景的原因。它导致原summary result=null、原driver不进入audit；即使完成剩余重放，已执行49项的8项失守也已足以否决候选。因此不追加仿真补测、不放宽上限、不重启原目录。以后注册前应将实际Runtime初值、每次pool启动/关闭和全部request计划放进同一个零物理计数检查，不能只手算回合数。

## T599独立部分证据恢复

原reader在评估器namespace查找失败，发生于任何历史Adam之前（0native/0Adam）；原代码、launch、log保留。独立adapter_001仅将assess解析到既有t591 namespace，正式科学数据/128次上限不变。adapter注册时generic run模块名还导致一次0native/0Adam导入拒绝，改唯一importlib模块名后才注册；未覆盖原已执行源。

恢复24.030486秒、0新增原生步、0新学习，128次丢弃历史Adam：最终完整ActorAdam joint hash与T598已保存proposal精确相同；全部52保存payload SHA核对、3完整接管含双32额外记忆/奖励/真实history逐步复算、5已完成精确原生重放核对通过。其余49是紧凑payload+原live完整数组哈希，不宣称独立全pose重建。

结果状态为PARTIAL_EVIDENCE_VERIFIED_NOT_COMPLETE_ORIGINAL_AUDIT，原T598完整audit仍缺失、原任务仍FAILED，不把这项恢复写成原完整审计PASS。

## 学习层面的结论

三条恢复历史的动作RMS由0.365882/0.093394/0.219605降至0.112414/0.063401/0.091837，恢复接管获得1例成功；存在真实学习能力。但39条保持历史最坏动作RMS升至0.270476。旧主干确实完整冻结（fixed SHA核对），新残差仍直接改变总动作，因此冻结参数不等于保持总策略，更不保证闭环不退化。

这个结果不支持立刻再加RL或步数。下一方案必须以总动作/实际成功ID保持为验收，分离“新分支拟合有效”与“保持可接受”的目标，并固定匹配对照；不能宣称独立GRU已经解决遗忘、泛化或原创性。当前只报告结果，不启动新学习。

## 保存与边界

原T598的124成员2,440,496字节RAM归档已完整远端逐成员SHA读回，archive SHA256 6a6bfca76c40c13e27cd176e8573c4b4d17464fdc26ee76b7e2e494764432863，原receipt/source_manifest不重做。原父、driver、inhibitor和两个worker实际PID均消失。新增恢复的全部小型原始文本、错误及SHA在RAW_EVIDENCE.json可逆bundle入Git；没有大二进制入普通Git。

完整407/fresh16/reload78均未执行；没有全场地/六米/八米通过或新合格视频。CPU14/15低优先、GPU关闭、Phase完全隔离；全部记录仍在global1GiB内、free≥40GiB。原T550保留。
