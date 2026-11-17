# T618 固定末段正负转向修正可恢复性

T617两已保存完整轨迹离线核验均超时，无成功恢复标签且原native exact缺测；不重启T617或再次扫教师。新机制只验证同一T616B真实74秒末段状态，一个固定有限转向干预能否改变结果。

固定三支delta=[0,+0.25,-0.25]，均从原起点执行同一冻结T616B Actor，决策370起至399共30决策6秒，把其raw location转向分量加delta并clip[-1,1]，drive不改；之后回到同一Actor自主控制。0支必须全部历史数组逐位精确既有T616B原始轨迹（新增pulse元数据不参与此历史字节等同性），防止把工程变化当干预效果。三支同初态/任务/原时钟/目标/安全/0.2mGOAL，不改接管时刻或幅值/时长/seed，不失败后加分支。完整自身120记忆从原起点carry，干预执行动作进入下一120输入；不复制别的hidden，不重置物理状态。

正负幅度预注册为归一化转向raw location的0.25，不等同0.25弧度或物理轮角。此是训练期动作干预，绝不声称纯自主部署或新RL更新。动作公式/所有candidate hidden/history/reward/scene/初末native积分独立审计，三条原生exact。修复T617接口合同，Pool发前记pending request，receive先落原native返回再派生标签；回归必须走缺request的真实返回载荷和同实际Session.step seam，不只测独立公式。

所有三支无论结果都执行+exact。若一个以上safeGOAL，按固定[0,+,-]顺序记录合格整条混合行为历史，后续纠正监督只用ownership==1实际干预段，完整实际前缀供学生自身BPTT；不当on-policy PPO经验。之后另注册有界学习/原34全保持和原起点新增成功才研究保留。本项0学习，不自动接旧学习预算，不修改拒绝模型的研究地位。三支均失败则没有成功标签，报告原始原因；不扩大动作网格。

最多384672物理/10命令/900main300audit/8MiB，0optimizer/0新教师。日22M660cmd9000秒50MiB与global1GiB/free40GiB保持，CPU14/15总200低优先GPUoff/Stage Phase全隔离。14:30原生15:00all与更早相对截止从严。全run独立审计、manifest及所有原始RAM远端逐SHA。
