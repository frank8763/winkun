# T320：人控转向序列的单一前进输入增强

注册2026-11-03；T319全部结束、独立audit PASS、37原件已完整远端逐成员SHA。不是续T317/T318/T319预算或重启旧目录。用户已接受训练期人工示范并要求错过窗口后继续，此处不等待额外人工输入。

## 唯一假设与边界

两条原人工轨迹均无碰撞、端点距离单调接近，原56s仍余0.75–0.78m。不是成功专家；T319固定表示小头拟合只降低14%同场景验证动作MSE，old20保持而原六米两中心仍失败。现在先检验**这两条具体行为的前进输入/时间裕度**，不宣称自主六米问题只因速度，也不推断道路一定可达。

每条原人控action序列保持steer列和100native/0.2s hold原样，只把drive列固定为允许值0.999；按原LEFT6完整起态/goal/scene/传感history真实仿真。先到原native终止即停，否则最多280hold/28000物理/56s。没有延长cap、缩放时间、扩大0.20m半径、忽略碰撞、附加转向反馈、规划器或路点。提高drive会改变动力学和后续空间位置，可能碰撞或错过目标，因此所有新状态/120输入/奖励必须真实重算，不能沿用原human标签或假设成功。

**行为来源明确记为HUMAN_DERIVED_OPEN_LOOP_THROTTLE_AUGMENTATION，不是新人工示范，不是单Actor自主导航，不是当前policy同策略数据。** 原人工超时结果不改。该固定行为仅用于收集训练候选数据；本任务0Actor/Adam/梯度/RL更新，不进行BC，也无运行时混合教师或路由。若成功只得到一条由人控派生的成功训练轨迹，仍须另训Actor并独立完成old275/原six54/fresh16/78，不能报告六米稳定或进入7/8。

## 文献与反证

- [DART，CoRL2017](https://proceedings.mlr.press/v78/laskey17a.html)，本轮查作者正式摘要、上轮已有相关资料。其在专家演示中注入扰动并由专家恢复；我们没有在线专家纠偏，只做固定输入变换，不是DART复现，不能借用其协变量偏移收益保证。重要反证是动作变动后的轨迹可能失效，必须真实原生重跑。
- [POfD，ICML2018](https://proceedings.mlr.press/v80/kang18a/kang18a.pdf)，本轮已读方法与算法1；支持少量不完美示范作为探索参考，但当前没有判别器/新奖励/策略梯度，不能称实现该算法。
- [Go-Explore/First return, then explore](https://arxiv.org/abs/2004.12919)，作者摘要/前轮作者资料已查；探索轨迹与最终稳健策略的区分值得保留。这里不引入archive/cell返回规划/目标路由，不称其复现或新颖性。
- [Fighting Copycat Agents，NeurIPS2020](https://papers.nips.cc/paper_files/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf)，问题与方法段和[作者实现README](https://github.com/AlvinWen428/fighting-copycat-agents)已查，未移植代码。两人控266/273行重复前动作，复制前动作的第二轨迹MSE仅.01462；低离线BC误差可能是捷径，不能当完成任务。未证明T319本身学了此捷径，也未训练对抗模块。以后真正Actor训练需报告纠正动作和自主闭环，不能把标准反复制机制称原创。

历史T194油门偏置试验是既有神经策略，其steer并非这里两条新人工序列；本轮不是重跑T194网格。只预注册两条已知源，各一个同样固定drive，**不再追加drive、时间、转向或种子网格**。失败不证明所有轨迹增强/RL不可行。

## 有界执行和保留

2条派生轨迹+各1次独立新环境同动作精确原生重生，全部4命令、≤112000物理、main300秒末15秒保护、独立audit120秒0物理；2重生不加独立样本。初态/渲染契约复用T317，预检只核实际源/变换/初态120 SHA，零新geometry/物理/参数提案。源/输入SHA、干净提交、GitHub完整根树核验后唯一启动。

每条保存全120 states/nextstates、真实3history/实际actions/counts/poses/原reward/初末native积分；原生重生12数组/积分精确。每决策原子快照，输入日志注明派生来源。原件≤512KiB×2加单个≤512KiB临时文件；replay中断前缀另存、从不覆写已完成原件。own2MiB=payload1.5MiB+metadata384KiB+buffer128KiB；global≤1GiB/free≥40GiB，注册最坏1073626768B，余量紧，不删原证据或复制整个模型/视频。

active只写本run，不同步源码/改源协议；最后全部父/timeout/inhibitor已退，独立原始变换/120history/reward/数组与原生收据审计，再全部原件RAM归档远端二进制逐成员SHA，最后源同步。成功/接触/超时/未执行/删失分列，原人控与Actor成绩不混合。Stage与Phase完全隔离，safeT255/T299/旧失败/高清视频保持。
