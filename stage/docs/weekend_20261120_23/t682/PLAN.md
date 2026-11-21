# T682：两来源的drive/steer参数因子拆分

T681两个原冻结整头变体field为10/24、4/24，均有超出发现4任务的新增，但已知olddev6_00/02/13/20全collision。T680四方向11906维梯度不稳定，不复活其训练/加seed或幅度。新问题：drive与steer两既有残差头是否分别承担新到达与旧碰撞的改变。此为原模型组件静态因果消融，非任意运行时拼算法/路由/teacher，不是BC/JVP约束/投影/新奖励。

固定两个parent即direction00_minus、direction03_plus，主干/GRU64/120因果输入不动。每parent完整2×2：00初始T501、11原双头variant从已精确验证原数据引用；新增10 drive-only(原variant drive+初始steer)与01 steer-only(初始drive+原variant steer)各一个固定单Actor。4新固定模型不随机抽方向、不缩放、不优化；每个保持一个5953参数头精确initial，另5953精确source；所有其他张量逐位相同。整个episode自身GRU、seedNone、原起点/原生时限/7m field边界和碰撞GOAL规则照常。

任务固定：olddev6_00/02/13/20 + 6相对T501原失败且source00成功field0_1/0_2/1_1/2_1/5_1/7_1，每模型10完整full+10exact，共40fresh40exact；4模型全测，不挑最好再补。两parent的11同任务各10原full均来自T679/T681，00 old4来自T501与field6来自T502，原parent full20与field baseline6逐SHA/数组/自有carry复核，场景reset/任务cap身份核验引用；T501 old4原始raw现为t629远端引用，本项只复核原case元数据/原auditPASS/384名单及恢复收据hash，不称本轮重读原old raw。不重跑消费/不计新独立样本。两parent是后验挑选成功源，不作独立随机重复/泛化。

逐parent×任务保存00/10/01/11终态Y=safeGOAL-collision，二因子结果交互I=Y11-Y10-Y01+Y00；每项safe/collision/timeout/boundary/censor、全路径。输出头在同一外部历史逐坐标可分不等于原生闭环结果可加。预检全10源历史验证每hybrid保留坐标与对应base/source完全相同，按自己完整GRU carry并clipGaussian确定性命令规则，非物理导数。独立audit重建四模型参数/全raw/积分/exact/每任务报告。

准入只为下一受限头研究：同一头在两parent消融模型均保4/4旧，并在6field有至少2个共同safe新增；要求恰好一个头满足这个跨parent合取，另头的全部结果照报。若两头均过则因素不唯一记AMBIGUOUS，不挑更好；若均不过记FAIL。不救阈值/幅度/seed/额外任务。即使PASS也不自动学习/晋升，只有4旧而非384；full407=null，原384全保+新增+同candidate407独立audit保留门不变。

0Actor/Value/optimizer；4静态消融模型完整保留；最坏2647464物理86cmd(80native+6管理)/1800main+300audit/root2400s40min/256MiB/global10GiB/free40GiB。CPU8/10/12native14audit/400%Nice19/4GiB无swap/CUDA关闭；资源重叠只暂停自己Stage cgroup、原deadline不延。先一手审查/真实入口0步预检/独立ledgerRESERVED/GitHub精确全树后唯一launch；活动时只读监督，不改源/协议/外写runs/并行source或backup。保所有失败raw与模型。
