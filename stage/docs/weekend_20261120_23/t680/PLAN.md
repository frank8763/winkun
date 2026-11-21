# T680：4个参数扰动对的估计稳定性，不强行训练11906维

T67964fresh+64exact完整独立auditPASS/中心未改；3旧任务和2新场地结果有变化，field1_offset1出现1safe、7_offset1出现2safe，但原旧多失败、field7_offset2全失败。support门只是机制可采，不是参数梯度可学、最好variant可保留或旧IID匹配预算优越性。下一先核查信号是否由单对方向支配。

对已固定4个11906维标准Gaussian方向epsilon及原尺度s、原Y=safeGOAL-collision，每任务antipair差Δ=Yplus-Yminus，g=mean4(Δepsilon)/(2s)。field4、old4、joint8按任务等权，分母不能因某对或某任务为零而删掉。原参数噪声不是动作分布，禁止给这些确定性动作假装on-policy Gaussian动作logp，不另加C/径向/排名回报。估计的是Gaussian平滑均值参数目标，非确定性中心效果。

固定报告每ID的4pair差与nonzero计数，三个任务均值梯度norm/每pair贡献/最大贡献占比、逐删一pair后分母3的norm/余弦、预先固定前2/后2方向块各均值的norm/余弦。零向量余弦=null，不制造稳定性。原64轨迹每文件SHA+模型/种子/正负映射核对；四共享方向跨任务/正负相关，exact不增独立样本。

估计可用门同时需：至少2field任务各有>=2个非零pair差、field均值非零、删除任一pair后均非零且对原field均值余弦>=.5、固定两半余弦>=.5、finite。门是助手预定工程可用性，不是假借论文置信定理。已观察任务差异稀少不作为调整门槛/重新抽seed/增pair数量的理由。只4方向无法辨识一般11906梯度，方向span rank<=4；线性Gaussian理论参照relativeMSE=(d+1)/4不是本环境方差估计。

0物理/原生命令/Actor/Value/optimizer，无模型导出，600main+600独立全部重构/root1800秒30min硬停/16MiB/global10GiB/free40GiB，CPU14/Nice19/4GiB无swap/CUDA关闭。独立data/source登记，先真实公式/删pair分母fixture预检、ledgerworstRESERVE、GitHub精确树后唯一launch；T679科学/driver/helper先退出。FAIL则不把4pair强行训练或从最好variant拼旧成功；停止此full参数估计路径，另有证据机制需重新定义，不做规模/seed救门。PASS仍仅有限稳定性，后续任何训练需另注册并samecandidate原384全保+新增+407/audit门。
