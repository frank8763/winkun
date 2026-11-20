# T674直接切线投影科学门失败

COMPLETE/独立auditPASS，0新增物理/Actor/Value/optimizer，23.038main+22.685audit秒。44独立旧结果半空间rank44，9active，stationarity约1.30e-19、complementarity1.01e-22、dualitygap-2.12e-22，实际表示后线性可行，位移改变17.96%。

仍未可用：field均值方向-5.73393e-7，非线性样本均值-5.70668e-7；旧dev6_08/19/26、hold7_03、wide117_80_LEFT_16非线性下降。数学投影正确并不等于非线性闭环保持，更未原生测试或导出candidate。独立审计重现否定结果，不是科学通过。

停止这条单次直接切线投影学习，不改容差/步长/权重网格救，不称所有约束方法被否定。下一T675先验证已有完整轨迹的goal/poses/remaining进展测量和失败同终态内差异，不能假装potential telescoping提供新完整return信息，不直接接Q/teacher或新奖励。T501保留。
