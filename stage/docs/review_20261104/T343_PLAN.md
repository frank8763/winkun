# T343：固定轮命令10，单次后轮力限120→240干预

Stage only，Phase完全隔离。用户明确要求提高最高速度，T341单独把轮命令5→10后，8秒直行后半程实际速度几乎相同(.08654/.08659m/s)。T342零物理原生配置检查确认同加载器后轮力限±120，T341已保存增益35、bias速度项-35、控制与实际actuator_velocity。以端点数据代入速度伺服公式，两工况未限幅力分别约127.55–130.24、302.89–305.83，均超过120。这是端点的机制线索，尚未直接采样力或证明唯一根因。

依据[MuJoCo速度执行器](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-velocity)和[力限文档](https://mujoco.readthedocs.io/en/stable/modeling.html#actuator-force-clamping)，新试验**两臂轮命令上限都固定10rad/s**；对照后轮actuator_forcerange±120，处理±240。只改两个后轮force_range，不改gain/gear/摩擦/阻尼/转向/质量/时间步或qpos/qvel；force_limited仍开启。原历史XML、已有环境与模型不修改。原生积分在干预瞬间必须完全不变，其他执行器force_range逐位不变，保存明确plant hash。

冻结相同T340 Actor、宽1.17m/80秒课程。每臂一次8秒全油门直行与一次自主六米中心，并各完整原生重生：4primary+4replay、2零物理预检geometry，共10命令/176000所有物理，0Actor/Adam。force1对照应精确复现T341 factor2原始全部数组/积分，是历史重复而非新增独立数据。实际force每决策端点直接采样，速度每native步；不得以命令或公式代替测量，端点采样不声称每个native步力都已记录。直行8秒未终止记行政诊断删失，碰撞/真超时/缺测分列。

唯一固定120→240，不扫描力限/seed/轮速/温度。main900秒、audit300秒、全部1800秒，own3.25MiB/global1GiB/free40GiB。复用原T341固定诊断引擎，新增真实端点力字段；每nav≤768KiB、每straight≤128KiB、一个partial≤768KiB、metadata768KiB，最坏≤3.25MiB。真实配置/输入SHA/zero-physics/action一致预检、本地提交/精确远端后唯一启动；T342审计/完整原始备份结束前不启动。所有父链退出后唯一audit/全原始RAM远端逐成员SHA。

此试验检验该工况力限是否限制车速，不能单独证明轮地滑移/阻尼不是问题。实际速度增加但导航接触仍说明需要新动力学下策略适应，不推广冻结旧策略；更高仿真力限不是已标定实车参数。原窄56秒六米、旧275、six54、16fresh、78发布重载均不在本任务内，不以提速或一中心成功晋升七八米。
