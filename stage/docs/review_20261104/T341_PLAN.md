# T341：真实轮速上限翻倍的单次诊断与策略迁移

用户2026-11-04明确要求继续六米并提高最高速度。先完成与新导航学习分离的动力学试验，不修改旧原件，不重启T340。Stage only，Phase全隔离。

固定T340最终Actor与宽1.17m/80秒课程，同一初态，仅令环境max_wheel命令上限乘2；原上限为匹配控制。零物理预检必须核验实际配置、后轮actuator_ctrlrange可覆盖新命令、初始积分/120输入和首动作一致。仅改变每个新建env的max_wheel，不修改历史XML、执行器增益/力限/摩擦/时间步，不按速度直接修改qpos/qvel，不将影片加速算物理提速。

两条件各一次8秒全油门直行诊断及一次自主导航中心，每项再一次完整原生重生：4unique+4replay，预检2geometry，共10命令，最多176000全部物理；0Actor/optimizer。直行在8秒主动截止记行政诊断删失，不能算导航超时/成功。实际速度逐native采样，保存全数组/历史、动作、轮执行器控制与速度、真实积分。测量轮命令上限、直行后半程实际速度及自主到达/接触/超时；不能预先宣称实际速度正好2倍。

主900秒、独立审计≤300秒、全部1800秒，结束即停，无速度/seed/噪声网格。own3.25MiB/global1GiB/free40GiB：两nav文件各≤768KiB、两straight各≤128KiB、一个partial≤768KiB、metadata≤768KiB，总3.25MiB。原速导航须精确复现T340零前缀成功，全部4个原生重生逐数组/积分精确才通过工程核验。源/输入SHA、协议、正式预检、本地提交/精确远端后唯一启动；全部进程退出后一次独立audit和全部RAM远端逐成员SHA。

依据[MuJoCo官方velocity actuator说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-velocity)与[control/force limits说明](https://mujoco.readthedocs.io/en/stable/modeling.html#actuator-force-clamping)，控制命令上限、执行器传动/控制限、实际车速应分开检验。已核对本项目StrictMuJoCoEnv._policy_to_commands：wheel=raw_ratio²*max_wheel，speed_gain只用于日志，不用于运动。本次只采用原生轮命令变化，绝不改日志系数充提速。实际原生被加载step_checked和配置也在预检保存后才能判定。

即使提速中心到达，仍是独立新动力学课程，不能当原窄六米验收或旧五米保持；本任务不执行old275/原six54/16fresh/78重载，不晋升七八米、不替换safeT255。若迁移碰撞，保留新最大轮速配置供后续另立策略适应，不能直接推广旧模型。若轮控限挡住预检则零物理停止并记录，不能临时改力矩/齿比继续。
