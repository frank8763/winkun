# Stage汇报材料（2026-11-02）

[下载完整素材包ZIP](https://github.com/frank8763/winkun/releases/download/stage-five-metre-videos-2026-11-02/stage_presentation_20261102-24f6efb1c99d.zip)，16.14MiB，11成员，包含两MP4、六页PPT文案、两张1920×1080 PNG与SVG、视频信息、审核与SHA清单。ZIP在RAM生成并完整远端读回逐成员验证；没有本地重复大压缩包。

## 两段真实五米成功MuJoCo录像

- [左侧5米1080p MP4](https://github.com/frank8763/winkun/releases/download/stage-five-metre-videos-2026-11-02/stage_5m_left_1080p.mp4)：视频46.84秒，原生到达44.434秒，最终余0.19999165m，无接触。
- [右侧5米1080p MP4](https://github.com/frank8763/winkun/releases/download/stage-five-metre-videos-2026-11-02/stage_5m_right_1080p.mp4)：视频46.04秒，原生到达43.636秒，最终余0.19993449m，无接触。

均1920×1080/25fps/H264，正常导航1倍速；0.8秒开头与1.6秒成功后展示暂停。全景与车辆近景、已走轨迹和状态字幕只用于显示。固定冻结T255 s2_reset，两个原五米中心任务；新原生复现全部10数组/120观测与最终积分匹配原验收，不是新训练或跨地图证据。

## 可复制到PPT的内容

- [六页文案与讲解句](PPT_CONTENT.md)
- [方法图SVG](stage_method.svg)，本地可重建PNG为stage_method.png
- [距离结果图SVG](stage_distance_results.svg)，本地PNG为stage_distance_results.png
- [视频参数与SHA](VIDEO_INFO.md)、[独立审核](VIDEO_AUDIT.json)

方法主线：自身成功经验整合+端到端深度策略梯度微调；120维因果观测→单Actor→连续驱动/转向→真实MuJoCo动力学。时间相关探索、PPO式裁剪、PCGrad和旧能力保持均为已知组件适配，不独立声称原创。选中模型每次清零GRU隐藏状态，不能宣称记忆收益。

已验证范围：原场景局部开发211/211、原64留出64/64、模型重载275逐例精确。六米最新0/54（左27接触、右27超时），旧275/275保持；16fresh/78复验未执行，七米/八米未进入。五米视频不替代或扩大此范围。

## 录制、备份与空间

T288共2全部命令44,035物理步、146.336秒，0优化器/导航训练更新；全原生/渲染/编码进程已退出。独立全帧解码/源模型/视频/原生数组SHA审核PASS。

23个原始输出文件已RAM归档完整远端逐成员SHA读回，见long_distance_20261101/release_receipt_T288与source_manifest_T288；两个独立MP4附件也分别完整远端读回。Git只版本控制文本/代码/SVG和收据，不存MP4或大ZIP；PNG为可重建本地预览并包含在下载包中。

当前newruns仍小于1GiB（约998.36MiB），空闲盘>40GiB；冻结模型及原失败证据未修改，没有在仓库外创建散落run。GitHub本次认证与源码同步已恢复，今晨新增报告也已补入同步源。
