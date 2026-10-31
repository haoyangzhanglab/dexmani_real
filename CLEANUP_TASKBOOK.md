# DexMani Real 清理任务状态

状态：T1–T7、HOME 标定桌面收口和静态盒体轴对齐均已完成（2026-11-01）。本文件记录当前行为、保留边界和仍暂缓的范围，不再作为待执行清单。

最初核查基线为 `23d3d9d`，实施起点为 `67960ce`；后续修改以当前源码和用户授权为准，不回退已有工作。日常用法见 [README](README.md)。

## 1. 已完成的行为

|任务|当前实现|
|---|---|
|T1 SAM 顺序协议|客户端、worker 和调用方均无 `frame_key`。每实例独占子进程和管道，顺序一问一答；保留 framing、错误传播、固定对象 ID、mask 边界及收尾。|
|T2 障碍物命名|按配置顺序生成 `obstacle_0`、`obstacle_1` 等名称，桌面仍为 `table`，无人工名称参数。|
|T3 录制选项|CLI 是三项录制选项的唯一输入来源；删除 `CollectionConfig`、`RealConfig.collection` 和覆盖合并。运行使用显式参数，归档字典不被业务读回。|
|T4 VR 朝向读取|运行只读取有限且位于 `[-180, 180]` 的 `theta_deg`，返回 `R_z(-theta)`；自动标定的采样、质量报告与低质量保存拒绝保持。|
|T5 录制来源|删除内部来源枚举复检；`collection_source` 字段、teleop VR 列、policy 标记及物理回放的来源限制保持。|
|T6 采样资产保存|直接保存已生成的 float32 点和法线，不重复转换或扫描；保留独占创建、采样边界和外部文件读取检查。|
|T7 retarget 静态复检|删除已由入口、VR 接收和设备反馈覆盖的 home pose、控制频率、reset qpos、landmarks 复检；保留转换、数组所有权、业务约束与求解状态语义。|
|HOME 标定桌面|所有装配 HomePlanner 的运行入口均读取有效桌面标定，连接设备前失败或完成准入；删除桌面开关、固定高度/余量及手 frame 高度代理。|
|静态盒体轴对齐|StaticCollisionBox 仅由基座系中心和沿 X/Y/Z 的完整边长定义，固定单位旋转；删除额外盒体四元数及无消费者的专用校验函数，保留倾斜标定桌面。|
|全链路复核|调用、配置、归档和外部边界已复核；VR 原始四元数的现有校验前移至 SDK 转换前，避免零四元数被转换成有效单位姿态。|

后两项经用户单独授权退出暂缓范围。仓内未发现现用非恒等旋转静态盒体配置，历史实验数据和报告未修改。

## 2. 运行入口与保留边界

- 录制默认仍为开启、`episodes`、每段 60 秒。相对目录基于仓库根目录，绝对路径直接使用，再拼接 TASK；手部禁用时禁止录制。
- CLI 打印与 Raw 归档包含实际录制选项，JSON 在设备连接和分段副作用前严格序列化。`--no-record` 不启动相机、不创建 recorder，非打印路径不生成录制快照。
- 遥操作、键盘点动、回放和策略入口使用现有 `load_table_plane()`；策略 session 向 HOME、点云及归档复用同一份平面，不重复读取或逐层校验。
- HOME 净间隙为真实几何到桌面的最小距离减去 `soft_clearance_m`。保留多候选、排序、路径采样和停稳；起始软间隙允许按原规则退出，回退容忍相对历史最佳间隙，退出后禁止重入，目标必须满足净间隙。
- MotionConstraints 的在线端点模型仍可不装配桌面；`PointCloudConfig.remove_table` 独立可选。工作空间、机械/操作限位、桌面碰撞间隙和点云裁剪 margin 不合并。
- 桌面尺寸、厚度、软间隙、接触链接及平面法向姿态保持可配置；URDF、FK、IK、EEF、VR 和相机的独立旋转职责保持。
- `source_camera_sequence`、来源时间、VR 序列、source/row/clock、模型来源和对象身份仍有消费者；读取缓存不会刷新来源年龄。Canonical 数据身份、配方和完整 URDF 保持，不改写历史数据。
- `Robot.validate_command`、XHand 公共发送入口、跨进程授权、运行代际、命令期限、反馈时效、停稳和制动约束保留。
- retarget 的骨长/掌退化、标定漂移、模型限位、求解输出/损失接受、速度和平滑、时间推进、reset/commit、失败不提交状态及诊断保持。输入形状和有限性由真实接收边界保证，dtype 转换和必要复制保留。
- 保留数组和文件所有权、拒绝覆盖、NPZ 首错语义、writer 发布边界及资源释放。SAM 的 IPC 检查与磁盘资产读取检查分别面对真实外部边界。
- 配置打印、离线工具按其实际消费者读取标定，不因 HOME 收口新增无关依赖；不重写 PolicyWorker、ObservationBuilder、控制框架或导出管线。

## 3. 仍暂缓的范围

|暂缓项目|保留的参数或机制|原因|
|---|---|---|
|仅保留 VR 遥操作|`KeyboardTeleopConfig`、键盘点动入口、`CartesianJog`|保留精细点动能力及共享按键/HOME。|
|仅保留 TCP server|`VRConfig.transport`、UDP/TCP client、client 重连|保留 VR 连接方式。|
|仅保留串口 XHand|`HandConfig.comm_type`、`ethercat_slave_position`、EtherCAT 生命周期|保留硬件通信能力。|
|固定相机型号或 L515 recipe|两项 `l515_*` 与型号分支|尚未决定缩小设备支持范围。|
|普通发送首次 CRC 即失败|`HandConfig.send_crc_limit` 及容忍状态|保留噪声串口可用性及停止阶段的独立职责。|
|固定 checkpoint 动作窗口|`ExecutionConfig.action_steps`|保留窗口实验及 sync/async/RTC。|
|固定去桌面及 hand_1024|`PointCloudConfig.remove_table`、`HandCloudConfig.num_points`|保留数值 recipe 和 checkpoint 能力。|
|HOME 只保留单候选|候选生成、排序、拒绝聚合|保留自动回零的成功覆盖范围。|
|删除离线分析或质量报告|时序分析、retarget diagnostics、calibration_capture、table.fit、episode_quality|保留研究与人工阅读用途。|
|改造导出 I/O 或控制架构|单遍导出、worker/builder/框架重写|需独立证据与 profiling。|

参数固化同样暂缓：retarget 的 `calibration_samples`、`calibration_timeout_s`、`calibration_max_relative_mad`、`bone_length_tolerance`、`max_evaluations`、`ftol_abs`，手部 `ki/kd`，桌面尺寸/厚度/接触链接、`eef_workspace_margin_m` 及其他损失权重、物理约束、反馈期限和研究参数。

`ftol_abs` 同时参与优化停止与解接受；arm 默认反馈率 25 Hz 与 `feedback_max_age_s=4/30` 不等价于 `4/feedback_hz`。

## 4. 已执行验证与限制

- SAM：真实 worker 配轻量模型的顺序请求、模型异常、EOF/断管、framing、mask/ID 和来源归档；场景选择与同帧复用链已复核。
- 录制：CLI 默认/覆盖/打印、fake 装配和时钟、真实循环限时/首错、实际分段入口至 Recorder/HDF5/MP4、配置与 VR 旋转归档、来源时间/序列及回放限制。
- 几何：HOME 清理完成 69 个姿态、11 次规划对照；静态盒体轴对齐完成四类场景、160 个姿态、12 次规划逐值对照，均使用真实 Pinocchio/hppfcl；软间隙规则、倾斜桌面、无桌面在线模型及非法配置另行验证。
- 标定与点云：VR 最小/完整 JSON、角度方向和非法输入、自动标定报告及低质量拒绝；四 HOME 入口的无效桌面标定提前失败、单次读取/复用；点云两种模式和相关 CLI 的无设备/无标定副作用。
- 数值与所有权：采样 NPY、实际资产准备/加载、外部非法资产、retarget 预处理/reset、反馈/IPC/commit 所有权和优化器故障检查；真实 FK/NLopt 固定输入前后结果一致。
- 本轮复核还验证了真实 VR SDK 的头/腕非法姿态拒绝、64 个单位姿态与 192 个缩放姿态，以及 worker 和朝向采样的坏帧处理；未发现可删除的冗余 SHA 扫描或无消费者兼容层。

验证按风险分配，静态检查、fake、真实数值库和真机结果分别报告。一次性基线、脚本和输出放仓库外，不新增永久测试平台或仓内工具缓存。未运行真实 SAM/GPU、实时设备、真机运动或完整 GUI 会话，不将离线结果称为物理验收。没有自动 push、merge 或创建 PR。
