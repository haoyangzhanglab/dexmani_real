# 策略部署

`dexmani_policy` 提供模型及观测／动作描述，本仓库负责读取观测、调度预测、执行控制和录制评估数据。

## 数据流

**传感器与机器人反馈 → 观测历史 → 模型推理 → 动作序列调度 → 动作映射 → 机器人**。

观测按模型需求组合图像、关节、触觉和点云。绝对关节动作仅按运行关节限位裁剪，保留输入的角度分支，不根据实测位置自动换成 2π 等价角。末端动作经共享局部 IK 转为关节目标；IK 允许保留跟踪残差。Raw 保存观测与指令提交尝试，不代表两设备原子执行或物理到位。

## 核心模块

| 模块 | 职责 |
| --- | --- |
| [session.py](../dexmani_real/deployment/session.py) | 装配模型、传感器和机器人，组织多段评估与结果汇总。 |
| [observation.py](../dexmani_real/deployment/observation.py) | 将观测历史编码为模型输入。 |
| [inference.py](../dexmani_real/deployment/inference.py) | 加载、预热、预测与关闭模型。 |
| [schedule.py](../dexmani_real/deployment/schedule.py) / [rollout.py](../dexmani_real/deployment/rollout.py) | 接续动作序列，推进单段观测、推理、控制与录制。 |

设备交互、运动约束和 Raw 存储复用 `robot`、`control`、`data` 模块。

## 配置来源与启动信息

| 配置领域 | 权威来源 |
| --- | --- |
| `observation_fields`、`action_mode`、`control_action_dim`、`n_obs_steps`、`n_action_steps`、`future_steps`、`supports_rtc`、`requires_prev_action` | Checkpoint 的模型契约，本地配置不覆盖。 |
| 模型使用的 PointCloud recipe、Fingertip Link 顺序、Depth shape／scale | Checkpoint。 |
| Arm／Hand、关节与运动限制、反馈 TTL、传感器和相机、环境与碰撞、HOME、遥操作 | RealConfig；相机等本地标定仍由现有标定入口加载。 |

| ExecutionConfig 参数 | 优先级 |
| --- | --- |
| `control_dt_s` | 非 `None` 的 RealConfig 值 > Checkpoint；两处缺失时报错。 |
| `action_steps` | 非 `None` 的 RealConfig 值 > Checkpoint `n_action_steps`。 |
| `mode`、`prefetch_steps`、`rtc_guidance_cap`、`hoi_max_age_s`、`hoi_max_skew_s` | RealConfig。 |

显式非法值直接报错，不回退到 Checkpoint。控制周期缺失时，在 `--config` 指定的 Python 实验文件中设置 `CONFIG.execution.control_dt_s`。覆盖值仍须通过现有控制频率、预测长度和 ODE 限制检查。

模型使用 `point_cloud` 时必须保存对应 recipe；`hand_pointcloud` 使用 RealConfig.hand_pointcloud；recipe 内有默认值的字段可以省略，未知字段和非法值拒绝。未使用的几何配置保留本地值。`fingertip_xyz` 使用 Checkpoint 的 `fingertip_link_names` 顺序。`pointcloud.workspace` 是模型点云裁剪范围，与 RealConfig 的机器人安全 `environment.workspace` 职责不同。

缺少手模板时运行 `python scripts/generate_hand_template.py --num-points 1024`；使用其他点数时替换该参数。

Depth 必须保存 `depth_config.shape`（height, width）和 `depth_scale_m`。shape 必须等于本地 `camera.height`、`camera.width`，不自动 resize。设备实际 depth scale 在相机启动后、机器人保持停止时核对；不自动修改比例、内参或坐标系。

HOI 由 `observation_fields` 中的 `object_pointcloud` 自动识别，无需额外 CLI 开关。使用 info.task_name 定位任务目录并校验模板结构；手模板点数取 RealConfig.hand_pointcloud，不从 checkpoint 覆盖；仍拒绝同时使用普通 `point_cloud`。非 HOI 模型不加载模板或启动 HOI Worker。

`--print-config` 只打印声明的 RealConfig，不加载模型或标定，因此不是最终有效部署配置。启动摘要另显示 Checkpoint、Action Mode、有效周期／频率与来源、Action Steps 与来源、Execution Mode、实际使用的几何／Depth recipe 和输出路径；`run_config.yaml` 继续使用原有归档结构。

指定 `--output` 或启用 `--record-raw` 后，输出目录、`run_config.yaml` 和 `session_result.json` 都是实验产物。创建或写入失败会结束会话并报告错误；目录不可用时不会推迟到下一次 B 重试。策略评估与遥操作中任何一段的程序、录制或收尾错误都会使最终退出码非零，后续成功的新段不清除此前失败。来源过期和操作取消仍通过各段的结束原因报告。

## 执行模式

| 模式 | 预测与动作接续 |
| --- | --- |
| sync | 等待预测，执行一段动作后再请求下一次预测。 |
| async | 执行当前序列时提前预测，按观测时间接续仍有效的新动作。 |
| rtc | 支持 RTC 的模型接收当前动作前缀，按已提交进度接续预测。 |

## ODE 的动作承接

加载模型后，`PolicyInfo.requires_prev_action` 标识 ODE。它只支持 `sync` 且 `execution.action_steps == n_action_steps`；默认从模型读取步数，异步、RTC 和缩短 chunk 会在连接设备前报错。其他策略保持原有调度与预测参数。

每个 Rollout 从 `prev_action=None` 开始。每条 `Robot.send_action()` 正常返回后，将裁剪后的 Arm 7 维、Hand 12 维目标合并为新的物理单位数组；完整执行 H 步后，把最后一次提交的目标传入下一次预测。不会提前使用预测末帧，也不会用实测关节位置覆盖目标。提交异常不推进承接值或动作游标，现有错误流程结束本段；中途停止立即终止并清空承接，下一段重新初始化。预热不发送指令，不改变承接值。

这里的提交目标与 Raw 的指令字段语义一致；正常返回不证明两设备原子执行、SDK 回执或物理到位。Arm 的 Ruckig 中间参考仍由驱动管理，不作为 ODE 起点。裁剪生效时，下一 chunk 从裁剪目标重新起步，这是执行约束适配，不能视为原始无约束积分曲线的连续延伸。

实机闭环、SDK 提交与伺服跟踪仍需单独验收。

`--inference-steps` 对 ODE 设置每个控制周期的积分子步数，不是整个 chunk 的场计算次数；它不改变 H 或控制周期。

## 来源时间与时效

execution.hoi_max_skew_s 默认 None，关闭额外来源时间差限制；物体点云与手几何或 joint_state 联合消费时，可配置有限正数秒值启用。HOI 原始相机时间、臂反馈时间、手反馈时间的 max−min 超限报告 hoi_source_skew 并停止本轮；不以推理完成时间替代来源时间。导出使用同一实验预算。此检查不代表物理同步，以下年龄检查仍独立生效。

最新关节与滞后物体点云的组合仍是实验近似；时间差预算只限制这种近似，不代表物理同步，关闭它也不会关闭年龄或跟踪有效性检查。


观测采用 latest-value 语义：有效样本可重复读取，读取不刷新来源时间。年龄按主机 `time.monotonic_ns()` 计算；Arm／Hand 使用 SDK 查询开始时间，Camera 使用新帧首次接收时间，VR 使用完整腕部／关键点观测中较早的接收时间。这些均非同步硬件采样时间，Hand SDK 也没有可靠的硬件采样序号或时间戳。

| 用途 | 默认预算 | 约定 |
| --- | --- | --- |
| Arm 反馈 | 160 ms（`4 / 25` s） | 默认 25 Hz 查询，查询周期须小于 TTL。 |
| Hand／模型或录制必需触觉 | 200 ms | 共用主机查询时效预算；两个触觉通道独立计龄。 |
| Camera／普通 Point Cloud | 约 133 ms（`4 / 30` s） | 点云继承相机来源时间与序列，处理消耗同一预算。 |
| HOI | 250 ms | 独立的 `execution.hoi_max_age_s`，仍按来源相机时间计龄。 |
| VR 来源 | 500 ms | 暂定掉线容忍度，不能由控制频率推导。 |
| VR／Keyboard 目标租约 | 250 ms | 与来源 TTL 独立，由控制循环续租。 |
| 策略命令租约 | 2 s | 命令到期后停止执行。 |

配置见 [config](../dexmani_real/config/)。HOI 预算来自已记录回放，依据与验证范围见[性能结论](object_reconstruction.md#当前实现与性能结论)；普通点云延迟和 VR 真实生产频率仍待测量。

共享 Ring 返回独立字段字典和只读数组；需要修改数据的消费者须自行复制。字段子集可复用同序列完整缓存，不重新复制大数组。并发复制重试后可能返回上一份完整快照，来源序列和时间保持不变，仍须通过消费者的时效检查。该协议限定 Linux x86_64、单写多读。

`ObservationNeeds` 只检查消费者实际需要的模态。必需输入缺失、过期或失效时结束本段，清空历史与动作计划；来源恢复后需要重新按 B。每段首次有效观测填满 `n_obs_steps`，不在同段重新填充恢复。关键机器人反馈始终须有效，耗时动作映射后还会复核反馈和输入时效。

模型需求由 `ObservationBuilder.needs` 定义；PolicySession 将它与共用的 `RAW_RECORDING_NEEDS` 一次性解析为传感器需求 `sensor_needs` 和读取需求 `read_needs`。前者决定相机、Depth 和 Worker 的初始化，后者决定派生 Ring 携带的模型图像及 Rollout 的读取字段。Rollout 直接接收解析结果，保留独立的 Raw 质量检查；HOI／普通点云模式下，Raw 仍另外读取原始 Camera RGB-D。

三种执行模式共用采样、动作提交和录制流程；只有动作选择与接续方式不同。控制循环等待下一控制步或反馈采样机会，键盘事件可唤醒等待，其他服务的协作检查间隔上限为 5 ms。这个等待上限不代表 SDK 调用或物理停止的延迟保证。

## 输入失效后的操作

观测失效会结束当前段并清空动作计划和历史，执行停止流程；Recorder 完成收尾、迟到预测消费完毕后才允许再次按 B。来源恢复本身不会恢复运动。检查终端的具体原因：

| 原因码 | 检查对象与处理 |
| --- | --- |
| `camera_source_stale`／`camera_source_unavailable_or_unmatched` | Camera RGB-D 来源及采集进程。 |
| `hand_tactile_aggregate_timestamp_ns_unavailable_or_stale` | 聚合触觉 aggregate 来源。 |
| `hand_tactile_dense_timestamp_ns_unavailable_or_stale` | 稠密触觉 dense 来源。 |
| `vr_stale`／`vr_unavailable` | VR 来源输入；不等同于机器人控制器故障。 |
| `pointcloud_stale`／`pointcloud_unavailable`、`hoi_stale`／`hoi_unavailable`／`hoi_pose_invalid` | 对应派生观测、相机来源和 Worker 日志。 |
| 必需 Worker 退出、SDK 通信故障、停止失败 | Session 故障退出；检查原因后重新启动程序，不能按 B 原地继续。 |

终端按段编号报告结束原因，包括未进入运行的准备失败；只有实际开始运行的段计入 `--num-episodes`。停止和收尾完成的提示出现后，检查来源，再显式按 B 建立新段。每个新段重新进行有界观测等待和 TTL 检查，首次有效观测填满 `n_obs_steps`，首个预测重置模型；不要求等待 `n_obs_steps` 个不同来源帧，也不复用旧历史和动作。

HOI 接纳监测和提交前最新状态检查仅读取 Episode ID、来源时间、相机序列和有效标志，不读取 RobotState、点云或图像。完整模型观测仍读取所需数据，并在动作映射后重新校验原输入时效。两条 HOI 读取路径共用请求匹配：请求必须存在且 ID 非零，结果 ID 必须匹配，来源时间和相机序列都必须严格晚于请求边界；不匹配统一报告 `hoi_unavailable`，位姿无效和过期分别报告 `hoi_pose_invalid`、`hoi_stale`。

Sensor Worker 与模型线程跨段保留；段间只回收本段 Recorder 和在途模型任务。HOI 段结束发布零 Episode ID，下一次 B 用新 ID 重置场景，旧 ID 结果不能进入新段。初次 HOI 观测等待保持 120 s，其他启动观测等待保持 1 s。取消、退出、急停仍优先，不增加自动重试或恢复状态。

“停止流程已完成”表示现有停止调用成功及资源收尾完成，不等于 Raw 保存成功，也不等于机器人整体物理停稳证明；Arm 新段准备仍检查实测停稳，Hand 没有等价证明。Teleop 和 Policy 均按当前段的 Recorder 最终结果显示录制情况：

| 当前段结果 | 终端提示 |
| --- | --- |
| 未创建 Recorder（未启用录制或准备提前结束） | 明确显示未启用或尚未开始录制。 |
| Raw 已发布，未截断 | 明确提示 Raw 保存成功及保存路径。 |
| Raw 已发布，有 `recording_truncated_reason` | 提示部分数据已保存、录制截断、原因及保存路径，不称为完整保存或完全失败。 |
| Raw 写入或收尾失败 | 用失败警告替代普通成功提示，列出错误和 Recorder 提供的 staging 路径；路径缺失时显示未提供，不探测可访问性或推断可恢复性。staging 不是正式发布的 Episode。 |
| Raw 未发布 | 只报告未发布，不推断是丢弃、零帧还是尚未执行发布。 |
| Writer pending | 明确显示仍在收尾、尚无最终结果，不显示正常完成或重启提示。 |

两入口共用 Raw 技术结果格式化；结束原因和任务结果分别显示。Policy 摘要直接使用已有时长与跳槽数，Teleop 不新增计时统计。在线任务结果显示 `unknown`，离线标注方式见 [README](../README.md#一次实验)，不会自动影响 Zarr 或训练选择。

仅停止和必要资源收尾均完成、且没有退出或急停请求时，才提示可以按 B。录制失败后若仍处于待命状态，先检查磁盘空间、目录权限及 Recorder 日志，再确认输入正常后开始下一段。停止失败或收尾异常继续按原规则传播，不显示停止成功或重新开始提示。提示只依据当前段结果；历史失败仍保留并影响最终退出码，不会把后续正常段显示为失败。

## Raw 录制与连续区间

- 模型所需 RGB／Depth 与 HOI 或普通点云同源；Raw 独立读取控制采样时最新的 Camera RGB-D，按普通 Camera TTL 检查并保留实际 `camera_ns`、`camera_sequence`。
- 录制要求两个触觉通道均在 Hand TTL 内，读取缓存不刷新来源时间；显式 reset 使旧值失效。模型不需要的相机／触觉只在启用录制时成为依赖。Raw 保留来源时间和相机序号，导出不做额外来源诊断扫描；联合 HOI 仅在显式设置 hoi_max_skew_s 时拒绝来源时间缺失、未来或时间差超预算，None 关闭该检查，不修补历史样本。
- 相机或触觉来源失效、质量拒绝、队列满都会结束本段。先停止设备，再排空已经接收的完整行；写入和关闭正常时保存有效前缀，保留结束原因。所有已启动的必需 Worker（包括仅用于录制的 Camera）退出都会终止会话。
- 文件／编码 I/O 失败结束本段，保留 staging 并报告，不发布失败录制。初始化失败时不开始运动；Writer 实际退出并完成收尾后才能重新按 B。发布与失败以 `EpisodeRecorder.result()` 为准，`poll_finish()` 仅表示收尾完成。停止、录制关闭和迟到预测的失败都会更新本段结果；程序错误、停止失败和无法回收的资源继续退出会话。
- RGB／Depth／状态逐行对应，MP4 PTS 相对首行控制时间。调度跳槽后，在下一条已保存行记录 `/timing/segment_starts`；非空录制从 0 开始、严格递增，`recording_truncated_reason` 仅作诊断。质量拒绝与队列满之后不再接收新行。
- Raw 保留 `termination_reason` 和可用的 `recording_truncated_reason`；详细错误仍由运行日志及已有 Policy 会话结果报告，不再复制到 Raw 的 `/meta/termination_details`。历史 Raw 无需改写。
- Raw 保留相机参数、深度与触觉尺度、坐标变换、关节顺序及来源时间。遥操作按当前控制行保存 `vr_wrist_pos`（float64，N×3）、`vr_wrist_quat_wxyz`（float64，N×4，wxyz 顺序）、`vr_landmarks`（float64，N×21×3）、`vr_recv_ts_ns` 和 `vr_sequence_id`（uint64，N）；保留 `vr_to_robot_rotation`，启用手部重定向时还保存 `hand_retargeting_calibration_json` 中的 `bone_lengths` 与 `palm_width`。采样频率与行对应不变，不另建高频 VR 流；`collection_source=policy_rollout` 仍不含 VR 字段。部署完整配置保存在已有 run_config.yaml；遥操作完整配置由研究者自行管理。
- 导出与回放必须有合法 `segment_starts`，不从时间间隔猜测断点。导出保留有效行，每个连续区间形成一个 Zarr `meta/episode_ends` 边界，训练窗口不跨断点；`raw_episode_count` 统计文件，`episode_count` 统计区间。区间按标称频率使用，周期抖动只告警、不重采样；控制时间倒退、非法数组和所选字段非有限值仍拒绝。

策略等待预测时仍可记录观测；没有指令提交尝试的行，其动作字段为 NaN。选择动作字段导出或进行物理回放时，这些行会被拒绝，不填充旧动作。

多区间回放须显式选择，如 `python scripts/replay_episode.py EPISODE --segment 0`。编号从 0 开始，非法选择在连接设备前报错；回放以所选区间首行检查起始姿态，不跨段或补过渡动作。批量导出可用 `--exclude` 排除不符合契约的 Episode；失败 staging 不发布。

## 结束与 HOME

| 场景 | 行为 |
| --- | --- |
| Teleop 输入／反馈不可用或目标过期 | 结束本段，确认停止并完成录制收尾；再次按 B 时重新准备、读取有效观测并建立锚点。 |
| Policy 反馈或模型输入不可用 | 结束当前 Episode；设备、录制和在途预测收尾后才可开始下一轮，旧预测不进入新段。 |
| HOME 的 `FeedbackUnavailable` 或 `HomeTimeout` | 本次 HOME 未完成；停止收尾成功且无硬故障后保留空闲会话，可重新按 H 或按原流程 B。Arm 失败时不执行 Hand HOME；任何阶段失败均不自动重试。 |

S/C 保存结束本段，D 丢弃本段，Q 保存退出，Esc 急停；准备和 HOME 期间停止指令同样优先。停止未确认、真实 SDK 故障、程序错误及底层普通 `TimeoutError` 退出会话或报错。Keyboard Jog 保持其退出边界。

SDK 接受停止请求不等于物理停稳；Arm 准备检查实测停稳，Hand 无等价证明。输入中断后的停止、新段开始和准备期间的操作指令优先级仍需实机验收。

返回[项目架构](../README.md)。


## HOME 规划与准入

三个入口使用同一个 `home_robot()`，顺序固定为停止、实测停稳、Arm 规划与执行、Arm 实测到位停稳、Hand 回位。
研究者须事先确认没有抓持物体，并验证 Arm HOME 附近的手部活动空间；触觉不用于自动判断手中无物。

- Direct 为当前实测 Arm 姿态到 HOME；只有明确几何／路径越限拒绝才尝试固定 Via。`arm.homing.via_qpos` 默认 None，不提供假定安全的姿态。
- 初始实测速度超过 HOME 限速、或候选总时长超过 HOME 预算时直接报错／超时，不进入几何备选。
- Via 两段都通过才执行，中间点速度和加速度均为零。模型／配置／程序错误不触发 Via，执行开始后也绝不改走 Via。
- Ruckig 离线生成 100 Hz 默认伺服周期下的参考点（实际取 `arm.servo_hz`），完整极值检查复用在线轨迹校验。父进程经 Pipe 一次传递样本，Owner 在 Mode 1 原样发送。
- 碰撞模型包含 Arm 自碰撞、已启用 Arm–Hand 对及静态环境；手部内部碰撞对仍未验证。XHand 固定为规划时实测姿态。
- HPP-FCL 使用统一安全余量。检查覆盖所有发送点，并按模型几何半径、关节运动界和余量沿 Ruckig 曲线细分。相邻检查区间的相对几何位移估计控制在余量的一半以内；这不是连续无碰撞证明，也不覆盖真实跟踪误差或异常制动。
- Owner 检查起点、跟踪误差、调度迟到及手部有效反馈时间戳；父进程检查手部保持偏差。短暂锁竞争沿用最后一份已验证的来源时间，不延长时效；父进程 I/O 阻塞导致时间戳过期时，Owner 停止，不能继续整条轨迹。不会补发追赶。

必须显式填写（默认 None）的实验参数：

| 参数 | 含义 |
| --- | --- |
| `arm.homing.max_jerk_deg_per_s3` | HOME jerk；速度、加速度沿用已有 HOME 配置，均不得超过普通 Arm 运动限制 |
| `arm.homing.collision_margin_m` | 所有已启用碰撞对的最小模型距离阈值 |
| `arm.homing.tracking_error_rad` | 实测位置与对应时刻离散参考的最大误差 |
| `arm.homing.hand_hold_tolerance_rad` | 停止准备及 Arm 路径期间允许的手部姿态变化 |
| `arm.homing.max_lateness_s` | 发送及 SDK 调用迟到预算，必须小于一个伺服周期 |
| `hand.home_max_step_deg` | 相对每个新鲜实测位置的最大指令增量，不是硬件速度限制 |
| `hand.home_tor_max` | Mode 3 HOME 输出上限，与逐关节原限制取较小值；不是 mA |
| `hand.home_timeout_s` | Arm 完成、手部反馈有效后的实际手部运动预算 |

`environment.home_model_confirmed=False` 默认拒绝 HOME。确认意味着所有固定障碍均已可信建模，或场景确实没有固定障碍；不会从点云桌面标定推测碰撞盒。
Arm 起点／到位停稳使用既有 `standstill_*` 参数，这些参数亦须实测验收。

Arm 规划前必须确认 Hand 停止接口实际接受了位置保持；因反馈不适合保持而进入 passive 时，拒绝 HOME。位置保持回执本身仍不证明物理停稳。

Hand 每次只基于提交后的新反馈生成小步目标，禁止旧缓存累积目标，连续三个提交后样本在容差内才成功。
CRC 写失败立即停止，无运动补发；停止接口保留既有有界重试。停止保持不提高 HOME 输出上限，普通命令恢复原逐关节配置。

`HomePathRejected` 表示未发送 HOME 运动目标，停止准备仍会发生。VR／Policy 保留停止的空闲会话；Jog 规划／配置拒绝后仅等待 R、Q、Esc，成功才重新准备 Jog。
取消和超时按原生命周期停止；设备、Owner 或停止错误继续传播，不能显示 HOME 成功。

离线测试不能验收 Mode 1 跟踪与调度、停止距离、URDF/SRDF 碰撞几何、环境标定、Via 姿态、手部保持／步长／输出和超时。
反馈时间戳是主机 SDK 查询起点，不证明硬件采样年龄。填写配置本身不构成可靠自动运行保证。

独立的 `scripts/tools/control_xhand.py` 是姿态诊断序列，使用固定的单动作 2 秒预算，不调用组合 HOME，也不据此声称经过 Arm 路径验证。
