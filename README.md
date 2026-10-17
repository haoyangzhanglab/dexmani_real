# DexMani Real

面向个人 PhD 研究的真实机器人灵巧操作仓库，当前实验系统为 **xArm7 + XHand + RealSense RGB-D + VR / HTS**。

本仓负责硬件与传感器接入、标定、遥操作采集、Raw 数据、离线处理及策略真机部署。模型、Dataset 和训练位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真位于 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

工作流：**标定 → 采集 → Raw → Canonical Zarr → 策略训练 → 真机评估**。

## 安装与配置

使用 Python >= 3.10，在源码 checkout 中安装；机器人资产按 checkout 路径读取。

```bash
python -m pip install -e .
python -m pip install -e ".[interactive,control]"
```

沿用实验机现有的硬件 SDK、几何与推理环境。离线几何处理需要 Pinocchio，retargeting 需要 NLopt 和所选后端；点云诊断视图需要 Open3D，交互数据查看需要 Rerun。键盘监听使用 pynput，需要桌面显示环境。

默认参数来自包内配置，`--config` 接受 YAML 字段覆盖，CLI 操作开关优先。可先打印声明配置，再编辑现场参数：

```bash
python examples/collect_teleop.py --print-config > local.yaml
python examples/collect_teleop.py --config local.yaml --print-config
python examples/run_policy.py --config local.yaml --print-config
```

`--help`、`--print-config` 不连接设备；打印配置也不读取现场标定或加载策略。实际运行入口检查所需配置、标定与执行条件。日志在运行入口初始化，可通过 `DEXMANI_LOG_DIR` 指定目录，默认写入 `~/.dexmani/logs/`。

## 实验工作流

### 标定

```bash
python examples/calibrate_camera.py --help
python examples/calibrate_vr_heading.py --help
python examples/pointcloud_process_example.py --help
```

相机外参、桌面平面、手部安装与 VR alignment 属于当前实验现场状态；物理设置变化后重新标定或确认。首次桌面标定在 pointcloud 示例中选择 table calibration。指尖 FK 使用当前手部安装补偿，碰撞与规划使用标称资产几何。

当前相机链路使用固定安装（eye-to-hand）外参，按相机 serial 绑定。采集允许缺相机外参；点云重建需要对应相机的内外参与 depth scale。标定在会话启动时读取。采集可指定 `--vr-transform` / `--camera-calibration`，部署可指定 `--camera-calibration`，相机和 VR 标定可指定 `--output`；桌面平面路径由 `environment.table.plane_path` 配置。

VR 朝向标定按 `--ref` 等待头部或手腕观测，头部使用自身接收时间去重；缺少头部追踪不影响手腕参考标定。VR 接收进程异常退出时不发布标定结果。遥操作启动会拒绝质量为 poor（朝向标准差 ≥ 5°）的标定，需要重新采集。

相机预览的 `MARKER OK` 表示当前预览帧检测成功；保存标定样本还需通过独立的多帧采样、静止和有效性检查。采样或求解结束后，键盘移动使用重新读取的机械臂反馈。

相机标定的键盘移动与旋转步长使用 `keyboard_teleop.delta_pos_m` / `delta_rpy_rad`，与键盘遥操作共用配置；默认每拍 0.008 m / 0.03 rad。采样后按 ENTER 比较五种手眼求解结果，通过激励和残差检查后保存；BACKSPACE 撤销最近样本，X 删除当前评估中位置残差最大的样本，删除后需重新求解。

### 采集

```bash
python examples/collect_teleop.py <task> --config local.yaml
python examples/collect_teleop.py <task> --config local.yaml --control-hz 20
```

任务名默认 `test`，用于录制元数据和 `policy.episodes_dir/<task>/` 输出目录。启动时先建立停止监听，再执行手部 HOME。

`--control-hz` 覆盖 `teleop.control_hz`（默认 16 Hz），同时设置目标、观测采样和录制频率；可按实验选择 10、16、20、30 Hz 等，不要求整除 xArm 的 100 Hz。XHand 每拍读取一次、每个有效动作发送一次，不重复发送到 100 Hz，也不静默丢拍降频。83 Hz 是手部文档上限，实际可用频率还受串口读写、IK 和采集负载限制。目标周期必须短于 arm/hand 的健康与反馈年龄预算；更低频率需显式调整这些预算。切换频率会改变输入 EMA 的时间响应，应在实验配置中记录。策略始终使用训练保存的 dt；键盘入口同样支持 `--control-hz`，松键会停止 jog。

| 按键 | 操作 |
|---|---|
| H | 保存并结束当前段，再执行整机 HOME；完成后等待新 B |
| T | 空闲、无 capture 且确认五指无接触时，请求复位触觉传感器 |
| B | 开始采集 |
| C | 暂停或恢复；恢复开始新 episode |
| S | 停止并保存，回到空闲；新 B 可开始 |
| D | 丢弃当前未发布 capture，回到空闲 |
| Q | 退出确认，再次 Q 保存退出；空闲且无 capture 时直接退出 |
| ESC | 急停 |

触觉保持 SDK 单位，采集和导出均不扣软件偏置。局部 NaN/Inf 或 SDK 标记不可信时，以一个 taxel 的完整三轴向量保持上一有效值；汇总力独立处理。Raw 保存控制与策略实际消费的有限触觉观测，元数据标记有界保持及其时限。保持值沿用原有效时间，其他 taxel 和新关节反馈继续更新；不做空间插值或额外平滑。连接、每轮开始和触觉复位后重新建立缓存。START 在 `hand.initial_state_timeout_s` 内等待有效初值，就绪后同次请求继续；S/Q/ESC 可取消，等待不授予运动权限。运行中任何必需触觉通道过期均进入故障停止，包括策略未使用触觉的情况。

有限值保证覆盖控制和模型实际消费的数值，包括关节状态与有界保持后的触觉。当前模型字段映射不消费 hand current / arm effort；这两项辅助诊断与记录字段缺测时仍保留 NaN，不缓存或填零。模型输入另有有限性检查，内部未初始化的缓存不能作为有效观测发布；这不保证全部 Raw 字段均有限。

T 对五个指尖各请求一次传感器复位，SDK 调用完成不代表已确认固件复位或零点效果；失败即停止后续调用。Raw 保存最近一次请求的时间和结果，未执行 T 也可采集。S/Q/ESC 在当前 SDK 调用返回后中断后续请求。连接、H 和 B 不自动复位传感器。键盘 jog 入口仅在 R 时执行 HOME。

遥操作从首个实际采样建立固定时间网格；读取、计算和下发可用本拍剩余时间。错过整拍、反馈过期或下发跨过截止时间会结束本轮并保存真实前缀，不追赶补发。无可执行目标时统一停止，录制开关不改变这一行为。同批停止/退出按键压过开始和恢复，S/D 同批优先保存。

遥操作默认位置比例为 `policy.vr_mapping.pos_scale=0.4`，位置 EMA 为 `policy.ema.alpha_pos=0.7`；同等机器人位移需要约 2.5 倍手部行程。参数统一由包内配置提供，实验调整通过 `local.yaml` 覆盖。旋转按腕姿态锚点等比例映射，不提供独立缩放配置。手部默认使用 TAG，将 `policy.hand_retargeting_type` 设为 `dexpilot` 可切换到 DexPilot；两个后端均使用当前 XHand 模型，参数分别由 `tag_retargeting` 和 `dexpilot_retargeting` 提供。

### 查看与导出

```bash
python examples/visualize_episode.py <episode> --info
python examples/visualize_episode.py <episode> --max-frames 100 --config local.yaml
python examples/export_policy_zarr.py episodes/<task> --config local.yaml
```

`--info` 查看元数据与数值概要，无需 Rerun；交互 viewer 顺序读取所选 RGB-D 帧。查看和导出使用当前点云、桌面与手部安装配置，相机几何来自对应 Raw。仅当 `pointcloud.remove_table: true` 时读取桌面平面。viewer 的点云耗时只统计已读取 RGB-D 的派生计算，不含读取和显示。

公共导出要求每段全部 **15 字段**完整，每行浮点数值均有限；缺字段、NaN/Inf、RGB-D 行数不符、相机几何缺失或点云无法重建时，整段拒绝并在日志中记录原因。完整性检查覆盖全部模态，不随模型所需字段而放宽，hand current / arm effort 的辅助 NaN 也会导致整段拒绝。Raw 原样保留，导出不删单帧、不插值、不填补缺测；触觉直接使用采集保存值，不再次保持或修补。下游按模型所需字段加载。

训练准入同时检查逐拍间隔与累计时间偏离，容差为 `dt/4`；单行会明确提示无间隔可检验。此容差用于导出筛选，不作为遥操作停机条件；Raw 保留实际时间，正常结束的采集也可能因时序超差被拒绝导出。训练 action 的机械臂部分表示本拍提交给执行器的绝对 `q_goal`，不要求下一拍到位；`action_ee` 是该目标的 FK。Mode 1 机械臂须有首点 ACCEPTED 和完整后续执行证据；正常替换尚未到达的目标有效。XHand 允许 ACCEPTED 或 CRC 未确认，Raw 保留真实发送状态。未发送、明确拒绝或调用异常仍整段拒绝。任务失败本身不排除技术有效数据。

默认输出为 `datasets/<task>.zarr`，目标必须不存在，可用 `--output` 指定新路径。Zarr 使用 Diffusion Policy 的 `data/*` 和 `meta/episode_ends` 结构，根属性仅保存数值处理、数据身份与 episode 名称。处理逻辑变化后从 Raw 重新导出。文件读写故障或处理逻辑异常会中止导出，已创建的 staging 保留供排查后重建。

导出按块处理，`--chunk-frames` 指定每批帧数上限（默认 100）；内存紧张时可调小，如 `--chunk-frames 16`。批次大小不改变数据行、采样频率或动作标签。`--pointcloud-num-points` 覆盖配置文件中的点数，按覆盖后的配置校验并导出。

Canonical 与部署统一使用 `fingertip_xyz`、`finger_tactile_force`、`taxel_tactile_force`，并提供 `camera_intrinsic` / `camera_extrinsic`。两条路径均保留实测关节角精度完成 FK，再转为模型使用的 float32。相机参数采用 Sim 的 float32、C-order K / [R|t] 布局，Real 外参方向为 `xarm_base` → color optical RDF（右、下、前）；Real Base 与 Sim World 的数值坐标不必相同。K/E 不包含完整畸变模型，Raw 保留 distortion model、coefficients 和 depth scale，图像与点云处理方式不变。字段 shape 与物理语义见 [Canonical 契约](dexmani_real/dataset/contracts.py)。旧 Canonical Zarr 须从 Raw 重新导出，不提供旧名别名或自动迁移。

### 策略部署

```bash
python examples/run_policy.py <policy/task/experiment> --config local.yaml
python examples/run_policy.py <policy/task/experiment> --execution-mode async --prefetch-steps 1 --n-action-steps 4
```

默认最多评估 20 轮，Q 提前退出会话，ESC 急停。H 回初始位、B 开始本轮、S 结束本轮；结束后保存结果并等待，整理场景后再次按 H、B 进行下一轮。T 在空闲且无接触时请求复位触觉传感器。可用 `--num-episodes N` 修改运行轮数上限，达到上限后退出；默认每轮运行预算 60 秒，可用 `--max-duration` 修改。

开始及 reset 后重新采样，复用运行时策略输入检查并核对 arm/hand HOME 姿态；启动容差默认机械臂 0.5°、手部 5°，机械臂 HOME 运动的收敛容差仍为 0.15°。准备失败保留具体原因，不进入 RUNNING 或消耗运行段数。启用 Raw 时仍要求可匹配且新鲜的 RGB-D 来源，单独报告录制来源失败。

策略负责保存的 horizon、观测历史、模态、normalizer、控制时间步和点云生产 recipe；encoder 内部采样点数可以不同于生产点数。Real 负责当前设备、现场状态和执行参数。入口一次性解析 `A_exec`：优先 `--n-action-steps`，否则读取策略保存的默认值，结果记为 `execution.action_steps`（由入口生成，不能在 YAML 指定），不覆盖模型配置。模型仍输出扣除观测历史偏移后的完整 `P = horizon - n_obs_steps + 1` 步未来动作；sync 要求 `A_exec <= P`，async/RTC 要求 `1 <= d <= A_exec` 且 `A_exec + d <= P`。

执行模式通过 `--execution-mode sync|async|rtc` 设置，CLI 优先于 YAML 的 `execution` 字段；async/RTC 使用 `--prefetch-steps d`，RTC 还需显式设置 `--rtc-guidance-cap`。当前 RTC 仅支持连续动作 BaseAgent DDIM 策略，ManiFlow 支持 sync/async。sync 与普通 async 在执行时根据当前反馈实现动作；async 在交接前 `d` 步查询，不提前冻结物理命令。RTC 保留前 `d` 步命令预实现与冻结，以对应物理 prefix 做 guidance，执行冻结命令前重新检查；`--rtc-guidance-cap 0` 保留冻结、关闭 guidance，可与正值比较。

sync 在推理完成后读取新反馈并开始执行，段内按策略时间步发送，段间保留实际推理等待。async/RTC 保持固定节拍，交接时读取反馈后消费已就绪的预测，并保持动作索引与前缀对齐；命令仍须在当前拍截止前完成下发。策略输入与执行配置校验、模型加载与 warmup、用于动作准入和 HOME 的几何模型构建均在设备连接前完成。warmup 用于初始化，不据此推算启动准入；运行时直接检查实际反馈、决策年龄和等待预算。

遥操作和策略共用 `ActionRealizer → RobotCommand → xArm 执行进程 → Ruckig → Mode 1`。专用进程唯一持有 xArm SDK，隔离主进程 XHand 原生调用对 Python 调度的影响；XHand、输入与 IK 留在原低频控制侧。`send_action` 提交机械臂目标后直接发送手部，不等待伺服 tick、机械臂 SDK 回执或到位；执行进程检查机械臂 SDK 返回结果并锁存故障，手部结果由当前调用返回。开始／恢复时确认新鲜实测静止状态，HOME 使用 Mode 0 路点及实测稳定等待。

安装控制依赖使用 `python -m pip install -e ".[control]"`；固定采用本地 Ruckig Community 0.17.3，仅使用单目标七关节速度/加速度约束，没有 jerk 调参或云端路点。源码安装需要 C++/CMake 构建环境。接口依据：[UFACTORY Servo Mode Guide](https://docs.supportarticle.ufactory.cc/support_articles/developer/ufactory-servo-mode-guide.html)、[Ruckig 0.17.3](https://github.com/pantor/ruckig/tree/v0.17.3)。

`arm` 下的执行参数独立于低频目标 dt，可通过现有 YAML 配置：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `servo_hz` / `feedback_hz` | 100 / 25 Hz | 参考发送 / 同步反馈查询频率 |
| `max_joint_velocity_deg_per_s` / `max_joint_acceleration_deg_per_s2` | 135 / 810 | 转为 rad/s、rad/s² 约束 Ruckig；HOME 保留适用 SDK 参数 |
| `servo_max_interval_s` / `sdk_timeout_s` | 0.020 / 0.020 s | 实际发送间隔及单次调用预算 / SDK GET、SET 响应超时 |
| `executor_timeout_s` | 3 s | 初始化额外等待、停止及关闭超时 |
| `tracking_error_rad` | 0.35 rad | 参考跟踪偏差上限 |

Ruckig 从连续参考位置和速度生成轨迹，按实际 monotonic 时间采样，检查完整轨迹制动极值；调度使用绝对 tick，不追赶补发过期点。低频提交截止与持续执行期限分开：后者检查当前代次、反馈、服务以及原始 VR 输入或策略决策年龄；新反馈和 heartbeat 不续期旧决策。WAIT 可在期限内继续当前参考并保持，但实测静止仍需独立确认。停止、故障、暂停和关闭撤销旧目标，恢复必须重新准备。执行计数、最近 SDK 调用和故障状态始终维护，不依赖录制是否开启。

这些默认值是工程预算，100 Hz 不是性能结论。反馈来自独立节奏的同步关节／状态／错误查询，使用查询开始时刻作为保守主机来源时间，缓存读取不刷新它；SDK effort 保持原单位。执行进程不能打断自身阻塞的 SDK 调用，关闭超时会明确报告仍未完成。发送间隔、停止响应、跟踪偏差和动作—反馈延迟需在完整真机负载下验证。关节位置伺服不提供力控/阻抗，Ruckig 也不证明运动路径无碰撞。

遥操作及各策略执行模式对 XHand 普通发送的偶发 CRC 均保留“提交未确认”并继续下一拍，不额外重发动作。连续发送 CRC 独立计时，正常发送响应清除异常段，成功读取不会清除；明确的空闲、停止和策略 WAIT 结束当前发送异常段。读取 CRC 仅在关节与全部触觉历史仍有效时回退，可信读取的主机时间不刷新。

运行期关节/触觉年龄及连续发送 CRC 共用 `hand.feedback_max_age_s`（默认约 133 ms）；初始反馈等待单独使用 `hand.initial_state_timeout_s`（默认 1 s）。达到运行期时限、过流、关节板故障或不明确的读取错误均进入故障停止；恢复一帧不自动续跑。停止命令保留独立的短时有界重试；停止失败保留故障和运动限制，SDK 接受不证明物理停止。SDK 长时间阻塞仍可能触发原有调度截止时间，需按实际设备确认这些预算。

健康的 bootstrap／sync 换段允许预算内 WAIT，不重复下发旧目标，也不保证上一已接受目标已经停止。async/RTC 预取期间继续执行当前有效计划。必要观测失效、计划过期、错过时隙/交接或 WAIT 超时会撤权、尝试停止并结束本轮；观测恢复后不自动续跑，新 B 仍须满足 HOME、新鲜度和模型空闲检查。论文评估应记录执行模式、预算与终止规则。

checkpoint 默认 `latest`；可用 `--checkpoint 80pct` 选择已保存的 80% milestone，也支持 `best` 或 checkpoint 文件名。百分比必须对应已有 checkpoint；`best` 需要实验中的 `best_ckpt.json`。模型推理参数与 Real 执行参数见 `--help`。保存产物时，`run_config.yaml` 记录实际配置、checkpoint、覆盖值、输出开关与源码版本。`--output` 指定部署输出位置。

Raw 与评估摘要可独立关闭：`--no-record` 关闭 Raw，`--no-results` 关闭摘要，两者同时使用时不创建会话产物。仅使用机器人状态且关闭 Raw 时不启动相机；RGB、点云、相机参数输入或 Raw 录制按需启动对应传感器。部署仅提供策略请求的字段；K/E 使用当前相机会话的 active color intrinsics 和 serial 对应的外参，一次计算后按观测历史展开，仍检查相机帧的新鲜度。仅请求相机参数不会启动点云 worker；请求外参时缺少有效标定会明确失败。

RealSense 保留 L515 和 D400 支持，启动只尝试一次；发现、配置或预热失败时清理资源并退出，由操作者排查后重新启动。相机、VR 和实时点云按需在独立进程中运行，通过父进程创建的匿名共享缓冲区传递观测；退出时先撤销运动并关闭机器人，再等待传感器进程退出，无法确认退出会报告清理失败。

评估摘要保存在 `session_result.json`，包含结束原因、时长、故障和推理、发送、跳过节拍的统计。启用摘要时，连接设备前写入初始未完成状态，撤销运动后保存每段结果，资源清理后更新会话结果。清理失败或模型关闭超时返回非零退出状态，episode 原始结束原因独立保留。模型关闭超时表示尚未完成；Python 线程或 CUDA 阻塞仍可能延迟进程退出。

Rollout Raw 默认用于评估 evidence，保留 bootstrap/WAIT 的无动作行与具体终止原因（关闭摘要时也保留）。公共导出按同一时间、完整性与动作提交规则判定，通常会拒绝这些含无动作行的 episode；未来用于再训练需另行定义派生片段和时间映射，不能填动作或裁行来通过当前导出。

启用 Raw 时，`data.h5/timing` 保存观测、query、命令序号及提交时序；`arm_execution` 保存录制开始和停止后的累计执行摘要。机械臂 `SUBMITTED` 只表示目标已提交，手部状态保留自身 SDK 返回结果。导出与 replay 共用整段校验：提交序号连续、每个目标至少有一个 SDK 接受点、所有调用已返回且接受、无取消缺口或故障；正常目标替换不要求到位。已确认的控制失败仍保存前缀和结束原因，执行或停止无法确认时保留 staging。新采集使用 `mode1_episode_summary_v1`，不再保存逐伺服点轨迹；旧 Raw 保持不变，须用对应历史代码处理。具体布局见 [Raw schema](dexmani_real/recording/storage/schema.py)，读取与延迟诊断见 [EpisodeReader](dexmani_real/recording/storage/reader.py)。

部署 joint、EEF 的最终关节目标及 RTC 冻结目标统一检查机械/关节限位、跳变、FK 最终 EEF 工作空间、自碰撞及当前配置的 static boxes。最终 FK 工作空间检查默认允许各边界超出 5 mm，可通过 `execution.eef_workspace_margin_m` 调整，设为 0 恢复严格边界；EEF 目标投影与 HOME 路径使用配置的工作空间边界。Policy 动作不做桌面碰撞检测，允许贴桌/接触操作，由操作员监督并准备硬件急停；H 先让手部回 HOME，再按实测手部姿态、工作空间、桌面和静态障碍检查机械臂的整条规划路径；找不到安全路径则报告原因，不启动机械臂回位。点云去桌仍使用当前桌面标定。EEF 先投影再 IK；拒绝就终止本轮并记录原因，不重规划或恢复。这些是最终目标检查，不证明真实连续运动轨迹安全，仍需现场验收。

`--no-record`、`--no-results` 只控制产物保存，部署仍连接设备并执行动作，不能用于离线验证。

### 轨迹回放

```bash
python examples/replay_episode.py episodes/<task>/<episode> --config local.yaml --output replay_results/<run>
```

按 Raw 的标称节拍逐行回放关节目标，要求 teleop 数据、非空有限的目标与关节状态、双设备持续发送状态，并检查目标是否满足当前限位。回放与导出共用下发状态及 episode 执行摘要校验，正常目标替换无需到位。回放连接机械臂与手，不启动相机或 VR。输出目录须新建或为空，保存 `session_result.json`、已采集的 `replay_data.npz` 与可计算的一致性指标 `metrics.json`。Q 保存前缀并退出，ESC 急停；完成后可按 H 执行规划回 HOME。

回放启动先执行手部 HOME，默认再以 0.5 s 渐变到首帧手部目标；`--hand-start-duration 0` 仅跳过渐变，启动时的手部 HOME 仍会执行。预备动作完成后，回放使用固定单调时钟时间槽；错过整槽或发送跨过本槽/反馈截止时间会停止并保存真实前缀，最后一帧也须满足截止时间。机械臂、手部观测分别保留消费样本的主机时间，提交开始/结束时间独立保存；机械臂目标提交和 SDK 接受均不代表到位。评价中的轨迹帧偏移用于跨运行对齐，提交前目标—状态差使用发送前消费的反馈；两者均不表示硬件响应延迟或发送后的跟踪误差。

会话结束阶段，先保存实测关节、目标、命令序号、发送状态、时间和执行摘要，再离线生成 EEF 位姿并原子更新 `replay_data.npz`。摘要包含手部预备动作期间的机械臂执行，控制行只包含正式回放。EEF 派生或其 NPZ 更新失败时，第一次保存的物理捕获仍保留，EEF 字段维持 NaN；指标写盘失败不影响已保存的 EEF。错误记录到 `session_result.json`；已进入回放流程的零帧失败也保存执行摘要和终止原因，不计算一致性指标。输入 Raw 始终保持原样。

## 其他入口

| 用途 | 入口 |
|---|---|
| 键盘 jog / HOME | `examples/keyboard_teleop.py` |
| XHand 动作与反馈示例 | `examples/xhand_control_example.py` |
| RGB-D 诊断 | `examples/realsense_record_example.py` |
| 点云诊断与桌面标定 | `examples/pointcloud_process_example.py` |

键盘 jog 不启动相机或 VR。XHand 示例按 HOME → fist → palm → V → OK → HOME 顺序执行动作，实测到位后各停留 1 秒；`--read-only` 仍连接设备并读取反馈。相机诊断使用 `--config`；RealSense 诊断只建立一次相机连接，退出时恢复可读取的原曝光优先级。具体操作与参数见各入口 `--help`。

## 数据与安全

**Raw 是实验 evidence，发布后不可变。** 采集保留接入的 RGB-D、关节与触觉观测。停止、晚帧、控制失败和缺测保留已采前缀及结束原因，仅操作者显式丢弃才删除当前 capture。写盘、编码或发布失败保留 staging 并报告失败；相机停帧不补黑图，触觉按 Raw 保存值导出。

每个 Raw episode 只使用 `data.h5` 与 `rgb.mp4`，HDF5 保存物理观测、尝试发送的目标、相对秒时间与发送状态，以及采集和相机元数据。RGB 按解码顺序与 HDF5 控制行对齐，实际采样时间以 HDF5 `timestamp` 为准；视频按标称帧率编码。保存的 depth 与 RGB 像素网格对齐，尺寸来自当前相机会话的标定；查看和点云重建复用同帧深度。Raw 保存物理相机的畸变参数，点云算法尚不支持的畸变模型不阻止录制或 RGB-D 查看（`--no-point-cloud`），但不能通过点云重建和公共导出。

录制端非阻塞提交当前控制行，单个后台线程完成编码和写盘，视频编码器在 START 就绪前打开；启动和结束时先撤销运动，再等待文件准备或发布。运行时负责停止设备并传入结束摘要，录制器不操作执行器。队列溢出、写入失败或执行结束无法确认时保留 staging，不能发布为完整 Raw。

Canonical 是可由 Raw 重建的训练缓存。训练侧只加载模型所需 observation/action，并检查这些字段的数值；训练筛选不回写 Raw。

xArm 控制器碰撞检测默认关闭（`arm.collision_sensitivity: 0`），在每次连接初始化时下发；已有 YAML 中该字段会覆盖默认值。关闭后不能依赖该检测触发碰撞停机。软件侧关节限位、自碰撞和环境几何检查仍按各入口执行，操作员须监督并准备硬件急停。

真机运行前确认机械限位、安装、标定与工作空间，保持操作者在场并可急停。设备 SDK 或 CUDA 阻塞会影响停止响应；离线检查不能证明现场安全。代码代理未经明确授权不得连接设备、执行运动、实时采集或标定写入。

## 开发

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

按改动补充纯离线逻辑核查。长期协作规则见 [AGENTS.md](AGENTS.md)，Claude 入口见 [CLAUDE.md](CLAUDE.md)。当前行为以源码、配置和 CLI 为准。
