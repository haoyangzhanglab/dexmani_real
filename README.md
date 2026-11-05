# DexMani Real

xArm7 + XHand + RealSense RGB-D + VR/HTS 的个人研究代码，支持遥操作、Raw 采集、Canonical 导出、sync / async / RTC 策略部署及物理回放。训练与推理见配套 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)。

## 安装与实验配置

使用 Python 3.10+，在仓库根目录安装和运行：

```bash
python -m pip install -e ".[interactive,control,geometry,visualization]"
```

设备 SDK、HTS、Pinocchio、NLopt 和配套 Policy 按实际实验环境安装。机器人模型和手模板从源码目录 `assets/` 加载，运行时保留 checkout。几何模型的独立环境见 [CordVIP 使用说明](third_party/cordvip_geometry/docs/usage.md)。

省略 `--config` 使用 dataclass 默认值。需要修改设备、模式、增益、频率或点数时，新建可信本地 Python 文件并导出 `CONFIG: RealConfig`；例如将下例保存为 `~/dexmani_config.py`，运行时传入 `--config ~/dexmani_config.py`：

```python
from dataclasses import replace
from dexmani_real.config.real import RealConfig

base = RealConfig()
CONFIG = replace(base,
    teleop=replace(base.teleop, control_hz=16.0),
    execution=replace(base.execution, mode="sync"),
)
```

常用控制精度参数：

| 配置字段 | 默认值及用途 |
| --- | --- |
| `arm.standstill_position_tolerance_rad` | `0.002618 rad`，实测位置稳定及 HOME 到位容差 |
| `arm.standstill_velocity_tolerance_rad_s` | `0.03 rad/s`，实测停稳速度阈值 |
| `environment.eef_workspace_tolerance_m` | `0.005 m`，EEF 端点及 jog 实测起点的工作区验收容差 |

停稳阈值由 HOME、伺服准备与相机标定共用；`homing` 保留回零超时、速度和加速度。EEF 端点及 jog 起点验收默认向外放宽名义边界 5 mm；键盘目标的 `workspace_command_margin_m` 向内收缩工作区，范围外参考只允许逐步退回、不扩大越界。VR 起点仍使用名义工作区，仅容纳计算舍入。

任务、路径、段数、时长和录制选择通过各入口的 CLI 指定。遥操作和键盘点动可在 Python 配置中设 `hand.enabled=False`；VR 遥操作还须使用 `--no-record` 关闭录制，实物手须移除或固定在配置的 HOME 姿态，供碰撞规划使用。策略启动与覆盖项见[策略部署](#策略部署)。

遥操作录制选项由 CLI 提供：`--record/--no-record` 默认开启，`--episodes-dir` 默认 `episodes`，`--max-duration` 默认每段 60 秒。相对录制目录相对仓库根目录，绝对路径直接使用，再拼接 TASK。录制时 TASK 须满足下文任务标签规则，且不能是 `.`、`..` 或包含路径分隔符；允许 `.research` 和首尾空格。`--no-record` 不使用 TASK，也不校验录制目录规则。例如：

```bash
python scripts/collect_teleop.py TASK --record --episodes-dir captures --max-duration 90
```

查看请求配置：

```bash
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
```

这些命令只打印声明配置的 JSON，不连接设备、不读取标定或加载 checkpoint。Policy 仅打印部署相关配置段；最终输入 recipe 从 checkpoint 恢复。HandConfig 构造时将 NumPy 整数转为 Python 整数，设备调用、JSON/YAML 归档使用同一表示；布尔和浮点不能用作整数配置。遥操作打印配置中的 `collection` 包含最终 CLI 录制选项；运行直接消费显式参数，Raw 不保存整份请求配置。启用分支负责业务校验，实测相机事实在连接后记录。

## 常用实验路径

将 TASK、EPISODE、POLICY 替换为任务、episode 目录和 Policy checkpoint 文件或目录。训练导出要求已发布 episode。

```bash
python scripts/collect_teleop.py TASK
python scripts/export_zarr.py episodes/TASK --output datasets/TASK.zarr
python scripts/run_policy.py POLICY --num-episodes 5
python scripts/replay_episode.py episodes/TASK/EPISODE
python scripts/visualize_episode.py episodes/TASK/EPISODE
python scripts/keyboard_teleop.py
```

键盘点动仅连接机械臂时，使用 `python scripts/keyboard_teleop.py --no-hand`；可与 `--config` 一起使用，覆盖配置中的 `hand.enabled`。禁用手部后，碰撞检查仍使用 `hand.home_qpos_deg` 指定的固定手姿态。

Episode 可视化默认按实测 `arm_qpos`、`hand_qpos` 通过 FK 重建手部表面点云；在 3D 区域切换到 `Hand Point Cloud` 可查看手部点云、指尖和 EEF 锚点，`xArm-base` 显示相机点云及指尖、EEF 锚点。两者共用基座坐标系和时间轴。手部点数由 `CONFIG.hand_pointcloud.num_points` 指定；`--no-point-cloud` 只关闭相机点云，手部重建不依赖相机外参。

VR 遥操作：B 空闲时开始新段；S/C 停止并保存；D 停止并丢弃；Q 保存当前段并退出，但保留已有 D 意图；Esc 急停并结束会话。保存决定确定后，迟到 D 不再逆转结果。H 仅在空闲且收尾完成时回初始位，活动段请先 S；T 仅在空闲时复位触觉，操作者须确认无接触。准备或收尾期间的 B/H/T 不会排队启动下一段。

Policy 只使用终端提示；其他保留音频的入口使用系统 `aplay` 或 `paplay` 播放提示，播放器不可用和后台播放失败只记录 warning；音频关闭的普通异常（包括线程未及时退出）由运行入口记录 warning 和诊断信息，不改变实验退出结果。关闭期间的 KeyboardInterrupt 立即锁存急停并保持失败退出；已有运行或资源收尾失败也不会被音频处理覆盖。一个后台线程播放，待播槽只保留最新提示；允许合并、漏播和延迟，不保证 FIFO 或急停音效抢占。退出取消待播及尾音，不等待排空。提示音不能证明机器人已停止、物理到位或任务成功。

物理回放使用 S/Q 停止并退出、Esc/Ctrl+C 急停并非零退出，收尾期间的急停同样生效；正常播完自动关闭。需要 HOME 时使用已有遥操作或键盘点动入口。

键盘点动使用 WASD、方向键、IJKL，R 回零，Q 退出；S/D 在该入口仍是方向键。启动及 HOME 后先完成伺服准备，再进入点动/hold。松键或 IK/约束拒绝时制动并 hold；有效候选不会只因名义 tick 已过而丢弃。IK 前后只比较方向键及 Q/R/Esc 的输入快照，无关键按下或释放不丢弃有效目标；Q/Esc 的取消仍由回调和实际发布边界执行。busy 期间仍维护按键释放，长按启动键不会穿过收尾自动启动。

Teleop 手部实测起点必须位于配置操作范围，范围外会列出关节及就位要求，只拒绝本段，不发送跳变目标。默认 distal 下限为拇指 10°、其余手指 5°；仅在 1e-6 rad 数值舍入范围内归一到边界后建立首帧参考，10 ms、4 rad/s 时经过 mapper 的最终目标每轴变化不超过 0.04 rad。默认显式 HOME 目标在范围内部留出 5° 到位容差；自定义范围或 HOME 时也需留出余量，HOME 成功不豁免实测范围检查。

## 策略部署

在上述 Real 环境中安装配套 `dexmani_policy` 及 checkpoint 所需模型依赖，确认设备与标定配置适合当前实验，保持 `hand.enabled=True`，从仓库根目录启动：

```bash
python scripts/run_policy.py /path/to/checkpoint.pt
```

路径接受 checkpoint 文件，或含 `checkpoints/latest.pt` 的目录；支持 `~`，相对路径基于调用工作目录。默认设备为 `cuda:0`，默认使用 sync；需要 async/RTC 时见[高级部署说明](docs/policy_deployment.md)。输入模态与动作表示从 checkpoint 恢复。

等待终端提示就绪后，B 开始一段；S/C 停止并保存已启用的录制；D 停止并丢弃本段；Q 保存当前段并退出，保留已接纳的 D 意图；Esc 急停并退出。连接后不自动回初始位；需要固定实验初姿时，在空闲且上一段收尾完成后按 H，活动段请先 S。T 仅在空闲且手部无接触时复位触觉。

常用覆盖项：

| 选择 | 默认来源与覆盖入口 |
| --- | --- |
| 控制周期、每次执行动作数 | 沿用 checkpoint；在 `--config` 指定的 Python 文件中设置 `CONFIG.execution.control_dt_s`、`CONFIG.execution.action_steps`。checkpoint 缺少周期时须明确填写 |
| 每段时长、段数 | CLI `--max-duration 60`、`--num-episodes 20`；每段仍须按 B 开始 |
| 录制与输出 | 默认仅终端输出；CLI `--record-raw` 开启完整 Raw，`--output DIR` 指定输出根目录；录制且未指定目录时使用 `rollouts/` |
| 权重、推理步数 | 沿用 checkpoint；CLI `--weights ema` 或 `--weights raw`、`--inference-steps N` 覆盖 |
| 推理设备、随机种子 | CLI `--device cuda:0`、`--seed 0`；种子每段重置 |

`--output` 或 `--record-raw` 会在输出根目录下建立会话目录，保存 `run_config.yaml` 与 `session_result.json`。配置文件写法见[安装与实验配置](#安装与实验配置)；调度约束、观测历史、IK、取消与资源收尾见[高级部署说明](docs/policy_deployment.md)。

## 数据、来源和执行语义

Raw 任务标签须为非空、非全空白字符串，不含 NUL；允许换行与制表符，录制边界不改写输入。策略 task 只作为元信息，开启 Raw 时检查，不参与会话目录拼装。遥操作 TASK 同时用于目录，仅额外禁止 `.`、`..` 和路径分隔符；允许 `.research` 和首尾空格。Canonical 可合并不同任务标签，不另存标签或来源列表。

Raw 使用 `data.h5` + `rgb.mp4`，遥操作默认保存在 `episodes/TASK/`；训练导出到 `datasets/TASK.zarr`。回放只打印提交数和耗时，不生成审计文件或输出目录，可重复执行同一命令。RawReader 和 `visualize_episode.py --info` 允许诊断显式 `.tmp_*` 路径；导出只检查请求 episode 和解析后 episode 的 basename，拒绝实际 `.tmp_episode` 及其别名，允许普通 `.tmp_workspace` 祖先及用作批量输入的临时工作目录。回放接受显式指定的临时目录，文件锁冲突、损坏或目标无效仍报错。诊断不会发布或改名数据。

Raw 将实测 state 与提交尝试 action 分开保存。组合命令按设备顺序提交，真实故障保持原错并停止；单次手部写 CRC 仍是一次提交尝试，下一周期使用最新目标。送入 arm mailbox 只代表目标发布，不保证每个目标进入 SDK 或物理到位。手部 warm start、阻尼和滤波参考由控制线程在提交尝试正常返回后更新；同一 VR 帧每个控制 tick 仍运行手映射和速度限制。dt 使用本次计算时刻减上次提交尝试参考时刻，未提交的候选不推进参考或时间。普通映射失败在原 `target_max_age_s`（默认 0.5 s）目标期限内重发完整旧 arm+hand，不续期；实际重发正常返回后推进手部参考时间。首目标等待不写动作行，等待期间重建尚未生效的启动参考，到期仍停止。SDK 调用正常返回也不证明手部物理执行。

Canonical 默认导出 [13 个字段](dexmani_real/data/export.py)，保留既有单位、输出 dtype、EEF 9D 与 EEF action 21D；用 `--fields` 选择实际所需字段，例如 `python scripts/export_zarr.py episodes/TASK --output datasets/rgb_state.zarr --fields rgb joint_state action`。只读取、检查和计算所选依赖：RGB/状态/action 无需外参或 FK，状态/action 不解码视频。Raw 浮点字段可使用不同浮点精度，转换后的训练值仍须有限；单位和关节顺序在读取相应字段时校验，单独读取图像不依赖动作元数据。导出动作时 policy 等待行的 NaN 仍会被拒绝。sync Policy Raw 的同行 action 是本次提交尝试，不代表由同行观测推理得到。正常保存、可导出和可物理回放是不同条件。相机外参 `camera_T_xarm_base_from_color` 从 color optical 坐标变换到 xArm base，平移单位为米。

采集端将 `timing/rows` 的时间字段写为主机单调时钟，单位 ns，缺失值为 -1；control_ns 是读取开始，observation_ns 是读取完成。原始时间数值保持不变。臂、手、触觉、相机保留来源时间，缓存不续期；同行不表示硬件同步。启用录制始终采集完整 Raw：实测关节、动作提交尝试、RGB、depth、aggregate、dense、来源时间及元数据，不随策略输入变化。Raw 动作表示提交尝试，不保证前缀包含最后一次实际提交。writer 收尾与保存决定见[高级部署说明](docs/policy_deployment.md#录制收尾与退出结果)。

teleop/jog/replay 从每轮开始计时，只等待剩余周期；慢计算降低实际频率，不补发积压 tick。Canonical 对采样间隔超过名义周期 ±25%、读取跨下一周期给出警告。control/observation 允许从 0 开始，须各自严格递增，observation 不早于本次 control；所选字段的动态来源时间为负、非因果、倒退或 camera_sequence 异常只汇总一次带定位的 warning，允许重复使用同一来源帧。静态相机参数不要求逐帧相机时间有效。Canonical 保留首段名义 dt 和 episode_ends，跨段周期允许 `rtol=1e-6, atol=0` 的表示误差，不另存质量统计，也不重采样。实际时序可查 Raw 的 `timing/rows` 和 `RawReader.timestamps_s`。可导出不代表数据质量适合当前实验。

导出先验证所选字段的元数据、结构、dtype、计数和时间规则，再逐块读取相应原始数组并检查转换后的有限值。未选字段的坏值不影响导出；仅导出点云时，也按每段实际深度尺寸分块读取。前置验证成功不代表全部数值已通过；坏数值可能在转换开始后甚至末块才发现，整段及整批拒绝语义不变。需要 RGB 时逐帧解码并核对帧数；成功时导出全部行，不截断、切分、插值、补帧或重采样。批量遇到不合格 episode 明确失败；可通过 `--exclude EPISODE` 排除整段后重试，不自动跳过或改写 Raw。排除名单可跨批次复用，不存在于当前输入的名称忽略，统计只计实际排除的唯一 episode。全部完成后才原子发布 Canonical；转换失败关闭资源并删除本次新建的 Canonical staging，清理失败另行记录且不覆盖原错；不删除源 Raw 或既有目标。

物理回放只要求 `data.h5` 中非空有限的 `action_arm_joint_target (N,7)`、`action_hand_joint_target (N,12)`，以及 `meta` 的 `control_hz`、`joint_units=rad` 和当前设备顺序的 JSON joint names。接受实数数值 dtype；不要求 timing、历史实测起点、RGB-D、触觉或完成标签。连接前一次校验整段当前 arm/hand 配置范围；当前手部配置可窄于公共额定硬件范围。超过边界 1e-6 rad 的目标拒绝，容差内微小越界归一到配置边界，归一后的目标仍须通过公共硬限位。坏行不筛除或补值。

replay 在 prepare 后读取一次当前 arm/hand 姿态并检查新鲜度，只与第一行目标比较，任一关节差大于 10° 即拒绝并关闭，实验者须先就位。随后顺序提交绝对目标，不执行 mapper、IK 或在线几何/碰撞检查，不加载 mesh 或桌面标定。实验者须确认同一机器人的布置适合该 demo。每行从实际开始计时，只等剩余周期；300 行每次耗时 63 ms 可全部顺序提交，慢了就变慢。每次发送附现有 2 s 目标租期，末行成功后固定可中断留驻 0.5 s，替代普通周期等待，不续租或证明到位。

回放等待和留驻期间继续服务 hand I/O、键盘与公共 fault/stop；replay 不重复读取整份状态或裁决旧目标期限。设备协调层在活动期间统一检查 hand 关节反馈，0.5 s 内短缺帧可继续，持续失联报错并停止；arm owner 的反馈与租期检查、真实 I/O 和关闭错误继续生效。无操作者输入的执行器停止非零退出。arm 邮箱仍是 latest-value，有序提交不代表每行都得到 SDK 回执或物理执行。

Canonical 可导出到 Raw 输入目录内的新目标（例如 `raw_runs/canonical.zarr`），已有目标及符号链接仍拒绝覆盖；导出前固定输入 episode 集合，staging 不进入本次输入集合。

Canonical 仅保存所选字段需要的数值配方。保存原始 depth 时各段尺度必须相同；仅保存 point_cloud 时逐段转为米，允许原始深度尺度和图像尺寸不同。实际保存的图像数组仍要求形状一致，所有 episode 保持统一 dt。

遥操作额外逐行保存 `vr_wrist_pos`、`vr_wrist_quat_wxyz`、`vr_landmarks`、`vr_recv_ts_ns`、`vr_sequence_id`，用于调试，不进入训练导出。VR 使用右手数据，21 个 landmark 保持 SDK 原顺序；来源时间取所需通道首次主机接收时间的较早者。接收端先校验并归一化原始姿态四元数，再转换为 FLU；零范数或非有限姿态被拒绝。位置用米、四元数 WXYZ；来源帧重复时 ID 和时间保持原值。每段 Raw 保存实际使用的 `vr_to_robot_rotation` 数值矩阵；手部形态标定和启动锚点只服务在线计算，不另行归档。逐次操作与失败控制时点用日志诊断。

触觉 aggregate `(5,3)` 与 dense `(5,120,3)` 按通道保持最后完整有效值和原来源时间，直到真实新样本恢复，不补零或刷新旧时间。实际依赖触觉的策略仍按 `hand.feedback_max_age_s` 检查所用输入年龄。导出所选触觉时，最大来源年龄超过 0.5 s 给出 warning，不保存统计或筛行。Real 数据使用 SDK sensor axes 和原生尺度，指序为 thumb/index/middle/ring/pinky；与仿真接触力 N 的单位和坐标不同，同名同形状不代表等价，也不假定 dense 求和等于 aggregate。RawReader 校验单位、指序和形状；Canonical 继续拒绝所选字段的缺失、非有限值及 RGB 帧数不匹配。

T 仅在空闲、无录制且手部无接触时显式执行。Robot 确认停止并检查取消后清空旧触觉缓存，保留关节，再向 `0x11`–`0x15` 各请求一次；记录操作时间和已执行每指返回码，CRC/已知触觉不可用不阻止其余指请求，操作者可再次按 T 重试。清缓存后取消无需恢复旧值，后续反馈自然补齐，下一次 B 再检查真实初值。返回码不证明物理零点校准效果，真实设备故障仍报错。

## 控制和几何边界

机械臂 SDK 由独立进程持有，XHand SDK 由父线程调用。命令期限为 teleop/jog 0.5 s、policy/replay 2 s，arm 异步邮箱发布（含 hold）和 hand 同步 SDK 发送前均检查同一期限；hand 反馈复核后若已到期，即使 arm 已发布也会取消并停止，不推进控制器参考。SDK 发送开始后跨期限不追溯否定此前发送。arm 反馈年龄预算为 4/30 s，hand 默认为 0.5 s，相机为显式 camera.max_frame_age_s（默认 4/30 s，与 fps 无关），VR 为 0.5 s；缓存读取不延长年龄。arm 默认反馈 25 Hz、伺服 100 Hz，SDK 阻塞时不保证达到调度频率。Ruckig 保留关节位置、速度、加速度、jerk 和制动极值约束，默认上限为 150 °/s、900 °/s²、12000 °/s³，实际参数以配置为准。arm 目标过期、停止请求与关节反馈失效会停止；真实 SDK 错误及停止确认失败仍是故障。来源时间检查和 SDK 返回成功均不构成端到端实时或物理到位保证，实际跟随及停止延迟需真机测量。

arm 外部目标仍严格检查格式、有限性和配置限位。完整轨迹（含制动段及段内速度为零的位置）与输出采样的限位比较仅允许 `1e-12 rad` 计算舍入误差；该数值与 HOME 到位、回放文件精度容差独立。通过检查后，采样点的微小越界归一到限位；step 同时推进参考并返回同一归一位置；发送失败或发送前过期后丢弃轨迹，不重试或复用未发送候选。

XHand 连接要求设备 ID 正确且取得有效关节反馈，不等待触觉。周期绝对目标的写 CRC 返回未确认，不阻塞重试旧目标；连续 5 次写 CRC 才进入现有停止路径，允许的非 CRC 命令返回会清零计数，读取成功不会清零。读 CRC 丢弃整帧且不更新来源时间。停止保留 0.25 s 有限 CRC 重试：实测关节在原有 `4/30 s` 内才 hold，否则 passive；退出仍请求 passive。软件等待不能强制中断阻塞的手部 SDK，后续关闭成功也不能抹去停止失败。

XHand 过流告警（SDK `1501035`、关节板 `110`）限频 warning 并继续运行；缺失触觉按通道保持，有效关节和其他通道继续更新。未知故障与非法关节/目标数据仍报错；关闭时汇总读写 CRC、触觉缺帧和过流计数。本机安装 SDK `1.5.2` 仅有原生实现，未确认缓存更新及帧来源时间，继续 `read_state(True)`，暂不启用发送后缓存复用。

xArm 连接只建立 SDK、校验设备并读取有效反馈，不代表已经准备运动；相机标定的直接连接保持只读。同一 SDK 会话仅在首次完整运动准备中配置碰撞灵敏度和 TCP 负载，失败或取消不记为完成，新连接重新配置；每次准备仍保留使能、显式发送模式命令并检查应答、对应加速度及最终 State 0，随后用新鲜反馈确认停稳。`api.mode` 是异步报告缓存，不用于跳过模式命令或拦截 servo；命令成功应答不代表实测到位。首个目标到达时再次检查速度并使用当前反馈建立 Ruckig 参考，后续周期延续规划状态。Mode 0 仅用于需要移动的显式 HOME，已到位时不为初始化参数进入运动模式。保留一个本地 cancel_epoch，恢复前确认停止、清旧邮箱、短锁内复核取消后清 stop。stopped 只代表停止处理完成，不能掩盖 SDK 失败。

实测起点通过共享函数容纳 `1e-6 rad` 的 float32 反馈舍入误差，仅将微小越界归一到配置边界后用于规划；Policy 初始姿态采用相同准入判断。原始反馈、observation、Raw 和来源时间戳保留不变，外部命令仍严格检查原限位；此容差不用于轨迹计算或传感器噪声。

xArm SDK 单次回复超时按用途区分：初始化配置、模式切换、HOME 命令和停止使用 1 s；反馈查询与伺服发送使用 20 ms，调用结束或抛错后恢复 1 s。超时是等待上限，正常回复立即返回；SDK `code=3` 表示回复超时，仍按设备故障处理。停止接口内部还会查询状态，回复预算不能视为物理停稳时间。

arm HOME 使用 `arm.homing.timeout_s`（默认 60 秒）的一次绝对期限，覆盖停止、实测停稳、单次 Mode 0 移动和到位确认。已到位时省略移动，两条路径均要求新鲜反馈、位置/速度收敛及连续 0.3 秒停稳；短暂偏离重新累计。独立 `home_arm()` 等待期间继续读取已连接手部的真实反馈并传播故障，不写手部目标、不等待触觉初值，也不清除此前手部命令的停止责任；后续 idle 停止按真实反馈年龄决定 hold/passive。H/R 在运行入口明确先 hand HOME，再 arm HOME。手部 HOME 的有限 CRC 补发和连续新反馈到位语义不变。HOME 结束保持停止；软件期限不能中断挂起的 native 调用。HOME 加速度独立配置为 `homing.max_acceleration_deg_per_s2`，默认 900°/s²，与改造前实际值一致。

`StaticCollisionBox` 由基座坐标系中心 `center_xyz_m` 和沿基座 X/Y/Z 轴的完整边长 `size_xyz_m` 定义，单位均为米，范围为 `center_xyz_m ± size_xyz_m / 2`。盒体固定轴对齐，按配置顺序命名为 `obstacle_0`、`obstacle_1` 等。桌面标定仅供点云处理，不再构造 HOME 碰撞盒。

基础 joint 映射和 arm HOME 不加载桌面标定、URDF 或碰撞网格；hand 禁用时也不依赖手部范围或 HOME 姿态。EEF/VR/jog 仍使用 IK、FK、workspace 和组合几何检查；禁用手部 I/O 时在这些能力入口验证固定手姿态及其范围。启用手部时保留设备配置检查。点云去桌面实际启用时继续加载 `environment.table_plane_path`，Policy 继续归档实际使用的平面系数和路径。

HOME 配置错误只结束本次操作；整机回位时手部可能已回位。操作者须事先确认直接关节回位空间，基础 HOME 不执行路径碰撞、桌面间隙或任务 workspace 检查，不提供绕障或碰撞保证，不增加软件确认弹窗。

joint Mapper 保留等价角选择和意图投影，移除 EEF workspace 与端点碰撞筛选；设备 API 仍严格拒绝非法确定目标。目标发布频率可以超过 arm.servo_hz，邮箱覆盖未消费的中间目标，提交次数不等于 SDK 执行次数；采样频率须能换算为有限且至少 1 ns 的整数周期，XHand 直接发送仍限制在 83 Hz 以内。键盘 jog 周期仅发送 arm 目标或 hold，频率按有效采样周期校验，启用手部时也允许配置 100 Hz；手部反馈仍独立以 30 Hz 服务，实际循环频率需测量。

目标端点与 Ruckig 关节极值检查不证明整条中间运动轨迹连续无碰撞；当前跳过手内碰撞的模型设定也不构成指间碰撞保护。FK 使用 URDF EEF，不能与固件 EEF 定义混用。手部重定向数学和默认参数见 [hand_retargeting.md](docs/hand_retargeting.md)。

手模板在 `assets/pointcloud_template/hand/hand_<点数>.npz`，默认 1024 点；更改模板点数、visual mesh 或安装模型后显式重建：

```bash
python scripts/generate_hand_template.py --num-points 1024 --overwrite
```

## 标定和查看

默认相机外参、桌面平面与 VR 朝向 JSON 位于 [config/calibration_data/](dexmani_real/config/calibration_data/)。纯 RGB-D 遥操作和不需要外参的 Policy Raw 录制允许默认外参文件或当前 serial 对应条目缺失，此时 Raw 不写入外参；点云、`camera_extrinsic` 等消费者仍要求有效外参。显式标定路径错误、文件损坏或当前条目无效均报错。相机标记须相对 EEF 刚性固定，摆位提供充分旋转激励；先退出厂商示教/独立点动控制连接，再只读采样。

```bash
python scripts/calibrate_camera.py collect --output calibration_samples/run_001
python scripts/calibrate_camera.py solve calibration_samples/run_001 --output camera_calibration.json
python scripts/calibrate_camera.py solve calibration_samples/run_001 --exclude 3 --output camera_calibration_clean.json
python scripts/calibrate_vr_heading.py --ref head
python scripts/tools/view_realsense.py
python scripts/tools/view_realsense.py --cloud processed
python scripts/tools/view_pointcloud.py
```

collect 的相机持续连接，Enter 为一组只读机械臂采样，完成立即断开臂；Q 结束。五帧原 RGB、检测、来源时间、实测位置/速度和样本 ID 保存为 NPZ，检查速度、帧新鲜度以及每次反馈相对采样起点的漂移。solve 不连接设备，按原 ID 排除样本，本批次不存在的 ID 忽略；质量合格后按真实 serial 仅写入 `T_base_from_camera`，保留其他相机条目。终端打印求解方法、样本数、位置/旋转 RMS、最差样本 ID 和输出位置；Python 调用返回含逐样本残差的 `HandEyeResult`，残差按使用的 sample ID 升序排列，`worst_sample` 为数组下标。VR 头朝向直接消费头帧，不要求右手就绪。

VR 运行读取 JSON 中任意有限的 `theta_deg`，例如 `{"theta_deg": 270.0}`，按度构造 `R_z(-theta)`；不限制角度表示区间，也不改写输入。NaN 和正负无穷被拒绝。ref、质量和时间报告不参与运行准入。手工角度的正确性由研究者确认；自动标定打印标准差、最大偏差和有效样本数，不按质量分级禁止保存；新文件只保存 theta_deg。Raw 归档实际使用的旋转矩阵。

solve 的外参输出不得覆盖 `capture.json` 或原始采样 NPZ。

采样使用逐帧 IPPE_SQUARE：重投影 RMS ≤ 1 px，至少四帧有效，帧间位置/旋转跨度 ≤ 5 mm / 3°；不一致则重新采样。collect 仅接受 none 或 Brown-Conrady 相机畸变模型。手眼求解固定 HORAUD，闭环 RMS ≤ 5 mm / 3° 才更新外参；这些内部残差不代表真实空间定位精度。solve 只需 `capture.json` 中的真实 `camera.serial` 和 NPZ 中的 EEF 与 Marker 位姿，不读取内参、不重算 RGB 或逐帧 PnP。

实时相机查看使用 view_realsense.py：默认 RGB-D，在 RGB-D 或点云窗口内按 P 开关点云、S 切换 raw/processed（默认 processed）；`--cloud raw/processed` 可在启动时打开点云并指定初始类型。Q/Esc 或关闭 RGB-D 窗口退出；关闭点云窗口仅关闭点云，按 P 可重新打开。预览不修改曝光优先级，不提供逐帧诊断面板。RGB-D 仅预览时不加载外参、桌面或点云处理器；启用对应点云类型时才加载所需资源。窗口需要桌面/OpenGL。

view_pointcloud.py 提供独立诊断流程：先在终端选择是否拟合桌面，RGB-D 预览按 Enter 继续，随后处理一帧点云、打印点数和本帧处理耗时，关闭相机后静态查看结果；Q/Esc 或关窗退出。桌面拟合固定采集五份不同深度帧，使用当前相机内参；仅按工作空间 XY 范围筛选，保留完整 XYZ，避免 Z 边界截断噪声造成高度偏移。每帧超过 30,000 点时沿完整索引范围确定性限点，在线点云仍按完整 XYZ 工作空间裁剪。拟合结果用于点云去桌面，可供本次处理使用，仅确认发布才改写共享桌面标定。`--config`、`--camera-calibration` 指定当前实验配置和外参。该流程复用当前点云算法和采集实现，预览不额外修改曝光设置。RGB-D 类型、网格和缓冲所有权由 RealSenseCamera.read 统一保证，RGB-D 工具、仅 RGB 的 ArUco 标定和按需部署使用同一读取边界。传感进程正常关闭预算为 5 s，包含 L515 在首帧后立即关闭时约 3.2 s 的 SDK stop；超时仍按既有流程终止进程并报告。

RGB-D 的非零 Brown 畸变需结合实际模型消费者判断：Real 反投影会反解 Brown，配套 Policy 的空间投影仅使用 K 和外参。使用空间投影融合前须核对 checkpoint 与相机参数；不支持的组合须明确拒绝，或另行使在线/离线几何共同支持畸变。不能仅凭同时存在 RGB 与 point_cloud 推断模型使用了空间投影。

点云默认使用基座系 XYZRGB，工作空间 `x=[0.15,0.85]、y=[-0.50,0.50]、z=[0,0.80] m`，去桌面间隙 10 mm。深度邻域支持、体素、半径密度和连通分量过滤后 FPS；空点云在 viewer 清除、导出拒绝。几何规则可能丢失细结构，参数和配方必须随实验保持一致。在线场景点云仍由独立进程构建，通过共享环携带同源图像、来源时间和相机序列；复制竞争退回旧缓存时不重复处理同一相机序列；空云不发布，尚未过期的上一份结果仍可能被使用。运行时 IPC 不再搬运硬件 color_frame_number，接收端去重和标定仍保留硬件帧号。

## 离线物体几何和分析

CordVIP 提供离线物体几何、实测点云和位姿分析，尚未接入主包 Policy 观测。使用流程、模型安装及输出字段见 [CordVIP 文档](scripts/cordvip/README.md)。

`--initial-selection` 可直接提供当前首帧原始 RGB 网格的 PNG/NPY mask，或引用 mask 的选择目录；两者均应用于当前指定首帧，目录附带的来源文件不参与校验或复制。`object generate --image PATH` 直接使用 RGBA。派生结果可写入 `rollouts/derived/objects` 等新目录，实际输入、Raw episode 内部及已有目标仍受保护。

Real 目前没有 `object_pointcloud` producer；请求该字段仍会明确报错，不能据此宣称 CordViP 已可直接真机部署。后续接入需要明确完整物体模板、坐标变换、点数/颜色、多目标选择与跟踪有效性；可见 scene cloud 不能替代完整模板。

```bash
python scripts/cordvip/track_object.py --episode episodes/TASK/EPISODE --scene scene.json --with-hand --output /tmp/cordvip/scene_001
python -m cordvip_geometry view /tmp/cordvip/scene_001
```

物体模型表面与语义点不代表实测接触；几何处理参数与配方需随实验归档。

物体资产按 `assets/pointcloud_template/object/<任务>/<物体>/` 组织，`reference.jpg`、`asset.json`、NPY 和 PLY 点云纳入 Git；`mesh/mesh_m.obj` 与 `third_party/weights/` 仅保留本地。克隆后可直接读取点云；FoundationPose 位姿估计需要先本地重建网格，具体步骤见[物体重建流程](docs/object_reconstruction.md)。

默认日志为终端 INFO、拒绝原因、完整异常 traceback 和简短段结果，需要留档时使用 shell 重定向。几何环境检查使用独立命令 `python -m cordvip_geometry check`。

开发约定见 [AGENTS.md](AGENTS.md)。

静态检查使用 `python -B -m ruff check --no-cache --select F dexmani_real`。SharedRing 同一 sequence、同一字段集合直接复用独立快照，调用方按只读使用；新序列复制竞争最多尝试两次，随后可复用相同字段集合的完整快照，保留原时间和 sequence；没有快照时不发送依赖该来源的动作，准备等待仍有期限。

遥操作运动学修复和三组离线对照见[手部重定向](docs/hand_retargeting.md#2026-11-05-离线比较与取舍)。实测手部仅允许 1e-6 rad 舍入归一；VR 起点不兼容名义工作空间时拒绝。jog 起点按端点验收范围接纳，内缩 margin 限制新目标；内缩范围外可向内退回，向外不扩大越界，纯旋转不夹带平移，超出端点范围仍拒绝。机械臂 IK 使用原生 Pinocchio，每个初值最多更新 50 次，再按实际候选误差与约束验收。本轮保留完整手部目标与骨长标定，删除了重复损失回算和逐段标定/成功锚点归档。
