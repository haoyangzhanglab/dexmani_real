# DexMani Real

xArm7 + XHand + RealSense RGB-D + VR/HTS 的个人研究代码，支持遥操作、Raw 采集、Canonical 导出、sync / async / RTC 策略部署及物理回放。训练与推理见配套 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)。

## 安装与实验配置

使用 Python 3.10+，在仓库根目录安装和运行：

```bash
python -m pip install -e ".[interactive,control,geometry,visualization]"
```

设备 SDK、HTS、Pinocchio、MPlib、NLopt 和配套 Policy 按实际实验环境安装。机器人模型和手模板从源码目录 `assets/` 加载，运行时保留 checkout。几何模型的独立环境见 [CordVIP 使用说明](third_party/cordvip_geometry/docs/usage.md)。

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

任务、路径、段数、时长、录制选择及 `--weights/--inference-steps/--seed/--device` 通过 CLI 指定。遥操作和键盘点动可在 Python 配置中设 `hand.enabled=False`；VR 遥操作还须使用 `--no-record` 关闭录制，实物手须移除或固定在配置的 HOME 姿态，供碰撞规划使用。Policy 要求启用 hand。Policy 配方从 checkpoint 恢复，其训练配置和运行快照可使用 YAML。

Policy 默认只保存 `run_config.yaml` 和 `session_result.json`；需要完整 Raw 时使用 `python scripts/run_policy.py EXPERIMENT --record-raw`。未录制时只启动模型实际输入和静态参数所需的传感器与标定。

遥操作录制选项由 CLI 提供：`--record/--no-record` 默认开启，`--episodes-dir` 默认 `episodes`，`--max-duration` 默认每段 60 秒。相对录制目录相对仓库根目录，绝对路径直接使用，再拼接 TASK。例如：

```bash
python scripts/collect_teleop.py TASK --record --episodes-dir captures --max-duration 90
```

查看请求配置：

```bash
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
```

这些命令只打印请求配置的 JSON，不连接设备、不读取标定或加载 checkpoint。遥操作的打印配置和 Raw 快照中的 `collection` 包含最终 CLI 录制选项，仅作为归档表示。启用分支负责业务校验，完整实验快照仍须可严格序列化；实测相机事实在连接后单独记录。

## 常用实验路径

将 TASK、EPISODE、EXPERIMENT 替换为任务、episode 目录和保存的 Policy 实验路径。训练导出要求已发布 episode。

```bash
python scripts/collect_teleop.py TASK
python scripts/export_zarr.py episodes/TASK --output datasets/TASK.zarr
python scripts/run_policy.py EXPERIMENT --checkpoint latest --num-episodes 5
python scripts/replay_episode.py episodes/TASK/EPISODE
python scripts/visualize_episode.py episodes/TASK/EPISODE
python scripts/keyboard_teleop.py
```

键盘点动仅连接机械臂时，使用 `python scripts/keyboard_teleop.py --no-hand`；可与 `--config` 一起使用，覆盖配置中的 `hand.enabled`。禁用手部后，碰撞检查仍使用 `hand.home_qpos_deg` 指定的固定手姿态。

Episode 可视化默认按实测 `arm_qpos`、`hand_qpos` 通过 FK 重建手部表面点云；在 3D 区域切换到 `Hand Point Cloud` 可查看手部点云、指尖和 EEF 锚点，`xArm-base` 显示相机点云及指尖、EEF 锚点。两者共用基座坐标系和时间轴。手部点数由 `CONFIG.hand_pointcloud.num_points` 指定；`--no-point-cloud` 只关闭相机点云，手部重建不依赖相机外参。

VR 遥操作与策略会话：B 空闲时开始新段；S/C 停止并保存；D 停止并丢弃；Q 保存当前段并退出，但保留已有 D 意图；Esc 急停并结束会话。保存决定确定后，迟到 D 不再逆转结果。H 仅在空闲且收尾完成时回初始位，活动段请先 S；T 仅在空闲时复位触觉，操作者须确认无接触。准备或收尾期间的 B/H/T 不会排队启动下一段。

音效使用系统 `aplay` 或 `paplay` 播放提示，播放器不可用、后台播放失败和尾音未及时退出只记录 warning；关闭函数实际抛出异常时仍报告错误并非零退出。一个后台线程播放，待播槽只保留最新提示；允许合并、漏播和延迟，不保证 FIFO 或急停音效抢占。退出取消待播及尾音，不等待排空。提示音不能证明机器人已停止、物理到位或任务成功。

物理回放使用 S/Q 停止并退出、Esc/Ctrl+C 急停并非零退出，收尾期间的急停同样生效；正常播完自动关闭。需要 HOME 时使用已有遥操作或键盘点动入口。

键盘点动使用 WASD、方向键、IJKL，R 回零，Q 退出；S/D 在该入口仍是方向键。启动及 HOME 后先完成伺服准备，再进入点动/hold。松键或 IK/约束拒绝时制动并 hold；有效候选不会只因名义 tick 已过而丢弃。busy 期间仍维护按键释放，长按启动键不会穿过收尾自动启动。

传感进程在尝试 start 前登记归属，启动后中断仍由统一收尾回收；明确无 PID 的对象不 join，无法确认退出的句柄继续保留并报告。Policy 按实际受管进程检查健康，静态参数相机关闭后不再参与检查。

设备和模型支持正常多轮复用。启用手部时，Teleop/Policy 启动先完成手部 HOME；仅模态或录制需要时复位触觉。每段 B 后准备设备并确认停稳，重新获取有效观测；Policy 可从当前有效姿态开始，不要求精确 HOME。策略训练分布可能要求特定初始姿态，由操作者按实验布置并按 H 回位。正常 S/C/D 结束后完成记录和模型任务回收，再接纳下一段；启动、prepare/HOME 取消或超时后退出会话，故障资源不重试复用。

策略周期、数值 recipe、输入模态和动作窗口从 checkpoint 恢复，不改变录制的完整多模态 schema。sync 等待预测后顺序执行动作；async 按原观测时间锚点裁掉过期动作，可继续旧计划的完整有效尾部；映射跨过名义 tick 后，只要来源、完整预测窗口和实验时长仍有效便可提交，每次控制机会至多提交一个候选；RTC 固定策略动作前缀并严格交接，错过承诺槽或承诺槽缺少必需观测即结束。

记预测长度为 P、`action_steps` 为 A、`prefetch_steps` 为 d：sync 要求 `1 <= A <= P`；async 要求 `1 <= d <= A <= P` 且 `d < P`；RTC 另要求 `A+d <= P`，guidance cap 须显式为有限非负值。EEF 前缀相同不保证未来 IK 分支相同。

history 保留最近 `n_obs_steps` 次已接纳的实际观测，窗口完整前不提交首次预测；sync 等待预测期间不重复入窗。预测等待与调度抖动可以使历史间隔不均，不因间隔清空重建；这放宽了模型输入的等间隔假设，与严格等间隔输入并不等价。来源与 Raw 采样时间保持原样。

首预测 reset RNG，预热后恢复 RNG；预热重复引用同一观测行时，手部几何只计算一次。

策略首次有效动作没有额外两秒预测等待期限。每轮正式开始建立一次实验时长（默认 60 秒）；API 显式传入 max_running_s=None 时，可在首个动作前持续等待，S/Q/Esc 仍有效。发送后的目标由 arm owner 的租期约束，owner 停止即结束本段，迟到预测不能恢复。启动取消计数在键盘启用前捕获，贯穿长初始化；内部停止不算用户取消。

每段结束先停止设备，再同步完成录制和模型回收；已知退出、急停或致命失败时，先 close 机器人再等待慢资源。writer 有自己的有界收尾预算，同一未完成模型任务跨 episode/session 收尾共用一次绝对期限；同一 arm owner 的重复 close 也只使用首次关闭预算的剩余时间。writer 正常关闭后按已确定的保存意图发布，失败 Raw staging 保留，零行不发布。运行失败与容器发布状态独立；后续模型或 close 失败不会反向否定已发布数据。线程、进程和 Future 的超时不代表已退出，仍保留句柄并报告未完成。

Policy Raw 直接使用结果的 episode 编号（如 `episode_002`），准备取消后允许编号空洞；既有数据不重命名。

`session_result.json` 保存会话起止时间、结束原因、episode 实际开始与持续时间、错误阶段和消息，以及录制路径、行数和截断原因。结果文件不参与设备复用判断，不再包含 outcome/shutdown 分类矩阵。错误、急停或资源关闭未完成均非零退出。真实 Future、线程和进程句柄在超时后仍被保留；模型 load/warmup/reset/predict/close 在同一个工作线程串行，未完成任务不重新获得完整清理预算。正式开始的段才计入 --num-episodes。

## 数据、来源和执行语义

任务标签须为非空、非全空白字符串，不含 ASCII 控制字符 0–31 或 DEL 127，录制与导出共用此规则且不改写输入；策略任务名在创建会话目录前也按此规则接纳。遥操作 TASK 同时用于目录，另外禁止路径分隔符、以点开头的目录名及首尾空白；这些目录限制不适用于一般自然语言标签。

Raw 使用 `data.h5` + `rgb.mp4`，默认在 `episodes/TASK/`；训练导出到 `datasets/TASK.zarr`，策略产物位于 `rollouts/`；回放只打印提交数和耗时，不生成审计文件或输出目录，可重复执行同一命令。RawReader 和 `visualize_episode.py --info` 允许诊断显式 `.tmp_*` 路径；导出只检查请求 episode 和解析后 episode 的 basename，拒绝实际 `.tmp_episode` 及其别名，允许普通 `.tmp_workspace` 祖先及用作批量输入的临时工作目录。回放接受显式指定的临时目录，文件锁冲突、损坏或目标无效仍报错。诊断不会发布或改名数据。

Raw 将实测 state 与提交尝试 action 分开保存。组合命令按设备顺序提交，部分失败保持原故障并停止；送入 arm mailbox 只代表目标发布，不保证每个目标进入 SDK 或物理到位。手部 warm start、阻尼和滤波参考由控制线程在提交正常返回后更新；同一 VR 帧每个控制 tick 仍运行手映射和速度限制。dt 使用本次计算时刻减上次实际提交参考时刻，未实际提交的候选不推进参考或时间。SDK 调用正常返回也不证明手部物理执行。

Canonical 保留 [固定 13 模态](dexmani_real/data/export.py)、原单位和 dtype、EEF 9D 与 EEF action 21D。示教导出目前只支持 teleop Raw。sync Policy Raw 的同行 action 是本次提交尝试，不代表由同行观测推理得到。正常保存、可导出和可物理回放是不同条件。相机外参 `camera_T_xarm_base_from_color` 从 color optical 坐标变换到 xArm base，平移单位为米。

`timing/rows` 使用主机单调时钟，单位 ns，缺失值为 -1；control_ns 是读取开始，observation_ns 是读取完成。臂、手、触觉、相机保留来源时间，缓存不续期；同行不表示硬件同步。每段固定按控制与录制需求并集读取。录制质量失败或队列满立即结束本段；健康 writer 排空已接纳前缀，关闭成功后按主线程已确定的保存意图发布。收尾开始后到达的 D 不追溯丢弃旧段，已接纳 D 不被 Q 覆盖。实际 writer I/O 或传感进程故障退出会话。零行不发布；线程仍存活或关闭失败保留 staging 与真实句柄。Raw 动作表示提交尝试，不保证前缀包含最后一次实际提交。

teleop/jog/sync/replay 从每轮开始计时，只等待剩余周期；慢计算降低实际频率，不补发积压 tick，sync 只在提交后推进动作游标。async 保留时间锚点，RTC 保留严格前缀与交接承诺。Canonical 时间质量采用警告：包括超过名义周期 ±25%、125 ms 断档、读取跨下一周期。control/observation 各自必须严格递增，observation 不早于本次 control；来源时间仍须因果有效且不倒退。episode_quality 保存平均周期、最大间隔、最大 readout、网格漂移和截断信息；触觉实际来源年龄通过 max_tactile_aggregate_age_ns、max_tactile_dense_age_ns 查看（单位 ns）。dt 保留名义周期，不重采样。实际时序通过 Raw 的 `timing/rows` 和 `RawReader.timestamps_s` 追溯；Canonical 保持固定训练字段，不增加时间数组。可导出不代表数据质量适合当前实验。

导出先验证元数据、结构、dtype、计数和时间规则；六个原始数值 dataset 随转换逐块各读取一次，整块有限值检查通过后才进入几何计算，转换结果仍检查有限值。前置验证成功不代表全部数值已通过；坏数值可能在转换开始后甚至末块才发现，整段及整批拒绝语义不变。保留 RGB 逐帧解码和流式写入，成功时导出全部帧，不截断、切分、插值、补帧或重采样。批量遇到不合格 episode 明确失败，错误标明 episode、原因及适用的行索引；可通过 `--exclude EPISODE` 排除整段后重试，不自动跳过或改写 Raw。全部完成后才原子发布 Canonical；准入或转换失败关闭资源并删除本次新建的 Canonical staging，清理失败另行记录且不覆盖原错；不删除源 Raw 或既有目标。

物理回放只要求 `data.h5` 中非空有限的 `action_arm_joint_target (N,7)`、`action_hand_joint_target (N,12)`，以及 `meta` 的 `control_hz`、`joint_units=rad` 和当前设备顺序的 JSON joint names。接受实数数值 dtype；不要求 timing、历史实测起点、RGB-D、触觉或完成标签。连接前一次校验整段当前 arm/hand 配置范围；当前手部配置可窄于公共额定硬件范围。超过边界 1e-6 rad 的目标拒绝，容差内微小越界归一到配置边界，归一后的目标仍须通过公共硬限位。坏行不筛除或补值。

replay 在 prepare 后读取一次当前 arm/hand 姿态并检查新鲜度，只与第一行目标比较，任一关节差大于 10° 即拒绝并关闭，实验者须先就位。随后顺序提交绝对目标，不执行 mapper、IK 或在线几何/碰撞检查，不加载 mesh、桌面标定或 HOME planner。实验者须确认同一机器人的布置适合该 demo。每行从实际开始计时，只等剩余周期；300 行每次耗时 63 ms 可全部顺序提交，慢了就变慢。每次发送附现有 2 s 目标租期，末行成功后固定可中断留驻 0.5 s，替代普通周期等待，不续租或证明到位。

回放等待和留驻期间继续服务 hand I/O、键盘与公共 fault/stop；replay 不重复读取整份状态或裁决反馈 TTL、旧目标期限。末帧 hand 没有新样本且接口无错时允许完成留驻；实际发送前的 hand 反馈检查、arm owner 的反馈与租期检查、真实 I/O 和关闭错误继续生效。无操作者输入的执行器停止非零退出。arm 邮箱仍是 latest-value，有序提交不代表每行都得到 SDK 回执或物理执行。

每次新导出生成独立 `source_id`，原样搬迁保留 ID，供人工归档和实验说明。它不是内容 hash，不保证 split/resume 使用同一数据；配套 Policy 训练恢复允许显式更换数据路径，并按其当前实现检查实际模型/数据兼容项。旧数据和 checkpoint 不自动改写。Canonical 保存影响数值结果的处理配方与完整 URDF；实验应记录 source_id、导出 Real SHA、训练 Policy SHA 及手模板来源。修改处理算法或模板后重新导出，已有未知来源须如实保留。

遥操作额外逐行保存 `vr_wrist_pos`、`vr_wrist_quat_wxyz`、`vr_landmarks`、`vr_recv_ts_ns`、`vr_sequence_id`，用于调试，不进入训练导出。VR 使用右手数据，21 个 landmark 保持 SDK 原顺序；来源时间取所需通道首次主机接收时间的较早者。接收端先校验并归一化原始姿态四元数，再转换为 FLU；零范数或非有限姿态被拒绝。位置用米、四元数 WXYZ；来源帧重复时 ID 和时间保持原值。`runtime_config_json`、VR 变换、手标定和每段一份初始锚点随 Raw 保存。逐次操作与失败控制时点用日志诊断。

触觉 aggregate `(5,3)` 与 dense `(5,120,3)` 分别保存 SDK 最后一份完整有效通道帧及来源时间，按各自来源年龄及配置的 hold 上限限时保持。数据使用 SDK sensor axes，原生物理尺度未知，指序为 thumb/index/middle/ring/pinky；不解释为牛顿或布尔接触，也不假定 dense 求和等于 aggregate。原生单位、指序、hold age 和实际触觉复位结果继续归档。RawReader 校验单位、指序和数据形状；Canonical 导出还校验数值及来源时间，两者均不依赖采集 TTL 元信息。这些记录不代表校准或真实接触。

## 控制和几何边界

机械臂 SDK 由独立进程持有，XHand SDK 由父线程调用。默认目标期限为 teleop/jog 0.5 s、policy/replay 2 s，hold 同样受期限限制。arm/hand 反馈年龄预算为 4/30 s，相机为显式 camera.max_frame_age_s（默认 4/30 s，与 fps 无关），VR 为 0.5 s；缓存读取不延长年龄。arm 默认反馈 25 Hz、伺服 100 Hz，SDK 阻塞时不保证达到调度频率。Ruckig 保留关节位置、速度、加速度、jerk 和制动极值约束，默认上限为 150 °/s、900 °/s²、12000 °/s³，实际参数以配置为准。目标过期、停止请求与反馈失效会停止；SDK 错误及停止确认失败仍是故障。来源时间检查和 SDK 返回成功均不构成端到端实时或物理到位保证，实际跟随及停止延迟需真机测量。

XHand 连接要求设备已发现且取得有效反馈。正常目标发送一次，CRC 未确认或其他 SDK 故障结束运行，不自动重发旧目标。停止保留有限的 CRC 重试：反馈新鲜时保持实测姿态，否则进入 passive；退出仍请求 passive。软件等待不能强制中断阻塞的手部 SDK，后续关闭成功也不能抹去停止失败。

XHand 过流告警（SDK `1501035`、关节板 `110`）不抛异常或终止运行，读写共享每秒最多一条 warning；有效关节和触觉反馈继续更新。其他故障码、CRC 和无效数据仍按原规则处理。该行为只改变软件对告警的处理，不改变固件电流保护或抓握驱动力。

xArm 初始化及流式准备进入 Mode 1，Mode 0 用于 HOME。父进程保留一个本地 cancel_epoch，不再有通用授权状态。正常重新开始依次等待停止确认并读取故障、停止态锁内清邮箱、短锁内复核取消/急停/closing/fault 后清 stop。owner 接纳新生命周期请求时失效旧 stopped；异常请求不恢复，只在关闭时收集真实故障。发送仍检查 stop/emergency/closing、目标租期和反馈。SDK 正常返回后才提交轨迹参考；stopped 仅代表停止处理结束，不能掩盖停止 SDK 失败。

xArm SDK 单次回复超时按用途区分：初始化配置、模式切换、HOME 命令和停止使用 1 s；反馈查询与伺服发送使用 20 ms，调用结束或抛错后恢复 1 s。超时是等待上限，正常回复立即返回；SDK `code=3` 表示回复超时，仍按设备故障处理。停止接口内部还会查询状态，回复预算不能视为物理停稳时间。

arm HOME 使用配置 arm.homing.timeout_s（默认 60 秒）的一次绝对期限，覆盖 planner 初始化、实测起点准备、规划、owner 停稳、全部路点和最终确认；不再累加分段估算预算。路径由父进程规划，owner 从同一 ArmConfig.home_qpos 获取唯一目标并复核末点。保留桌面/自碰撞检查、规划后实测起点复核、位置/速度收敛和 dwell；SDK wait=False 返回不代表到位。独立 hand HOME 保留现有 2 秒预算。startup/prepare 与执行器 stop/close 仍使用各自预算；软件期限不能中断挂起的 native 调用。

`StaticCollisionBox` 由基座坐标系中心 `center_xyz_m` 和沿基座 X/Y/Z 轴的完整边长 `size_xyz_m` 定义，单位均为米，范围为 `center_xyz_m ± size_xyz_m / 2`。盒体固定轴对齐，按配置顺序命名为 `obstacle_0`、`obstacle_1` 等。标定桌面名为 `table`，按平面法向计算姿态，支持倾斜桌面。

遥操作、键盘点动、策略会话首次使用 arm HOME 才加载桌面标定并构造 planner，会话内复用。未使用 HOME 的运行、启动手部 HOME、触觉初始化和单纯 RGB-D 录制不依赖桌面标定；实际启用点云去桌面时仍须在该 worker 启动前加载，之后 HOME 复用同一平面。未使用的平面保持未加载；Policy 实际使用后将系数和路径写入运行配置快照。

缺失/无效 HOME 标定或专用配置、干净的几何路径拒绝仅结束本次 HOME：遥操作和 Policy 保持待命，键盘点动重新准备并恢复 hold；手部可能已完成回位，不能据此认定整机完全没动。设备、超时、停止或清理失败仍按错误结束，不自动换直线或跳过检查。

HOME 只检查当前关节位置到配置 HOME 的直接路径，按现有 1° 最大关节步长采样，检查关节范围、工作空间、碰撞与桌面软间隙。不重选等价终点、不绕行、不从软间隙自动脱困；直接路径拒绝时由操作者调整姿态。离散检查不是连续碰撞证明，SDK 返回也不是到位证明。environment.table_plane_path 仍服务 HOME 与点云，实际使用的平面参数继续归档。

目标端点与 Ruckig 关节极值检查不证明整条中间运动轨迹连续无碰撞；当前跳过手内碰撞的模型设定也不构成指间碰撞保护。FK 使用 URDF EEF，不能与固件 EEF 定义混用。手部重定向数学和默认参数见 [hand_retargeting.md](docs/hand_retargeting.md)。

手模板在 `assets/pointcloud_template/hand/hand_<点数>.npz`，默认 1024 点；更改模板点数、visual mesh 或安装模型后显式重建：

```bash
python scripts/generate_hand_template.py --num-points 1024 --overwrite
```

## 标定和查看

默认相机外参、桌面平面与 VR 朝向 JSON 位于 [config/calibration_data/](dexmani_real/config/calibration_data/)。相机标记须相对 EEF 刚性固定，摆位提供充分旋转激励；先退出厂商示教/独立点动控制连接，再只读采样。

```bash
python scripts/calibrate_camera.py collect --output calibration_samples/run_001
python scripts/calibrate_camera.py solve calibration_samples/run_001 --output camera_calibration.json
python scripts/calibrate_camera.py solve calibration_samples/run_001 --exclude 3 --output camera_calibration_clean.json
python scripts/calibrate_vr_heading.py --ref head
python scripts/tools/view_realsense.py
python scripts/tools/view_realsense.py --cloud processed
python scripts/tools/view_pointcloud.py
```

collect 的相机持续连接，Enter 为一组只读机械臂采样，完成立即断开臂；Q 结束。五帧原 RGB、检测、来源时间、实测位置/速度和样本 ID 保存为 NPZ，速度、帧新鲜度及前后漂移仍检查。solve 不连接设备，按原 ID 排除样本，保存逐样本残差和 used/excluded/worst IDs；质量合格后写入原 serial 映射格式外参，报告使用同名 `.result.json`。VR 头朝向直接消费头帧，不要求右手就绪。

VR 运行读取 JSON 中有限且位于 `[-180, 180]` 的 `theta_deg`，例如 `{"theta_deg": 30.0}`，构造 `R_z(-theta)`。ref、质量和时间报告不参与运行准入。手工角度的正确性由研究者确认；自动标定在保存时拒绝低质量结果，并输出完整报告。Raw 归档实际使用的旋转矩阵。

solve 的外参输出与派生报告不得覆盖 `capture.json` 或原始采样 NPZ。

实时相机查看使用 view_realsense.py：默认 RGB-D，在 RGB-D 或点云窗口内按 P 开关点云、S 切换 raw/processed（默认 processed）；`--cloud raw/processed` 可在启动时打开点云并指定初始类型。Q/Esc 或关闭 RGB-D 窗口退出；关闭点云窗口仅关闭点云，按 P 可重新打开。预览不修改曝光优先级，不提供逐帧诊断面板。RGB-D 仅预览时不加载外参、桌面或点云处理器；启用对应点云类型时才加载所需资源。窗口需要桌面/OpenGL。

view_pointcloud.py 提供独立诊断流程：先在终端选择是否拟合桌面，RGB-D 预览按 Enter 继续，随后处理一帧点云、打印点数和本帧处理耗时，关闭相机后静态查看结果；Q/Esc 或关窗退出。桌面拟合固定采集五份不同深度帧，使用当前相机内参；拟合结果可供本次处理使用，仅确认发布才改写共享桌面标定。`--config`、`--camera-calibration` 指定当前实验配置和外参。该流程复用当前点云算法和采集实现，预览不额外修改曝光设置。RGB-D 类型、网格和缓冲所有权由 RealSenseCamera.read 统一保证，预览、标定和采集入口使用同一读取边界。

RGB-D 的非零 Brown 畸变需结合实际模型消费者判断：Real 反投影会反解 Brown，配套 Policy 的空间投影仅使用 K 和外参。使用空间投影融合前须核对 checkpoint 与相机参数；不支持的组合须明确拒绝，或另行使在线/离线几何共同支持畸变。不能仅凭同时存在 RGB 与 point_cloud 推断模型使用了空间投影。

点云默认使用基座系 XYZRGB，工作空间 `x=[0.15,0.85]、y=[-0.50,0.50]、z=[0,0.80] m`，去桌面间隙 10 mm。深度邻域支持、体素、半径密度和连通分量过滤后 FPS；空点云在 viewer 清除、导出拒绝。几何规则可能丢失细结构，参数和配方必须随实验保持一致。在线场景点云仍由独立进程构建，通过共享环携带同源图像、来源时间和相机序列；复制竞争退回旧缓存时不重复处理同一相机序列；空云不发布，尚未过期的上一份结果仍可能被使用。运行时 IPC 不再搬运硬件 color_frame_number，接收端去重和标定仍保留硬件帧号。模型线程构建方案尚缺完整推理和调度性能验收，未启用。

## 离线物体几何和分析

CordVIP 提供离线物体几何、实测点云和位姿分析，尚未接入主包 Policy 观测。使用流程、模型安装及输出字段见 [CordVIP 文档](scripts/cordvip/README.md)。

```bash
python scripts/cordvip/track_object.py --episode episodes/TASK/EPISODE --scene scene.json --with-hand --output /tmp/cordvip/scene_001
python -m cordvip_geometry view /tmp/cordvip/scene_001
```

物体模型表面与语义点不代表实测接触；几何处理参数与配方需随实验归档。

物体资产按 `assets/pointcloud_template/object/<任务>/<物体>/` 组织，`reference.jpg`、`asset.json`、NPY 和 PLY 点云纳入 Git；`mesh/mesh_m.obj` 与 `third_party/weights/` 仅保留本地。克隆后可直接读取点云；FoundationPose 位姿估计需要先本地重建网格，具体步骤见[物体重建流程](docs/object_reconstruction.md)。

默认日志为终端 INFO、拒绝原因、完整异常 traceback 和简短段结果，需要留档时使用 shell 重定向。几何环境检查使用独立命令 `python -m cordvip_geometry check`。

开发约定见 [AGENTS.md](AGENTS.md)。

离线回归使用 `python -B -m pytest -p no:cacheprovider tests`；测试注入 fake driver，不连接真实设备。静态检查使用 `python -B -m ruff check --no-cache --select F dexmani_real tests`。SharedRing 复制竞争最多尝试两次，随后可复用相同字段集合的完整快照，保留原时间和 sequence；没有快照时不发送依赖该来源的动作，准备等待仍有期限。
