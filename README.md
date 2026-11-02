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

任务、路径、段数、时长、录制选择及 `--weights/--inference-steps/--seed/--device` 通过 CLI 指定。无手运行只需在 Python 配置中设 `hand.enabled=False`；VR 遥操作还须使用 `--no-record` 关闭录制，实物手须移除或固定在配置的 HOME 姿态，供碰撞规划使用。Policy 配方从 checkpoint 恢复，其训练配置和运行快照可使用 YAML。

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

将 TASK、EPISODE、EXPERIMENT 替换为任务、已发布 episode 和保存的 Policy 实验路径。

```bash
python scripts/collect_teleop.py TASK
python scripts/export_zarr.py episodes/TASK --output datasets/TASK.zarr
python scripts/run_policy.py EXPERIMENT --checkpoint latest --num-episodes 5
python scripts/replay_episode.py episodes/TASK/EPISODE --output replay_results/run_001
python scripts/visualize_episode.py episodes/TASK/EPISODE
python scripts/keyboard_teleop.py
```

Episode 可视化默认按实测 `arm_qpos`、`hand_qpos` 通过 FK 重建手部表面点云；在 3D 区域切换到 `Hand Point Cloud` 可查看手部点云、指尖和 EEF 锚点，`xArm-base` 显示相机点云及指尖、EEF 锚点。两者共用基座坐标系和时间轴。手部点数由 `CONFIG.hand_pointcloud.num_points` 指定；`--no-point-cloud` 只关闭相机点云，手部重建不依赖相机外参。

VR 遥操作与策略会话：B 空闲时开始新段；S/C 停止并保存；D 停止并丢弃；Q 保存当前段并退出，但保留已有 D 意图；Esc 急停并结束会话。保存决定确定后，迟到 D 不再逆转结果。H 仅在空闲且收尾完成时回初始位，活动段请先 S；T 仅在空闲时复位触觉，操作者须确认无接触。准备或收尾期间的 B/H/T 不会排队启动下一段。

音效使用系统 `aplay` 或 `paplay` 播放提示，普通播放或关闭失败仅 warning，不影响实验结果和退出码。提示音不能证明机器人已停止、物理到位或任务成功。

物理回放使用 S 停止、H 回零、Q 退出；键盘点动使用 WASD、方向键、IJKL，R 回零，Q 退出。键盘在启动及 R 回零结束后完成伺服准备并进入 RUNNING/hold，再提示点动就绪；方向键只更新目标。松键或 jog 目标被 IK/约束拒绝时均制动并 hold，下一轮按当前键态继续，不撤权、不要求松开再按。hold 也有制动距离；运动候选错过本轮提交机会时丢弃，制动 hold 仍立即提交。普通输入过期按取消处理，SDK/停止失败仍是故障。

设备和模型在会话内复用。启用手部时，Teleop/Policy 启动先完成手部 HOME；仅模态或录制需要时复位触觉，反馈就绪不证明零点标定完成。每段 B 后仍有伺服准备延迟；VR 重新锚定，Policy 重新采集观测历史。未回收的 writer 或模型任务禁止开始新段。

策略周期、数值 recipe、输入模态和动作窗口从 checkpoint 恢复，不改变录制的完整多模态 schema。sync 等待预测后顺序执行动作；async 按时间锚点裁掉过期动作，可继续旧计划的有效尾部；RTC 固定策略动作前缀并严格交接，错过承诺槽即结束。记预测长度为 P、`action_steps` 为 A、`prefetch_steps` 为 d：sync 要求 `1 <= A <= P`；async 要求 `1 <= d <= A <= P` 且 `d < P`；RTC 另要求 `A+d <= P`，guidance cap 须显式为有限非负值。EEF 前缀相同不保证未来 IK 分支相同。多帧 history 断档后清空重积累，窗口完整前不提交新预测；单帧模式不变。首预测 reset RNG，预热后恢复 RNG；sync 的预测等待不得超过已有目标期限。

每段结束先撤权并停止设备，再完成录制和模型回收；writer 与模型各有完整的独立收尾预算。writer 完整关闭且计数一致后才按首次保存意图发布；失败保留 staging，零行不发布。Raw 记录执行结束时的状态，完整的异常段可保存为 failed；发布后的模型回收错误只进入 episode/session，不回写 Raw。退出先释放设备，再完成慢速收尾与尾音等待。

`session_result.json` 保存会话 outcome/reason/shutdown、急停请求、episode 终止原因、独立错误及 Raw 发布结果。关闭阶段真正的 KeyboardInterrupt 立即锁存急停和跨进程 emergency/stop 信号，仍尝试剩余资源并保存 Policy 最终结果，非零退出；已有核心主错及最终原因继续保留，中断事实独立可见。等待中断不会被当作关闭完成，未回收资源保留句柄并报告 pending，不重新开始等待预算。Python 等待期限无法强制终止阻塞 native 调用；结果文件写入失败仍是核心失败。已进入运行的段计入 `--num-episodes`，准备失败或取消不计；技术正常不代表任务成功。

## 数据、来源和执行语义

任务标签须为非空、非全空白字符串，不含 ASCII 控制字符 0–31 或 DEL 127，录制与导出共用此规则且不改写输入；策略任务名在创建会话目录前也按此规则接纳。遥操作 TASK 同时用于目录，另外禁止路径分隔符、以点开头的目录名及首尾空白；这些目录限制不适用于一般自然语言标签。

Raw 使用 `data.h5` + `rgb.mp4`，默认在 `episodes/TASK/`；训练导出到 `datasets/TASK.zarr`，策略和回放产物位于 `rollouts/`、`replay_results/`。显式 `.tmp_*` 路径和指向临时目录的链接也会被读取入口拒绝。

Raw 将实测 state 与提交尝试 action 分开保存。组合命令按设备顺序提交，部分失败保持原故障并停止；送入 arm mailbox 只代表目标发布，不保证每个目标进入 SDK 或物理到位。手部 warm start、阻尼和滤波参考由控制线程在提交正常返回后更新；同一 VR 帧每个控制 tick 仍运行手映射和速度限制。dt 使用本次计算时刻减上次实际提交参考时刻，迟到丢弃的候选不推进参考或时间。SDK 调用正常返回也不证明手部物理执行。

Canonical 保留 [固定 13 模态](dexmani_real/data/export.py)、原单位和 dtype、EEF 9D 与 EEF action 21D。示教导出目前只支持 teleop Raw。sync Policy Raw 的同行 action 是本次提交尝试，不代表由同行观测推理得到。正常保存、可导出和可物理回放是不同条件。相机外参 `camera_T_xarm_base_from_color` 从 color optical 坐标变换到 xArm base，平移单位为米。

`timing/rows` 使用主机单调时钟，单位 ns，缺失值为 -1；`control_ns` 是采样起点，`observation_ns` 是快照完成时刻。臂、手、触觉和相机保留各自来源时间，缓存读取不刷新年龄；相机来源为首次主机接收时间，不等于曝光时刻。同行数据不代表硬件同步。录制和模型输入共用当前快照，启用录制就要求完整采集来源有效。

Teleop/jog/async 跳过错过的控制时点，不追赶旧动作，Raw 因此可能有时间缺口。**Canonical 只接纳完整发布、正常关闭、结构及计数一致、`technical_status == "normal"` 且全部行有效的 teleop episode。** 必需模态、有限值、动作、时间和来源时间检查任一失败就拒绝整段；不要求任务成功。网格及相邻采样间隔容差仍为 dt/4，readout 小于 dt，触觉来源年龄须小于 Raw 记录的 hold 上限。合法时间抖动和重复相机帧号不会删行。`episode_quality` 仅记录整段最大网格偏差和 readout，段长由 `meta/episode_ends` 确定。

导出保留分块数值扫描、RGB 逐帧解码和流式写入；接纳后导出全部帧，不截断、切分、插值、补帧或重采样。批量遇到不合格 episode 明确失败，错误标明 episode、原因及适用的行索引；可通过 `--exclude EPISODE` 排除整段后重试，不自动跳过或改写 Raw。全部完成后才原子发布 Canonical；准入或转换失败仅保留隐藏 staging，不留下可见半成品，也不覆盖已有输出。

物理回放要求完整发布、技术正常、合法非空有限目标和首行实测起点；含无命令 NaN 行的 Policy Raw 会被拒绝，不筛行或补值。

每次新导出生成独立 `source_id`，原样搬迁保留 ID，供人工归档和实验说明。它不是内容 hash，不保证 split/resume 使用同一数据；配套 Policy 训练恢复允许显式更换数据路径，并按其当前实现检查实际模型/数据兼容项。旧数据和 checkpoint 不自动改写。Canonical 保存影响数值结果的处理配方与完整 URDF；实验应记录 source_id、导出 Real SHA、训练 Policy SHA 及手模板来源。修改处理算法或模板后重新导出，已有未知来源须如实保留。

遥操作额外逐行保存 `vr_wrist_pos`、`vr_wrist_quat_wxyz`、`vr_landmarks`、`vr_recv_ts_ns`、`vr_sequence_id`，用于调试，不进入训练导出。VR 使用右手数据，21 个 landmark 保持 SDK 原顺序；来源时间取所需通道首次主机接收时间的较早者。接收端先校验并归一化原始姿态四元数，再转换为 FLU；零范数或非有限姿态被拒绝。位置用米、四元数 WXYZ；来源帧重复时 ID 和时间保持原值。`runtime_config_json`、VR 变换、手标定和每段一份初始锚点随 Raw 保存。逐次操作与失败控制时点用日志诊断，软超时只保存一份固定大小摘要。

触觉 aggregate `(5,3)` 与 dense `(5,120,3)` 分别保存 SDK 最后一份完整有效通道帧及来源时间，按各自来源年龄及配置的 hold 上限限时保持。数据使用 SDK sensor axes，原生物理尺度未知，指序为 thumb/index/middle/ring/pinky；不解释为牛顿或布尔接触，也不假定 dense 求和等于 aggregate。原生单位、指序、hold age 和实际触觉复位结果继续归档；读取仍校验数值语义，不据说明文案推断校准或真实接触。

## 控制和几何边界

机械臂 SDK 由独立进程持有，XHand SDK 由父线程调用。默认目标期限为 teleop/jog 0.5 s、policy/replay 2 s，hold 同样受期限限制。arm/hand 反馈年龄预算为 4/30 s，相机为 4/fps，VR 为 0.5 s；缓存读取不延长年龄。arm 默认反馈 25 Hz、伺服 100 Hz，SDK 阻塞时不保证达到调度频率。Ruckig 保留关节位置、速度、加速度、jerk 和制动极值约束，默认上限为 150 °/s、900 °/s²、12000 °/s³，实际参数以配置为准。过期、撤权与反馈失效会停止；SDK 错误及停止确认失败仍是故障。来源时间检查和 SDK 返回成功均不构成端到端实时或物理到位保证，实际跟随及停止延迟需真机测量。

XHand 连接要求设备已发现且取得有效反馈。正常目标发送一次，CRC 未确认或其他 SDK 故障结束运行，不自动重发旧目标。停止保留有限的 CRC 重试：反馈新鲜时保持实测姿态，否则进入 passive；退出仍请求 passive。软件等待不能强制中断阻塞的手部 SDK，后续关闭成功也不能抹去停止失败。

XHand 过流告警（SDK `1501035`、关节板 `110`）不抛异常或终止运行，读写共享每秒最多一条 warning；有效关节和触觉反馈继续更新。其他故障码、CRC 和无效数据仍按原规则处理。该行为只改变软件对告警的处理，不改变固件电流保护或抓握驱动力。

xArm 初始化进入 Mode 1，Mode 0 仅用于规划 HOME。伺服使用邮箱中最新目标，发送前检查权限、期限及反馈。纯代际锁争用只跳过本次发送；SDK 正常返回后才推进轨迹参考。

xArm SDK 单次回复超时按用途区分：初始化配置、模式切换、HOME 命令和停止使用 1 s；反馈查询与伺服发送使用 20 ms，调用结束或抛错后恢复 1 s。超时是等待上限，正常回复立即返回；SDK `code=3` 表示回复超时，仍按设备故障处理。停止接口内部还会查询状态，回复预算不能视为物理停稳时间。

HOME 保留路径搜索、等价关节分支、逐段停稳和碰撞检查。HOME 桌面盒体/软间隙与在线目标检查职责不同；在线环境使用 static_boxes，允许抓取中的桌面接触。

`StaticCollisionBox` 由基座坐标系中心 `center_xyz_m` 和沿基座 X/Y/Z 轴的完整边长 `size_xyz_m` 定义，单位均为米，范围为 `center_xyz_m ± size_xyz_m / 2`。盒体固定轴对齐，按配置顺序命名为 `obstacle_0`、`obstacle_1` 等。标定桌面名为 `table`，按平面法向计算姿态，支持倾斜桌面。

遥操作、键盘点动、策略会话和物理回放首次使用 arm HOME 才加载桌面标定并构造 planner，会话内复用。未使用 HOME 的运行、启动手部 HOME、触觉初始化和单纯 RGB-D 录制不依赖桌面标定；实际启用点云去桌面时仍须在该 worker 启动前加载，之后 HOME 复用同一平面。未使用的有效平面元数据保持未加载，实际使用后记录系数和路径。

缺失/无效 HOME 标定或专用配置、干净的几何路径拒绝仅结束本次 HOME，保持停止并允许调整后重试；手部可能已完成回位，不能据此认定整机完全没动。已完成回放后的单次 HOME 拒绝记录在 `home_results`，不改变原回放完成结果。设备、超时、停止或清理失败仍按错误结束，不自动换直线或跳过检查。

HOME 净间隙仍为机器人几何到桌面的最小距离减去 `table.soft_clearance_m`。起始姿态在软间隙内时仍允许按既有规则退出，退出后不能重新进入，目标必须满足净间隙要求。`environment.table_plane_path` 是 HOME 和点云共用的标定来源；点云 `remove_table` 仍独立可选，去桌面不会执行机器人碰撞检查。在线端点控制不装配桌面碰撞模型。workspace、米/deg/rad 和物理高度/间隙保持明确。

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

点云默认使用基座系 XYZRGB，工作空间 `x=[0.15,0.85]、y=[-0.50,0.50]、z=[0,0.80] m`，去桌面间隙 10 mm。深度邻域支持、体素、半径密度和连通分量过滤后 FPS；空点云在 viewer 清除、导出拒绝。几何规则可能丢失细结构，参数和配方必须随实验保持一致。

## 离线物体几何和分析

CordVIP 提供离线物体几何、实测点云和位姿分析，尚未接入主包 Policy 观测。使用流程、模型安装及输出字段见 [CordVIP 文档](scripts/cordvip/README.md)。

```bash
python scripts/cordvip/track_object.py --episode episodes/TASK/EPISODE --scene scene.json --with-hand --output /tmp/cordvip/scene_001
python -m cordvip_geometry view /tmp/cordvip/scene_001
```

物体模型表面与语义点不代表实测接触；几何处理参数与配方需随实验归档。

物体资产按 `assets/pointcloud_template/object/<任务>/<物体>/` 组织，`reference.jpg`、`asset.json`、NPY 和 PLY 点云纳入 Git；`mesh/mesh_m.obj` 与 `third_party/weights/` 仅保留本地。克隆后可直接读取点云；FoundationPose 位姿估计需要先本地重建网格，具体步骤见[物体重建流程](docs/object_reconstruction.md)。

默认日志为终端 INFO、拒绝原因、完整异常 traceback 和简短段结果，需要留档时使用 shell 重定向。几何环境检查使用独立命令 `python -m cordvip_geometry check`。

相关离线回归（需安装真实数值、视频及几何依赖，不连接设备）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/test_audio.py tests/test_errors.py tests/test_interaction.py tests/test_policy.py tests/test_data.py
```

这些测试不证明真实 SDK 中断响应、物理停稳或 native 阻塞回收。开发约定见 [AGENTS.md](AGENTS.md)。
