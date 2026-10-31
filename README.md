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

设备、控制和感知等实验参数通过 `--config` 加载可信 Python 文件中的 `CONFIG: RealConfig`。

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
python scripts/evaluate_replay.py --episode episodes/TASK/EPISODE --replay replay_results/run_001
python scripts/visualize_episode.py episodes/TASK/EPISODE
python scripts/keyboard_teleop.py
```

采集和策略会话使用 B 开始、S 保存并结束本段、D 丢弃本段、Q 保存并退出、Esc 急停。H 只在无活动段时回初始位，T 只在空闲、手部无接触时复位触觉。物理回放使用 S 停止、H 回零、Q 退出；键盘点动使用 WASD、方向键、IJKL，R 回零，Q 退出。

设备和模型在会话内复用。启动时手部 HOME、触觉复位和真实反馈就绪后才 warmup；SDK 请求成功与反馈就绪不证明已完成触觉零点标定。Teleop 每段顺序完成准备、等待 streaming 准备后的新观测、用实测姿态和 VR 输入重建锚点及最终准入，再进入 ACTIVE 控制；初始预计算观测不作为准入锚点。本段先撤权并请求停止，再关闭录制，由监督方唯一决定发布或丢弃；writer 无发布权。成功关闭且计数一致的零行段提示“未录到数据”，不发布并清理临时目录；非空技术失败段仍保存已录数据。写入/关闭失败、计数不一致或超时不会发布，临时目录保留。收尾期间的 B/H 等准备按键失效，未退出的 writer/模型任务阻止下一段授权。Python 等待期限不能终止阻塞 native 调用。

策略周期、数值 recipe、模态和动作窗口从 checkpoint 恢复。sync 等待预测后推进实际动作步；async 按时间锚点交接，可消费旧预测仍有效的尾部；RTC 保留冻结前缀、引导与严格交接时刻，错过动作不会补发。async/RTC 要求 `1 <= prefetch_steps <= action_steps` 且二者之和不超过预测窗口；RTC guidance cap 须显式为有限非负值。

`session_result.json` 保存会话 outcome/reason/shutdown、错误及 episodes 原始结果，包括每段主因、补充清理错误、时长、跳步及 Raw 结果；段数和终止原因计数由 episodes 计算，不重复持久化。时长完成、预测超时、操作者结束、目标准入和设备故障分别分类。已进入 RUNNING 的段计入 `--num-episodes`，准备失败或取消不计。会话正常结束不代表任务成功率。未消费的退休预测报错会终止会话，不回写已封存 Raw。

## 数据、来源和执行语义

Raw 使用 `data.h5` + `rgb.mp4`，默认在 `episodes/TASK/`；训练导出到 `datasets/TASK.zarr`，策略和回放产物位于 `rollouts/`、`replay_results/`。显式 `.tmp_*` 路径和指向临时目录的链接也会被读取入口拒绝。

Raw 将实测 state 与提交尝试 action 分开保存。组合命令按设备顺序提交，部分失败保持原故障并停止；送入 arm mailbox 只代表目标发布，不保证每个目标进入 SDK 或物理到位。手部 warm start、阻尼和滤波仅在提交正常返回并在锁内复核授权与期限后，从实际 RobotCommand 更新。

Canonical 保留 [固定 13 模态](dexmani_real/data/export.py)、原字段/单位/dtype、EEF 9D 与 EEF action 21D。内部 FK 使用独立 SE(3)，仅在数据和 policy 边界编码 rot6d。默认导出固定周期的有效连续前缀，技术失败段保守排除末行；正常保存、可导出和可物理回放的集合不同。结构损坏或保留范围内坏帧明确失败，不重采样或改写来源时间。

`control_ns` 是采样起点，`observation_ns` 是快照完成时刻；臂、手、触觉、相机的来源时间独立保留。相机使用 depth/color 首次主机接收时间中的较早者，读取缓存不会延长年龄；这不等同于曝光时刻。相机与点云 owner、五帧相机历史、防撕裂和同源配对继续保留。额外录制相机不会缩短纯控制动作期限。

每次新导出生成独立 `source_id`，原样搬迁保留 ID。split/resume 必须匹配同一数据身份，旧数据和 checkpoint 不自动改写。Canonical 保存影响数值结果的处理配方与完整 URDF；实验应记录 source_id、导出 Real SHA、训练 Policy SHA 及手模板来源。修改处理算法或模板后重新导出，已有未知来源须如实保留。

遥操作额外逐行保存 `vr_wrist_pos`、`vr_wrist_quat_wxyz`、`vr_landmarks`、`vr_recv_ts_ns`、`vr_sequence_id`，用于调试，不进入训练导出。接收端先校验并归一化原始姿态四元数，再转换为 FLU；零范数或非有限姿态被拒绝。位置用米、四元数 WXYZ；来源帧重复时 ID 和时间保持原值。`runtime_config_json`、VR 变换、手标定、初始锚点及失败时控制时点随 Raw 保存。

## 控制和几何边界

机械臂 SDK、模式、伺服和实测反馈仍由单独 arm owner 持有。运行代际/撤权、命令期限、反馈时效、controller 状态与 SDK 返回码在实际消费处检查。Ruckig 保留 jerk、关节位置及制动极值约束；默认伺服 100 Hz、速度 150 °/s、加速度 900 °/s²、jerk 12000 °/s³，参数以 Python 配置为准。调度延迟后从实际发送时刻安排下一步，实际跟随和停止延迟需真机测量。hand SDK 的阻塞调用不受 arm 期限强制中断。

HOME 保留路径搜索、等价关节分支、逐段停稳和碰撞检查。HOME 桌面盒体/软间隙与在线目标检查职责不同；在线环境使用 static_boxes，允许抓取中的桌面接触。

`StaticCollisionBox` 由基座坐标系中心 `center_xyz_m` 和沿基座 X/Y/Z 轴的完整边长 `size_xyz_m` 定义，单位均为米，范围为 `center_xyz_m ± size_xyz_m / 2`。盒体固定轴对齐，按配置顺序命名为 `obstacle_0`、`obstacle_1` 等。标定桌面名为 `table`，按平面法向计算姿态，支持倾斜桌面。

遥操作、键盘点动、策略会话和物理回放在装配 HOME 时都要求有效桌面标定；缺失或无效会在设备连接前失败。

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
python scripts/tools/view_realsense.py --calibrate-table
python scripts/tools/view_realsense.py --save-dir snapshots
python scripts/tools/view_realsense.py --benchmark-frames 20
```

collect 的相机持续连接，Enter 为一组只读机械臂采样，完成立即断开臂；Q 结束。五帧原 RGB、检测、来源时间、实测位置/速度和样本 ID 保存为 NPZ，速度、帧新鲜度及前后漂移仍检查。solve 不连接设备，按原 ID 排除样本，保存逐样本残差和 used/excluded/worst IDs；质量合格后写入原 serial 映射格式外参，报告使用同名 `.result.json`。VR 头朝向直接消费头帧，不要求右手就绪。

VR 运行读取 JSON 中有限且位于 `[-180, 180]` 的 `theta_deg`，例如 `{"theta_deg": 30.0}`，构造 `R_z(-theta)`。ref、质量和时间报告不参与运行准入。手工角度的正确性由研究者确认；自动标定在保存时拒绝低质量结果，并输出完整报告。Raw 归档实际使用的旋转矩阵。

solve 的外参输出与派生报告不得覆盖 `capture.json` 或原始采样 NPZ；其他新结果仍可保存在采样目录内。

相机查看统一使用 view_realsense.py：默认 RGB-D，`--cloud raw/processed` 在启动时选择固定点云类型；Q/Esc 或关闭任一窗口退出。预览不修改曝光优先级，不提供运行中模式切换和逐帧诊断面板。`--save-dir PATH` 保存单帧原始 RGB-D 到一个 `rgbd_*.npz`，打印文件路径后退出；`--benchmark-frames N` 测量 N 个新深度帧；`--calibrate-table` 采集五帧拟合桌面并在终端明确确认后更新共享平面。这三项互斥，执行后退出。窗口需要桌面/OpenGL，代码清理测试不自动连接真实设备。

标定样本与原始快照均以独占方式创建 NPZ，拒绝覆盖。写入或关闭失败时清理本次新建文件；连续失败时保留首个异常和原 traceback，后续关闭或删除错误另记日志。

快照可用 `np.load(path, allow_pickle=False)` 读取：`rgb` 为 RGB uint8，`depth_aligned_to_color_raw` 为对齐到 color 网格的 uint16 原始计数；`K`、`distortion_model`、`distortion_coeffs` 描述该网格，`depth_scale_m_per_unit` 是每计数的米数。另保存实际 `camera_serial`、`source_timestamp_ns`、`source_clock` 和 depth/color frame number。来源时钟为 `host_monotonic_oldest_rgbd_receive`，时间取两通道首次主机接收时刻中的较早者，不是文件创建时间或曝光时间。深度换算为 `depth_raw.astype(np.float32) * np.float32(depth_scale_m_per_unit)`。原始快照无需外参、桌面文件或点云处理配置；需要基座点云时，离线显式加载对应 serial 的相机标定后重建。

点云默认使用基座系 XYZRGB，工作空间 `x=[0.15,0.85]、y=[-0.50,0.50]、z=[0,0.80] m`，去桌面间隙 10 mm。深度邻域支持、体素、半径密度和连通分量过滤后 FPS；空点云在 viewer 清除、导出拒绝。几何规则可能丢失细结构，参数和配方必须随实验保持一致。

## 离线物体几何和分析

CordVIP 流程为同一 Raw 首帧人工确认 mask → 顺序全部行 → 保存逐行结果 → 独立查看。支持单/多物体、无资产 mask-only、独立 observed/model、法向、手 FK 与 SAM RPC；深度支持量参与 pose-valid，pose 丢失不删除独立实测点。

逐帧结果统一保存在 `frames/*.npz`，点云和法线仅保存 camera 坐标，包含状态、失败原因、门控前候选位姿与已计算的深度支持数据；候选存在不代表 `pose_valid=True`。每帧仍保留 K，以及手几何启用时的 `T_base_camera`、link_ids 等事实。由 `T_base_camera` 取 R、t，基座系点为 `(points_camera @ R.T + t).astype(np.float32)`，法线为 `(normals_camera @ R.T).astype(np.float32)`。场景、配置、模型来源和结束摘要各保存一次。可按需用 `numpy.load(path, allow_pickle=False)` 读取 NPZ 离线计算指标。

离线查看在换行时更新 RGB 叠加和几何，`v` 只切换点云，静止时仅处理窗口事件和渲染。`--no-3d` 不解压点云；`--headless` 只读取并检查 RGB、源 row/time/clock，不读取对象字段或场景配置，也不加载 OpenCV/Open3D。

```bash
python scripts/cordvip/track_object.py --episode episodes/TASK/EPISODE --scene scene.json --with-hand --output outputs/scene_001
python -m cordvip_geometry view outputs/scene_001
python scripts/analyze_timing.py episodes/TASK/EPISODE
```

物体生成先保存未定尺度 mesh，实测定尺度后才一次采样最终点/法向。初始 mask 仅可复用于相同 RGB-D、相机几何、行和来源时间。流程、模型安装与结果字段见 [CordVIP 文档](scripts/cordvip/README.md)。同行关联不证明手/相机物理同步，模型表面与语义点不代表实测接触。

默认日志为终端 INFO、拒绝原因、完整异常 traceback 和简短段结果，需要留档时使用 shell 重定向。回放指标通过 evaluate_replay.py 显式计算，分析失败不回写控制结果；指标按索引比较实测轨迹，不代表目标跟踪或时间对齐。analyze_timing.py 只报告采样间隔、readout 和来源年龄，明确列出缺失/非法来源，不推断曝光或 SDK 延迟。几何环境检查使用独立命令 `python -m cordvip_geometry check`。

开发约定见 [AGENTS.md](AGENTS.md)。
