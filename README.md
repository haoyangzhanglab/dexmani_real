# DexMani Real

DexMani Real 是面向灵巧操作研究的真实机器人运行时，覆盖 xArm7（7 DoF）、
XHand（12 DoF）、Quest/HTS 手部跟踪与 RealSense RGB-D 的遥操作、数据采集、
物理回放、离线处理和 learned-policy 部署。

> 这是安全敏感的真实硬件软件。除明确标注为离线的命令外，运行前必须确认
> 硬件连接、工作空间、标定状态、急停条件和操作者授权。

## 当前能力

- VR 或键盘控制 xArm7，并可选联动 XHand。
- 通过共享内存协调 arm、hand、VR、camera、recorder 与 policy 进程。
- 常规 arm/hand joint command 统一经过安全门，并在设备 worker 的 SDK 边界再次
  校验；homing 使用独立的碰撞检查路径和专用 queue。
- VR/键盘遥操作与物理回放的轨迹回放环节保留机器人自碰撞和静态障碍检查，但桌面不作为
  动作拒绝条件，以允许近桌面的精细抓取；homing（含回放 return_home）保留桌面安全验证。
- XHand 已知的抓取接触过流码 `1501035` 在发送位置命令时会被接受，不中止抓取，并记入
  episode 质量指标；同一码在状态读取失败时不生成伪造的新鲜反馈。持续数据不可用仍由
  新鲜度和 watchdog 边界处理。
- RealSense 相机按设备原生频率连续采集；16 Hz 控制网格只选择最新严格因果帧，
  不再将相机发布节拍绑定到控制频率。
- 事务式写入 depth-to-color aligned RGB-D raw episode v22；保留 native depth/color 几何与时序 provenance。
- 将 aligned raw v22 episode 清洗为 processed HDF5 v7，再导出 Policy Zarr v3。
- 物理回放已记录 episode，并保存回放轨迹与一致性指标。
- 通过可替换 `PolicyRuntime` 运行 joint/EE-action learned policy；仓库包含无模型的
  deterministic fake 实现和 `dexmani_policy` 集成。启动时 `DeploymentManifest`
  fail-closed 校验 checkpoint 契约与部署配置（action_key/点云 N/stride/observation）。

## 导航

| 目标 | 入口 | 主要实现 |
|---|---|---|
| 理解仓库与修改约束 | [`AGENTS.md`](AGENTS.md)、[`code_style.md`](code_style.md) | [`repo_map.md`](repo_map.md)、[`user_design.md`](user_design.md) |
| VR 遥操作与采集 | [`examples/collect_teleop.py`](examples/collect_teleop.py) | [`teleop/session.py`](dexmani_real/teleop/session.py)、[`teleop/loop.py`](dexmani_real/teleop/loop.py)、[`teleop/operator_controls.py`](dexmani_real/teleop/operator_controls.py)、[`teleop/control_grid.py`](dexmani_real/teleop/control_grid.py)、[`teleop/action_proposal.py`](dexmani_real/teleop/action_proposal.py) |
| 键盘遥操作 | [`examples/keyboard_teleop.py`](examples/keyboard_teleop.py) | [`teleop/keyboard_session.py`](dexmani_real/teleop/keyboard_session.py) |
| 物理回放 | [`examples/replay_episode.py`](examples/replay_episode.py) | [`replay/`](dexmani_real/replay) |
| raw episode 读取/录制 | — | [`recording/frame.py`](dexmani_real/recording/frame.py)、[`recording/recorder.py`](dexmani_real/recording/recorder.py)、[`recording/hdf5_writer.py`](dexmani_real/recording/hdf5_writer.py)、[`recording/reader.py`](dexmani_real/recording/reader.py) |
| 离线清洗与 Zarr 导出 | [`examples/process_episodes.py`](examples/process_episodes.py)、[`examples/export_policy_zarr.py`](examples/export_policy_zarr.py) | [`data/`](dexmani_real/data) |
| learned-policy 部署 | [`examples/run_policy.py`](examples/run_policy.py) | [`deployment/`](dexmani_real/deployment)、[`integrations/`](dexmani_real/integrations) |
| 相机、桌面与 VR 标定 | [`examples/`](examples) | [`calibration/`](dexmani_real/calibration)、[`sensor/`](dexmani_real/sensor)、[`config/`](dexmani_real/config) |
| 点云完整链路 | [`docs/pointcloud_pipeline.md`](docs/pointcloud_pipeline.md) | [`sensor/pointcloud.py`](dexmani_real/sensor/pointcloud.py)、[`sensor/pointcloud_worker.py`](dexmani_real/sensor/pointcloud_worker.py) |

完整的逐文件职责见 [`repo_map.md`](repo_map.md)。

## 核心架构

```text
RealSense / Quest-HTS / xArm7 / XHand
                 │
                 ▼
        device-specific workers
                 │
                 ▼
 RuntimeChannels: typed rings + queues + lifecycle state
          │                         │
          ├─ teleop coordinator     └─ inference worker → policy plan ring
          │                                      │
          └──────────────────────────────────────┤
                                                 ▼
                                      SafetyGate validation
                                                 │
                                      command publication
                                                 │
                                      device workers + SDK checks

 teleop fixed-grid samples → RecorderIO → data.h5 + depth.h5 + rgb.mp4
 raw episode → offline processing → processed HDF5 → Policy Zarr
```

必须保持的边界：

- 跨进程状态通过 `RuntimeChannels`；固定 wire shape/dtype 由
  [`ipc/schema.py`](dexmani_real/ipc/schema.py) 定义，机器人模型 shape 与关节顺序由
  [`robot_spec.py`](dexmani_real/robot_spec.py) 定义。
- xArm、XHand、RealSense 和 HTS SDK 对象只存在于各自 owner/worker 内。
- teleop、replay 和 deployment 负责动作决策；候选先由
  [`control/safety_gate.py`](dexmani_real/control/safety_gate.py) fail-closed 校验，再由
  [`control/publication.py`](dexmani_real/control/publication.py) 发布；设备
  worker 在 [`robot/command_validation.py`](dexmani_real/robot/command_validation.py) 再次校验。
  arm/hand homing 分别由 [`control/arm_home.py`](dexmani_real/control/arm_home.py) 和
  [`control/hand_home.py`](dexmani_real/control/hand_home.py) 拥有。
- Recorder 只持久化选定的固定网格样本，不拥有机器人控制。
- `run_generation`、freshness、heartbeat、safety state 与 worker 侧检查共同使
  暂停、回零或故障前的旧命令失效。

### 控制与录制边界

- [`teleop/action_proposal.py`](dexmani_real/teleop/action_proposal.py) 只计算并限幅
  EEF、arm 与 hand proposal；它不发布命令、不访问 shared memory，也不写录制数据。
- [`teleop/loop.py`](dexmani_real/teleop/loop.py) 构造资源、等待 readiness 并调度两个
  窄职责模块：[`teleop/operator_controls.py`](dexmani_real/teleop/operator_controls.py) 处理
  BEGIN/pause/home/quit 与录制决策，[`teleop/control_grid.py`](dexmani_real/teleop/control_grid.py)
  完成单个 causal tick 的读取、proposal、校验、发布和采样。pause、VR/hand feedback
  异常、BEGIN audio gate 或录制终结进入 command-quiescence；恢复前必须重新锚定，期间不发布命令。
  同步 home 是有意的控制静默区间，返回后由 loop 同时重锚 coordinator 和 control-grid
  时钟，不计作遥操作栅格丢失。
- RecorderIO 从 fixed-size shared-memory record 按逻辑 sequence 严格、连续地取得所有权；
  它只复制尚未确认的 slot，缺失或超出环容量时丢弃 active episode，绝不跳过样本。随后它将
  record 解码为不可变、拥有自身数组副本的 `EpisodeFrame`，并拥有 episode transaction、
  camera sidecar、验证和有界 finalize。其轮询是非执行器的服务循环；周期性批量持久化允许
  超过单次轮询周期，健康性以 sample backlog、sequence 连续性和 writer 状态为准，而不是
  轮询相位。其中
  [`recording/hdf5_writer.py`](dexmani_real/recording/hdf5_writer.py) 是单个
  `data.h5` handle、dataset append 与 offset 的唯一 owner。两者都不拥有机器人命令或
  episode 的开始/停止决策。
- 相机标定的纯 ArUco/hand-eye 计算、运动控制与 side-effect lifecycle 分别位于
  [`calibration/camera/solver.py`](dexmani_real/calibration/camera/solver.py)、
  [`calibration/camera/control.py`](dexmani_real/calibration/camera/control.py) 与
  [`calibration/camera/session.py`](dexmani_real/calibration/camera/session.py)。

## 环境

项目声明 Python `>=3.10`，目标开发环境是 conda 环境 `real_robot`。从仓库根目录：

```bash
conda activate real_robot
python -m pip install -e .
```

`pyproject.toml` 提供基础 Python 依赖，但不是完整的硬件/研究环境锁文件。
实际工作流还可能需要对应设备或功能的外部包，例如 xArm SDK、XHand controller、
RealSense SDK、HTS hand-tracking SDK、Pinocchio、MPlib、NLopt、
`dex-retargeting`、Open3D、PyTorch/PyTorch3D、Rerun，以及外部
`dexmani_policy` 仓库。按当前任务安装这些依赖，不要假设一次基础安装即可连接硬件。

配置的唯一合并优先级是：

```text
CLI override > YAML file > dexmani_real/config/defaults.py
```

运行时配置由 [`config/runtime.py`](dexmani_real/config/runtime.py) 校验、冻结并生成
SHA-256；部署配置由 [`deployment/config.py`](dexmani_real/deployment/config.py)
独立解析。`pointcloud` 是与 EEF `policy.workspace` 分离的感知配置段；实时 worker、离线
重建和 shadow 检查只从该段派生参数，并把策略 ID 与配置 SHA-256 写入 processed/Zarr
语义。可在不启动硬件的情况下查看遥操作解析结果：

```bash
python examples/collect_teleop.py --print-config
```

## 常用命令

### 硬件工作流

| 用途 | 命令 | 主要副作用 |
|---|---|---|
| VR 遥操作采集 | `python examples/collect_teleop.py --task-name <task>` | 连接 arm/hand/VR/camera；写 raw episode |
| VR 遥操作但不录制 | `python examples/collect_teleop.py --task-name <task> --no-record` | 连接 arm/hand/VR；不启动 camera/recorder |
| 键盘遥操作 | `python examples/keyboard_teleop.py` | 连接并控制 xArm7，可选 XHand |
| 物理回放 | `python examples/replay_episode.py episodes/<task>/episode_*` | 预检后控制 xArm7/XHand；写 `replay_results/` |
| 回放 processed HDF5 | `python examples/replay_episode.py episodes_processed/<task>/episode_<timestamp>.h5 --processed` | 同上；`--processed` 显式确认跳过记录期模型(URDF/SRDF)provenance，workspace/碰撞预检仍按当前模型执行；含 risky bridge 的压缩轨迹拒绝回放 |
| learned policy | `python examples/run_policy.py --deployment-config <file.yml>` | 启动 arm、可选 hand、inference 与 coordinator（active/pending 调度 + EE→IK + delta/collision 安全门）；请求 `point_cloud` 时同时连接 camera |
| 相机标定 | `python examples/calibrate_camera.py --hand-geometry <absent or secured-home>` | 连接 xArm/RealSense；更新相机标定；参数必须反映真实 XHand 安装状态 |
| VR 朝向标定 | `python examples/calibrate_vr_heading.py` | 连接 HTS；更新 VR transform |
| RealSense 点云交互诊断 | `python examples/realsense_record_example.py` | 只连接相机；GUI 切换完整 RAW/处理后点云，不写标定 |
| 桌面点云与标定诊断 | `python examples/pointcloud_process_example.py` | 只连接相机；多帧拟合桌面；仅在显式确认后备份并更新共享桌面标定 |
| XHand 独立诊断 | `python examples/xhand_control_example.py` | 连接并控制 XHand |

支持 argparse 的入口应先用 `--help` 查看当前参数；
`examples/xhand_control_example.py` 没有 `--help` 模式，执行即进入硬件流程。
不要从旧文档复制硬件地址或运动参数。

相机标定的 `--hand-geometry` 是物理事实声明：未安装 XHand 时使用 `absent`；已安装时，
只有在它实际固定于配置的 home 姿态时才能使用 `secured-home`。它不绕过碰撞检查。

### Learned policy 实时点云

部署配置的 `observation_fields` 包含 `point_cloud` 时，lifecycle 才启动 camera 与独立
point-cloud worker。worker 始终读取最新的 depth-to-color aligned RGB-D，旧帧不会排队；inference 组装
点云 T 历史窗（`n_obs_steps` 帧，跨帧 `camera_generation` 一致），每帧为 xArm-base
`float32 (N, 6)`，列语义为 `xyzrgb`，RGB 范围为 `[0,1]`。`pointcloud_num_points` 只允许
`1024`、`2048`、`4096`、`8192`，也可通过 `--pointcloud-num-points` 覆盖。

实时、离线处理和 shadow 诊断共用 `PointCloudConfig` 与同一个生产 builder。处理顺序为：
aligned depth 范围/局部支持与边缘四方向支撑过滤 → 使用缓存 color-camera ray 的桌面高度
迟滞裁减（在反投影前，以可靠高点连通保护物体低处表面）→ color-frame 反投影 → xArm-base
变换与 workspace 裁减 → XYZ/RGB 均值体素 →
单次 radius graph 邻居密度/连通域过滤 → 空间候选上限 → 确定性固定 N 采样。感知
workspace 当前为 x `[0.0,0.8]`、y `[-0.5,0.5]`、z `[0.0,0.8]` m；体素 RGB 是其源
aligned 像素 RGB 的均值。processed HDF5 和 Policy Zarr 同时保存并校验算法、配置 SHA-256
与桌面平面身份，禁止混合不同点云语义的数据。

示例 deployment YAML：

```yaml
deployment:
  runtime_target: dexmani_real.integrations.dexmani_policy:DexManiPolicyRuntime
  checkpoint: /path/to/checkpoints/best.pt
  model_config_path: /path/to/dp3.yaml
  action_key: action            # action | action_ee（须与 checkpoint 一致）
  hand_enabled: true
  observation_fields: arm_qpos,hand_qpos,point_cloud
  pointcloud_num_points: 2048
  replan_stride_steps: 8        # 不得超过模型 n_action_steps
  use_ema: true
```

点云缺失、过期、shape/dtype 错误、非有限值或颜色越界时 inference fail closed，不发布
新的 plan。实时路径当前仅支持静态 `eye_to_hand` 标定；`eye_in_hand` 需要另行建立与
相机帧同步的机械臂位姿合同。离线 IPC、合成 RGB-D 和已保存的真实 L515 帧均已验证；
实时 worker 的 compute/source-to-publish p95 仍须在完整硬件部署中记录。
`pointcloud_process_example.py` 同时报告纯构建与 capture-to-cloud 的 p50/p95/max，并报告
深度滤波、桌面裁减、反投影、体素和离群过滤等逐阶段 p95；纯构建目标为 p95 < 40 ms。

### L515 RGB/Depth 时序限制

暗场下 RGB 曾因 Auto-Exposure Priority=ON 把曝光拉到 ~60 ms 而降帧到 ~16.7 Hz。
根因已确认并修复：`RealSenseConfig.auto_exposure_priority` 默认 `0.0`（OFF，Auto
Exposure 仍 ON），RGB 恢复 30 Hz，亮度由增益补偿、几乎不变（暗场噪声上升）。

深度与颜色流仍然不是同时曝光（两路曝光/时间戳存在 skew）。processed v7 的点云将
depth-to-color aligned 像素 RGB 聚合为体素颜色，但这不表示同步曝光；运动物体仍可能出现
颜色时间错位。

RGB-D 点云/桌面交互诊断入口为 `examples/realsense_record_example.py` 与
`examples/pointcloud_process_example.py`。前者可在完整 RAW 点云和 canonical 处理后点云之间
切换：`p` 显示点云，`s` 切换 RAW/PROCESSED，`f/r` 分别冻结和重置。后者先询问是否执行
桌面标定；选择后会提示清空桌面并采集 5 帧，以确定性 RANSAC 重新拟合桌面。新拟合在本次
运行立即使用，只有操作者输入 `y` 才会备份并原子更新
`dexmani_real/config/desk_plane.json`。该文件同时服务感知裁减和桌面碰撞几何。

### 离线数据工作流

先审计一个任务目录或单个 episode，不写输出：

```bash
python examples/process_episodes.py \
  episodes/<task> \
  --dry-run
```

传入 `episodes/<task>/episode_*` 时只处理该 episode；传入 `episodes/<task>` 时处理其直接子目录中的全部 episode。

确认审计结果后去掉 `--dry-run`，默认发布到
`episodes_processed/<task>/`。默认 profile 为 `rgb_pc`；可通过
`--profile` 选择 `joint`、`rgb`、`pointcloud` 或 `rgb_pc`。后两种 profile 可用
`--pointcloud-num-points` 选择 `1024`、`2048`、`4096` 或 `8192`，默认 `1024`。

raw episode 可视化只展示文件实际存储的相机与状态数据。需要检查 canonical 点云时，先用
`process_episodes.py` 生成 processed HDF5 v7，再运行
`python examples/visualize_episode_processed.py <processed.h5>`；该入口会校验 `(N,6)`、
`float32`、xArm-base 坐标系、RGB 范围、算法/采样语义、配置哈希与桌面标定身份。

发布时逐 episode 显示 tqdm 进度（stderr），终端只打印精简汇总，不再向 stdout 输出 JSON。
需要机器可读报告时加 `--write-report`，发布成功后在
`episodes_processed/<task>/process_log/episode_*.json` 为每个 episode 落一份（含 config、
决策、输出与校验）；默认不生成。损坏或审计失败（硬无效帧过多、压紧产生危险跳变等）的
episode 会自动跳过并打印 warning 与原因；`--annotations` 显式 `include: true` 的 episode
不会被自动跳过，其失败会阻断整批。随后导出一个全新的 Zarr 目标：

```bash
python examples/export_policy_zarr.py \
  --input-root episodes_processed/<task> \
  --output dataset/<task>.zarr \
  --task-name <task>
```

输出目标已存在时会拒绝覆盖。可视化 raw episode：

```bash
python examples/visualize_episode.py episodes/<task>/episode_*
```

可视化处理后的 processed HDF5（rgb/depth 与预计算点云已在文件内，离线、不连硬件）：

```bash
python examples/visualize_episode_processed.py episodes_processed/<task>/episode_<timestamp>.h5
```

## 数据布局

```text
episodes/<task>/episode_<timestamp>/
├── data.h5       # fixed-grid robot/action/VR/quality data and metadata
├── depth.h5      # grid-aligned uint16 depth stream
└── rgb.mp4       # grid-aligned RGB stream

episodes_processed/<task>/
├── episode_<timestamp>.h5
└── process_log/                 # only with --write-report
    └── episode_<timestamp>.json

dataset/<task>.zarr/
├── data/*
└── meta/episode_ends
```

正式 raw writer 写 schema v22；depth 通过 SDK 对齐到 color 像素网格后保存，并同时保存两路帧号、设备时间戳、native intrinsics、distortion 与 `T_color_from_depth` 作为 provenance。其他 raw schema 需要在运行时之外显式迁移。离线处理默认保守：
硬无效行可被移除，时序异常先审计；压紧后产生危险动作跳变时默认拒绝该轨迹。

## 开发与验证

安全的最低成本检查：

```bash
python -m compileall -q dexmani_real examples tests
python -m pytest -q
python tools/check_architecture.py
git diff --check
git diff --stat
git status --short
```

`tests/` 提供纯函数、配置快照、IPC ABI、ring buffer、schema、生命周期、安全失败路径和
架构边界的离线回归门禁；`tests/fixtures/contracts/` 冻结需要显式审查的 ABI/schema/架构
基线。example 程序不是测试，测试和静态检查通过也不等于完成真实硬件验证。仓库级
agent 工作约定见 [`AGENTS.md`](AGENTS.md)，具体编码规范见
[`code_style.md`](code_style.md)。
