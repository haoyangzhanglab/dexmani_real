# DexMani Real

面向个人 PhD 论文实验的真实机器人灵巧操作仓库，当前硬件为 **xArm7 + XHand + RealSense RGB-D + VR / HTS**。

本仓负责设备接入、标定、遥操作采集、Raw 数据、离线处理与真机部署。模型与训练见 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真见 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

**标定 → 遥操作采集 → Raw → Canonical Zarr → 策略训练 → 真机评估**

## 安装与配置

在已有的 Python ≥ 3.10 实验环境中，从仓库根目录安装；机器人资产从源码 checkout 路径读取：

```bash
python -m pip install -e ".[interactive,control]"
```

纯离线处理可用 `python -m pip install -e .`；按需追加可选依赖组 `geometry`（手部点云）和 `visualization`（Rerun 查看工具）。设备 SDK、Pinocchio、NLopt 和策略推理依赖沿用实验环境；笛卡尔在线 IK 另需 MPlib，键盘交互需要桌面会话。完整依赖见 [pyproject.toml](pyproject.toml)。

配置优先级为 **CLI 覆盖 > `--config` YAML > 包内默认值**。先查看配置，再将实验覆盖项保存为 `local.yaml`：

```bash
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
```

`--print-config` 不连接设备、不加载策略或现场标定。使用前核对设备地址、serial、工作空间及标定路径，配置字段见 [experiment.py](dexmani_real/config/experiment.py)。日志默认写入 `~/.dexmani/logs/`，可用 `DEXMANI_LOG_DIR` 调整。

## 实验工作流

以下 `test`、episode 目录和 `policy/task/experiment` 均为示例，使用时替换并准备 `local.yaml`。各入口参数见 `--help`。

**标定、采集、部署、回放和实时诊断会连接真实设备。** 运行前确认安装、标定、限位与工作空间，保持操作者监督及硬件急停可用。xArm 控制器碰撞检测默认关闭（`arm.collision_sensitivity: 0`）；软件检查和离线验证不能证明真机安全。

### 1. 标定

物理设置变化后，重新标定或确认相机外参、桌面平面、手部安装与 VR alignment。相机采用固定安装的 eye-to-hand 外参，按 serial 绑定。

| 项目 | 入口与要求 |
| --- | --- |
| 相机外参 | [calibrate_camera.py](scripts/calibrate_camera.py)；会移动机械臂，须用 `--hand-geometry` 声明手未安装（`absent`）或已物理固定在配置 HOME 手形（`secured-home`） |
| VR 朝向 | [calibrate_vr_heading.py](scripts/calibrate_vr_heading.py)；`--ref head` 面向机器人 +X，`--ref wrist` 将手腕指向机器人 +X |
| 桌面平面 | [view_pointcloud.py](scripts/tools/view_pointcloud.py)；选择 table calibration，确认后发布结果 |

相机标定 jog 允许桌面接触，HOME 单独检查路径净空；采样、求解或目标被拒绝后，须释放全部 jog 键再继续移动。采集可用 `--vr-transform`、`--camera-calibration` 指定标定，部署支持后者；桌面路径由 `environment.table.plane_path` 指定。

### 2. 遥操作与采集

```bash
python scripts/collect_teleop.py test --config local.yaml
```

启动先执行手部 HOME。默认录制到 `episodes/test/`，每段最多 60 秒，控制与录制频率为 16 Hz；可通过 `--episodes-dir`、`--max-duration`、`--control-hz` 或配置调整。改变频率需核对反馈新鲜度预算。手部默认使用 TAG，可配置为 DexPilot。

| 按键 | 行为 |
| --- | --- |
| H | 保存并结束当前段，整机 HOME 后等待 B |
| B | 空闲时开始新一段；启用录制时创建 capture |
| S | 停止并保存，回到空闲 |
| D | 丢弃当前未发布 capture，回到空闲 |
| T | 空闲且无 capture 时请求复位触觉，操作前确保五指无接触 |
| Q / ESC | Q 停止、保存并退出；ESC 请求急停 |

`--no-record` 关闭录制、相机和录制时长限制，**遥操作仍会执行**。`--no-hand` 仅用于手未安装或已固定在 HOME 的调试，须同时关闭录制。

### 3. 离线查看与导出

```bash
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --info
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --max-frames 100
python scripts/export_zarr.py episodes/test --config local.yaml
```

`--info` 无需 Rerun；交互查看可用 `--no-point-cloud` 关闭点云，缺相机外参时仍可查看 RGB-D 和机器人状态。点云处理使用 Raw 保存的相机几何与当前现场配置。

导出默认写入 `datasets/test.zarr`，`--output` 必须指向 Raw 之外的新路径。仅导出字段完整、数值和来源时间有效且采集技术正常的整段数据；无效段拒绝并记录原因，其余段继续，I/O 或处理错误则中止并保留 staging。字段和处理逻辑见 [Canonical](dexmani_real/data/canonical/schema.py) 与[导出实现](dexmani_real/data/canonical/export.py)；缓存变化后从 Raw 重新导出。

### 4. 策略部署

```bash
python scripts/run_policy.py policy/task/experiment --config local.yaml
python scripts/run_policy.py policy/task/experiment --config local.yaml --execution-mode async --prefetch-steps 1 --n-action-steps 4
```

实验参数为不带 `experiments/` 前缀的 `policy/task/experiment`，或绝对实验目录。`policy` 是方法身份（如 `sat`），`ddp/sat` 仅用于 Policy 的 Hydra 配置选择与 DDP 启动；新 DDP 实验同样以 `sat/task/experiment` 部署。训练 artifact 决定输入、预处理、动作表示和控制周期；Real 提供当前设备、标定与执行配置。

默认 checkpoint 为 `latest`；选择 `--checkpoint best` 时，EMA/raw 与 NFE 默认取选点记录，可由 `--weights`、`--inference-steps` 覆盖。Real 执行长度取 YAML/CLI 的 `execution.action_steps`（CLI 为 `--n-action-steps`），未设置时取保存的 `n_action_steps`，不继承仿真选点的执行长度。默认使用 `sync`；`async` 需设置 `prefetch_steps`，`rtc` 还需设置 `rtc_guidance_cap`，并由模型支持。执行参数见 [execution.py](dexmani_real/config/execution.py)。LAMP 真机闭环尚未验证。

H 回 HOME、B 开始本轮、S 结束本轮，整理场景后再次按 H、B；Q 退出，ESC 请求急停。T 仅在空闲、无 capture 且五指无接触时请求复位触觉。默认最多 20 轮、每轮 60 秒，可用 `--num-episodes`、`--max-duration` 调整。

产物默认位于 `rollouts/policy/task/experiment/session_*/`，包含实际配置、评估摘要与 Raw。方法、任务与 run 均须为单层目录名，输出不得通过符号链接逃逸 `--output` 根目录；不兼容带 `ddp/` 的旧方法名或迁移旧产物。`--no-record` 仅关闭 Raw，**仍连接设备、执行动作并保存配置和结果**。从 `session_result.json` 查看本轮收尾时记录的结果与 Raw 路径。

从本仓库根目录、使用已安装两个仓库及 Real 离线依赖（含 fpsample）的环境执行：

```bash
python scripts/check_deployment.py
```

该入口使用临时产物与 PolicySession 替身，检查 DDP 方法名到会话目录、路径拒绝、best 推理默认值与 Real 执行参数的职责及既有模式/模态限制；不加载模型、不连接设备或外部服务。Policy 侧的全部 DDP 配方和原启动入口检查见[验证指南](../dexmani_policy/docs/验证指南.md#基础设施与跨仓库离线检查)。通过只证明名称与接口一致性，不代表所有方法支持真机部署或已验证真机安全。

### 5. 真机轨迹回放

```bash
python scripts/replay_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --output replay_results/test_run
```

回放技术正常且目标完整的 teleop Raw，连接机械臂与手，不启动相机或 VR。启动先执行手部 HOME，再默认用 0.5 秒渐变到首帧手部目标；`--hand-start-duration 0` 仅跳过渐变。

输出目录须新建或为空，保存结果、回放数据与可计算的误差指标。Q 保存前缀并退出，ESC 请求急停，完成后可按 H 回 HOME；输入 Raw 保持不变。实现见 [replay/](dexmani_real/replay/)。

## 设备工具

| 用途 | 入口与边界 |
| --- | --- |
| 键盘 jog / HOME | [keyboard_teleop.py](scripts/keyboard_teleop.py)；连接机器人，R 执行 HOME，不启动相机或 VR |
| XHand 动作与反馈 | [control_xhand.py](scripts/tools/control_xhand.py)；默认执行动作，`--read-only` 仍连接手 |
| RGB-D 诊断 | [view_realsense.py](scripts/tools/view_realsense.py)；连接相机，交互查看 RGB-D 与点云 |
| 点云诊断 / 桌面标定 | [view_pointcloud.py](scripts/tools/view_pointcloud.py)；连接相机，`--save-dir` 保存诊断数据 |
| 物体重建与跟踪 | [CORDViP 示例](scripts/cordvip/README.md)；独立依赖，按说明区分离线与实时模式 |

查看工具统一使用 Rerun，冻结、时间浏览与显隐在 viewer 内操作。关闭 viewer 不会停止实时相机；按终端提示退出。键盘 jog 与相机标定被拒绝移动后，须释放全部 jog 键才能重新操作。

## 数据与安全边界

- **Raw 发布后不可变，Canonical 是可重建的训练缓存。** Raw 使用 `data.h5` 和 `rgb.mp4`，保存实测观测、最终提交尝试目标、来源时间及结束原因；发布不证明任务成功、目标送达或设备停止。
- 停止、晚帧和控制失败保留已采前缀；仅操作者显式重录或丢弃时删除当前 capture。写入、编码、关闭或发布失败保留 staging；收尾为 `pending` 不能视为资源已关闭，须查看最终结果。
- 技术失败或缺目标的采集不能严格导出或回放；不补模态、不删坏帧或填动作来满足训练要求。字段、坐标与时间语义见 [Raw 定义](dexmani_real/data/raw/schema.py)，收尾行为见 [Recorder](dexmani_real/data/recorder.py)。
- HOME 使用当前工作空间、障碍与桌面配置，无可行路径时拒绝。晚帧、反馈过期或决策过期时停止，不追赶补发；SDK 或 CUDA 阻塞仍可能影响停止响应。

## 代码导航

| 职责 | 起点 |
| --- | --- |
| 配置与标定 | [config/experiment.py](dexmani_real/config/experiment.py)、[calibration/](dexmani_real/calibration/) |
| 设备与进程通信 | [robot/](dexmani_real/robot/)、[sensor/](dexmani_real/sensor/)、[ipc/](dexmani_real/ipc/) |
| 遥操作与运行 | [teleop/](dexmani_real/teleop/)、[runtime/](dexmani_real/runtime/) |
| 采集与数据处理 | [data/](dexmani_real/data/) |
| 运动规划与运动学 | [planning/](dexmani_real/planning/) |
| 部署与回放 | [deployment/](dexmani_real/deployment/)、[replay/](dexmani_real/replay/) |

开发约定见 [AGENTS.md](AGENTS.md)，Claude 阅读入口见 [CLAUDE.md](CLAUDE.md)。具体行为以当前源码与解析后的配置为准。
