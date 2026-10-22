# DexMani Real

面向个人 PhD 研究的真实机器人灵巧操作仓库，当前实验系统为 **xArm7 + XHand + RealSense RGB-D + VR / HTS**。

本仓负责设备接入、标定、遥操作采集、Raw 数据、离线处理及策略真机部署。模型、Dataset 与训练位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真位于 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

工作流：**标定 → 遥操作采集 → Raw → Canonical Zarr → 策略训练 → 真机评估**。

## 目录导航

| 目录 | 职责 |
| --- | --- |
| [robot/](dexmani_real/robot/)、[sensor/](dexmani_real/sensor/) | 设备接口、配置与采集 worker |
| [runtime/](dexmani_real/runtime/)、[teleop/](dexmani_real/teleop/)、[deployment/](dexmani_real/deployment/)、[replay/](dexmani_real/replay/) | 运行协调、遥操作、部署与回放 |
| [data/](dexmani_real/data/) | 异步录制；`raw/` 原始记录与相机几何；`canonical/` 派生处理与导出 |
| [calibration/](dexmani_real/calibration/)、[config/](dexmani_real/config/)、[planning/](dexmani_real/planning/) | 现场标定、实验配置与运动规划 |
| [ipc/](dexmani_real/ipc/)、[utils/](dexmani_real/utils/) | 进程间通信与通用辅助函数 |
| [scripts/](scripts/) | 主要运行入口；[tools/](scripts/tools/) 设备与感知工具；[cordvip/](scripts/cordvip/README.md) 物体重建与跟踪 |

## 安装与配置

使用 Python ≥ 3.10，在仓库根目录以 editable 方式安装；机器人资产按源码 checkout 路径读取：

```bash
python -m pip install -e ".[interactive,control]"
```

仅需基础离线处理依赖时使用 `python -m pip install -e .`。依赖声明见 [pyproject.toml](pyproject.toml)，硬件和推理后端沿用实验机环境：

| 功能 | 额外依赖 |
| --- | --- |
| xArm、XHand、RealSense、VR / HTS | 对应 SDK、驱动及设备连接配置 |
| FK、IK、碰撞检查与 HOME 规划 | Pinocchio、MPlib |
| TAG / DexPilot 手部重定向 | NLopt 及所选后端的几何依赖 |
| 点云诊断视图 / episode 交互查看 | Open3D / Rerun |
| 策略部署 | 已安装的 `dexmani_policy` 及训练实验对应的推理依赖 |

键盘交互使用 pynput，需要桌面显示会话。安装不验证设备连接或现场标定。

配置优先级为 **CLI 覆盖 > `--config` YAML > 包内默认值**。先查看完整声明配置：

```bash
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
```

将本次实验需要覆盖的字段保存为 `local.yaml`，例如：

```yaml
teleop:
  control_hz: 16.0
pointcloud:
  num_points: 1024
```

```bash
python scripts/collect_teleop.py --config local.yaml --print-config
```

字段以当前输出及[配置加载入口](dexmani_real/config/experiment.py)为准，未知字段直接报错。示例 YAML 仅展示覆盖方式，运行前仍须核对设备地址、serial、工作空间与现场标定。

XHand 默认通过 `/dev/ttyUSB0` 串口连接。使用 EtherCAT 时，同时设置 `hand.comm_type: ethercat` 和 `hand.device_name` 为实际网卡接口名；显式设置 `hand.device_name: null` 才使用所选传输的设备发现，各入口均不自动改写设备名。

各入口的 `--help` 不连接设备；支持 `--print-config` 的入口仅打印声明配置，不读取现场标定或加载策略，也不证明运行条件已满足。日志默认写入 `~/.dexmani/logs/`，可用 `DEXMANI_LOG_DIR` 指定位置。

## 实验工作流

以下任务名 `test`、episode 目录及 `policy/task/experiment` 均为示例，使用时替换为实际名称；`local.yaml` 须提前准备。

### 1. 标定

相机外参、桌面平面、手部安装与 VR alignment 属于当前实验状态；物理设置变化后重新标定或确认。相机采用固定安装的 eye-to-hand 外参，按设备 serial 绑定；坐标约定见[数据与安全边界](#数据与安全边界)。

| 标定 | 入口与要求 |
| --- | --- |
| 相机外参 | [calibrate_camera.py](scripts/calibrate_camera.py)；会连接并移动机械臂、采集图像，须声明 `--hand-geometry absent` 或 `secured-home` |
| VR 朝向 | [calibrate_vr_heading.py](scripts/calibrate_vr_heading.py)；`--ref head` 面向机器人 +X，`--ref wrist` 将手腕指向机器人 +X |
| 桌面平面 | [view_pointcloud.py](scripts/tools/view_pointcloud.py)；启动后选择 table calibration，确认后发布拟合结果 |

先查看各入口的具体操作：

```bash
python scripts/calibrate_camera.py --help
python scripts/calibrate_vr_heading.py --help
python scripts/tools/view_pointcloud.py --help
```

相机标定中的 `absent` 表示手未安装，`secured-home` 表示已物理固定在配置 HOME 手形；两者均按固定 HOME 手形做碰撞检查。标定在线 jog 允许桌面接触，HOME 独立检查桌面路径净空。相机与 VR 标定支持 `--output`；桌面文件由 `environment.table.plane_path` 指定。

采集可通过 `--vr-transform`、`--camera-calibration` 选择标定，部署支持 `--camera-calibration`。采集允许缺相机外参，点云重建须使用数据对应的相机几何。VR 质量为 `poor` 的标定不能用于遥操作启动。具体采样和求解操作以入口提示为准，采样或求解后先释放全部 jog 键再继续移动。

### 2. 遥操作与采集

```bash
python scripts/collect_teleop.py test --config local.yaml
python scripts/collect_teleop.py test --config local.yaml --control-hz 20
```

任务名默认 `test`，录制默认写入 `episodes/test/`，根目录由 `control.episodes_dir` 配置。启动时建立停止监听并执行手部 HOME，随后等待操作者开始。

| 按键 | 遥操作行为 |
| --- | --- |
| H | 保存并结束当前段，执行整机 HOME，完成后等待新的 B |
| B | 开始采集 |
| C | 暂停或恢复；恢复开始新的 episode |
| S | 停止并保存，回到空闲 |
| D | 丢弃当前未发布 capture，回到空闲 |
| T | 空闲且无 capture 时请求复位触觉传感器；操作前确保五指无接触 |
| Q | 暂停并进入退出确认，再按 Q 保存退出；空闲且无 capture 时直接退出 |
| ESC | 请求急停 |

`--control-hz` 同时设置目标、观测采样与录制频率，默认 16 Hz；机械臂伺服频率独立。调频须满足设备与反馈新鲜度预算，并会改变 EMA 的时间响应，应记录实际配置。错过时隙或反馈过期时停止并保留真实前缀，不追赶补发。

手部默认使用 TAG，`control.hand_retargeting_type: dexpilot` 切换到 DexPilot。位置映射与 EMA 分别由 `control.vr_mapping`、`control.ema` 配置，定义见[控制配置](dexmani_real/config/control.py)。

`--no-record` 关闭录制及相机采集，遥操作仍会执行；`--no-hand` 仅用于手未安装或已固定在 HOME 的调试，须同时关闭录制。

### 3. 离线查看与导出

```bash
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --info
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --max-frames 100
python scripts/export_zarr.py episodes/test --config local.yaml
```

`--info` 打印数据概要，无需启动 Rerun；交互查看可用 `--no-point-cloud` 关闭点云。查看和导出使用 Raw 保存的相机几何，以及当前点云、桌面、手部安装与指尖配置；仅启用去桌时读取桌面平面。

公共导出要求每段全部 **15 字段**完整、浮点数值有限，并通过时序及动作执行证据检查。缺字段、NaN/Inf、RGB-D 行数不符、相机几何缺失或非法、空点云均整段拒绝，记录原因并保留原 Raw；任务失败本身不排除技术有效的数据。导出不删坏帧、不填补缺测、不修改 Raw。

默认输出 `datasets/test.zarr`，目标必须是 Raw 之外的新路径。`--output` 指定路径，`--exclude` 排除整段 episode，`--chunk-frames` 调整处理批大小，`--pointcloud-num-points` 覆盖点数。同一缓存要求 task、dt、depth scale 和数组 shape/dtype 一致；文件系统或处理异常保留 staging。

缓存从 Raw 重新导出至新路径，不自动改写旧缓存。新导出生成新的 `data_revision`，已有训练产物仍须通过来源检查，不得复制旧 revision 绕过检查。字段与数值表示见 [Canonical 定义](dexmani_real/data/canonical/schema.py)，处理与存储细节见[处理入口](dexmani_real/data/canonical/processing.py)和[导出入口](dexmani_real/data/canonical/export.py)。

在线与离线共用[点云处理](dexmani_real/sensor/pointcloud/processing.py)；部署采用训练 artifact 保存的点云 recipe 和当前现场标定。更换采样表示后须重新导出并训练，不能仅凭点数相同沿用模型。

### 4. 策略部署

```bash
python scripts/run_policy.py policy/task/experiment --config local.yaml
python scripts/run_policy.py policy/task/experiment --config local.yaml --execution-mode async --prefetch-steps 1 --n-action-steps 4
```

实验输入为不带 `experiments/` 前缀的 `policy/task/experiment`，也可使用有效的绝对实验目录。策略 artifact 提供模型输入、动作表示和控制 dt；Real 提供当前设备、标定与执行配置。每段执行步数按 `--n-action-steps` > YAML 的 `execution.action_steps` > 策略默认值确定。

| 执行模式 | 行为与配置 |
| --- | --- |
| `sync`（默认） | 执行一段后等待下一次推理，保留真实段间等待 |
| `async` | 当前段执行时预取下一段，需设置 `prefetch_steps` |
| `rtc` | 预取并冻结交接前已承诺的动作前缀，需设置 `prefetch_steps` 与 `rtc_guidance_cap` |

预取步数表示距交接还剩多少个控制步时启动下一次推理，须满足策略 horizon 的容量约束。策略与模式兼容性由 [Real 预检](dexmani_real/deployment/preflight.py)和 [Policy 部署边界](https://github.com/haoyangzhanglab/dexmani_policy/blob/main/docs/项目架构.md#部署边界)检查，具体选项见 `--help`。LAMP 真机闭环尚未验证；离线预检不作为真机验证。

H 回 HOME、B 开始本轮、S 结束本轮；整理场景后再次按 H、B 开始下一轮。Q 提前退出，ESC 请求急停，T 在空闲且无接触时请求复位触觉。部署不使用 C/D。

默认最多 20 轮正式运行，每轮运行预算 60 秒，分别用 `--num-episodes`、`--max-duration` 修改；未进入正式运行的失败或取消准备不占用轮数。这些预算不保证阻塞 SDK 或模型调用能立即返回。

checkpoint 默认 `latest`，支持 `best`、已有百分比 milestone（如 `80pct`）及实验 `checkpoints/` 内的路径；由 Policy 统一解析。`--weights`、`--inference-steps` 可覆盖推理设置。

默认产物位于 `rollouts/policy/task/experiment/session_*/`，`--output` 指定根目录；`run_config.yaml` 保存实际配置与模型信息，`session_result.json` 保存评估摘要。

Rollout Raw 按会话内录制启动尝试依次命名为 `episode_001`、`episode_002` 等。准备失败或取消时，已启动且正常收尾的采集继续保存，包括零行采集；后续启动使用新编号。评估摘要保留准备尝试，查找采集时使用摘要中的 Raw 路径，不能按正式运行轮次推算目录名。

`--no-record` 与 `--no-results` 分别关闭 Raw 和摘要，同时关闭时不创建会话产物；**它们不会关闭设备连接或动作执行**。相机按策略输入与 Raw 录制需求启动，仅使用机器人状态且关闭 Raw 时不启动相机。

### 5. 真机轨迹回放

```bash
python scripts/replay_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --output replay_results/test_run
```

回放读取当前格式的 teleop Raw，按标称节拍执行已记录关节目标；会连接机械臂与手，不启动相机或 VR。启动先执行手部 HOME，再默认用 0.5 秒渐变到首帧手部目标；`--hand-start-duration 0` 仅跳过渐变。

输出目录须新建或为空，保存 `session_result.json`、`replay_data.npz` 和可计算的 `metrics.json`。Q 保存前缀并退出，ESC 请求急停，完成后可按 H 规划回 HOME。晚帧或发送超时会停止，输入 Raw 保持不变；结果中的轨迹帧偏移用于跨运行对齐，不表示硬件响应延迟。

## 设备工具与其他入口

| 用途 | 入口 | 运行边界 |
| --- | --- | --- |
| 键盘 jog / HOME | [keyboard_teleop.py](scripts/keyboard_teleop.py) | 连接机器人，不启动相机或 VR；净运动意图归零停止 jog，R 执行 HOME |
| XHand 动作与反馈 | [control_xhand.py](scripts/tools/control_xhand.py) | 默认执行动作；`--read-only` 仍连接手并读取反馈 |
| RGB-D 诊断 | [view_realsense.py](scripts/tools/view_realsense.py) | 连接相机，交互查看 RGB-D 与点云 |
| 点云诊断 / 桌面标定 | [view_pointcloud.py](scripts/tools/view_pointcloud.py) | 连接相机；`--save-dir` 保存本次 RGB-D、点云及重建 metadata |
| 物体重建与跟踪 | [CORDViP 示例](scripts/cordvip/README.md) | 使用独立的 `cordvip_geometry` 依赖与入口，区分离线数据和实时相机模式 |

各入口的参数与按键以 `--help` 和交互提示为准。键盘遥操作与相机标定中，触界、IK 或准入拒绝后须释放全部 jog 键才能重新操作。

## 数据与安全边界

**Raw 发布后不可变，Canonical 是可重建的训练缓存。** Raw episode 使用 `data.h5` 和 `rgb.mp4`，保留物理观测、目标、真实 dispatch、时钟和终止原因。RGB 与深度按保存行对齐，实际采样时刻以 HDF5 时间为准；视频使用标称帧率编码。历史格式使用对应版本代码读取，不改写 Raw 来满足新格式。

停止、晚帧、控制失败保留已采前缀及结束原因；仅操作者显式重录或丢弃才删除当前 capture。写盘、编码、发布或执行结束无法确认时保留 staging，不能报告为完整 Raw。

规划、IK、工作空间、静态障碍和相机外参统一使用 xArm-base 坐标。RGB-D 使用对齐后的 color 像素网格和内参，深度使用数据保存的 depth scale；Raw 外参保存 color→xArm-base 变换，Canonical 和模型的 `camera_extrinsic` 使用其逆变换 base→color 的 `[R|t]`。

训练 action 表示本拍提交的绝对关节目标，`action_ee` 是该目标的 FK，均不代表下一拍已到达。Rollout Raw 保留 bootstrap/WAIT 的无动作行及终止原因，通常会被公共导出拒绝；不能通过填动作或裁行冒充当前训练格式。

触觉按明确时限保存上一有效三轴向量保持后的消费值，保留原有效时间，过期停止；导出直接使用保存值，不再次修补或扣偏置。hand current / arm effort 等辅助缺测保留 NaN，这些 Raw 可留存，但不满足完整导出要求。

运动入口检查关节限位、FK 工作空间、自碰撞与静态障碍，并按入口采用相应手形和桌面接触规则；整机 HOME 额外检查路径并等待实测稳定。目标准入、SDK 接受、实测到位和真实停止是不同结果；`CRC_UNCONFIRMED` 不代表 ACK 或关节到位。

xArm 控制器碰撞检测默认关闭（`arm.collision_sensitivity: 0`），不能依赖它触发碰撞停机；软件几何检查仍执行。真机运行前确认安装、标定、限位和工作空间，保持操作者监督及硬件急停可用。SDK 或 CUDA 阻塞可能影响停止响应，离线检查不能证明真机安全。

## 开发

默认运行无设备副作用的检查：

```bash
python -m compileall -q dexmani_real scripts
git diff --check
```

按改动补充纯逻辑、假时钟或 SDK/model 替身验证，本仓库不使用 pytest 或 Ruff。代码代理的长期规则见 [AGENTS.md](AGENTS.md)，Claude 阅读入口见 [CLAUDE.md](CLAUDE.md)。
