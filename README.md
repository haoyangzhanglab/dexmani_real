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
| 手部模型点云（完整导出或相应策略输入） | Trimesh；可选依赖组 `geometry`，沿用仓库 XHand URDF/mesh |
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

最短覆盖示例：[teleop](examples/teleop.yaml)、[sync](examples/sync.yaml)、[async](examples/async.yaml)、[RTC](examples/rtc.yaml)。采集入口另解析同一 YAML 的 `collection` 局部选项；部署入口由 artifact 决定控制 dt 和输入表示，执行窗口须满足该策略 horizon。

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

任务名默认 `test`，录制默认写入 `episodes/test/`，根目录由 `collection.episodes_dir` 或 `--episodes-dir` 配置。每段默认最多录制 60 秒，可用 `collection.max_duration_s` 或 `--max-duration` 修改。连接机器人前建立停止监听；手部 HOME 成功后启动 VR，待其就绪再等待操作者开始。

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

手部默认使用 TAG，`teleop.hand_retargeting_type: dexpilot` 切换到 DexPilot。位置映射与 EMA 分别由 `teleop.vr_mapping`、`teleop.ema` 配置；手启用开关为 `hand.enabled`，工作空间属于 `environment`，共用 IK 容差属于 `motion`。定义见[配置入口](dexmani_real/config/experiment.py)。

`--record/--no-record` 覆盖 `collection.enabled`。关闭录制时不启动相机，不应用录制时长限制，遥操作仍会执行；`--no-hand` 仅用于手未安装或已固定在 HOME 的调试，须同时关闭录制。

### 3. 离线查看与导出

```bash
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --info
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --max-frames 100
python scripts/export_zarr.py episodes/test --config local.yaml
```

`--info` 打印数据概要，无需启动 Rerun；交互查看可用 `--no-point-cloud` 关闭点云。查看和导出使用 Raw 保存的相机几何，以及当前点云、桌面、手部安装与指尖配置；仅启用去桌时读取桌面平面。

公共导出要求每段全部 **13 字段**完整、浮点数值有限，并通过真实时间、dispatch、触觉新鲜度及技术结束检查。包含两类触觉和手部模型点云；不导出 arm_qvel、arm_effort、hand_current，也不生成 object_pointcloud、segmentation 或 done。缺字段、NaN/Inf、RGB-D 行数不符、相机几何缺失或非法、空点云均整段拒绝，记录原因并保留原 Raw；任务失败本身不排除技术有效的数据。导出不删坏帧、不填补缺测、不修改 Raw。

默认输出 `datasets/test.zarr`，目标必须是 Raw 之外的新路径。`--output` 指定路径，`--exclude` 排除整段 episode，`--chunk-frames` 调整处理批大小，`--pointcloud-num-points` 覆盖点数。同一缓存要求 task、dt、depth scale 和数组 shape/dtype 一致；文件系统或处理异常保留 staging。

缓存从 Raw 重新导出至新路径，不自动改写旧缓存。Policy 按实际 episode 顺序、长度、训练划分与训练配方检查恢复条件。字段与数值表示见 [Canonical 定义](dexmani_real/data/canonical/schema.py)，处理与存储细节见[处理入口](dexmani_real/data/canonical/processing.py)和[导出入口](dexmani_real/data/canonical/export.py)。

在线与离线共用[点云处理](dexmani_real/sensor/pointcloud/processing.py)；部署采用训练 artifact 保存的完整点云 recipe 和当前现场标定，缺字段不使用当前默认值补齐。更换采样表示后须重新导出并训练，不能仅凭点数相同沿用模型。

`hand_pointcloud` 来自当前 XHand URDF/mesh 的固定表面样本，随实测关节 FK 变换为 base-frame XYZRGB，默认 1024 点、seed 0；`hand_pointcloud.num_points` 可明确设为 512。它使用固定材质色，不需要相机或分割。手点云 recipe 保存点数、种子、采样方法、link 选择及颜色规则；模型映射由当前源码与 URDF 定义，不计算资产哈希或比较历史模板身份。场景点云另存自身处理 recipe。depth 保留 uint16 原计数，米值为计数乘保存的 depth scale；不可统一当作毫米。

### 4. 策略部署

```bash
python scripts/run_policy.py policy/task/experiment --config local.yaml
python scripts/run_policy.py policy/task/experiment --config local.yaml --execution-mode async --prefetch-steps 1 --n-action-steps 4
```

实验输入为不带 `experiments/` 前缀的 `policy/task/experiment`，也可使用有效的绝对实验目录。策略 artifact 提供模型输入、动作表示和控制 dt；Real 提供当前设备、标定与执行配置。每段执行步数按 `--n-action-steps` > YAML 的 `execution.action_steps` > 策略默认值确定。

CLI 与 Python 调用共用 [run_policy](dexmani_real/deployment/run.py) 入口；Python 调用通过 `runtime.execution.action_steps` 指定执行步数，留空时使用策略默认值。入口先完成输入与调度参数预检，再保存配置和准备资源。加载后的 Policy 在连接设备前检查模型是否支持所选模式。修改交接、预取和时隙计算时，从无设备 I/O 的 [schedule.py](dexmani_real/deployment/schedule.py) 开始；动作转换与目标准入位于 [planning/action.py](dexmani_real/planning/action.py)。

[会话流程](dexmani_real/deployment/session.py)负责设备、操作员输入和多轮实验；[单轮流程](dexmani_real/deployment/episode.py)负责准备、推理调度、执行和当前 capture 的结束。停止立即撤销软件运动授权，取消中的模型准备无需等待 reset 返回即可结束。旧模型任务成功完成并回收后，才接受下一轮开始；迟到模型异常会结束会话，并保留原有结束事实和已发布 Raw。

| 执行模式 | 行为与配置 |
| --- | --- |
| `sync`（默认） | 执行一段后等待下一次推理，保留真实段间等待 |
| `async` | 当前段执行时预取下一段；推理迟到时继续消费未过期的旧预测尾部，需设置 `prefetch_steps` |
| `rtc` | 预取并冻结交接前已承诺的动作前缀，需设置 `prefetch_steps` 与 `rtc_guidance_cap` |

预取步数表示距交接还剩多少个控制步时启动下一次推理，须满足策略 horizon 的容量约束。[Policy 部署入口](https://github.com/haoyangzhanglab/dexmani_policy/tree/main/dexmani_policy/deployment)检查训练 artifact 和模型能力，[Real 预检](dexmani_real/deployment/preflight.py)负责现场输入与执行预算，具体选项见 `--help`。LAMP 真机闭环尚未验证；离线预检不作为真机验证。

H 回 HOME、B 开始本轮、S 结束本轮；整理场景后再次按 H、B 开始下一轮。Q 提前退出，ESC 请求急停，T 在空闲且无接触时请求复位触觉。部署不使用 C/D。

普通 `async` 的 `action_steps` 是名义执行/重规划长度；新结果提前完成仍等待计划交接，迟到接管时跳过新预测中已错过的动作。旧预测耗尽或决策过期时结束本轮，不重复末动作、不补发、不延长有效期。执行模式的任务效果需要在对应实验中评估。

默认最多 20 轮正式运行，每轮运行预算 60 秒，分别用 `--num-episodes`、`--max-duration` 修改；未进入正式运行的失败或取消准备不占用轮数。这些预算不保证阻塞 SDK 或模型调用能立即返回。

checkpoint 默认 `latest`，支持 `best`、已有百分比 milestone（如 `80pct`）及实验 `checkpoints/` 内的路径；由 Policy 统一解析。`--weights`、`--inference-steps` 可覆盖推理设置。

默认产物位于 `rollouts/policy/task/experiment/session_*/`，`--output` 指定根目录；`run_config.yaml` 保存实际配置与模型信息，`session_result.json` 保存评估摘要。会话关闭同时报告未回收模型任务和模型关闭的结果，模型关闭成功不会掩盖旧任务失败。

Rollout Raw 按会话内录制启动尝试依次命名为 `episode_001`、`episode_002` 等。准备失败或取消时，已启动且正常收尾的采集继续保存，包括零行采集；后续启动使用新编号。评估摘要区分是否发布与实际行数：零行不等于未发布。查找采集时使用摘要中的 Raw 路径，不能按正式运行轮次推算目录名。

`--no-record` 与 `--no-results` 分别关闭 Raw 和摘要，同时关闭时不创建会话产物；**它们不会关闭设备连接或动作执行**。相机按策略输入与 Raw 录制需求启动，仅使用机器人状态且关闭 Raw 时不启动相机。

无录制时，手点云仅准备 FK/模板，depth-only 仅消费深度和来源 header，K/E-only 不复制图像；录制开启时同时保存关节、两类触觉与 RGB-D。depth 与手点云输入需要完整 artifact recipe 及真实模型支持；当前模型未声明支持的字段在连接设备前拒绝，不因 Real 已能生成该字段而放宽模型检查。部署 depth 保持训练的 raw counts 表示，并在相机 ready 后核对实际 scale。

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

键盘 jog 与相机标定在连接机器人之前启动退出/急停监听；连接和相机就绪等待期间的取消会撤销软件运动授权。

## 数据与安全边界

**Raw 发布后不可变，Canonical 是可重建的训练缓存。** Raw episode 使用 `data.h5` 和 `rgb.mp4`，保留必要物理观测、目标、真实 dispatch、来源时钟和终止原因。RGB 与深度按保存行对齐，实际采样时刻以 HDF5 时间为准；视频使用标称帧率编码。读取和导出使用当前格式，不转换旧时间布局或补齐缺失元数据，不改写或迁移已发布 Raw。

停止、晚帧、控制失败保留已采前缀及结束原因；仅操作者显式重录或丢弃才删除当前 capture。写盘、编码、close 或发布失败保留 staging。I/O 完整的停止未确认或部分发送记录可以发布 Raw，但保留失败事实，并拒绝严格导出或回放；文件发布不证明真实停止。

规划、IK、工作空间、静态障碍和相机外参统一使用 xArm-base 坐标。RGB-D 使用对齐后的 color 像素网格和内参，深度使用数据保存的 depth scale；Raw 外参保存 color→xArm-base 变换，Canonical 和模型的 `camera_extrinsic` 使用其逆变换 base→color 的 `[R|t]`。

训练 action 表示本拍提交的绝对关节目标，`action_ee` 是该目标的 FK，均不代表下一拍已到达。Rollout Raw 保留 bootstrap/WAIT 的无动作行及终止原因，通常会被公共导出拒绝；不能通过填动作或裁行冒充当前训练格式。

聚合与 taxel 两类触觉均按明确时限保持上一有效三轴向量，以各自最老原有效时间判断过期；Raw 保存消费值、来源时间和处理语义，导出不再次修补或扣偏置。触觉沿用 SDK 原生数值，未经独立标定不能称为 Newton 或 Sim 等价接触力。arm_qvel、arm_effort、hand_current 不默认持久化；控制所需的反馈继续读取。

运动入口检查关节限位、FK 工作空间、自碰撞与静态障碍，并按入口采用相应手形和桌面接触规则；整机 HOME 额外检查路径并等待实测稳定。目标准入、SDK 接受、实测到位和真实停止是不同结果；`CRC_UNCONFIRMED` 不代表 ACK 或关节到位。

HOME 限于受限桌面场景，依次检查 direct、proximal_first、distal_first 三条确定性路线，并保留必要的等价角对齐。候选全部失败时明确拒绝，需要操作者处理；不提供任意障碍环境中的自动绕行。

xArm 控制器碰撞检测默认关闭（`arm.collision_sensitivity: 0`），不能依赖它触发碰撞停机；软件几何检查仍执行。真机运行前确认安装、标定、限位和工作空间，保持操作者监督及硬件急停可用。SDK 或 CUDA 阻塞可能影响停止响应，离线检查不能证明真机安全。

## 开发

默认运行无设备副作用的检查：

```bash
python -m compileall -q dexmani_real scripts
git diff --check
```

编译检查不证明真机安全；几何、配置与数据处理等行为按具体改动补充纯离线验证。代码代理的长期规则见 [AGENTS.md](AGENTS.md)，Claude 阅读入口见 [CLAUDE.md](CLAUDE.md)。
