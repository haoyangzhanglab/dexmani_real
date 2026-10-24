# DexMani Real

面向个人 PhD 论文实验的真实机器人灵巧操作仓库，当前系统为 **xArm7 + XHand + RealSense RGB-D + VR / HTS**。以科学正确性、实验复现和研究迭代为目标，编码与协作约定见 [AGENTS.md](AGENTS.md)。

本仓负责设备接入、标定、遥操作采集、Raw 数据、离线处理及策略真机部署。模型、Dataset 与训练位于 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy)，仿真位于 [dexmani_sim](https://github.com/haoyangzhanglab/dexmani_sim)。

**标定 → 遥操作采集 → Raw → Canonical Zarr → 策略训练 → 真机评估**

## 安装与配置

在已有的 Python ≥ 3.10 实验环境中，从仓库根目录安装；机器人资产按源码 checkout 路径读取：

```bash
python -m pip install -e ".[interactive,control]"
```

仅需基础离线处理依赖时使用 `python -m pip install -e .`。依赖声明见 [pyproject.toml](pyproject.toml)，设备和推理后端沿用实验机环境：

| 功能 | 额外依赖 |
| --- | --- |
| xArm、XHand、RealSense、VR / HTS | 对应 SDK、驱动及设备配置 |
| FK、碰撞检查与 HOME 规划 | Pinocchio |
| 笛卡尔目标在线 IK | Pinocchio、MPlib |
| 手部模型点云 | Trimesh，可选依赖组 `geometry` |
| TAG / DexPilot 重定向 | NLopt 及所选后端的几何依赖 |
| 点云诊断 / episode 交互查看 | Open3D / Rerun |
| 策略部署 | 已安装的 `dexmani_policy` 及训练实验对应的推理依赖 |

离线 FK、HOME 规划和关节动作部署无需 MPlib。键盘交互使用 pynput，需要桌面显示会话。安装不验证设备连接或现场标定。

配置优先级为 **CLI 覆盖 > `--config` YAML > 包内默认值**。先查看声明配置：

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

字段以当前输出及[配置入口](dexmani_real/config/experiment.py)为准，未知字段直接报错；采集还读取同一 YAML 中的 `collection` 选项。示例不替代设备地址、serial、工作空间与现场标定的核对。

XHand 默认通过 `/dev/ttyUSB0` 串口连接。使用 EtherCAT 时，设置 `hand.comm_type: ethercat` 并将 `hand.device_name` 设为实际网卡接口名；显式设为 `null` 才使用所选传输的设备发现。

各入口的 `--help` 不连接设备；`--print-config` 只打印声明配置，不读取现场标定或加载策略。日志默认写入 `~/.dexmani/logs/`，可用 `DEXMANI_LOG_DIR` 指定位置。

## 实验工作流

以下任务名 `test`、episode 目录和 `policy/task/experiment` 均为示例；使用时替换，提前准备 `local.yaml`。

**标定、采集、部署和回放会连接真实设备。** 运行前确认安装、标定、限位与工作空间，保持操作者监督及硬件急停可用。xArm 控制器碰撞检测默认关闭（`arm.collision_sensitivity: 0`），不能依赖它触发碰撞停机；软件几何检查仍执行。

### 1. 标定

相机外参、桌面平面、手部安装与 VR alignment 属于当前实验状态，物理设置变化后重新标定或确认。相机采用固定安装的 eye-to-hand 外参，按设备 serial 绑定。

| 标定 | 入口与要求 |
| --- | --- |
| 相机外参 | [calibrate_camera.py](scripts/calibrate_camera.py)；会移动机械臂并采集图像，须声明 `--hand-geometry absent` 或 `secured-home` |
| VR 朝向 | [calibrate_vr_heading.py](scripts/calibrate_vr_heading.py)；`--ref head` 面向机器人 +X，`--ref wrist` 将手腕指向机器人 +X |
| 桌面平面 | [view_pointcloud.py](scripts/tools/view_pointcloud.py)；选择 table calibration，确认后发布拟合结果 |

```bash
python scripts/calibrate_camera.py --help
python scripts/calibrate_vr_heading.py --help
python scripts/tools/view_pointcloud.py --help
```

`absent` 表示手未安装，`secured-home` 表示已物理固定在配置 HOME 手形；两者均按固定 HOME 手形检查碰撞。标定 jog 允许桌面接触，HOME 独立检查路径净空。采样、求解或目标被拒绝后，先释放全部 jog 键再继续移动。

采集通过 `--vr-transform`、`--camera-calibration` 选择标定，部署支持 `--camera-calibration`；桌面文件由 `environment.table.plane_path` 指定。采集允许缺相机外参，点云重建需要数据对应的相机几何；质量为 `poor` 的 VR 标定不能用于遥操作启动。

### 2. 遥操作与采集

```bash
python scripts/collect_teleop.py test --config local.yaml
```

默认录制到 `episodes/test/`，每段最多 60 秒；用 `--episodes-dir`、`--max-duration` 或 YAML 的 `collection` 选项调整。入口先建立停止监听，手部 HOME 成功后启动 VR，待其就绪再等待操作者开始。

| 按键 | 遥操作行为 |
| --- | --- |
| H | 保存并结束当前段，整机 HOME 后等待新的 B |
| B | 开始采集 |
| C | 暂停或恢复；恢复开始新的 episode |
| S | 停止并保存，回到空闲 |
| D | 丢弃当前未发布 capture，回到空闲 |
| T | 空闲且无 capture 时请求复位触觉；操作前确保五指无接触 |
| Q | 暂停并进入退出确认，再按 Q 保存退出；空闲且无 capture 时直接退出 |
| ESC | 请求急停 |

`--control-hz` 同时设置目标、观测采样与录制频率，默认 16 Hz；机械臂伺服频率独立。调频影响反馈新鲜度预算与 EMA 时间响应，须记录实际配置。晚帧或反馈过期时停止并保留前缀，不追赶补发。

手部默认使用 TAG，通过 `teleop.hand_retargeting_type: dexpilot` 切换到 DexPilot；其余映射和运动参数查阅[配置入口](dexmani_real/config/experiment.py)。

`--no-record` 关闭录制、相机和录制时长限制，**遥操作仍会执行**。`--no-hand` 仅用于手未安装或已固定在 HOME 的调试，须同时关闭录制。

### 3. 离线查看与导出

```bash
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --info
python scripts/visualize_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --max-frames 100
python scripts/export_zarr.py episodes/test --config local.yaml
```

`--info` 无需启动 Rerun；交互查看可用 `--no-point-cloud` 关闭点云。处理使用 Raw 保存的相机几何，以及当前点云、桌面、手部安装与指尖配置；仅启用去桌时读取桌面平面。

公共导出要求每段全部 **13 字段**完整、浮点有限，并通过来源时间、dispatch、触觉新鲜度和技术结束检查，包含手部模型点云与两类触觉。任一检查不通过均整段拒绝并记录原因，保留原 Raw；任务失败本身不排除技术有效的数据。完整字段见 [Canonical 定义](dexmani_real/data/canonical/schema.py)。

默认输出 `datasets/test.zarr`，目标须为 Raw 之外的新路径。`--output` 指定路径，`--exclude` 排除整段 episode，`--chunk-frames` 调整处理批大小；I/O 或处理失败保留 staging。缓存格式或处理逻辑变化后，从 Raw 重新导出，不迁移旧缓存或补齐缺失字段。

在线与离线共用[点云处理](dexmani_real/sensor/pointcloud/processing.py)。部署使用训练 artifact 保存的处理 recipe 和当前现场标定，缺字段不以当前默认值补齐；更换数值表示后须重新导出并训练。

### 4. 策略部署

```bash
python scripts/run_policy.py policy/task/experiment --config local.yaml
python scripts/run_policy.py policy/task/experiment --config local.yaml --execution-mode async --prefetch-steps 1 --n-action-steps 4
```

实验输入为不带 `experiments/` 前缀的 `policy/task/experiment`，或有效的绝对实验目录。策略 artifact 决定模型输入、动作表示与控制 dt；Real 提供当前设备、标定与执行配置。执行步数按 `--n-action-steps` > YAML 的 `execution.action_steps` > 策略默认值确定。

| 模式 | 行为与配置 |
| --- | --- |
| `sync`（默认） | 执行一段后等待下一次推理，保留真实段间等待 |
| `async` | 提前推理；迟到时继续消费仍有效的旧预测尾部，需设置 `prefetch_steps` |
| `rtc` | 提前推理并冻结已承诺的动作前缀，需设置 `prefetch_steps` 与 `rtc_guidance_cap` |

预取窗口须满足策略 horizon 容量，模型须支持所选模式与输入；这些条件在连接设备前检查。旧预测耗尽或决策过期时结束本轮，不重复末动作、不补发。LAMP 真机闭环尚未验证。

H 回 HOME、B 开始本轮、S 结束本轮；整理场景后再次按 H、B。Q 提前退出，ESC 请求急停，T 在空闲且无接触时请求复位触觉；部署不使用 C/D。

默认最多 20 轮正式运行，每轮运行预算 60 秒，用 `--num-episodes`、`--max-duration` 调整；准备失败或取消不占轮数。预算不保证阻塞 SDK 或模型调用立即返回。checkpoint 默认 `latest`，其余推理选项见 `--help`。

默认产物位于 `rollouts/policy/task/experiment/session_*/`；`run_config.yaml` 保存实际配置与模型信息，`session_result.json` 保存评估摘要。按摘要中的 Raw 路径定位采集，正式轮次与录制尝试不一一对应；准备失败或取消时，已启动且正常收尾的采集也会保留，包括零行采集。

`--no-record` 与 `--no-results` 分别关闭 Raw 和摘要，**均不关闭设备连接或动作执行**；同时关闭时不创建会话产物。相机按策略输入与录制需求启动。

### 5. 真机轨迹回放

```bash
python scripts/replay_episode.py episodes/test/episode_YYYYMMDD_HHMMSS --config local.yaml --output replay_results/test_run
```

回放读取当前格式的 teleop Raw，按标称节拍执行已记录关节目标；会连接机械臂与手，不启动相机或 VR。启动先执行手部 HOME，再默认用 0.5 秒渐变到首帧手部目标；`--hand-start-duration 0` 仅跳过渐变。

输出目录须新建或为空，保存 `session_result.json`、`replay_data.npz` 和可计算的 `metrics.json`。Q 保存前缀并退出，ESC 请求急停，完成后可按 H 回 HOME；晚帧或发送超时会停止，输入 Raw 保持不变。

## 设备工具

| 用途 | 入口 | 运行边界 |
| --- | --- | --- |
| 键盘 jog / HOME | [keyboard_teleop.py](scripts/keyboard_teleop.py) | 连接机器人，不启动相机或 VR；净运动意图归零停止 jog，R 执行 HOME |
| XHand 动作与反馈 | [control_xhand.py](scripts/tools/control_xhand.py) | 默认执行动作；`--read-only` 仍连接手并读取反馈 |
| RGB-D 诊断 | [view_realsense.py](scripts/tools/view_realsense.py) | 连接相机，交互查看 RGB-D 与点云 |
| 点云诊断 / 桌面标定 | [view_pointcloud.py](scripts/tools/view_pointcloud.py) | 连接相机；`--save-dir` 保存 RGB-D、点云及重建 metadata |
| 物体重建与跟踪 | [CORDViP 示例](scripts/cordvip/README.md) | 独立的 `cordvip_geometry` 依赖，区分离线数据和实时相机模式 |

各入口的参数与交互以 `--help` 和提示为准。键盘 jog 与相机标定在连接机器人前启动退出/急停监听；触界、IK 或准入拒绝后须释放全部 jog 键才能重新操作。

## 数据与安全边界

**Raw 发布后不可变，Canonical 是可重建的训练缓存。** Raw 使用 `data.h5` 和 `rgb.mp4`，保存实测观测、提交目标、真实 dispatch、来源时间与终止原因。RGB-D 按保存行对齐，视频按标称帧率编码，实际采样时刻以 HDF5 时间为准。

停止、晚帧和控制失败保留真实前缀；仅操作者显式重录或丢弃才删除当前 capture。写入、编码、关闭或发布失败保留 staging。I/O 完整的部分发送或停止未确认记录可以发布 Raw，但拒绝严格导出或回放；发布不证明任务成功或设备停止。

| 概念 | 数值含义 |
| --- | --- |
| 坐标 | 规划与运动使用 xArm-base；Raw 相机外参为 color→base，Canonical 为其逆变换 base→color 的 `[R\|t]` |
| 深度 | 对齐到 color 像素网格的 `uint16` 原计数，乘保存的 depth scale 得到米值，不能统一当作毫米 |
| 动作 | 本拍提交的绝对关节目标；`action_ee` 是该目标的 FK，均不表示下一拍已到达 |
| 触觉 | 聚合与 taxel 分别保持有界上一有效三轴向量，并保留原有效时间；使用 SDK 原生值，未经标定不能称为 Newton 或 Sim 等价接触力 |
| 手点云 | 当前 XHand URDF/mesh 的固定表面样本随实测关节 FK 变换，使用固定材质色，无需相机或分割 |

导出不修补缺测、不删坏帧，不再次修补触觉或扣偏置。Rollout Raw 的无动作行保留原样，不裁行或填动作以满足训练格式。arm_qvel、arm_effort、hand_current 不默认持久化；控制仍读取所需关节位置与速度，手部电流可通过诊断入口读取。

目标准入、SDK 接受、实测到位与真实停止是不同结果，`CRC_UNCONFIRMED` 不表示 ACK 或到位。HOME 限于当前受限桌面场景，无可行路径时拒绝并交由操作者处理。SDK 或 CUDA 阻塞可能影响停止响应，离线验证不能证明真机安全。

## 代码导航

从对应的 [scripts/](scripts/) 入口开始，按任务追踪以下模块；字段布局与实现细节以源码为准。

| 修改内容 | 起点 |
| --- | --- |
| 配置与覆盖 | [config/experiment.py](dexmani_real/config/experiment.py) |
| 设备与观测 | [robot/](dexmani_real/robot/)、[sensor/](dexmani_real/sensor/)、[runtime/](dexmani_real/runtime/) |
| 遥操作与采集 | [teleop/](dexmani_real/teleop/)、[data/recorder.py](dexmani_real/data/recorder.py) |
| 策略执行与调度 | [deployment/run.py](dexmani_real/deployment/run.py)、无设备 I/O 的 [schedule.py](dexmani_real/deployment/schedule.py) |
| Raw 与导出 | [Raw 定义](dexmani_real/data/raw/schema.py)、[Canonical 定义](dexmani_real/data/canonical/schema.py)、[导出入口](dexmani_real/data/canonical/export.py) |
| 动作准入、运动学与标定 | [planning/action.py](dexmani_real/planning/action.py)、[planning/kinematics/](dexmani_real/planning/kinematics/)、[calibration/](dexmani_real/calibration/) |
| 回放与进程通信 | [replay/](dexmani_real/replay/)、[ipc/](dexmani_real/ipc/) |

## 开发

编码约定、改动核对及验证方式统一见 [AGENTS.md](AGENTS.md#验证与交付)，按改动选择必要的离线检查。Claude 使用 [CLAUDE.md](CLAUDE.md) 按任务定位，无需全篇重复阅读。
