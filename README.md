# DexMani Real

面向 xArm7 + XHand 真机灵巧操作实验的研究代码，使用 RealSense RGB-D 和 VR/HTS 采集示教，支持训练数据导出、策略部署和物理回放。策略训练与推理由配套 [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy) 提供。

主要流程：**配置与标定 → 采集示教 → 查看与导出训练集 → 训练策略 → 真机评估**。

## 安装与实验配置

使用 Python 3.10+，以下命令均在仓库根目录运行：

```bash
python -m pip install -e ".[interactive,control,geometry,visualization]"
```

设备 SDK（xArm、XHand、RealSense）、HTS、Pinocchio 和 NLopt 需按实际实验环境另行安装。运行时会读取 `assets/` 中的机器人模型和手模板，请保留源码目录。

设备和运行参数统一由 [RealConfig](dexmani_real/config/real.py) 配置。需要修改时，将下例保存为 `~/dexmani_config.py`，按实际设备调整：

```python
from dataclasses import replace
from dexmani_real.config.real import RealConfig

base = RealConfig()
CONFIG = replace(
    base,
    arm=replace(base.arm, ip="192.168.1.111"),
    hand=replace(base.hand, device_name="/dev/ttyUSB0"),
)
```

运行时通过 `--config ~/dexmani_config.py` 使用该文件；下文示例省略此参数。未指定时使用默认配置。相机 serial、HOME、关节范围、工作空间和控制频率等参数见 [config/](dexmani_real/config/)。任务名、路径、段数、时长和录制开关通过各脚本的命令行参数指定。

连接设备前可查看配置，以下命令不会连接设备或加载策略：

```bash
python scripts/collect_teleop.py --config ~/dexmani_config.py --print-config
python scripts/run_policy.py --config ~/dexmani_config.py --print-config
```

## 标定和查看

默认标定保存在 [calibration_data/](dexmani_real/config/calibration_data/)。首次使用或改变设备布置后，核对 VR 朝向；使用相机点云前核对外参，启用去桌面时还需标定桌面平面。

| 目的 | 命令 |
| --- | --- |
| 查看相机画面 | `python scripts/tools/view_realsense.py` |
| 标定 VR 朝向，面向机器人 +X | `python scripts/calibrate_vr_heading.py --ref head` |
| 拟合桌面并检查点云 | `python scripts/tools/view_pointcloud.py` |

相机外参分为采样和离线求解两步：

```bash
python scripts/calibrate_camera.py collect --output calibration_samples/run_001
python scripts/calibrate_camera.py solve calibration_samples/run_001
```

采样时标记须相对机器人末端刚性固定。在多个位置与朝向摆位，每次采样前退出厂商示教或独立工具的控制连接，在采样终端按 Enter 采集一组、Q 结束。求解成功后更新默认相机外参，再进行桌面拟合与点云检查。

## 采集与导出训练数据

下文 `TASK` 为任务名，`EPISODE` 为该任务下已保存的 episode 目录名。

### 采集示教

```bash
python scripts/collect_teleop.py TASK
```

默认开启录制，每段时长默认 60 秒，保存到 `episodes/TASK/`。每段 Raw 数据包含 `data.h5` 和 `rgb.mp4`。可用 `--max-duration 90` 调整时长，`--episodes-dir DIR` 更改录制根目录；仅遥操作时加 `--no-record`。

等待终端提示就绪后再操作。**VR 遥操作和策略部署共用以下按键**，保存与丢弃仅在启用录制时生效：

| 按键 | 操作 |
| --- | --- |
| B | 开始新段，每段均需手动开始 |
| S / C | 停止并保存本段 |
| D | 停止并丢弃本段 |
| Q | 退出会话，保存尚未丢弃的当前段 |
| H | 空闲时回初始位，先手后臂 |
| T | 空闲且手部无接触时复位触觉 |
| Esc | 急停并结束会话 |

连接后不会自动回初始位。B/H/T 须在上一段收尾完成后使用；活动段先按 S 停止。收尾等待期间继续处理退出和设备反馈，Q/Esc 转入设备关闭后继续回收 writer；首次收尾请求固定保存/丢弃意图与等待预算，后续 S/C/D 不会修改该请求。writer 观察到保存请求后非阻塞排空已接纳帧，队列为空即进入保存收尾；丢弃请求停止消费剩余帧。正在执行的取帧、写入或关闭仍正常返回。

遥操作和策略部署在连接成功后先内部停止，再启动传感器、标定与预热；内部停止不撤销用户操作。初始化阻塞后的读 CRC 不刷新缓存时间，后续有效帧由现有有界观测等待接纳；正式运动前仍检查反馈新鲜度。HOME 普通取消后保持停止，停止处理及本次回复均确认后保留会话，可再次显式 H/B；Q 退出，急停、真实故障或取消收尾超时仍关闭。停止确认仅表示 SDK 停止处理完成，不证明物理停稳。

手部 HOME 的停止、初始关节反馈等待、目标发送、至多一次 CRC 补发和到位确认共用 `hand.home_timeout_s`。CRC 不刷新关节缓存时间；补发也须等待原始时间戳仍新鲜的关节反馈。等待期间继续检查取消和故障，触觉缺失不阻止 HOME。到位须由最后一次提交完成后的三个不同新鲜关节样本确认。阻塞 SDK 调用无法被该期限强制中断，返回后会复核取消和超时；最终停止收尾仍执行。

设备关闭失败时保留尚未释放的对象，须再次关闭成功后才能重新连接；重试关闭不向已断开的手部重复发送停止。

XHand 普通停止仅用新鲜、机械范围内的实测位置保持；容差内的微小舍入越界可归一，明显越界、无效或陈旧反馈改为 passive 请求。判断不使用实验收窄的操作范围，也不修改原始反馈；passive 不等于物理硬停止。

### 查看数据

```bash
python scripts/visualize_episode.py episodes/TASK/EPISODE
```

可视化包含 RGB-D、关节、触觉和手部点云，需要图形桌面。只查看数据结构时加 `--info`。

### 导出训练集

```bash
python scripts/export_zarr.py episodes/TASK --output datasets/TASK.zarr
```

将已保存的 Raw episodes 转为训练用 Zarr，交给 `dexmani_policy` 训练。默认导出全部模态；仅需 RGB、关节状态和动作时，加 `--fields rgb joint_state action`，这些字段不依赖相机外参或几何重建。

输出路径须尚不存在。不合格 episode 会使整批导出失败；检查原因后，可用 `--exclude EPISODE` 排除整段再导出，源 Raw 保持不变。

## 策略部署

在上述环境中安装配套 `dexmani_policy` 及 checkpoint 所需模型依赖，核对当前设备与标定，并保持 `hand.enabled=True`：

```bash
python scripts/run_policy.py /path/to/checkpoint.pt --num-episodes 5
```

也可传入包含 `checkpoints/latest.pt` 的目录。默认使用 `cuda:0` 和 `sync` 模式；输入模态、动作表示和控制周期沿用 checkpoint，周期缺失时须在配置中指定。等待就绪后按 B 开始，每段仍需手动触发，操作键同[采集示教](#采集示教)。

默认只在终端输出。需要同时保存会话结果和完整 Raw 数据时：

```bash
python scripts/run_policy.py /path/to/checkpoint.pt --num-episodes 5 --record-raw --output rollouts/TASK
```

`--output DIR` 单独使用时仅保存配置与会话结果；`--record-raw` 开启录制，未指定输出目录时使用 `rollouts/`。每段时长默认 60 秒，可用 `--max-duration` 修改；推理设备可用 `--device` 修改。

async/RTC、执行参数覆盖与运行机制见[策略部署说明](docs/policy_deployment.md)。

## 键盘点动与物理回放

| 目的 | 命令 | 操作 |
| --- | --- | --- |
| 键盘点动 | `python scripts/keyboard_teleop.py` | WASD、方向键、IJKL 点动；R 回零；Q 退出；Esc 急停 |
| 仅连接机械臂点动 | `python scripts/keyboard_teleop.py --no-hand` | 使用相同按键；碰撞检查使用配置中的固定手姿态 |
| 物理回放 | `python scripts/replay_episode.py episodes/TASK/EPISODE` | 自动播放；S/Q 停止退出；Esc/Ctrl+C 急停 |

键盘点动按固定时钟网格调度，迟到时跳过错过的周期，不补发；启动及 HOME 完成后重新建立起点。恢复网格时相邻实际执行间隔可能缩短，不保证硬实时或固定实测频率。松键减速保持，成功提交保持后清除 IK 参考。

## 控制和几何边界

本项目机械臂限位适用于已确认的 XS1305（SDK v1.18.4 的 71305 范围），ArmConfig 可在设备有效限位内收窄运行范围。

运行前核对 HOME、关节范围与工作空间。HOME 是直接关节回位，物理回放按录制的绝对目标执行，均不做路径绕障。回放前需将机器人就位至首行目标附近，各关节偏差不超过 10°，并确认当前布置适合该示教。仅连接机械臂点动时，实物手须移除或固定在配置的 HOME 姿态。

回放顺序发送各行，普通行的目标期限覆盖该行计划等待，再加现有 2 秒裕量，因此 0.2 Hz 的 5 秒行间等待不会自行耗尽租期。超出期限仍停止，等待期间继续检查反馈和 S/Q/Esc。末行保留发送返回后约 0.5 秒留驻及有限租期，不要求确认物理到位。

Raw 中 state 是实测值，action 是指令提交尝试；发送、保存或导出成功均不证明机器人实际到位。在线端点和关节限位检查也不保证整条运动路径无碰撞。

EE 遥操作、策略和 jog 共用有界局部 IK，返回的是下一条可执行关节目标，允许尚未到达输入末端位姿。`motion.ik_target_speed_deg_s` 默认 75°/s（不超过设备速度上限），`motion.ik_max_lead_deg` 默认 10°；均为离线原型参数。尚无提交参考（包括清除后重启）时，使用一个名义控制周期的目标变化预算；已有参考时按上次成功提交后的时间差计算，至多一个周期，不积累暂停时间。固定姿态偏好取本段首次成功提交时使用的实测起点，不强制 HOME。参考超前失配结束当前段，普通遥操作几何拒绝仍在原期限内重发旧 arm/hand。详细公式和离线结果见[连续性验证](docs/hand_retargeting.md#共享-ee-ik-连续性与离线验证)。

## 进一步阅读

| 主题 | 入口 |
| --- | --- |
| 策略执行模式与录制行为 | [策略部署说明](docs/policy_deployment.md) |
| VR 到 XHand 的映射与参数 | [手部重定向](docs/hand_retargeting.md) |
| XHand 离线对照与验收限制 | [重定向实验记录](docs/xhand_retargeting_experiment_2026-11-06.md) |
| 离线物体几何与位姿分析 | [CordVIP 脚本说明](scripts/cordvip/README.md)，[独立环境安装](third_party/cordvip_geometry/docs/usage.md) |
| 物体网格与点云资产制作 | [物体重建流程](docs/object_reconstruction.md) |
| 开发约定 | [AGENTS.md](AGENTS.md) |

CordVIP 当前用于离线分析，尚未接入主包的策略观测。完整命令行选项请查看各脚本的 `--help`。
