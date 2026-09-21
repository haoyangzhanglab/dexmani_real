# DexMani Real

面向个人 PhD 灵巧操作实验的真实机器人代码：**采数据、部署 learned policy、做真机 evaluation**。
当前硬件是 xArm7、XHand、RealSense RGB-D 和 VR / HTS；模型训练与推理由相邻的
`dexmani_policy` 负责。本仓库负责真机实验与数据采集。

> 本仓库会控制真实硬件。home、teleoperation、replay、policy rollout 和 camera
> calibration 都不是离线测试。运行前检查工作空间、标定、机械限位与急停，并保持操作者在场。
> 软件检查通过不代表真机安全验证通过。

## 安装与配置

要求 Python >=3.10。通用依赖安装：

```bash
python -m pip install -e .
```

xArm、XHand、RealSense、HTS SDK，以及运动学/碰撞规划和 `dexmani_policy` 依赖，
按实验机环境安装。普通 import 不应连接设备；SDK 连接与控制由 owning worker 显式执行。

配置解析顺序是 **CLI > YAML > `config/defaults.py`**：

```bash
python examples/collect_teleop.py --print-config
```

经常调整的 IP、serial、频率、相机尺寸、数据路径、task、控制参数留在实验配置。

运动入口要求在实验 YAML 显式提供 `safety.max_dispatch_delay_s`（有限正秒数）。
当前没有经过真机 timing validation 的默认值；缺失配置会在设备启动前拒绝运行。
该预算从每个 endpoint 的执行时机开始，覆盖准备、FIFO 等待和 XHand 最终目标 slew；
FULL/CRC 重试不刷新 deadline。过期撤销该 generation，必须由操作者重新开始；
keyboard/calibration jog 必须先释放按键。HOME 排队使用现有 request queue budget，
已开始的长轨迹仍使用自身的执行/abort timeout。

## 研究入口

| 工作流 | 命令 | 硬件 / 输出 |
|---|---|---|
| VR collection | `python examples/collect_teleop.py --task-name <task> --operator <name>` | 真机；raw episode |
| Keyboard jog / home | `python examples/keyboard_teleop.py` | 真机 |
| Physical replay | `python examples/replay_episode.py <episode>` | 真机；raw float64 sent targets |
| Policy rollout | `python examples/run_policy.py <policy/task/experiment>` | 真机；rollout session |
| Camera calibration | `python examples/calibrate_camera.py --hand-geometry {absent,secured-home}` | 真机；标定 |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | VR / HTS；标定 |
| Policy export | `python examples/export_policy_zarr.py episodes/<task> --dry-run` | 只读预检；移除 `--dry-run` 后导出 |
| XHand diagnostics | `python examples/xhand_diagnostics.py` | 连接真机；读取 joints/tactile/identity，无运动 |
| Inspect raw | `python examples/visualize_episode.py <episode> --info` | 离线；去掉 `--info` 可视化 |

完整参数与按键以各入口的 `--help` 为准。

### Teleoperation / recording

```bash
# 标准采集
python examples/collect_teleop.py --task-name <task> --operator <name>

# 无手、无录制的 arm 调试
python examples/collect_teleop.py --task-name <task> --operator <name> --no-hand --no-record
```

`B` 开始、`C` 暂停、`S` 停止并保存、`D` 丢弃、`H` 归位、`Q` 退出、`ESC` 急停。
退出录制时按提示选择保存/丢弃；不要把 GUI 关闭或进程退出当作物理急停。
数据写入 `episodes/<task>/episode_*`。相机和 recorder 在 teleop 中不承担控制输入职责：
录制失败不应突然中断人工控制，但会使会话结果失败。arm / hand / VR 控制故障仍走停机路径。
自动中断时，非空且已安全关闭的录制前缀保存在 `incomplete_*`，不是可直接训练的完整 episode；
显式丢弃仍丢弃。最终路径与录制结果查看日志。
recorder 进程自己 drain 最后一帧、关闭视频/HDF5 并发布结果；关闭期间使用原始固定 deadline，
不以普通循环 heartbeat 判死。无法确认关闭时保留 staging，停止后不复用其 IPC；
上一 episode 的结果未确认前不开始下一次录制。键盘监听持续到录制收尾完成。

### Policy rollout

```bash
python examples/run_policy.py <policy/task/experiment> \
  --num-episodes 2 --inference-steps 4 --seed 0
```

`B` 开始 trial、`S` 停止并保存、`H` 归位、`Q` 退出、`ESC` 急停；`--num-episodes` 指 trial 次数。
该入口连接真实硬件。`--artifact` 是 experiment 的 `checkpoints/` 下的部署文件名；
不填时使用 `deployment_latest.pt`。训练 checkpoint 先在 `dexmani_policy` 中通过其
public deployment export 命令导出，不能当作部署文件直接传入。

policy 子进程拥有模型 / CUDA，先 load / warmup，再启动硬件 workers。
每个 chunk 同步推理，按 policy control period 逐条发送，不跳动作、不补发追赶；
下一 chunk 等上一条动作完整经过一个周期后再观测和推理。慢推理可能造成实际动作间隔变长。

输出位于 `rollouts/<policy>/<task>/<experiment>/session_*/`，包含 raw episode 和运行配置。
**时间分析使用 HDF5 的实际 timestamp，不使用 MP4 固定帧率推断控制时间。**
试验次数与保存成功的 episode 数不同；recording 故障不应被解释成机器人故障。

## 数据

当前路径：

```text
raw episode → offline geometric / sensor transforms → Policy Zarr
```

```bash
python examples/export_policy_zarr.py episodes/<task> --dry-run
python examples/export_policy_zarr.py episodes/<task>
# 新输出位置（目标必须不存在），使用实验几何配置
python examples/export_policy_zarr.py episodes/<task> --config experiment.yaml --output datasets/<task>_v2.zarr
```

默认输出为 `datasets/<task>.zarr`。转换保留每个完整 episode 的行数与边界、实际 source timestamps、
RGB-D、点云、FK/EE/fingertips、joint/EE actions 和触觉 validity。数值按 episode、图像按有界 chunk
处理；正式 export 不先完整 dry-run 一遍。`--dry-run` 执行相同转换和 admission，但不写输出。
技术损坏会阻断整个输出；`--annotations` YAML 可显式排除整个 episode（`include: false`），
或提供 `task_name`。一个输出只允许一个 task，必须与输入任务目录名一致；`--task-name` 可显式覆盖
raw task label，但不能与 annotation 冲突。未知 episode annotation 会报错。已有输出、输入内部路径、
源数据目录和已有 Zarr 内部均拒绝覆盖/写入。

policy_eval 的同步、非均匀时序不能被静默解释成 fixed-dt teleop 数据。
转换和 fixed-rate replay 仍拒绝这种输入；需要 time-aware conversion 才能改变此限制。
触觉 validity 必须保存，不能把传感器无效数据解释成零接触力。

raw 当前只支持 v30。旧 v29 episode 可独立复制转换，不修改原始数据：

```bash
python examples/convert_raw_v29.py episodes_old/<task>/episode_x episodes/<task>/episode_x
```

转换只去掉旧命令 ID、重复行号、恒定 SOURCE 标记和恒定 sample-valid 标记，保留科学数据；
输出路径必须不存在。运行时不维护旧 schema 分支。更早的格式使用相应历史版本代码离线处理。

## 代码追踪与安全 owner

当前主要路径：

```text
teleop/session.py → teleop/loop.py → control_loop/grid.py
                                       ↓
                        robot/projection.py（一次软投影）
                                       ↓
                 robot/commands.py（检查目标、提交有序 FIFO）
                                       ↓
                     robot/{arm,hand}_worker.py → SDK

policy: deployment/session.py 启动进程并独立处理键盘
        deployment/runner.py → observation.py → dexmani_policy.predict(array_dict)
        → 同一 robot command path

recording: control loop → build_episode_frame → RecorderClient.add_frame(frame)
          → sample ring → IO worker → EpisodeRecorder.add_frame(frame) → HDF5/video
```

主进程拥有进程启动、运行时长上限、critical worker 健康检查与清理；独立键盘 listener
在模型推理、home 和录制关闭阻塞时响应停止。policy 子进程拥有模型/CUDA 与 trial 计数/结果；
SDK 由各自 arm/hand worker 持有。`SafetyState` 为 DISARMED / ARMED / RUNNING / FAULT。
共享 run generation/start 原子读取，让 parent 能在推理阻塞时执行运行时长上限；
terminal cause 保留第一次实际停止的原因，供晚返回的 runner 收尾。

命令只在 FIFO 成功提交时分配 sequence。generation 只用于暂停、停止或阻塞推理结束后拒绝旧动作，
不是科研 episode 身份。撤销阻止新的 software admission；已经 admission 的 SDK 调用仍可能晚返回，
arm 与 hand 也不是物理事务。SDK acceptance 不等于物理收敛，尤其不能让 XHand 中间限速点提前确认最终目标。

必须保留的硬边界：机械限位、非有限数阻断、实际速度 / 步长限制、急停、SDK 错误、旧命令撤销、
实验依赖的工作空间 / 碰撞约束，以及 worker 停止后才释放共享内存。
软 projection 负责生成可执行目标；worker / driver 负责最终 SDK 输入与物理状态。
不要用多层重复 validation 替代明确的 ownership。

## 离线检查

```bash
python -m compileall -q dexmani_real examples
python -m pytest -q
git diff --check
```

pytest 需要相应的离线媒体、运动学依赖，但不应连接硬件。保留的重点是数值几何、转换、
动作限制、deadline/revocation、录制边界和 public policy 数据接口。
离线检查不启动真机、CUDA 或真实 checkpoint。

## 真机 commissioning（需单独授权，离线测试不代替）

先测量准备、排队、SDK 与 hand slew 延迟，再填写 `safety.max_dispatch_delay_s`。
低速检查启动/停止、慢推理时的 parent timeout、home 与长录制关闭时的 S/Q/ESC、
人为 backlog 导致的 deadline 撤销、arm/hand 错误和单侧先行、RealSense/触觉失效、
真实 checkpoint rollout、物理急停和安全断开。deadline 保护 software admission，
已经 admission 的 vendor 调用仍可能执行或晚返回；真机停机距离/延迟必须实测。
