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

支持 `--config` 的入口按 **CLI > 显式指定的 YAML > `config/defaults.py`** 解析，
不会自动查找实验 YAML。可在连接设备前查看采集入口的解析结果：

```bash
python examples/collect_teleop.py --print-config
python examples/collect_teleop.py --config experiment.yaml --print-config
```

经常调整的 IP、serial、频率、相机尺寸、数据路径、task、控制参数留在实验配置。

VR 遥操作的网格频率和轮询频率由 `teleop.control_hz` / `teleop.executor_poll_hz` 配置。
learned policy 的动作周期唯一来自 `PolicySpec.control_dt_s`。
流式命令一次只有一条待采用；下一条必须等本次命令包含的执行器都明确采用。
XHand 的首次限速点采用与到达最终目标分开记录；手部归位使用
`hand.home_command_ack_timeout_s` 等待最终目标，机械臂 HOME 保留规划、请求和执行超时。

## 研究入口

| 工作流 | 命令 | 硬件 / 输出 |
|---|---|---|
| VR collection | `python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name>` | 真机；raw episode |
| Keyboard jog / home | `python examples/keyboard_teleop.py --config experiment.yaml` | 真机 |
| Physical replay | `python examples/replay_episode.py <episode> --config experiment.yaml` | 真机；raw float64 sent targets |
| Policy rollout | `python examples/run_policy.py <policy/task/experiment>` | 真机；rollout session |
| Camera calibration | `python examples/calibrate_camera.py --config experiment.yaml --hand-geometry absent` | 真机；标定；需声明 XHand 物理状态 |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | VR / HTS；标定 |
| Policy export | `python examples/export_policy_zarr.py episodes/<task> --dry-run` | 只读预检；移除 `--dry-run` 后导出 |
| XHand diagnostics | `python examples/xhand_diagnostics.py` | 连接真机；读取 joints/tactile/identity，无运动 |
| Inspect raw | `python examples/visualize_episode.py <episode> --info` | 离线；去掉 `--info` 可视化 |

除 XHand diagnostics 外，参数以各入口的 `--help` 为准，交互按键见会话提示。
`xhand_diagnostics.py` 没有离线 `--help`，执行该脚本会启动 SDK 诊断并连接/发现设备。
相机标定的 `--hand-geometry` 是物理状态声明：未安装 XHand 才可用 `absent`；
安装的手已物理固定在配置的 home 姿态时用 `secured-home`。两者均使用固定 home 手模型做碰撞检查。

### Teleoperation / recording

```bash
# 标准采集
python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name>

# 无手、无录制的 arm 调试
python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name> --no-hand --no-record
```

`B` 开始、`C` 暂停/恢复、`S` 停止并保存、`D` 丢弃、`H` 归位、`Q` 退出、`ESC` 急停。
退出录制时按提示选择保存/丢弃；不要把 GUI 关闭或进程退出当作物理急停。
数据写入 `episodes/<task>/episode_*`。录制模式下，`B` 必须等相机和 RecorderIO 可用。
相机或 recorder 失败会使当前 demonstration 无效，并撤销运动回到 ARMED；仍可手动归位或退出。
只有显式 `--no-record` 才允许不录制的遥操作。`C` 保留同一 episode，撤销旧命令后，
等待暂停之后的新 robot/VR 反馈再重新锚定；不会把暂停改成停止或丢弃。
自动中断时，非空且已安全关闭的录制前缀保存在 `incomplete_*`，不是可直接训练的完整 episode；
显式丢弃仍丢弃。最终路径与录制结果查看日志。
recorder 进程自己 drain 最后一帧、关闭视频/HDF5 并发布结果；关闭期间使用原始固定 deadline，
不以普通循环 heartbeat 判死。无法确认关闭时保留 staging，停止后不复用其 IPC；
上一 episode 的结果未确认前不开始下一次录制。键盘监听持续到录制收尾完成。

### Policy rollout

`run_policy.py --config experiment.yaml` 使用同一配置解析器。
硬件启动前写入完整 `run_config.yaml`，包括解析后的 runtime、PolicySpec、checkpoint、seed 和 Git 状态。

```bash
python examples/run_policy.py <policy/task/experiment> --config experiment.yaml \
  --num-episodes 2 --inference-steps 4 --seed 0
```

`B` 开始 episode、`S` 停止并保存、`H` 归位、`Q` 退出、`ESC` 急停；`--num-episodes` 指实际开始的 episode 次数。
该入口连接真实硬件。`--artifact` 是 experiment 的 `checkpoints/` 下的部署文件名；
不填时使用 `deployment_latest.pt`。训练 checkpoint 先在 `dexmani_policy` 中通过其
public deployment export 命令导出，不能当作部署文件直接传入。

policy 子进程拥有模型 / CUDA，先 load / warmup，再启动硬件 workers。
每个 chunk 同步推理，按 policy control period 逐条发送，不跳动作、不补发追赶；
下一次发布必须同时满足上一条命令已采用、且距离上次成功发布至少一个周期。
慢推理或采用延迟后，后续周期从实际成功发布时刻重新计算；不会突发补发。

输出位于 `rollouts/<policy>/<task>/<experiment>/session_*/`，包含 raw episode 和运行配置。
**时间分析使用 HDF5 的实际 timestamp，不使用 MP4 固定帧率推断控制时间。**
运行次数与保存成功的 episode 数不同。必需的 camera/recorder 失败会结束无效 evaluation 并返回非零，
只有实际硬件故障才声明 FAULT。结果分开保存 technical_status、termination_reason 和 task_success（可离线标注）。

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

固定周期训练导出拒绝非均匀 anchor 间隔（包括暂停缺口），不静默重采样或压缩时间。
policy_eval 保持其独立的非均匀时序，不直接作为 fixed-dt teleop 数据导出/回放。
触觉 validity 必须保存，不能把传感器无效数据解释成零接触力。

raw 当前只支持 v31；v30 缺少采用证据，不能直接视为 v31，也不支持自动补造证据。
v31 保存 command_id、run_id、issued 时间、每个执行器的 presence、adopted 及真实采用时间。
正常动作只有在命令包含的全部执行器采用后才写入；held/failure 行不重复发送。
部分采用或未知采用仅作诊断，并使整个 episode 默认不可训练导出或物理回放。
物理 replay 使用实际 anchor 间隔与新的共同采用 command ID，保留 float64 目标及执行器 presence。

## 代码追踪与安全 owner

当前主要路径：

```text
teleop/session.py → teleop/loop.py → control_loop/grid.py
                                       ↓
                        robot/projection.py（一次软投影）
                                       ↓
                 robot/commands.py（检查目标、发布单条耦合命令）
                                       ↓
                     robot/{arm,hand}_worker.py → SDK

policy: deployment/session.py 启动进程并独立处理键盘
        deployment/runner.py → observation.py → dexmani_policy.predict(array_dict)
        → 同一 robot command path

recording: control loop → build_episode_frame → RecorderClient.add_frame(frame)
          → sample ring → IO worker → EpisodeRecorder.add_frame(frame) → HDF5/video
```

主进程拥有进程启动、运行时长上限、critical worker 健康检查与清理；独立键盘 listener
在模型推理、home 和录制关闭阻塞时响应停止。policy 子进程拥有模型/CUDA 与 episode 计数/结果；
SDK 由各自 arm/hand worker 持有。`SafetyState` 为 DISARMED / ARMED / RUNNING / FAULT。
共享 run_id/start 原子读取，让 parent 能在推理阻塞时执行运行时长上限；
terminal cause 保留第一次实际停止的原因，供晚返回的 runner 收尾。

command_id 在同一个 RuntimeChannels 生命周期中单调分配；run_id 是运动授权 epoch，
用于暂停、停止及阻塞推理返回后拒绝旧动作，不是科研 episode 身份。
撤销阻止新的 SDK admission；已经 admission 的调用仍可能晚返回，arm/hand 并非物理原子事务。
录制中的待采用命令在撤销后使用新的 worker 状态完成有界核算，再允许后续命令覆盖证据。
SDK adoption 不等于物理收敛；XHand 的 adopted 与最终目标 reached 分开。

必须保留的硬边界：机械限位、非有限数阻断、实际速度 / 步长限制、急停、SDK 错误、旧命令撤销、
实验依赖的工作空间 / 碰撞约束，以及 worker 停止后才释放共享内存。
软 projection 负责生成可执行目标；worker / driver 负责最终 SDK 输入与物理状态。
不要用多层重复 validation 替代明确的 ownership。

## 离线检查

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

仓库没有提交的测试目录。对改动的纯逻辑可运行一次性离线 smoke checks，重点包括数值几何、
转换、动作限制和数据接口；不新增测试框架。Ruff 或依赖缺失时报告跳过，不为检查升级实验环境。
离线检查不运行硬件入口，不启动真机、CUDA 或真实 checkpoint。

## 真机 commissioning（需单独授权，离线测试不代替）

测量真实工作负载的发布、SDK 采用与 hand slew 延迟，验证采用核算和归位超时。
低速检查 C 暂停/恢复与重新锚定、S/D/Q/H、慢推理时的 parent timeout、
长录制关闭时的 S/Q/ESC、arm/hand 单侧先行及故障、RealSense/触觉失效、
真实 checkpoint rollout、物理急停和安全断开。
软件撤销阻止后续 admission；已经 admission 的 vendor 调用仍可能执行或晚返回，
真机停机距离/延迟必须实测。
