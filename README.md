# DexMani Real

面向个人 PhD 灵巧操作实验的真实机器人代码：**采数据、部署 learned policy、做真机 evaluation**。
当前硬件是 xArm7、XHand、RealSense RGB-D 和 VR / HTS；模型训练与推理由相邻的
`dexmani_policy` 负责。本仓库不以通用机器人框架、多租户或长期内部 API 兼容为目标。

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

目前配置解析顺序仍是 **CLI > YAML > `config/defaults.py`**：

```bash
python examples/collect_teleop.py --print-config
```

经常调整的 IP、serial、频率、相机尺寸、数据路径、task、控制参数留在实验配置。
不要为了假想硬件组合、存储后端或未来 serving 需求继续增加配置层。

## 研究入口

| 工作流 | 命令 | 硬件 / 输出 |
|---|---|---|
| VR collection | `python examples/collect_teleop.py --task-name <task> --operator <name>` | 真机；raw episode |
| Keyboard jog / home | `python examples/keyboard_teleop.py` | 真机 |
| Physical replay | `python examples/replay_episode.py <episode>` | 真机；processed 输入加 `--processed` |
| Policy rollout | `python examples/run_policy.py <policy/task/experiment>` | 真机；rollout session |
| Camera calibration | `python examples/calibrate_camera.py --hand-geometry {absent,secured-home}` | 真机；标定 |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | VR / HTS；标定 |
| Offline processing | `python examples/process_episodes.py episodes/<task> --dry-run` | 只读预检；移除 `--dry-run` 后转换 |
| Policy export | `python examples/export_policy_zarr.py episodes_processed/<task> --dry-run` | 只读预检；移除 `--dry-run` 后导出 |
| Inspect raw | `python examples/visualize_episode.py <episode> --info` | 离线；去掉 `--info` 可视化 |
| Inspect processed | `python examples/visualize_episode_processed.py <processed.h5> --info` | 离线 |

完整参数与按键以各入口的 `--help` 为准。

### Teleoperation / recording

```bash
# 标准采集
python examples/collect_teleop.py --task-name <task> --operator <name>

# 无手、无录制的 arm 调试
python examples/collect_teleop.py --task-name <task> --operator <name> --no-hand --no-record
```

数据写入 `episodes/<task>/episode_*`。相机和 recorder 在 teleop 中不承担控制输入职责：
录制失败不应突然中断人工控制，但会使会话结果失败。arm / hand / VR 控制故障仍走停机路径。
自动中断时，非空且已安全关闭的录制前缀保存在 `incomplete_*`，不是可直接训练的完整 episode；
显式丢弃仍丢弃。最终路径与录制结果查看日志。

### Policy rollout

```bash
python examples/run_policy.py <policy/task/experiment> \
  --num-episodes 2 --inference-steps 4 --seed 0
```

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
raw episode → offline geometric / sensor processing → processed HDF5 → Policy Zarr
```

中间 processing 有真实工作：相机几何、点云、坐标变换、FK / fingertip / 触觉准备。
processed 文件是可检查的离线结果，而不是另一套运行时协议。batch report 是普通 YAML 摘要，
不具有独立的 schema / migration / loader 体系。

policy_eval 的同步、非均匀时序不能被静默解释成 fixed-dt teleop 数据。
目前 processing 和 fixed-rate replay 仍拒绝这种输入；需要 time-aware conversion 才能改变此限制。
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
                         control/projection.py + safety_gate.py
                                       ↓
                         control/publication.py → command FIFO
                                       ↓
                          robot/{arm,hand}_worker.py → SDK

policy: deployment/lifecycle.py → executor.py
        → inference/observation.py → dexmani_policy.predict(array_dict)
        → 同一 command path

recording: control loop → recording/sample.py + client.py
          → IO worker → recorder.py → HDF5 / video
```

启动器直接创建 `context.Process`；process name 就是 readiness key，没有第二套 process spec。
命令只在 FIFO 成功提交时分配 sequence。generation 只用于暂停、停止或阻塞推理结束后拒绝旧动作，
不是科研 episode 身份。SDK acceptance 不等于物理收敛，尤其不能让 XHand 中间限速点提前确认最终目标。

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
动作限制、停止后不再发送旧动作，以及已经发生过的实验回归。
具体异常文案、内部日志或已删除的 abstraction 不应成为必须长期保留的架构。
