# DexMani Real

面向个人 PhD 灵巧操作实验的真实机器人代码，用于 demonstration 采集、learned policy 部署与真机 evaluation。硬件为 xArm7、XHand、RealSense RGB-D 和 VR / HTS；模型训练与 policy implementation 位于相邻的 `dexmani_policy` 仓库。

本仓库会控制真实硬件。运行前检查机械限位、实验环境、标定与急停，并保持操作者在场。真机操作需明确授权；离线检查通过不代表真机安全验证通过。

## 安装与配置

要求 Python >= 3.10。在实验环境中安装本仓库，以下命令均从仓库根目录执行：

    python -m pip install -e .

硬件 SDK、运动学/点云库和 `dexmani_policy` 依赖按实验机环境安装。普通 import 和配置解析不应连接设备，真实连接由对应 worker 显式建立。

支持 `--config` 的入口按 CLI > YAML > [默认配置](dexmani_real/config/defaults.py) 解析。连接设备前可查看采集配置：

    python examples/collect_teleop.py --print-config
    python examples/collect_teleop.py --config experiment.yaml --print-config

Teleop、键盘控制和 policy 分别使用各自的控制周期，replay 使用录制频率。动作生产频率不得高于相关执行器 worker 的服务频率；worker 频率不等于设备物理伺服频率。

## 常用入口

| 工作流 | 命令 | 说明 |
|---|---|---|
| VR collection | `python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name>` | 真机；raw episode |
| Keyboard jog / home | `python examples/keyboard_teleop.py --config experiment.yaml` | 真机 |
| Physical replay | `python examples/replay_episode.py <episode> --config experiment.yaml` | 真机 |
| Policy rollout | `python examples/run_policy.py <policy/task/experiment> --config experiment.yaml` | 真机；evaluation session |
| Camera calibration | `python examples/calibrate_camera.py --config experiment.yaml ...` | 真机；标定 |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | VR / HTS |
| Canonical Zarr export | `python examples/export_policy_zarr.py episodes/<task>` | 离线 |
| XHand diagnostics | `python examples/xhand_diagnostics.py` | 连接真机；诊断 |
| Inspect raw | `python examples/visualize_episode.py <episode> --info` | 离线 |

完整参数以各入口 `--help` 为准。

## Teleop 与录制

Teleop 默认录制，显式无录制调试使用 `--no-record`。

| 按键 | 操作 |
|---|---|
| B | 开始 |
| C | 暂停 / 恢复 |
| S | 停止并保存 |
| D | 丢弃 |
| H | Return-home |
| Q | 退出 |
| ESC | 急停 |

暂停撤销当前动作权限，恢复时从新鲜的机器人与 VR 观测重新锚定。发生暂停或控制异常的 episode 可保留用于诊断，但不作为 clean training demonstration 导出。

录制依赖 camera 与 RecorderIO；它们失败时不会静默降级为无录制运行。Teleop 通过 `policy.max_record_duration_s` 配置正常 episode 预算，policy eval 通过 `--max-duration` 配置。录制预算必须严格小于 Recorder 的 hard guard，启动前会校验；过长时应缩短 episode，而不是提高资源保护上限。

Raw episode 关闭、验证后才原子发布。启动时若发现 `.tmp_<episode_name>` 残留目录，只会给出 warning：它们不是已发布的 raw episodes，可能来自录制中断，需人工检查处理，不会自动删除或恢复。

## 运动安全边界

正常控制发送最新的绝对目标；`run_id` 防止暂停、停止或超时后的旧动作重新执行。记录的 action 表示高层目标，不承诺设备已消费或物理到达。

xArm 依赖目标限位和 SDK/controller 的速度、加速度控制；XHand 直接接收经过限位检查的绝对位置目标。

Cartesian IK 检查目标姿态的机器人自碰撞，并使用最终手部目标或新鲜实测手姿态。`--no-hand` 要求手已拆除或固定在配置的 home 姿态。

正常 teleop / eval / replay 不检查当前位置到目标的完整运动路径或环境碰撞。完整路径与环境避碰由 planned return-home 负责，实验仍需操作者现场监督。

xArm HOME 以实测收敛判断完成；XHand HOME 的 SDK 发送成功不代表物理到位。后续机械臂规划使用新鲜实测手姿态，实际手部运动与机械臂路径仍需真机验证。

## 数据与训练缓存

Raw episode 是实验 source of truth，保留机器人状态、RGB-D 与标定、XHand 电流/触觉及其有效性、VR 源数据、真实时间戳、绝对动作目标和必要诊断。控制与录制复用当前观测快照，各模态按自身时间戳检查新鲜度。

Canonical Zarr 是从 raw 重建的全量训练缓存，包含 learning-relevant modalities；`dexmani_policy` 在加载时选择模型输入。点云、FK 和 fingertips 在离线转换时生成，Zarr 不保存 runtime timing arrays 或 validity masks。`action` 与 `action_ee` 表示同一个最终目标，分别使用关节和末端空间描述。

Raw-to-Zarr 按完整 episode 接收或拒绝，不修复、切分、重采样或删除坏行。暂停、异常控制行、触觉无效、录制不完整或转换失败等情况会整条拒收并报告原因。批量导出可继续处理其他正常 episodes，但必须汇总拒绝原因。

## Policy evaluation

Policy artifact 定义模型输入、关节顺序、动作周期和训练时的预处理；当前 Real 配置提供现场标定与硬件几何。RGB 预处理由 Policy 负责，点云使用 artifact 配置与当前标定。模型输入语义保持一致，重新标定不要求重新训练。

Policy worker 持有 model / CUDA，使用同步 inference 和本地 action chunk。推理期间动作权限已失效时，丢弃返回的旧动作。Pointcloud-only policy 不依赖源 RGB-D 帧仍驻留；同时输入 RGB 和点云时才匹配源帧。

Evaluation 数据用于评估与诊断，不能直接当作 teleop BC demonstration 导入训练缓存。任务成功与否需离线判断。

## 开发与离线检查

开发约束见 [AGENTS.md](AGENTS.md)。源码、schema 和 resolved configuration 定义实现行为，README 只保留稳定工作流与操作约定。

纯逻辑修改使用一次性离线 smoke checks；仓库不维护 committed tests 目录，不运行真机入口作为测试。

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

部署相关修改可在已安装依赖的环境中运行离线回归：

    python -m dexmani_real.deployment.smoke_test

可选工具缺失时报告未完成的检查，不为检查安装或升级实验环境依赖。涉及实际运动、急停、暂停恢复、传感器失效和 shutdown 的行为，需另行授权真机验证。
