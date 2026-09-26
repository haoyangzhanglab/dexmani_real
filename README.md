# DexMani Real

面向个人 PhD 灵巧操作实验的真实机器人代码，用于 demonstration 采集、learned policy 部署与真机 evaluation。硬件为 xArm7、XHand、RealSense RGB-D 和 VR / HTS；模型训练与 policy implementation 位于相邻的 `dexmani_policy` 仓库。

本仓库会控制真实硬件。运行前检查机械限位、实验环境、标定与急停，并保持操作者在场。真机操作需明确授权；离线检查通过不代表真机安全验证通过。

## 安装与配置

要求 Python >= 3.10。在实验环境中安装本仓库，以下命令均从仓库根目录执行：

    python -m pip install -e .

硬件 SDK、运动学/点云库和 `dexmani_policy` 依赖按实验机环境安装。普通 import 和配置解析不应连接设备；采集、部署和回放工作流由对应 worker 建立连接，诊断脚本在显式启动后连接设备。

支持 `--config` 的入口按 CLI > YAML > [ExperimentConfig 默认值](dexmani_real/config/experiment.py) 解析，每次解析创建独立配置。连接设备前可查看采集配置：

    python examples/collect_teleop.py --print-config
    python examples/collect_teleop.py --config experiment.yaml --print-config

已接受的相机、桌面和 VR 标定状态保存在 `dexmani_real/calibration/state/`，由对应的标定入口显式更新。

相机标定中，SPACE 采样，ENTER 求解并仅保存通过质量检查的结果，R 执行 planned HOME，Q 结束。退出是否成功还取决于 worker 正常停止与 DISARMED 检查。

Teleop、键盘控制和 policy 分别使用各自的控制周期，replay 使用录制频率。动作生产频率不得高于相关执行器 worker 的服务频率；worker 频率不等于设备物理伺服频率。

## 常用入口

| 工作流 | 命令 | 说明 |
|---|---|---|
| VR collection | `python examples/collect_teleop.py --config experiment.yaml --task-name <task> --operator <name>` | 真机；raw episode |
| Keyboard jog / home | `python examples/keyboard_teleop.py --config experiment.yaml` | 真机 |
| Physical replay | `python examples/replay_episode.py <episode> --config experiment.yaml` | 真机 |
| Policy rollout | `python examples/run_policy.py <policy/task/experiment> --config experiment.yaml` | 真机；evaluation session |
| Camera calibration | `python examples/calibrate_camera.py --config experiment.yaml --hand-geometry absent` | 真机；仅在手已拆除时用 `absent`，固定在 home 的手用 `secured-home` |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | VR / HTS |
| Canonical Zarr export | `python examples/export_policy_zarr.py episodes/<task>` | 离线 |
| XHand diagnostics | `python examples/xhand_diagnostics.py` | 连接真机；诊断 |
| RGB-D diagnostics | `python examples/realsense_record_example.py` | 连接相机；实时 RGB-D / 点云显示 |
| Point-cloud diagnostics | `python examples/pointcloud_process_example.py` | 连接相机；点云检查与可选桌面标定，`--save-dir` 保存诊断快照 |
| Inspect raw | `python examples/visualize_episode.py <episode> --info` | 离线 |

带参数解析的入口可通过 `--help` 查看完整参数。`xhand_diagnostics.py` 和 `realsense_record_example.py` 不支持 `--help`，直接运行会进入硬件诊断流程。

键盘控制使用 WASD/方向键和 IJKL 微调，R 执行 planned HOME，Q 退出，ESC 急停。下节的 B/C/S/D/H 按键用于 VR teleop。

物理回放只接受当前 raw schema 的 teleop episode；加载时拒绝失败控制行、空轨迹或非有限的动作/机器人状态。启动前需在操作者监督下将机器人置于录制起始姿态附近：机械臂和手部每个关节的误差均不得超过 10°，机械臂按限位内等价角比较。回放按录制频率逐条发布目标，Q 停止并保留已采集数据，结束后的 H 执行机器人 planned HOME。默认输出为 `replay_results/<episode_name>_replay/`，`--output` 必须指向不存在或为空的目录；采集到数据后保存 `replay_data.npz` 并计算一致性指标 `metrics.json`。

## Teleop 与录制

Teleop 默认录制，显式无录制调试使用 `--no-record`。无手调试需同时使用 `--no-hand` 并禁用录制，且手必须已拆除或固定在配置的 home 姿态。

| 按键 | 操作 |
|---|---|
| B | 开始 |
| C | 暂停 / 恢复 |
| S | 停止并保存 |
| D | 丢弃 |
| H | 停止并保存当前录制，然后 Return-home |
| Q | 请求退出；录制中先暂停并等待保存/丢弃选择 |
| ESC | 急停 |

录制中按 Q 后，可用 S 保存、D 丢弃，或 H 保存并回 home，然后退出。超过 `policy.quit_save_timeout_s` 未选择时，尝试保存为异常 episode 后退出。

暂停撤销当前动作权限，恢复时从新鲜的机器人与 VR 观测重新锚定。发生暂停或控制异常的 episode 可保留用于诊断，但不作为 clean training demonstration 导出。

录制依赖相机与录制 worker。已启动的必需 worker 意外退出或必需录制资源失效时，会撤销动作权限并结束会话，CLI 返回非零。相机、VR、点云、录制或 policy 进程失败属于实验失败；arm/hand 故障、急停或无法确认子进程停止按物理 FAULT 处理。

Teleop 通过 `policy.max_record_duration_s` 配置正常 episode 预算，policy eval 通过 `--max-duration` 配置。录制预算必须严格小于录制器的帧数硬上限，启动前会校验；过长时应缩短 episode。

## 运动安全边界

正常控制发送最新的绝对目标；`run_id` 防止暂停、停止或超时后的旧动作重新执行。记录的 action 表示高层目标，不承诺设备已消费或物理到达。

xArm 依赖目标限位和 SDK/controller 的速度、加速度控制；XHand 直接接收经过限位检查的绝对位置目标。XHand 正常撤权时使用新鲜实测姿态保持，反馈过期时回退到 passive 模式；急停与退出时也请求 passive 模式。

Cartesian IK 检查目标姿态的机器人自碰撞，并使用最终手部目标或新鲜实测手姿态。`--no-hand` 要求手已拆除或固定在配置的 home 姿态。

正常 teleop / eval / replay 不检查当前位置到目标的完整运动路径或环境碰撞。完整路径与环境避碰由 planned return-home 负责，实验仍需操作者现场监督。

xArm 与 XHand HOME 均以实测收敛判断完成。XHand 默认要求连续三个不同的新鲜反馈样本中，每个关节的目标误差均不超过 5°；命令提交与收敛共用超时预算，默认 2 秒。后续机械臂规划使用新鲜实测手姿态，实际手部运动与机械臂路径仍需真机验证。

## 数据与训练缓存

Raw episode 是实验 source of truth，保留机器人状态、RGB-D 与标定、XHand 电流/触觉及其有效性、VR 源数据、真实时间戳、绝对动作目标和必要诊断。控制与录制复用当前观测快照，各模态按自身时间戳检查新鲜度。

每个已发布 episode 包含 `data.h5`（控制行与元数据）、`depth.h5`（对齐到彩色图像的原始深度）和 `rgb.mp4`。深度值乘以相机提供的 `depth_scale` 才得到米。当前 raw schema 为 33，读取时要求版本完全匹配；旧布局需在仓库外显式迁移。

Raw 先写入 `.tmp_<episode_name>`，关闭文件并通过结构验证后，才经 fsync 和原子重命名持久化发布。录制失败时释放 writer 并删除本次 staging；清理失败只记录错误并留下原目录，不覆盖原始录制异常。启动时发现残留 staging 只警告，需人工检查处理，不自动删除或恢复。

Canonical Zarr 是从 raw 重建的全量训练缓存，包含 learning-relevant modalities；`dexmani_policy` 在加载时选择模型输入。点云、FK 和 fingertips 在离线转换时生成，Zarr 不保存 runtime timing arrays 或 validity masks。`action` 与 `action_ee` 表示同一个最终目标，分别使用关节和末端空间描述。

Raw-to-Zarr 按完整 episode 接收或拒绝，不修复、切分、重采样或删除坏行。暂停、异常控制行、触觉无效、录制不完整或转换失败等情况会整条拒收并报告原因。批量导出可继续处理其他正常 episodes，但必须汇总拒绝原因。

导出默认写入 `datasets/<task_name>.zarr`，可用 `--output` 指定新目标。`--dry-run` 执行相同的完整转换与校验而不创建 Zarr，仍需相应的运动学和点云依赖。

Zarr 在目标父目录下写入 staging，完成后再次确认目标未占用，再在同一文件系统内重命名发布。它是可从 raw 重建的缓存，不逐块 fsync；导出失败时清理本次 staging，raw 保持不变。

导出拒绝覆盖已有目标；解析符号链接后，目标不能位于输入目录、仓库的 `episodes/`、`episodes_processed/`、`rollouts/` 或已有 Zarr 内部。`episodes_processed/` 仅作为历史数据保护目录保留，当前流程直接从 raw 生成 Zarr。dry-run 同样检查目标路径。单个异常 episode 或没有可接收 episode 时返回失败。

## Policy evaluation

Policy artifact 定义模型输入、关节顺序、动作周期和训练时的预处理；当前 Real 配置提供现场标定与硬件几何。RGB 预处理由 Policy 负责，点云使用 artifact 配置与当前标定。模型输入语义保持一致，重新标定不要求重新训练。

Policy worker 持有 model / CUDA，使用同步 inference 和本地 action chunk。推理期间动作权限已失效时，丢弃返回的旧动作。Pointcloud-only policy 不依赖源 RGB-D 帧仍驻留；同时输入 RGB 和点云时才匹配源帧。

操作顺序为 H 回到初始姿态、布置场景、B 开始、S 停止；Q 退出，ESC 急停。HOME 完成后需要新的 B 才能开始；HOME 阻塞期间 S/Q 仍会立即撤销动作权限。`--num-episodes` 按实际开始并结束的 episode 计数，录制失败的 episode 也计入预算。

会话输出位于 `rollouts/<policy>/<task>/<experiment>/session_*/`，包含 `run_config.yaml` 和实际保存的 episode 目录。Policy 模式不使用 C/D；同批收到 S/Q 时会忽略 H/B。

会话执行结果由 CLI 退出码表示：正常完成或操作者退出且资源干净关闭时为 0；必需 worker/operator 异常、物理故障、急停、非正常 shutdown 或共享内存关闭失败时为非零。共享内存仅在确认全部子进程停止后释放。Policy 的未完成 active episode 若被异常中断，会按技术无效结束。

Evaluation 数据用于评估与诊断，不能直接当作 teleop BC demonstration 导入训练缓存。任务成功与否需离线判断。

## 开发与离线检查

开发约束见 [AGENTS.md](AGENTS.md)。源码、schema 和 resolved configuration 定义实现行为，README 只保留稳定工作流与操作约定。

纯逻辑修改使用一次性离线 smoke checks；仓库不维护 committed tests 目录，不运行真机入口作为测试。

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

可选工具缺失时报告未完成的检查，不为检查安装或升级实验环境依赖。涉及实际运动、急停、暂停恢复、传感器失效和 shutdown 的行为，需另行授权真机验证。
