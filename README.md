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
| VR collection | `python examples/collect_teleop.py --config experiment.yaml --task-name <task>` | 真机；raw episode |
| Keyboard jog / home | `python examples/keyboard_teleop.py --config experiment.yaml` | 真机 |
| Physical replay | `python examples/replay_episode.py <episode> --config experiment.yaml` | 真机 |
| Policy rollout | `python examples/run_policy.py <policy/task/experiment> --config experiment.yaml` | 真机；evaluation session |
| Camera calibration | `python examples/calibrate_camera.py --config experiment.yaml --hand-geometry absent` | 真机；仅在手已拆除时用 `absent`，固定在 home 的手用 `secured-home` |
| VR heading calibration | `python examples/calibrate_vr_heading.py` | VR / HTS |
| Canonical Zarr export | `python examples/export_policy_zarr.py <raw_task_dir>` | 离线；输入为包含当前 Raw episodes 的任务目录 |
| XHand diagnostics | `python examples/xhand_diagnostics.py` | 连接真机；诊断 |
| RGB-D diagnostics | `python examples/realsense_record_example.py` | 连接相机；实时 RGB-D / 点云显示 |
| Point-cloud diagnostics | `python examples/pointcloud_process_example.py` | 连接相机；点云检查与可选桌面标定，`--save-dir` 保存诊断快照 |
| Inspect raw | `python examples/visualize_episode.py <episode> --info` | 离线 |

带参数解析的入口可通过 `--help` 查看完整参数。`xhand_diagnostics.py` 和 `realsense_record_example.py` 不支持 `--help`，直接运行会进入硬件诊断流程。

键盘控制使用 WASD/方向键和 IJKL 微调，R 执行 planned HOME，Q 退出，ESC 急停。下节的 B/C/S/D/H 按键用于 VR teleop。

物理回放只接受当前 `format="dexmani.raw"` 的 teleop episode；加载时拒绝空轨迹或非有限的动作/机器人状态。启动前需在操作者监督下将机器人置于录制起始姿态附近：机械臂和手部每个关节的误差均不得超过 10°，机械臂按限位内等价角比较。回放按录制频率逐条发布目标，Q 停止并保留已采集数据，结束后的 H 执行机器人 planned HOME。默认输出为 `replay_results/<episode_name>_replay/`，`--output` 必须指向不存在或为空的目录；采集到数据后保存 `replay_data.npz` 并计算一致性指标 `metrics.json`。

## Teleop 与录制

Teleop 默认录制，显式无录制调试使用 `--no-record`，此时不启动相机 worker 或录制 writer。无手调试需同时使用 `--no-hand` 并禁用录制，且手必须已拆除或固定在配置的 home 姿态。

| 按键 | 操作 |
|---|---|
| B | 开始 |
| C | 暂停 / 恢复 |
| S | 停止并保存 |
| D | 丢弃 |
| H | 停止并保存当前录制，然后 Return-home |
| Q | 请求退出；录制中先暂停并等待保存/丢弃选择 |
| ESC | 急停 |

录制中按 Q 后，可用 S 保存、D 丢弃，或 H 保存并回 home，然后退出。超过 `policy.quit_save_timeout_s` 未选择时，丢弃 capture 后退出。

C 暂停或传感器失效触发的暂停会撤销动作权限，并将当前 capture 标为 discard-only。恢复时从新鲜机器人与 VR 观测重新锚定，只恢复控制，不重启或分段录制；此后不再接收录制行。S/H/Q 收尾时会丢弃该 capture 并明确记录原因，不能误认为已保存 demonstration。控制/发布失败、连续性中断和异常结束同样丢弃 teleop capture。正常 Q 后等待保存/丢弃选择本身不使 clean capture 失效。

录制依赖相机 worker 和控制进程内的单一 writer 线程。控制步复用当前观测的 RGB-D，通过容量为 16 行的本地 FIFO 非阻塞提交；不再复制整帧 RGB-D 到录制共享内存。writer 线程从打开到关闭独占 HDF5/PyAV，负责编码、写盘和发布；小型数值数组不压缩，深度仍使用 gzip level 1。START 等待 writer 就绪后才允许动作，保存/丢弃只在撤销动作权限后阻塞等待。

队列满或 writer 异常属于硬录制失败，不静默丢行；控制循环检测到后撤销动作权限并结束会话，CLI 返回非零。已启动的必需 worker 意外退出同样结束会话。相机、VR、点云、录制或 policy 失败属于实验失败；arm/hand 故障、急停或无法确认子进程停止按物理 FAULT 处理。

Teleop 通过 `policy.max_record_duration_s` 配置正常 episode 预算，policy eval 通过 `--max-duration` 配置。录制预算必须严格小于录制器的帧数硬上限，启动前会校验；过长时应缩短 episode。

## 运动安全边界

正常控制发送最新的绝对目标；`run_id` 防止暂停、停止或超时后的旧动作重新执行。记录的 action 表示高层目标，不承诺设备已消费或物理到达。

xArm 依赖目标限位和 SDK/controller 的速度、加速度控制；XHand 直接接收经过限位检查的绝对位置目标。XHand 正常撤权时使用新鲜实测姿态保持，反馈过期时回退到 passive 模式；急停与退出时也请求 passive 模式。

Cartesian IK 检查目标姿态的机器人自碰撞，并使用最终手部目标或新鲜实测手姿态。`--no-hand` 要求手已拆除或固定在配置的 home 姿态。

正常 teleop / eval / replay 不检查当前位置到目标的完整运动路径或环境碰撞。完整路径与环境避碰由 planned return-home 负责，实验仍需操作者现场监督。

xArm 与 XHand HOME 均以实测收敛判断完成。XHand 默认要求连续三个不同的新鲜反馈样本中，每个关节的目标误差均不超过 5°；命令提交与收敛共用超时预算，默认 2 秒。后续机械臂规划使用新鲜实测手姿态，实际手部运动与机械臂路径仍需真机验证。

会话结束时，确认全部子进程停止且没有物理故障后，安全状态变为 DISARMED。相机、VR、点云、录制或 policy 失败，以及共享内存关闭失败，仍会导致会话失败；DISARMED 不表示实验成功。arm/hand 异常退出、急停或已锁存的物理故障保持 FAULT；无法确认子进程停止时也进入 FAULT，并保留共享内存。

## 数据与训练缓存

当前 Raw 是不可变的实验 source of truth，不做原地改写：episode 在 `data.h5` 保存控制行、aligned Z16 深度和 metadata，RGB 使用 `rgb.mp4`；metadata 使用 `format="dexmani.raw"` 和最终 `termination_reason`，不保存 validity 字段或全局递增版本。正常 runtime / Reader / replay / export 只支持当前格式，不保留 legacy Raw compatibility branch。Reader 验证已有的已知字段，忽略额外未知字段；消费者通过 `require_fields(...)` 要求自己所需的能力。旧 Raw 格式不属于当前支持范围，仓库不再提供 v34 迁移 CLI；需要时应在仓库外或历史版本中单独处理，不向当前生产路径加入兼容逻辑。

控制与录制复用同一观测快照。Raw 保留机器人状态、RGB-D、XHand 电流/触觉、实际发布的绝对关节目标和真实物理标定；不保存 VR、逐帧时间戳、transport bookkeeping 或软件 provenance。深度值乘以 `depth_scale` 才得到米。

`action_arm_joint_target` / `action_hand_joint_target` 的有限值表示本控制步确实发布的目标，NaN 表示没有发布相应动作。不得用前一条目标补齐未发布动作；实际发布的 hold 必须保留。Teleop 只发布完整、连续且成功发布动作的 demonstration。Policy rollout 是评估证据：普通 IK 不可执行步仍保存观测和 NaN-action 行；健康的录制前缀可在异常结束时连同 termination reason 保存，不能为了排除失败结果而删除 rollout。

XHand 触觉保留 SDK-native、去软件 bias 后的数值，SI 牛顿单位尚未验证。Raw 和导出均不额外缩放；aggregate/dense 缺测分别使用 NaN，运行时 validity 仍相互独立。Raw 在 START 快照当前 resolved hand mount；离线 FK 只读取 Raw 安装参数。

START 必须用当前相机 serial 解析完整 eye-to-hand 标定，验证 aligned RGB-D 几何、depth scale 和 base-from-color 变换。缺失或 eye-in-hand 均拒绝录制。彩色网格畸变模型必须受 canonical 反投影支持：当前仅接受 `none` / `distortion.none` 和 `brown_conrady` / `distortion.brown_conrady`。

Raw 先写入 `.tmp_<episode_name>`；writer 关闭文件并结构验证后，经 fsync 和原子重命名发布。失败时仅清理本次拥有的 staging；无法确认释放或清理失败时保留 ownership 并拒绝新 START，原始错误不被覆盖。残留 staging 只警告，需人工检查，不自动恢复或删除。

Canonical Zarr 使用 `format="dexmani.real.canonical"`，保持完整多模态缓存：`joint_state`、`arm_qvel`、`arm_effort`、`hand_current`、`eef_pose`、`fingertip_points`、`contact_force`、`tactile_force`、`rgb`、`depth`、`point_cloud`、`action`、`action_ee`。Root attrs 只保存 format/task/dt；单位、坐标系、顺序和 semantic ID 位于各数组 attrs，派生模态保存实际 recipe。一个 Zarr 的 Z16 depth 必须使用唯一 `scale_m_per_unit`，不混合不同深度尺度。

`action` 与 `action_ee` 来自同一个最终发布目标，后者由最终机械臂关节目标 FK 得到。Fingertip recipe 保存模型身份、link 列表及 Raw mount 来源，不记录工作站绝对 URDF 路径。Point-cloud attrs 保存数值 recipe 和离线使用的 audited table plane；保存到 policy runtime 的契约不携带历史桌面标定，部署仍使用当前 Real 标定。

点云训练和部署只传递数值 recipe，不匹配 semantic ID 或 derivation 说明；保留 shape/dtype、有限性、点数及数值参数合法性检查。实时与离线调用同一个点云处理函数，部署恢复训练参数，其他模态仍检查其语义契约。

Raw-to-Canonical 在写入前预检所有候选 episode：空 episode、必需文件/字段缺失、Raw 数据形状/dtype 或必要元数据错误，以及任一必需浮点数组中的 NaN/Inf，均整条拒绝并跳过，包括触觉、电流和 effort 缺测。日志报告拒绝原因和首个异常行（从 0 开始）；输出目录中的 `export_report.json` 保存接受、自动拒绝和显式排除清单。全部被拒绝时不生成 Zarr。Raw 中的 NaN 仍保留为缺测证据，不修改原数据；不补值、切分、重采样或删帧。

不支持的 Raw 格式/相机畸变模型、非 teleop 来源、跨 episode 契约不兼容、I/O 错误，以及 RGB 解码、FK、点云生成和 Zarr 写入错误仍终止整个导出并清理 staging。导出要求所有浮点输出有限，不使用动作幅度、触觉大小或任务成败等经验阈值筛选。`dexmani_policy` 继续只加载选中的数组，并在 normalizer 拟合和模型构造前拒绝其中的 NaN/Inf。

导出默认使用项目处理参数和当前桌面标定 `dexmani_real/calibration/state/table_plane.json`，将实际使用的平面参数保存到点云 export provenance。可用 `--config export.yaml` 覆盖处理参数；指定其他桌面平面时，在 YAML 中设置 `environment.table.plane_abcd` 和 `plane_path: null`，或用 `pointcloud.remove_table: false` 关闭桌面剔除。启用桌面剔除时，标定文件缺失或无效会使导出失败。

```bash
python examples/export_policy_zarr.py episodes/<task>
# 人工筛选只排除完整 episode，可重复 --exclude；不修改 Raw 或 task 名称。
python examples/export_policy_zarr.py episodes/<task> --output datasets/<task>_curated.zarr --exclude episode_unwanted
```

导出不再提供 `--dry-run`、task-name override 或 annotation rewriting。真实导出通过 owned staging 完成相同校验，失败不会留下已发布的部分 Zarr。拒绝覆盖已有目标；目标不能位于输入、仓库 `episodes/`、`episodes_processed/`、`rollouts/` 或已有 Zarr 内部。

Canonical 是可重建派生缓存，只支持当前 `format="dexmani.real.canonical"`。字段名和 semantic ID 的含义不可静默改变；不同表示使用新的描述性身份，算法选择记录实际 recipe。新增字段不影响不请求它的消费者。旧缓存直接从当前 Raw 重新导出，不维护格式兼容分支。

## Policy evaluation

保存的 Policy 实验 `config.yaml` 定义模型、输入模态、动作周期、归一化模式与推理预处理；普通训练 checkpoint 提供 raw/EMA 权重和 fitted normalizer。不需要 deployment export。当前 Real 配置提供现场标定与硬件几何，点云采用保存的数值 recipe；RGB deterministic resize/center crop 由 Policy 执行。

```bash
conda run --no-capture-output -n real_robot python examples/run_policy.py <experiment_dir> --checkpoint best
conda run --no-capture-output -n real_robot python examples/run_policy.py <policy/task/run> --checkpoint latest --weights raw --inference-steps 10
```

`--checkpoint` 默认为 best（实验根目录 `best_ckpt.json`），也可选 latest 或 checkpoints 中的文件名。`--weights ema|raw` 与 `--inference-steps` 显式覆盖保存的选择；不存在的 best/EMA 会报错。需要 hand_enabled=true 和训练时保存的非空 real_runtime。Parent 不加载模型权重；完成 pure preflight 后创建 session/SHM，policy restore/warmup 就绪后才启动硬件 workers。

`--config experiment.yaml` 只设置当前 Real 硬件与运行参数；Policy 始终读取所选实验目录里的 `config.yaml`。未显式覆盖的 weights/NFE，best 使用 selection record，latest/文件名使用保存的 `eval.use_ema` / `eval.inference_steps`。

Policy worker 持有 model / CUDA，使用同步 inference 和本地 action chunk。推理期间动作权限已失效时，丢弃返回的旧动作。Pointcloud-only policy 不依赖源 RGB-D 帧仍驻留；同时输入 RGB 和点云时才匹配源帧。保存的 `real_runtime.modality_contracts` 记录训练所用表示；指尖 FK 恢复保存的 link 列表并使用当前安装参数。Canonical 中存在某个模态不等于 live Real 已支持：当前未实现的 live 模态会在模型/硬件启动前明确拒绝。Real deployment 只支持由当前 canonical contract 训练并保存完整 `modality_contracts` 的实验。

操作顺序为 H 回到初始姿态、布置场景、B 开始、S 停止；Q 退出，ESC 急停。HOME 完成后需要新的 B 才能开始。B 每次只尝试一次：检查新鲜 arm/hand HOME 状态与完整 padded observation，recorder START 后再次检查，失败则丢弃准备中的录制并等待新的 B；HOME 阻塞期间 S/Q 仍会立即撤销动作权限。`--num-episodes` 按实际开始并结束的 episode 计数，录制失败的 episode 也计入预算。

每个实际开始的 episode 重置 Policy 固定 seed、EEF IK fallback RNG 和本地统计；结束时记录 timing/clipping/IK summary。`--max-duration` 从成功进入 RUNNING 起计时，是 cooperative runner budget；S/Q/ESC 与 run_id worker fence 负责撤销权限，包括推理阻塞期间的撤销。

会话输出位于 `rollouts/<policy>/<task>/<experiment>/session_*/`，包含 `run_config.yaml` 和实际保存的 episode 目录。Policy 模式不使用 C/D；同批收到 S/Q 时会忽略 H/B。

会话执行结果由 CLI 退出码表示：正常完成或操作者退出且资源干净关闭时为 0；必需 worker/operator 异常、物理故障、急停、非正常 shutdown 或共享内存关闭失败时为非零。正常 S/Q 允许 policy 完成本地 episode 结束与录制收尾；若尚未正常结束的 active episode 被外部关闭打断，即使已收到 S/Q，也以明确的异常 termination reason 保存健康录制前缀。

正常 Q 的 policy 收尾与后续 worker 停止共用 `safety.shutdown_timeout_s`，默认 65 秒；录制 STOP 等待默认 60 秒。录制收尾耗时会减少其他 worker 的剩余等待时间。

Evaluation 数据用于评估与诊断，不能直接当作 teleop BC demonstration 导入训练缓存。任务成功与否需离线判断。

## 开发与离线检查

开发约束见 [AGENTS.md](AGENTS.md)。源码、schema 和 resolved configuration 定义实现行为，README 只保留稳定工作流与操作约定。

纯逻辑修改使用一次性离线 smoke checks；仓库不维护 committed tests 目录，不运行真机入口作为测试。

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

可选工具缺失时报告未完成的检查，不为检查安装或升级实验环境依赖。涉及实际运动、急停、暂停恢复、传感器失效和 shutdown 的行为，需另行授权真机验证。
